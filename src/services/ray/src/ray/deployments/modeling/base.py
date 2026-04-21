import asyncio
import contextvars
import gc
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import ray
import torch
from accelerate import dispatch_model
from pydantic import BaseModel, ConfigDict
from ray import serve
from torch.amp import autocast
from torch.cuda import max_memory_allocated, memory_allocated, reset_peak_memory_stats

from transformers.modeling_utils import _get_device_map
from nnsight.modeling.mixins import RemoteableMixin
from nnsight.schema.request import RequestModel
from nnsight.modeling.mixins.remoteable import StreamTracer

# Per-call request binding. Concurrent __call__ invocations each set this
# on entry; legacy helpers (respond/log/stream_send) read from it so they
# see the right request without needing the signature changed. ContextVars
# propagate through asyncio.to_thread, so worker threads also see it.
_current_request: contextvars.ContextVar[Optional["BackendRequestModel"]] = (
    contextvars.ContextVar("_ndif_current_request", default=None)
)
from nnsight.modeling.language import LanguageModel
from nnsight.modeling.hf_serve.vanilla_server import VanillaBatchServer
from nnsight.intervention.tracing.globals import Globals
from ....types import MODEL_KEY
from ....logging import set_logger
from ....metrics import (
    ExecutionTimeMetric,
    GPUMemMetric,
    ModelLoadTimeMetric,
    RequestResponseSizeMetric,
)
from ....providers.objectstore import ObjectStoreProvider
from ....providers.socketio import SioProvider
from ....schema import BackendRequestModel, BackendResponseModel, BackendResultModel
from ...nn.backend import RemoteExecutionBackend
from ...nn.ops import StdoutRedirect
from ...nn.security.protected_objects import protect_persistent_objects
from ...nn.security.protected_environment import (
    WHITELISTED_MODULES,
    WHITELISTED_MODULES_DESERIALIZATION,
    Protector,
)
from .util import kill_thread, load_with_cache_deletion_retry, remove_accelerate_hooks


class BaseModelDeployment:
    def __init__(
        self,
        model_key: MODEL_KEY,
        cuda_devices: str,
        execution_timeout: float | None,
        dispatch: bool,
        dtype: str | torch.dtype,
        *args,
        extra_kwargs: Dict[str, Any] = {},
        **kwargs,
    ) -> None:
        super().__init__()

        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

        ObjectStoreProvider.connect()

        self.model_key = model_key
        self.execution_timeout = execution_timeout
        self.dispatch = dispatch
        self.dtype = dtype
        self.extra_kwargs = extra_kwargs

        self.cached = False

        self.logger = set_logger(model_key)

        self.runtime_context = ray.get_runtime_context()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        torch.set_default_dtype(torch.bfloat16)

        self.model = self.load_from_disk()

        # Thread-local sandbox. Used three ways:
        #   1. ``with self.execution_protector:`` — import whitelist on
        #      this thread only (non-LM execute_single path).
        #   2. ``self.execution_protector(target_globals)`` — factory per
        #      nnsight's ``worker_context`` contract: activates TLS and
        #      shadows risky builtins in the user frame's globals.
        #   3. ``pre()`` uses its own short-lived Protector for
        #      deserialization whitelist.
        self.execution_protector = Protector(WHITELISTED_MODULES)

        if dispatch:
            self.model._module.requires_grad_(False)

        torch.cuda.empty_cache()

        _force_sequential = os.getenv("NDIF_FORCE_SEQUENTIAL") == "1"

        if isinstance(self.model, LanguageModel) and not _force_sequential:
            # Sandbox user intervention code in each mediator worker
            # thread. The batch server's bg generation thread runs no
            # user code and is intentionally left unwrapped.
            self.batch_server = VanillaBatchServer(
                self.model,
                mediator_timeout=30.0,
                worker_context=self.execution_protector,
            )
            self.batch_server.start()
            self.model_lock = None
        else:
            self.batch_server = None
            self.model_lock = asyncio.Lock()

        self.thread_pool = ThreadPoolExecutor(max_workers=1)

        self.kill_switch = asyncio.Event()
        self.execution_ident = None

        # Build and protect the persistent-objects dict once. Walking the
        # envoy tree (hundreds of modules for 7B+ models) and wrapping each
        # reference costs O(modules) and would otherwise repeat per request.
        self._persistent_objects = protect_persistent_objects(
            self.model._remoteable_persistent_objects()
        )

        StreamTracer.register(self.stream_send, self.stream_receive)

    @property
    def request(self) -> Optional["BackendRequestModel"]:
        """Current request for the calling task/thread. Backed by a
        ContextVar set in ``__call__`` so concurrent requests don't alias."""
        return _current_request.get()

    @request.setter
    def request(self, value):
        # Legacy code (cleanup()) sets this to None between requests. With
        # the ContextVar the scope is already bounded by __call__, so this
        # is a no-op. Keeping the setter so old writes don't raise.
        pass

    def load_from_disk(self):
        start = time.time()
        torch.cuda.synchronize()
        self.logger.info(f"Loading model from disk for model key {self.model_key}...")

        model = load_with_cache_deletion_retry(
            lambda: RemoteableMixin.from_model_key(
                self.model_key,
                device_map="auto",
                dispatch=self.dispatch,
                torch_dtype=self.dtype,
                **self.extra_kwargs,
            )
        )
        torch.cuda.synchronize()
        load_time = time.time() - start

        ModelLoadTimeMetric.update(load_time, self.model_key, "disk")

        devices = set()

        for param in model._module.parameters():
            devices.add(f"{param.device.type}:{param.device.index}")

        self.logger.info(
            f"Model loaded from disk in {load_time} seconds on devices: {devices}"
        )

        return model

    async def to_cache(self):
        self.logger.info(f"Saving model to cache for model key {self.model_key}...")
        # torch.cuda.synchronize()
        await self.cancel()

        if self.batch_server is not None:
            self.batch_server.stop()

        remove_accelerate_hooks(self.model._module)

        self.model._module = self.model._module.cpu()
        # torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        self.cached = True

    def from_cache(self, cuda_devices: str):
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

        torch.cuda.synchronize()
        start = time.time()

        self.logger.info(f"Loading model from cache for model key {self.model_key}...")

        device_map = _get_device_map(self.model._module, "auto", None, None, None, None)

        remove_accelerate_hooks(self.model._module)

        self.model._module = dispatch_model(self.model._module, device_map)

        if self.batch_server is not None:
            # Thread was stopped in to_cache and cannot be restarted;
            # reinstate with the same worker_context so mediator workers
            # on the rehydrated server remain sandboxed.
            self.batch_server = VanillaBatchServer(
                self.model,
                mediator_timeout=30.0,
                worker_context=self.execution_protector,
            )
            self.batch_server.start()

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        load_time = time.time() - start

        devices = set()

        for param in self.model._module.parameters():
            devices.add(f"{param.device.type}:{param.device.index}")

        self.logger.info(
            f"Model loaded from cache in {load_time} seconds on devices: {devices}"
        )

        ModelLoadTimeMetric.update(load_time, self.model_key, "cache")

        self.cached = False

    async def __call__(self, request: BackendRequestModel) -> None:
        """Executes the model service pipeline:

        1.) Pre-processing
        2.) Execution
        3.) Post-processing
        4.) Cleanup

        Args:
            request (BackendRequestModel): Request.
        """

        if self.cached:
            raise LookupError("Failed to look up actor")

        token = _current_request.set(request)
        _rid = request.id[:8]
        _t0 = time.time()
        print(f"[BATCH-PROF t=0.000] __call__ ENTER rid={_rid}", flush=True)

        result = None
        try:
            _t_pre_start = time.time()
            # Offload pickle/unpickle to a worker thread so the event loop
            # stays free to admit other concurrent requests. ContextVar
            # propagates so helpers inside pre() still see the right
            # request via self.request.
            inputs = await asyncio.to_thread(self.pre, request)
            _t_pre_end = time.time()
            print(
                f"[BATCH-PROF t={_t_pre_end-_t0:.3f}] pre DONE rid={_rid} "
                f"pre_ms={(_t_pre_end-_t_pre_start)*1000:.0f}",
                flush=True,
            )

            if self.batch_server is not None:
                result = await self.execute_batched(request, inputs)
            else:
                async with self.model_lock:
                    result = await self._execute_single_with_timeout(inputs)

            _t_exec_end = time.time()
            print(
                f"[BATCH-PROF t={_t_exec_end-_t0:.3f}] execute DONE rid={_rid}",
                flush=True,
            )
            await asyncio.to_thread(self.post, request, result)

        except Exception as e:
            self.exception(request, e)

        finally:
            _current_request.reset(token)
            del result

            self.cleanup()

    async def _execute_single_with_timeout(self, inputs):
        """Fallback path timeout wrapper: run execute_single in a thread,
        race against kill_switch + execution_timeout. Preserves the
        existing kill_thread mechanism for non-LM models."""
        job_task = asyncio.create_task(asyncio.to_thread(self.execute_single, inputs))
        kill_task = asyncio.create_task(self.kill_switch.wait())

        done, pending = await asyncio.wait(
            [job_task, kill_task],
            timeout=self.execution_timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

        if job_task in done:
            return await job_task
        elif kill_task in done:
            kill_thread(self.execution_ident)
            raise Exception("Your job was cancelled or preempted by the server.")
        else:
            kill_thread(self.execution_ident)
            raise Exception(
                f"Job took longer than timeout: {self.execution_timeout} seconds"
            )

    async def cancel(self):
        if self.execution_ident is not None:
            self.kill_switch.set()

    # Ray checks this method and restarts replica if it raises an exception
    def check_health(self):
        pass

    ### ABSTRACT METHODS #################################

    def pre(self, request: "BackendRequestModel") -> RequestModel:
        """Deserialize the request body into a RequestModel.

        Runs off the event loop via ``asyncio.to_thread`` so concurrent
        calls don't serialize on the unpickle step. Reads only
        ``self._persistent_objects`` (immutable after __init__), so it's
        safe to run on any worker thread.
        """
        _t_start = time.perf_counter()
        _rid = request.id[:8]

        raw = request.request
        _t_ray_start = time.perf_counter()
        if isinstance(raw, ray.ObjectRef):
            raw = ray.get(raw)
        _t_ray_done = time.perf_counter()

        _t_unpickle_start = time.perf_counter()
        with Protector(WHITELISTED_MODULES_DESERIALIZATION):
            req_model = RequestModel.deserialize(
                raw, self._persistent_objects, request.zlib
            )
        _t_unpickle_done = time.perf_counter()

        _t_respond_start = time.perf_counter()
        self._respond(
            request,
            status=BackendResponseModel.JobStatus.RUNNING,
            description="Your job has started running.",
        )
        _t_respond_done = time.perf_counter()

        print(
            f"[BATCH-PROF pre_detail rid={_rid} "
            f"total_ms={(_t_respond_done-_t_start)*1000:.0f} "
            f"ray_get_ms={(_t_ray_done-_t_ray_start)*1000:.0f} "
            f"unpickle_ms={(_t_unpickle_done-_t_unpickle_start)*1000:.0f} "
            f"respond_ms={(_t_respond_done-_t_respond_start)*1000:.0f}]",
            flush=True,
        )
        return req_model

    def execute_single(self, request: RequestModel) -> Any:
        """Fallback execution path for non-LanguageModel deployments.
        Runs the full forward pass on the calling thread with the
        Protector sandbox active around RemoteExecutionBackend.

        Args:
            request (RequestModel): Deserialized request.

        Returns:
            (saves, gpu_mem, execution_time) tuple.
        """

        self.execution_ident = threading.current_thread().ident

        with autocast(device_type="cuda", dtype=torch.get_default_dtype()):
            if torch.cuda.is_available():
                reset_peak_memory_stats()
                model_memory = memory_allocated()

            execution_time = time.time()

            # Execute object.
            with StdoutRedirect(self.log):
                result = RemoteExecutionBackend(
                    request.interventions, self.execution_protector
                )(request.tracer)

            execution_time = time.time() - execution_time

            # Compute GPU memory usage
            if torch.cuda.is_available():
                gpu_mem = max_memory_allocated() - model_memory
            else:
                gpu_mem = 0

        return result, gpu_mem, execution_time

    def _compile_trace(self, request_model: RequestModel) -> list:
        """Atomic compile + entries-build. Called from a worker thread via
        asyncio.to_thread. Safe for concurrent invocation because:

          * ``init_interleaver=False`` — ``_setup_interleaver`` does NOT
            reset ``model._interleaver`` state (shared with the bg thread).
          * ``mediators=request_model.tracer.mediators`` — reads from the
            tracer's own per-request list, not the shared interleaver.

        Matches the pattern in nnsight/modeling/hf_serve/api/server.py:/v1/nnsight/generate.
        """
        try:
            Globals.enter()
            _args, kwargs = request_model.tracer._setup_interleaver(
                request_model.interventions, init_interleaver=False
            )
            entries = self.batch_server.build_entries(
                kwargs, mediators=request_model.tracer.mediators
            )
            request_model.tracer.mediators.clear()
            return entries
        finally:
            Globals.exit()

    async def execute_batched(
        self,
        request: "BackendRequestModel",
        request_model: RequestModel,
    ) -> Any:
        """Batched execution path for LanguageModel deployments.

        Compiles the trace in a worker thread (unblocking the event loop
        so concurrent requests can compile in parallel), submits each
        invoke to the shared VanillaBatchServer, and awaits all per-invoke
        futures. Cross-request batching is achieved by the batch server
        mixing concurrent traces' tokens in one forward pass.

        Args:
            request: The frontend request (for logging / response channel).
            request_model: Deserialized nnsight RequestModel.

        Returns:
            (merged_saves, 0, execution_time) tuple. gpu_mem is always
            0 on the batched path (shared-GPU metric is not meaningful).
        """

        start = time.time()
        _rid = request.id[:8]
        print(f"[BATCH-PROF eb_entry rid={_rid}]", flush=True)

        # Compile off the event loop. When this returns, we have the
        # per-invoke entries ready to submit.
        _t_compile_start = time.time()
        entries = await asyncio.to_thread(self._compile_trace, request_model)
        _t_submit_start = time.time()
        # submit_async MUST run on the event loop — it binds the returned
        # Future to asyncio.get_running_loop(). Microseconds per call.
        futures = [self.batch_server.submit_async(e) for e in entries]
        _t_submitted = time.time()
        print(
            f"[BATCH-PROF eb_sync_done rid={_rid} "
            f"compile_ms={(_t_submit_start-_t_compile_start)*1000:.0f} "
            f"submit_ms={(_t_submitted-_t_submit_start)*1000:.0f}]",
            flush=True,
        )

        per_invoke_saves = await asyncio.gather(*futures)
        _t_gather_done = time.time()
        print(
            f"[BATCH-PROF eb_gather_done rid={_rid} "
            f"gather_ms={(_t_gather_done-_t_submitted)*1000:.0f}]",
            flush=True,
        )

        merged: dict = {}
        for saves in per_invoke_saves:
            if not saves:
                continue
            if "__error__" in saves:
                raise RuntimeError(saves["__error__"])
            merged.update(saves)

        execution_time = time.time() - start
        return merged, 0, execution_time

    def post(self, request: "BackendRequestModel", result: Any) -> None:
        """Save the result, notify the client, record metrics.

        Takes ``request`` explicitly so it's safe under concurrency and
        can run off the event loop via ``asyncio.to_thread`` (the boto3
        upload inside ``BackendResultModel.save`` is a blocking socket
        call and should not hold the event loop).
        """

        saves = result[0]
        gpu_mem: int = result[1]
        execution_time_s: float = result[2]

        result_object = BackendResultModel(
            id=request.id,
            **saves,
        ).save()

        self._respond(
            request,
            status=BackendResponseModel.JobStatus.COMPLETED,
            description="Your job has been completed.",
            data=(result_object.url(), result_object._size),
        )

        RequestResponseSizeMetric.update(request, result_object._size)
        GPUMemMetric.update(request, gpu_mem)
        ExecutionTimeMetric.update(request, execution_time_s)

    def exception(self, request: "BackendRequestModel", exception: Exception) -> None:
        """Handles exceptions that occur during model execution."""

        description = traceback.format_exc()
        self._respond(
            request,
            status=BackendResponseModel.JobStatus.ERROR,
            description=f"{description}\n{str(exception)}",
        )

        # Special handling for CUDA device-side assertion errors
        if "device-side assert triggered" in str(exception):
            self.restart()

    def restart(self):
        """Restarts the Ray serve deployment in response to critical errors.

        This is typically called when encountering CUDA device-side assertion errors
        or other critical failures that require a fresh replica state.
        """
        ray.kill(
            ray.get_actor(f"ModelActor:{self.model_key}", namespace="NDIF"),
            no_restart=False,
        )

    def cleanup(self):
        """Performs cleanup operations after request processing.

        This method:
        1. Disconnects from socketio if connected
        2. Zeros out model gradients
        3. Forces garbage collection
        4. Clears CUDA cache

        This cleanup is important for preventing memory leaks and ensuring
        the replica is ready for the next request.
        """
        self.kill_switch.clear()
        self.execution_ident = None

        SioProvider.disconnect()

        self.model._model.zero_grad()
        gc.collect()
        torch.cuda.empty_cache()

    def log(self, *data):
        """Logs data during model execution.

        This method is used to send log messages back to the client through
        the websocket connection. It joins all provided data into a single string
        and sends it as a LOG status response.

        Args:
            *data: Variable number of arguments to be converted to strings and logged.
        """
        description = "".join([str(_data) for _data in data])
        self.respond(status=BackendResponseModel.JobStatus.LOG, description=description)

    def stream_send(self, data: Any):
        """Sends streaming data back to the client.

        This method is used to send intermediate results or progress updates
        during model execution. It wraps the data in a STREAM status response.

        Args:
            data (Any): The data to stream back to the client.
        """

        response = self.request.create_response(
            BackendResponseModel.JobStatus.STREAM, self.logger, data=data
        )

        SioProvider.emit(
            "stream", data=(self.request.session_id, response.pickle(), self.request.id)
        )

    def stream_receive(self, *args):
        """Receives streaming data from the client.

        This method establishes a websocket connection if needed and waits
        for data from the client. It has a 5-second timeout for receiving data.

        Returns:
            The deserialized data received from the client.
        """
        return SioProvider.sio.receive(5)[1]

    def _respond(self, request: "BackendRequestModel", **kwargs) -> None:
        """Send a response for an explicit request. Safe under concurrency
        because the request is passed in rather than read from shared
        state. Callers that already have the request in a local variable
        (pre/post/exception) use this directly."""
        try:
            request.create_response(**kwargs, logger=self.logger).respond()
        except:
            self.logger.exception("Error responding to client")

    def respond(self, **kwargs) -> None:
        """Legacy variant that reads the request from the current
        ContextVar. Kept for backward compat with log/stream_send paths
        that don't have the request in local scope.
        """
        request = _current_request.get()
        if request is None:
            self.logger.warning(
                "respond() called with no request in context; dropping %s",
                kwargs.get("status"),
            )
            return
        self._respond(request, **kwargs)


class BaseModelDeploymentArgs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_key: MODEL_KEY
    cuda_devices: str

    execution_timeout: float | None = None
    device_map: str | None = "auto"
    dispatch: bool = True
    dtype: str | torch.dtype = "bfloat16"


_MAX_CONCURRENCY = int(os.getenv("NDIF_MAX_CONCURRENCY", "32"))


@ray.remote(
    num_cpus=2, num_gpus=0, max_restarts=-1, max_concurrency=_MAX_CONCURRENCY
)
class ModelActor(BaseModelDeployment):
    """Ray remote actor for model execution.

    ``NDIF_MAX_CONCURRENCY`` (default 32) caps in-flight ``__call__``
    invocations. Extra calls queue in Ray's per-actor runtime. With
    batching enabled (LM deployments) the batch server groups whatever
    is in flight; setting the cap to 1 forces one request at a time and
    collapses every forward to ``bs=1``.
    """

    pass
