import asyncio
import contextvars
import gc
import os
import threading
import time
from typing import Any, Dict, Optional, Set

import ray
import torch
from accelerate import dispatch_model
from pydantic import BaseModel, ConfigDict
from torch.amp import autocast
from torch.cuda import max_memory_allocated, memory_allocated, reset_peak_memory_stats
from transformers.modeling_utils import _get_device_map

from nnsight.intervention.tracing.globals import Globals
from nnsight.modeling.hf_serve.vanilla_server import VanillaBatchServer
from nnsight.modeling.language import LanguageModel
from nnsight.modeling.mixins import RemoteableMixin
from nnsight.modeling.mixins.remoteable import StreamTracer
from nnsight.schema.request import RequestModel

from opentelemetry import trace

from .....common.logging import set_logger
from .....common.metrics import (
    ExecutionTimeMetric,
    GPUMemMetric,
    ModelLoadTimeMetric,
    RequestResponseSizeMetric,
)
from .....common.tracing import (
    TracingContext,
    init_tracing,
    set_request_attributes,
    trace_span,
)
from .....common.providers.objectstore import ObjectStoreProvider
from .....common.providers.socketio import SioProvider
from .....common.schema import (
    BackendRequestModel,
    BackendResponseModel,
    BackendResultModel,
)
from .....common.types import MODEL_KEY
from ...nn.backend import RemoteExecutionBackend
from ...nn.ops import StdoutRedirect
from ...nn.security import (
    Protector,
    WHITELISTED_MODULES,
    WHITELISTED_MODULES_DESERIALIZATION,
)
from ...nn.security.protected_objects import protect
from .util import kill_thread, load_with_cache_deletion_retry, remove_accelerate_hooks


# Per-call request binding. Concurrent __call__ invocations each set this
# on entry; helpers (respond / log / stream_send) read from it so they
# see the right request without needing the signature threaded through.
# ContextVars propagate through asyncio.to_thread, so worker threads
# spawned for pre()/post() also see the right value.
_current_request: contextvars.ContextVar[Optional["BackendRequestModel"]] = (
    contextvars.ContextVar("_ndif_current_request", default=None)
)


class BaseModelDeployment:
    def __init__(
        self,
        model_key: MODEL_KEY,
        execution_timeout: float | None,
        dispatch: bool,
        dtype: str | torch.dtype,
        gpu_mem_bytes_by_id: Dict[int, int] | None = None,
        *args,
        trace_context: Optional[Dict[str, str]] = None,
        extra_kwargs: Dict[str, Any] = {},
        **kwargs,
    ) -> None:
        super().__init__()

        init_tracing("ndif-ray")

        parent_ctx = TracingContext.extract(trace_context)

        with trace_span(
            "model_actor.init",
            parent_context=parent_ctx,
            attributes={
                "ndif.model.key": model_key,
                "ndif.model.gpu_mem_bytes_by_id": str(gpu_mem_bytes_by_id or {}),
                "ndif.model.dispatch": dispatch,
                "ndif.model.dtype": str(dtype),
            },
        ) as span:
            self._init_trace_context = trace_context

            span.add_event("connecting_providers")
            ObjectStoreProvider.connect()
            SioProvider.connect()

            self.model_key = model_key
            self.execution_timeout = execution_timeout
            self.dispatch = dispatch
            self.dtype = dtype
            self.extra_kwargs = extra_kwargs
            self.gpu_mem_bytes_by_id = gpu_mem_bytes_by_id or {}

            self.cached = False

            self.logger = set_logger(model_key)

            span.add_event("getting_ray_runtime_context")
            self.runtime_context = ray.get_runtime_context()

            if isinstance(dtype, str):
                self.dtype = getattr(torch, dtype)

            torch.set_default_dtype(torch.bfloat16)
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True

            # Set the default CUDA device to the first target GPU BEFORE any CUDA
            # call. This ensures the CUDA context (~400MiB) is created on the
            # target GPU rather than always landing on GPU 0.
            if self.gpu_mem_bytes_by_id:
                first_gpu = next(iter(self.gpu_mem_bytes_by_id))
                torch.cuda.set_device(first_gpu)

                # Set per-process memory fraction for each target GPU
                for gpu_id, mem_bytes in self.gpu_mem_bytes_by_id.items():
                    total = torch.cuda.get_device_properties(gpu_id).total_memory
                    fraction = min(mem_bytes / total, 1.0)
                    torch.cuda.set_per_process_memory_fraction(fraction, gpu_id)

            span.add_event("loading_model")
            self.model = self.load_from_disk()

            span.add_event("building_persistent_objects")
            # Pre-wrap every module reference once at init. Walking the
            # envoy tree (hundreds of modules for 7B+ models) and
            # synthesizing a ProtectedObject subclass per module is O(N)
            # at warmup; thanks to the per-type class cache in
            # protected_objects.protect(), it collapses to
            # O(unique_module_types). Per-request unpickling then becomes
            # a pure dict lookup on self.persistent_objects.
            self.persistent_objects = self.model._remoteable_persistent_objects()
            for key, value in self.persistent_objects.items():
                if isinstance(value, torch.nn.Module):
                    self.persistent_objects[key] = protect(value)

            # Thread-local sandbox. Used three ways:
            #   1. ``with self.execution_protector:`` — import whitelist
            #      on this thread only (non-LM execute_single path).
            #   2. ``self.execution_protector(target_globals)`` — factory
            #      per nnsight's ``worker_context`` contract: activates
            #      TLS and shadows risky builtins in the user frame's
            #      globals. Used for mediator workers in the batched
            #      execution path.
            #   3. ``pre()`` uses its own short-lived Protector for the
            #      deserialization whitelist.
            self.execution_protector = Protector(WHITELISTED_MODULES)

            if dispatch:
                self.model._module.requires_grad_(False)

            torch.cuda.empty_cache()

            # Cross-request batching for LanguageModel deployments. The
            # batch server's bg generation thread runs no user code; user
            # interventions are dispatched onto fresh mediator threads
            # which the worker_context wraps in our TLS Protector.
            #
            # Non-LM deployments (or LM with NDIF_FORCE_SEQUENTIAL set)
            # fall back to the execute_single path serialized via
            # model_lock.
            if isinstance(self.model, LanguageModel):
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

            self.kill_switch = asyncio.Event()
            self.execution_ident = None
            self._request_count = 0

            StreamTracer.register(self.stream_send, self.stream_receive)

    @property
    def request(self) -> Optional["BackendRequestModel"]:
        """Current request for the calling task/thread. Backed by a
        ContextVar set in ``__call__`` so concurrent requests don't alias.
        """
        return _current_request.get()

    @request.setter
    def request(self, value):
        # Legacy code (cleanup() in earlier versions) set this to None
        # between requests. With the ContextVar the scope is already
        # bounded by __call__, so this is a no-op. Keeping the setter so
        # any remaining writes from upstream code don't raise.
        pass

    def _build_max_memory(self) -> Optional[Dict[int, int]]:
        """Build a max_memory dict that restricts model placement to target GPUs.

        Returns a dict mapping GPU index to max memory in bytes. Target GPUs
        get their allocated budget (capped at total device memory). Non-target
        GPUs get 0 bytes to prevent any allocation. Returns None if no target
        GPUs are set, which lets accelerate use all available GPUs.
        """
        if not self.gpu_mem_bytes_by_id:
            return None

        num_gpus = torch.cuda.device_count()
        max_memory = {}
        for i in range(num_gpus):
            if i in self.gpu_mem_bytes_by_id:
                total = torch.cuda.get_device_properties(i).total_memory
                max_memory[i] = min(self.gpu_mem_bytes_by_id[i], total)
            else:
                max_memory[i] = 0
        return max_memory

    def _verify_device_placement(self, module: torch.nn.Module, source: str):
        """Verify and log that model parameters are on the expected GPUs."""
        devices: Set[str] = set()
        for param in module.parameters():
            devices.add(f"{param.device.type}:{param.device.index}")

        span = trace.get_current_span()
        span.set_attribute("ndif.model.devices", str(sorted(devices)))

        self.logger.info(f"Model loaded from {source} on devices: {devices}")

        if self.gpu_mem_bytes_by_id:
            expected = {f"cuda:{gpu}" for gpu in self.gpu_mem_bytes_by_id.keys()}
            actual_cuda = {d for d in devices if d.startswith("cuda:")}
            if actual_cuda and not actual_cuda.issubset(expected):
                span.add_event(
                    "device_placement_mismatch",
                    {
                        "expected": str(sorted(expected)),
                        "actual": str(sorted(actual_cuda)),
                    },
                )
                self.logger.warning(
                    f"Device placement mismatch! Expected GPUs {list(self.gpu_mem_bytes_by_id.keys())}, "
                    f"but model is on {actual_cuda}"
                )

    def load_from_disk(self):
        parent_ctx = TracingContext.extract(self._init_trace_context)
        with trace_span(
            "model_actor.load",
            parent_context=parent_ctx,
            attributes={
                "ndif.model.key": self.model_key,
                "ndif.model.load_source": "disk",
            },
        ) as span:
            start = time.time()
            torch.cuda.synchronize()
            self.logger.info(
                f"Loading model from disk for model key {self.model_key} "
                f"with gpu_mem_bytes_by_id {self.gpu_mem_bytes_by_id}..."
            )

            max_memory = self._build_max_memory()

            model = load_with_cache_deletion_retry(
                lambda: RemoteableMixin.from_model_key(
                    self.model_key,
                    device_map="auto",
                    max_memory=max_memory,
                    dispatch=self.dispatch,
                    torch_dtype=self.dtype,
                    attn_implementation="eager",
                    **self.extra_kwargs,
                )
            )
            torch.cuda.synchronize()
            load_time = time.time() - start

            span.set_attribute("ndif.model.load_time_s", load_time)
            ModelLoadTimeMetric.update(load_time, self.model_key, "disk")

            self._verify_device_placement(model._module, "disk")

            self.logger.debug(f"Model loaded from disk in {load_time} seconds")

            return model

    async def to_cache(self, trace_context: Optional[Dict[str, str]] = None):
        parent_ctx = TracingContext.extract(trace_context)
        with trace_span(
            "model_actor.to_cache",
            parent_context=parent_ctx,
            attributes={"ndif.model.key": self.model_key},
        ) as span:
            await self.cancel()

            # Stop the batch server's bg generation thread before moving
            # weights to CPU. Restarted in from_cache() once weights are
            # back on the target GPU.
            if self.batch_server is not None:
                self.batch_server.stop()

            span.add_event("remove_accelerate_hooks")
            remove_accelerate_hooks(self.model._module)

            # Reset per-process memory fractions before releasing GPU memory
            for gpu_id in self.gpu_mem_bytes_by_id:
                torch.cuda.set_per_process_memory_fraction(1.0, gpu_id)

            span.add_event("move_to_cpu")
            self.model._module = self.model._module.cpu()

            span.add_event("gc_collect")
            gc.collect()
            torch.cuda.empty_cache()

            self.cached = True

    def from_cache(
        self,
        gpu_mem_bytes_by_id: Dict[int, int],
        trace_context: Optional[Dict[str, str]] = None,
    ):
        """Restore model from CPU cache onto the specified GPU(s)."""
        parent_ctx = TracingContext.extract(trace_context)
        with trace_span(
            "model_actor.load",
            parent_context=parent_ctx,
            attributes={
                "ndif.model.key": self.model_key,
                "ndif.model.load_source": "cache",
            },
        ) as span:
            self.gpu_mem_bytes_by_id = gpu_mem_bytes_by_id

            if self.gpu_mem_bytes_by_id:
                first_gpu = next(iter(self.gpu_mem_bytes_by_id))
                torch.cuda.set_device(first_gpu)

                for gpu_id, mem_bytes in self.gpu_mem_bytes_by_id.items():
                    total = torch.cuda.get_device_properties(gpu_id).total_memory
                    fraction = min(mem_bytes / total, 1.0)
                    torch.cuda.set_per_process_memory_fraction(fraction, gpu_id)

            torch.cuda.synchronize()
            start = time.time()

            self.logger.info(
                f"Loading model from cache for model key {self.model_key} "
                f"with gpu_mem_bytes_by_id {gpu_mem_bytes_by_id}..."
            )

            max_memory = self._build_max_memory()

            device_map = _get_device_map(self.model._module, "auto", max_memory, None)

            remove_accelerate_hooks(self.model._module)

            self.model._module = dispatch_model(self.model._module, device_map)

            # Re-create the batch server: VanillaBatchServer's bg thread
            # was stopped in to_cache() and Python threads cannot be
            # restarted. Re-create with the same worker_context so
            # mediator workers on the rehydrated server stay sandboxed.
            if isinstance(self.model, LanguageModel):
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

            self._verify_device_placement(self.model._module, "cache")

            span.set_attribute("ndif.model.load_time_s", load_time)
            ModelLoadTimeMetric.update(load_time, self.model_key, "cache")

            self.logger.debug(f"Model loaded from cache in {load_time} seconds")

            self.cached = False

    async def __call__(self, request: BackendRequestModel) -> None:
        """Executes the model service pipeline:

        1.) Pre-processing
        2.) Execution (batched if LanguageModel, else single via model_lock)
        3.) Post-processing
        4.) Cleanup

        Args:
            request (BackendRequestModel): Request.
        """

        if self.cached:
            raise LookupError("Failed to look up actor")

        parent_ctx = TracingContext.extract(request.trace_context)
        token = _current_request.set(request)

        with trace_span("model_actor.call", parent_context=parent_ctx) as span:
            set_request_attributes(span, request)
            result = None
            try:
                # Offload pickle/unpickle to a worker thread so the event
                # loop stays free to admit other concurrent requests.
                # ContextVar propagates so helpers inside pre() still see
                # the right request via self.request.
                inputs = await asyncio.to_thread(self.pre, request)

                if self.batch_server is not None:
                    result = await self.execute_batched(request, inputs)
                else:
                    async with self.model_lock:
                        result = await self._execute_single_with_timeout(inputs)

                await asyncio.to_thread(self.post, request, result)

            except Exception as e:
                span.set_status(trace.StatusCode.ERROR, str(e))
                span.record_exception(e)
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
        ``self.persistent_objects`` (immutable after __init__), so it's
        safe to run on any worker thread.
        """
        with trace_span(
            "model_actor.pre", attributes={"ndif.model.key": self.model_key}
        ) as span:
            self._respond(
                request,
                status=BackendResponseModel.JobStatus.RUNNING,
                description="Your job has started running.",
            )

            raw = request.request
            if isinstance(raw, ray.ObjectRef):
                raw = ray.get(raw)

            span.add_event("deserializing_request")
            with Protector(WHITELISTED_MODULES_DESERIALIZATION):
                req_model = RequestModel.deserialize(
                    raw, self.persistent_objects, request.zlib
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

        # Compile off the event loop. When this returns, we have the
        # per-invoke entries ready to submit.
        entries = await asyncio.to_thread(self._compile_trace, request_model)

        # submit_async MUST run on the event loop — it binds the returned
        # Future to asyncio.get_running_loop(). Microseconds per call.
        futures = [self.batch_server.submit_async(e) for e in entries]

        per_invoke_saves = await asyncio.gather(*futures)

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
        with trace_span(
            "model_actor.post", attributes={"ndif.model.key": self.model_key}
        ) as span:
            saves = result[0]
            gpu_mem: int = result[1]
            execution_time_s: float = result[2]

            span.add_event("saving_result")
            result_object = BackendResultModel(
                id=request.id,
                **saves,
            ).save(compress=request.compress)

            span.set_attribute("ndif.response_size_bytes", result_object._size)
            span.set_attribute("ndif.request.compress", request.compress)

            span.add_event("sending_response")
            self._respond(
                request,
                status=BackendResponseModel.JobStatus.COMPLETED,
                description="Your job has been completed.",
                data=(result_object.url(), result_object._size),
            )

            span.set_attribute("ndif.execution_time_s", execution_time_s)
            span.set_attribute("ndif.gpu_mem_bytes", gpu_mem)

            RequestResponseSizeMetric.update(request, result_object._size)
            GPUMemMetric.update(request, gpu_mem)
            ExecutionTimeMetric.update(request, execution_time_s)

    def exception(self, request: "BackendRequestModel", exception: Exception) -> None:
        """Handles exceptions that occur during model execution."""

        self._respond(
            request,
            status=BackendResponseModel.JobStatus.ERROR,
            description=str(exception),
        )

        # Special handling for CUDA device-side assertion errors
        if "device-side assert triggered" in str(exception):
            self.restart()

    def restart(self):
        """Restarts the Ray serve deployment in response to critical errors."""
        ray.kill(
            ray.get_actor(f"ModelActor:{self.model_key}", namespace="NDIF"),
            no_restart=False,
        )

    def cleanup(self):
        """Performs cleanup operations after request processing.

        - Zeros out model gradients.
        - Clears nnsight Globals and interleaver state.
        - Runs full garbage collection every 5 requests.
        - Clears CUDA cache.

        NOTE: deliberately does NOT call upstream's ``clear_set_attrs()``.
        That mechanism is incompatible with our batched execution path
        because user attribute writes mutate the real shared module
        synchronously, corrupting concurrent peers in the same fused
        forward before rollback can land. We refuse writes outright in
        ``ProtectedObject.__setattr__`` instead. See
        ``docs/TODO_batched_attr_writes.md`` for the analysis.
        """
        with trace_span(
            "model_actor.cleanup", attributes={"ndif.model.key": self.model_key}
        ) as span:
            self.kill_switch.clear()
            self.execution_ident = None

            span.add_event("zero_grad")
            self.model._model.zero_grad(set_to_none=True)

            span.add_event("clearing_globals")
            Globals.clear()
clear_set_attrs()
            self.model.interleaver.cancel()

            self._request_count += 1
            if self._request_count % 5 == 0:
                span.add_event("gc_collect")
                gc.collect()
                torch.cuda.empty_cache()

    def log(self, *data):
        """Logs data during model execution.

        Sends log messages back to the client through the websocket
        connection. Joins all provided data into a single string and
        sends it as a LOG status response.
        """
        try:
            self.execution_protector.__exit__(None, None, None)
            description = "".join([str(_data) for _data in data])
            self.respond(
                status=BackendResponseModel.JobStatus.LOG, description=description
            )
        finally:
            self.execution_protector.__enter__()

    def stream_send(self, data: Any):
        """Sends streaming data back to the client.

        Wraps the data in a STREAM status response.
        """
        request = _current_request.get()
        if request is None:
            return

        response = request.create_response(
            BackendResponseModel.JobStatus.STREAM, self.logger, data=data
        )

        SioProvider.emit(
            "stream", data=(request.session_id, response.pickle(), request.id)
        )

    def stream_receive(self, *args):
        """Receives streaming data from the client.

        Establishes a websocket connection if needed and waits for data
        from the client. 5-second timeout for receiving data.
        """
        return SioProvider.sio.receive(5)[1]

    def _respond(self, request: "BackendRequestModel", **kwargs) -> None:
        """Send a response for an explicit request. Safe under concurrency
        because the request is passed in rather than read from shared
        state. Callers that already have the request in a local variable
        (pre / post / exception) use this directly.
        """
        try:
            request.create_response(**kwargs, logger=self.logger).respond()
        except Exception:
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

    execution_timeout: float | None = None
    device_map: str | None = "auto"
    dispatch: bool = True
    dtype: str | torch.dtype = torch.bfloat16
    gpu_mem_bytes_by_id: Dict[int, int] | None = None
    trace_context: Optional[Dict[str, str]] = None


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
    collapses every forward to ``bs=1`` — useful for A/B-testing batching
    benefits.
    """

    pass
