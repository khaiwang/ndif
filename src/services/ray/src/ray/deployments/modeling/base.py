import asyncio
import gc
import threading
import time

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set

import ray
import torch
from accelerate import dispatch_model
from pydantic import BaseModel, ConfigDict

from torch.amp import autocast
from torch.cuda import max_memory_allocated, memory_allocated, reset_peak_memory_stats
from transformers.modeling_utils import _get_device_map

from nnsight.modeling.mixins import RemoteableMixin
from nnsight.modeling.mixins.remoteable import StreamTracer
from nnsight.schema.request import RequestModel

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
from ....types import MODEL_KEY
from ...nn.backend import RemoteExecutionBackend
from ...nn.ops import StdoutRedirect
from ...nn.security.protected_environment import (
    WHITELISTED_MODULES,
    WHITELISTED_MODULES_DESERIALIZATION,
    Protector,
)
from ...nn.security.protected_objects import protect, clear_set_attrs
from nnsight.intervention.tracing.globals import Globals
from .pinned_pool import PinnedBufferPool, unique_named_tensors
from .relay_buffer import get_relay_buffer
from .util import kill_thread, load_with_cache_deletion_retry, remove_accelerate_hooks


def _set_single_gpu_device_map(module: torch.nn.Module, gpu_id: int) -> None:
    """Set ``hf_device_map`` on *module* to map every sub-module to *gpu_id*."""
    module.hf_device_map = {name: str(gpu_id) for name, _ in module.named_modules()}


def _reassign_module_data(
    module: torch.nn.Module,
    gpu_views: Dict[str, torch.Tensor],
    target_device: torch.device,
) -> None:
    """Point every parameter/buffer in *module* at the corresponding GPU tensor.

    Tensors present in *gpu_views* are reassigned directly.  Buffers not in
    *gpu_views* that are still on CPU are moved to *target_device*.
    """
    for name, param in module.named_parameters():
        if name in gpu_views:
            param.data = gpu_views[name]
    for name, buf in module.named_buffers():
        if name in gpu_views:
            buf.data = gpu_views[name]
        elif buf.device.type != "cuda":
            buf.data = buf.data.to(target_device, non_blocking=True)


def _reassign_module_data_cpu(
    module: torch.nn.Module,
    cpu_views: Dict[str, torch.Tensor],
) -> None:
    """Point every parameter/buffer in *module* at the corresponding CPU tensor."""
    for name, param in module.named_parameters():
        if name in cpu_views:
            param.data = cpu_views[name]
    for name, buf in module.named_buffers():
        if name in cpu_views:
            buf.data = cpu_views[name]
        elif buf.device.type != "cpu":
            buf.data = buf.data.cpu()


class BaseModelDeployment:
    def __init__(
        self,
        model_key: MODEL_KEY,
        execution_timeout: float | None,
        dispatch: bool,
        dtype: str | torch.dtype,
        gpu_mem_bytes_by_id: Dict[int, int] | None = None,
        spawn_timestamp: float | None = None,
        *args,
        extra_kwargs: Dict[str, Any] = {},
        **kwargs,
    ) -> None:
        super().__init__()
        init_start = time.time()

        provider_start = time.time()
        ObjectStoreProvider.connect()
        SioProvider.connect()
        provider_time = time.time() - provider_start

        self.model_key = model_key
        self.execution_timeout = execution_timeout
        self.dispatch = dispatch
        self.dtype = dtype
        self.extra_kwargs = extra_kwargs
        self.gpu_mem_bytes_by_id = gpu_mem_bytes_by_id or {}
        self.target_gpus = list(self.gpu_mem_bytes_by_id.keys())

        self.cached = False

        self.logger = set_logger(model_key)

        self.runtime_context = ray.get_runtime_context()

        if isinstance(dtype, str):
            self.dtype = getattr(torch, dtype)

        torch.set_default_dtype(torch.bfloat16)

        # Set the default CUDA device to the first target GPU BEFORE any CUDA
        # call. This ensures the CUDA context (~400MiB) is created on the
        # target GPU rather than always landing on GPU 0.
        cuda_init_start = time.time()
        if self.target_gpus:
            torch.cuda.set_device(self.target_gpus[0])

            # Set per-process memory fraction for each target GPU
            for gpu_id, mem_bytes in self.gpu_mem_bytes_by_id.items():
                total = torch.cuda.get_device_properties(gpu_id).total_memory
                fraction = min(mem_bytes / total, 1.0)
                torch.cuda.set_per_process_memory_fraction(fraction, gpu_id)
        cuda_init_time = time.time() - cuda_init_start

        # Use cudaHostRegister allocator for pin_memory=True. This avoids
        # power-of-2 rounding and enables multi-threaded page registration.
        try:
            torch.cuda.memory._set_allocator_settings(
                "pinned_use_cuda_host_register:True,pinned_num_register_threads:8"
            )
        except Exception:
            self.logger.warning(
                "cudaHostRegister allocator not available, using default"
            )

        self._pinned_pool: PinnedBufferPool | None = None
        self._bg_pin_thread: threading.Thread | None = None

        # CPU-only load: model stays on CPU until dispatch_to_gpu() is called
        self.model = self.load_from_disk()

        # Start background pinned pool allocation — overlaps with security setup
        self._start_pinned_preallocation()

        security_start = time.time()
        self.persistent_objects = self.model._remoteable_persistent_objects()

        for key, value in self.persistent_objects.items():
            if isinstance(value, torch.nn.Module):
                self.persistent_objects[key] = protect(value)

        self.execution_protector = Protector(WHITELISTED_MODULES, builtins=True)

        if dispatch:
            self.model._module.requires_grad_(False)

        security_time = time.time() - security_start

        self.request: BackendRequestModel

        self.thread_pool = ThreadPoolExecutor(max_workers=1)

        self.kill_switch = asyncio.Event()
        self.execution_ident = None

        StreamTracer.register(self.stream_send, self.stream_receive)

        init_time = time.time() - init_start
        ray_overhead = init_start - spawn_timestamp if spawn_timestamp else 0.0
        ModelLoadTimeMetric.update(ray_overhead, self.model_key, "ray_actor_overhead")
        ModelLoadTimeMetric.update(
            provider_time, self.model_key, "init_provider_connect"
        )
        ModelLoadTimeMetric.update(cuda_init_time, self.model_key, "init_cuda_context")
        ModelLoadTimeMetric.update(security_time, self.model_key, "init_security_setup")
        ModelLoadTimeMetric.update(init_time, self.model_key, "init_cpu_total")
        self.logger.info(
            f"ModelActor.__init__ completed (CPU) in {init_time:.2f}s "
            f"(ray_overhead={ray_overhead:.2f}s, provider={provider_time:.2f}s, "
            f"cuda_init={cuda_init_time:.2f}s, "
            f"load_from_disk=see 'disk_cpu_load' metric, security={security_time:.2f}s)"
        )

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
        """Verify and log that model parameters are on the expected GPUs.

        Args:
            module: The model module to check.
            source: Description of load source (e.g., 'disk', 'cache') for logging.
        """
        devices: Set[str] = set()
        for param in module.parameters():
            devices.add(f"{param.device.type}:{param.device.index}")

        self.logger.info(f"Model loaded from {source} on devices: {devices}")

        if self.gpu_mem_bytes_by_id:
            expected = {f"cuda:{gpu}" for gpu in self.gpu_mem_bytes_by_id.keys()}
            actual_cuda = {d for d in devices if d.startswith("cuda:")}
            if actual_cuda and not actual_cuda.issubset(expected):
                self.logger.warning(
                    f"Device placement mismatch! Expected GPUs {list(self.gpu_mem_bytes_by_id.keys())}, "
                    f"but model is on {actual_cuda}"
                )

    def load_from_disk(self):
        """Load model from disk to CPU only. GPU dispatch happens separately."""
        start = time.time()
        self.logger.info(
            f"Loading model from disk (CPU) for model key {self.model_key} "
            f"targeting GPUs {self.target_gpus}..."
        )

        model = load_with_cache_deletion_retry(
            lambda: RemoteableMixin.from_model_key(
                self.model_key,
                device_map="cpu",
                dispatch=self.dispatch,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
                **self.extra_kwargs,
            )
        )
        load_time = time.time() - start

        ModelLoadTimeMetric.update(load_time, self.model_key, "disk_cpu_load")

        self.logger.info(f"Model loaded from disk to CPU in {load_time:.2f}s")

        return model

    def _start_pinned_preallocation(self, reuse_ok: bool = False) -> None:
        """Ensure a pinned buffer pool is ready for the next eviction cycle.

        Args:
            reuse_ok: If True, keep the existing pool when its layout still
                matches the current model (same parameter names, shapes, dtypes).
                This avoids an expensive free+realloc of pinned memory between
                evict/reload cycles where the model architecture hasn't changed.
        """
        try:
            if (
                reuse_ok
                and self._pinned_pool is not None
                and self._pinned_pool.is_ready
            ):
                if self._pinned_pool.matches(self.model._module):
                    self.logger.info("Reusing existing pinned buffer pool")
                    return

            if self._pinned_pool is not None:
                self._pinned_pool.release()
            pool = PinnedBufferPool()
            pool.preallocate_async(self.model._module)
            self._pinned_pool = pool
        except Exception:
            self.logger.warning(
                "Failed to start pinned buffer pre-allocation", exc_info=True
            )
            self._pinned_pool = None

    def dispatch_to_gpu(self):
        """Move model from CPU to target GPUs. Called by controller after evictions complete."""
        start = time.time()
        torch.cuda.synchronize()
        self.logger.info(f"Dispatching model to GPUs {self.target_gpus}...")

        max_memory = self._build_max_memory()
        module = self.model._module

        if len(self.target_gpus) == 1:
            target_device = torch.device(f"cuda:{self.target_gpus[0]}")
            remove_accelerate_hooks(module)

            pool = self._pinned_pool
            if pool is not None and pool.wait_ready(timeout=120.0):
                # Pinned mirror path: copy mmap'd CPU -> pinned views, then DMA -> GPU
                self.logger.info("dispatch_to_gpu: creating pinned mirror")
                for name, tensor in unique_named_tensors(module):
                    pinned_view = pool.get(name)
                    if pinned_view is not None:
                        pinned_view.copy_(tensor.data)
                # Swap param.data to pinned views (frees mmap'd memory)
                _reassign_module_data_cpu(module, pool._views)
                # DMA pinned -> GPU
                gpu_views = pool.transfer_to_device(target_device)
                torch.cuda.synchronize()
                _reassign_module_data(module, gpu_views, target_device)
                self.logger.info("dispatch_to_gpu: mirror created, pool persists")
            else:
                # Fallback: multi-stream .to(device), no mirror
                self.logger.info("dispatch_to_gpu: fallback (no mirror)")
                self._pinned_pool = None
                num_streams = 4
                streams = [
                    torch.cuda.Stream(device=target_device) for _ in range(num_streams)
                ]
                gpu_views: Dict[str, torch.Tensor] = {}
                for i, (name, tensor) in enumerate(unique_named_tensors(module)):
                    with torch.cuda.stream(streams[i % num_streams]):
                        gpu_views[name] = tensor.data.to(
                            target_device, non_blocking=True
                        )
                torch.cuda.synchronize()
                _reassign_module_data(module, gpu_views, target_device)

            _set_single_gpu_device_map(module, self.target_gpus[0])
        else:
            # Multi-GPU: use accelerate dispatch_model, no mirror support
            if self._pinned_pool is not None:
                self._pinned_pool.release()
                self._pinned_pool = None
            remove_accelerate_hooks(module)
            device_map = _get_device_map(module, "auto", max_memory, None)
            self.model._module = dispatch_model(module, device_map)
            torch.cuda.synchronize()

        gc.collect()
        torch.cuda.empty_cache()

        dispatch_time = time.time() - start
        ModelLoadTimeMetric.update(dispatch_time, self.model_key, "disk_gpu_dispatch")

        self._verify_device_placement(self.model._module, "disk")

        self.logger.info(
            f"Model dispatched to GPUs in {dispatch_time:.2f}s "
            f"(mirror={'yes' if self._pinned_pool is not None else 'no'})"
        )

    def _async_pin_upgrade(self, module: torch.nn.Module) -> None:
        """Background thread: allocate pinned pool, copy unpinned CPU data into it."""
        try:
            pool = PinnedBufferPool()
            pool.preallocate(module)
            if pool.is_ready:
                for name, tensor in unique_named_tensors(module):
                    pinned_view = pool.get(name)
                    if pinned_view is not None:
                        pinned_view.copy_(tensor.data)
                _reassign_module_data_cpu(module, pool._views)
                self._pinned_pool = pool
                gc.collect()
                self.logger.info(
                    "_async_pin_upgrade: pinned mirror created from unpinned cache"
                )
            else:
                pool.release()
                self.logger.warning("_async_pin_upgrade: pool allocation failed")
        except Exception:
            self.logger.warning("_async_pin_upgrade failed", exc_info=True)

    async def to_cache(self):
        self.logger.info(f"Saving model to cache for model key {self.model_key}...")
        cache_start = time.time()

        cancel_start = time.time()
        await self.cancel()
        cancel_time = time.time() - cancel_start

        # Reset per-process memory fractions before releasing GPU memory
        for gpu_id in self.gpu_mem_bytes_by_id:
            torch.cuda.set_per_process_memory_fraction(1.0, gpu_id)

        hook_start = time.time()
        remove_accelerate_hooks(self.model._module)
        hook_time = time.time() - hook_start

        transfer_start = time.time()
        module = self.model._module
        used_mirror = False

        pool = self._pinned_pool
        if pool is not None and pool.is_ready:
            # Mirror path: data already in pinned pool (identical to GPU,
            # inference is read-only). Just swap param.data to pinned views.
            _reassign_module_data_cpu(module, pool._views)
            used_mirror = True
            self.logger.info("to_cache: zero-copy via pinned mirror")
        else:
            # No mirror: relay buffer GPU -> unpinned CPU, then async pin upgrade
            relay = get_relay_buffer()
            relay.ensure_allocated()
            cpu_views = relay.gpu_to_cpu(module)
            _reassign_module_data_cpu(module, cpu_views)
            self.logger.info("to_cache: relay buffer gpu_to_cpu complete")

        transfer_time = time.time() - transfer_start

        cleanup_start = time.time()
        gc.collect()
        torch.cuda.empty_cache()
        cleanup_time = time.time() - cleanup_start

        # If we used relay (no mirror), start background thread to upgrade
        # unpinned CPU tensors to a pinned pool for fast reload.
        if not used_mirror:
            self._bg_pin_thread = threading.Thread(
                target=self._async_pin_upgrade, args=(module,), daemon=True
            )
            self._bg_pin_thread.start()

        self.cached = True

        cache_time = time.time() - cache_start
        ModelLoadTimeMetric.update(transfer_time, self.model_key, "to_cache_gpu_to_cpu")
        ModelLoadTimeMetric.update(cleanup_time, self.model_key, "to_cache_cleanup")
        ModelLoadTimeMetric.update(cache_time, self.model_key, "to_cache_total")
        self.logger.info(
            f"Model cached in {cache_time:.2f}s "
            f"(cancel={cancel_time:.2f}s, hooks={hook_time:.2f}s, "
            f"gpu_to_cpu={transfer_time:.2f}s, cleanup={cleanup_time:.2f}s, "
            f"mirror={used_mirror})"
        )

    def from_cache(self, gpu_mem_bytes_by_id: Dict[int, int]):
        """Restore model from CPU cache onto the specified GPU(s).

        Uses max_memory targeting to ensure the model is placed on exactly
        the requested GPUs, rather than relying on CUDA_VISIBLE_DEVICES
        (which cannot be changed after CUDA context initialization).

        Args:
            gpu_mem_bytes_by_id: Dict mapping GPU index to allocated bytes.
        """
        self.gpu_mem_bytes_by_id = gpu_mem_bytes_by_id
        self.target_gpus = list(gpu_mem_bytes_by_id.keys())

        # Switch default CUDA device to the new target GPU before any CUDA ops
        if self.target_gpus:
            first_gpu = self.target_gpus[0]
            torch.cuda.set_device(first_gpu)

            # Set per-process memory fraction for each target GPU
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
        module = self.model._module

        if len(self.target_gpus) == 1:
            target_device = torch.device(f"cuda:{self.target_gpus[0]}")

            # Wait for background pin upgrade if still running
            if self._bg_pin_thread is not None and self._bg_pin_thread.is_alive():
                self.logger.info("from_cache: waiting for async pin upgrade...")
                self._bg_pin_thread.join()
            self._bg_pin_thread = None

            pool = self._pinned_pool
            use_fast_path = pool is not None and pool.is_ready

            if use_fast_path:
                # Pinned fast path: DMA from pinned pool -> GPU
                self.logger.info("from_cache: single-GPU pinned fast path")

                remove_accelerate_hooks(module)

                gpu_views = pool.transfer_to_device(target_device)
                torch.cuda.synchronize()
                _reassign_module_data(module, gpu_views, target_device)
            else:
                # Fallback: relay buffer CPU -> GPU
                self.logger.info("from_cache: single-GPU relay fallback")

                remove_accelerate_hooks(module)

                relay = get_relay_buffer()
                relay.ensure_allocated()
                gpu_views = relay.cpu_to_gpu(module, target_device)
                torch.cuda.synchronize()
                _reassign_module_data(module, gpu_views, target_device)

            _set_single_gpu_device_map(module, self.target_gpus[0])
        else:
            # Multi-GPU: use dispatch_model for cross-device hooks.
            # Release pool if exists (can't mirror multi-GPU)
            if self._pinned_pool is not None:
                self._pinned_pool.release()
                self._pinned_pool = None

            self.logger.info("from_cache: multi-GPU path")

            device_map = _get_device_map(module, "auto", max_memory, None)

            remove_accelerate_hooks(module)

            self.model._module = dispatch_model(module, device_map)

            torch.cuda.synchronize()

        gc.collect()
        torch.cuda.empty_cache()

        load_time = time.time() - start

        self._verify_device_placement(self.model._module, "cache")

        self.logger.info(f"Model loaded from cache in {load_time:.2f}s")

        ModelLoadTimeMetric.update(load_time, self.model_key, "cache")

        self.cached = False

        # Keep existing pinned pool if layout matches; otherwise re-allocate
        self._start_pinned_preallocation(reuse_ok=True)

    def release_mirror(self):
        """Release the pinned mirror pool, freeing CPU memory.

        Called by the controller when CPU memory is tight. The model continues
        serving from GPU; future evictions will fall back to relay buffer.
        """
        if self._pinned_pool is not None:
            self._pinned_pool.release()
            self._pinned_pool = None
            self.logger.info("Pinned mirror released (controller request)")

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

        self.request = request

        try:
            result = None

            inputs = self.pre()

            job_task = asyncio.create_task(asyncio.to_thread(self.execute, inputs))
            kill_task = asyncio.create_task(self.kill_switch.wait())

            done, pending = await asyncio.wait(
                [job_task, kill_task],
                timeout=self.execution_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()

            if job_task in done:
                result = await job_task
            elif kill_task in done:
                kill_thread(self.execution_ident)
                raise Exception("Your job was cancelled or preempted by the server.")
            else:
                kill_thread(self.execution_ident)
                raise Exception(
                    f"Job took longer than timeout: {self.execution_timeout} seconds"
                )

            self.post(result)

        except Exception as e:
            self.exception(e)

        finally:
            del request
            del result

            self.cleanup()

    async def cancel(self):
        if self.execution_ident is not None:
            self.kill_switch.set()

    # Ray checks this method and restarts replica if it raises an exception
    def check_health(self):
        pass

    ### ABSTRACT METHODS #################################

    def pre(self) -> RequestModel:

        self.respond(
            status=BackendResponseModel.JobStatus.RUNNING,
            description="Your job has started running.",
        )

        """Logic to execute before execution."""
        with Protector(WHITELISTED_MODULES_DESERIALIZATION):
            request = self.request.deserialize(self.persistent_objects)

        return request

    def execute(self, request: RequestModel) -> Any:
        """Execute request.

        Args:
            request (BackendRequestModel): Request.

        Returns:
            Any: Result.
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

    def post(self, result: Any) -> None:
        """Logic to execute after execution with result from `.execute`.

        Args:
            request (BackendRequestModel): Request.
            result (Any): Result.
        """

        saves = result[0]
        gpu_mem: int = result[1]
        execution_time_s: float = result[2]

        result_object = BackendResultModel(
            id=self.request.id,
            **saves,
        ).save(compress=self.request.compress)

        self.respond(
            status=BackendResponseModel.JobStatus.COMPLETED,
            description="Your job has been completed.",
            data=(result_object.url(), result_object._size),
        )

        RequestResponseSizeMetric.update(self.request, result_object._size)
        GPUMemMetric.update(self.request, gpu_mem)
        ExecutionTimeMetric.update(self.request, execution_time_s)

    def exception(self, exception: Exception) -> None:
        """Handles exceptions that occur during model execution.

        This method processes different types of exceptions and sends appropriate error responses
        back to the client. For NNsight-specific errors, it includes detailed traceback information.
        For other errors, it includes the full exception traceback and message.

        Args:
            exception (Exception): The exception that was raised during __call__.
        """

        self.respond(
            status=BackendResponseModel.JobStatus.ERROR,
            description=str(exception),
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

        self.model._model.zero_grad()
        self.request = None

        Globals.clear()
        clear_set_attrs()
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

    def respond(self, **kwargs) -> None:
        """Sends a response back to the client.

        This method handles sending responses through either websocket
        or object store, depending on whether a session_id exists.

        If session_id exists:
        1. Establishes websocket connection if needed
        2. Sends response through websocket

        If no session_id:
        1. Saves response to object store

        Args:
            **kwargs: Arguments to be passed to create_response, including:
                - status: The job status
                - description: Human-readable status description
                - data: Optional additional data
        """

        try:
            self.request.create_response(**kwargs, logger=self.logger).respond()
        except:
            self.logger.exception("Error responding to client")


class BaseModelDeploymentArgs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_key: MODEL_KEY

    execution_timeout: float | None = None
    device_map: str | None = "auto"
    dispatch: bool = True
    dtype: str | torch.dtype = torch.bfloat16
    gpu_mem_bytes_by_id: Dict[int, int] | None = None
    spawn_timestamp: float | None = None


@ray.remote(num_cpus=2, num_gpus=0, max_restarts=-1)
class ModelActor(BaseModelDeployment):
    """Ray remote actor for model execution.

    This actor handles the actual model inference and is managed by the
    ModelDeployment class. It inherits from BaseModelDeployment to provide
    the core model functionality.
    """

    pass
