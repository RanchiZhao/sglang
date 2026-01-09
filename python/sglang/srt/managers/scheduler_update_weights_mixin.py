from __future__ import annotations

import logging
import os
import traceback
from typing import TYPE_CHECKING, Tuple

import torch

from sglang.srt.constants import (
    GPU_MEMORY_ALL_TYPES,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed import get_moe_ep_group, get_moe_tp_group, get_tp_group
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.managers.io_struct import (
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
    UpdateWeightsFromAwexReqInput,
    UpdateWeightsFromAwexReqOutput,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerUpdateWeightsMixin:

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """In-place update of the weights from disk."""
        success, message = self.tp_worker.update_weights_from_disk(recv_req)
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return UpdateWeightFromDiskReqOutput(success, message, 0)

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        """Initialize the online model parameter update group."""
        success, message = self.tp_worker.init_weights_update_group(recv_req)
        return InitWeightsUpdateGroupReqOutput(success, message)

    def destroy_weights_update_group(self, recv_req: DestroyWeightsUpdateGroupReqInput):
        """Destroy the online model parameter update group."""
        success, message = self.tp_worker.destroy_weights_update_group(recv_req)
        return DestroyWeightsUpdateGroupReqOutput(success, message)

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter."""
        success, message = self.tp_worker.update_weights_from_distributed(recv_req)
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return UpdateWeightsFromDistributedReqOutput(success, message)

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        """Update the online model parameter from tensors."""
        import os
        import time
        profile_enabled = os.environ.get("SLIME_BASELINE_PROFILE", "0") == "1"
        deep_profile_enabled = os.environ.get("SLIME_DEEP_PROFILE", "0") == "1"

        # Deep profiling: measure Ray latency from submit timestamp
        if deep_profile_enabled and hasattr(recv_req, '_submit_ts') and recv_req._submit_ts:
            ray_latency = time.time() - recv_req._submit_ts
            try:
                if torch.distributed.get_rank(group=self.tp_cpu_group) == 0:
                    log_msg = f"[Ray Latency] {ray_latency*1000:.1f}ms"
                    print(log_msg, flush=True)
                    try:
                        with open("/mnt/hisys-data/yqzhao/deep_profile.log", "a") as f:
                            f.write(log_msg + "\n")
                            f.flush()
                            os.fsync(f.fileno())
                    except Exception:
                        pass
            except Exception:
                pass

        if profile_enabled:
            t_start = time.time()
        
        # [6.1] Worker processing
        worker = self.draft_worker or self.tp_worker
        success, message = worker.update_weights_from_tensor(recv_req)
        
        if profile_enabled:
            worker_time = time.time() - t_start
            t_barrier_start = time.time()
        
        # TODO extract common code b/t update_weights_from_distributed and update_weights_from_tensor later
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        
        # [6.2] TP Barrier - wait for all workers to finish
        torch.distributed.barrier(group=self.tp_cpu_group)
        
        if profile_enabled:
            barrier_time = time.time() - t_barrier_start
            total_time = time.time() - t_start
            # Only print from rank 0 scheduler to reduce noise
            try:
                if torch.distributed.get_rank(group=self.tp_cpu_group) == 0:
                    log_msg = (
                        f"[SGLang Scheduler Profile] worker={worker_time*1000:.1f}ms "
                        f"barrier={barrier_time*1000:.1f}ms total={total_time*1000:.1f}ms"
                    )
                    print(log_msg, flush=True)
                    # Also write to shared storage for reliability
                    try:
                        with open("/mnt/hisys-data/yqzhao/sglang_profile.log", "a") as f:
                            f.write(log_msg + "\n")
                            f.flush()
                            os.fsync(f.fileno())
                    except Exception:
                        pass
            except Exception:
                # Fallback if group rank check fails
                pass
        
        return UpdateWeightsFromTensorReqOutput(success, message)

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        success, message = self.tp_worker.update_weights_from_ipc(recv_req)
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromIPCReqOutput(success, message)

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter)

    def release_memory_occupation(
        self: Scheduler, recv_req: ReleaseMemoryOccupationReqInput
    ):
        assert (
            self._is_no_request()
        ), "release_memory_occupation should be called only when no ongoing request."

        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.add(tag)

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
            self.flush_cache()

            if self.disaggregation_mode == DisaggregationMode.DECODE:
                if hasattr(self, "disagg_decode_prealloc_queue"):
                    self.disagg_decode_prealloc_queue.release_memory_occupation()
            elif self.disaggregation_mode == DisaggregationMode.PREFILL:
                if hasattr(self, "disagg_prefill_bootstrap_queue"):
                    self.disagg_prefill_bootstrap_queue.release_memory_occupation()

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.stashed_model_static_state = _export_static_state(
                self.tp_worker.model_runner.model
            )
            torch.distributed.barrier(self.tp_cpu_group)
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_CUDA_GRAPH)

        torch.get_device_module().synchronize()

        return ReleaseMemoryOccupationReqOutput()

    def resume_memory_occupation(
        self: Scheduler, recv_req: ResumeMemoryOccupationReqInput
    ):
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.remove(tag)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)
            torch.distributed.barrier(self.tp_cpu_group)
            _import_static_state(
                self.tp_worker.model_runner.model,
                self.stashed_model_static_state,
            )
            del self.stashed_model_static_state

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)

            if self.disaggregation_mode == DisaggregationMode.DECODE:
                if hasattr(self, "disagg_decode_prealloc_queue"):
                    self.disagg_decode_prealloc_queue.resume_memory_occupation()
            elif self.disaggregation_mode == DisaggregationMode.PREFILL:
                if hasattr(self, "disagg_prefill_bootstrap_queue"):
                    self.disagg_prefill_bootstrap_queue.resume_memory_occupation()

        return ResumeMemoryOccupationReqOutput()

    def check_weights(self: Scheduler, recv_req: CheckWeightsReqInput):
        try:
            self.tp_worker.model_runner.check_weights(action=recv_req.action)
            return CheckWeightsReqOutput(success=True, message="Success.")
        except Exception as e:
            logger.warning(f"check_weights see error: {e}")
            traceback.print_exc()
            return CheckWeightsReqOutput(success=False, message=f"{e}")

    def save_remote_model(self: Scheduler, params):
        url = params["url"]

        self.tp_worker.model_runner.save_remote_model(url)

        if self.draft_worker is not None:
            draft_url = params.get("draft_url", None)
            assert (
                draft_url is not None
            ), "draft_url must be provided when draft model is enabled"
            self.draft_worker.model_runner.save_remote_model(draft_url)

    def save_sharded_model(self: Scheduler, params):
        self.tp_worker.model_runner.save_sharded_model(
            path=params["path"],
            pattern=params["pattern"],
            max_size=params["max_size"],
        )

    # Awex integration methods
    _awex_receiver = None  # Initialized at scheduler startup if enabled

    def _init_awex_receiver(self: Scheduler):
        """
        Initialize awex receiver if enabled.

        IMPORTANT: This must be called during Scheduler.__init__ (not lazily)
        because the awex reader.initialize() registers 'num_infer_engines' to
        MetaServer, which the training side waits for before sending weights.

        Only tp_rank == 0 AND node_rank == 0 schedulers need to register.
        """
        if self._awex_receiver is not None:
            return

        try:
            from sglang.srt.managers.awex_integration import (
                AwexWeightReceiver,
                is_awex_enabled,
            )
        except ImportError as e:
            logger.debug(f"[Scheduler] awex integration not available: {e}")
            return

        if not is_awex_enabled(self.server_args):
            return

        # Only tp_rank == 0 AND node_rank == 0 needs to initialize
        # (registers num_infer_engines to MetaServer)
        node_rank = getattr(self.server_args, "node_rank", 0)
        tp_rank = getattr(self, "tp_rank", 0)
        if node_rank != 0 or tp_rank != 0:
            logger.info(
                f"[Scheduler] Skipping awex init for node_rank={node_rank}, tp_rank={tp_rank}"
            )
            return

        logger.info(
            f"[Scheduler] Awex enabled, creating receiver (node_rank=0, tp_rank=0)..."
        )

        try:
            self._awex_receiver = AwexWeightReceiver(self, self.server_args)
            logger.info("[Scheduler] Awex receiver created, initializing...")

            # Initialize immediately to register num_infer_engines with MetaServer
            self._awex_receiver.initialize()
            logger.info("[Scheduler] Awex receiver initialized and registered with MetaServer")
        except Exception as e:
            logger.error(f"[Scheduler] Failed to initialize awex receiver: {e}")
            import traceback
            traceback.print_exc()
            self._awex_receiver = None
            raise

    def update_weights_from_awex(
        self: Scheduler, recv_req: UpdateWeightsFromAwexReqInput
    ) -> UpdateWeightsFromAwexReqOutput:
        """
        Receive and update weights via awex optimized path.

        This method replaces the baseline chunk-based update with a single
        awex call that handles the entire weight transfer efficiently.

        Expected savings: ~12s (Ray round-trip overhead elimination)
        """
        self._init_awex_receiver()

        if self._awex_receiver is None:
            msg = "Awex receiver not initialized - is enable_awex set?"
            logger.error(msg)
            return UpdateWeightsFromAwexReqOutput(success=False, message=msg)

        try:
            # Flush cache before receiving new weights
            if recv_req.flush_cache:
                self.flush_cache()

            # Receive weights via awex
            success = self._awex_receiver.receive_weights(recv_req.step_id)

            # Barrier to ensure all workers have received weights
            torch.distributed.barrier(group=self.tp_cpu_group)

            return UpdateWeightsFromAwexReqOutput(
                success=success,
                message="Success" if success else "Failed to receive weights",
            )
        except Exception as e:
            logger.error(f"[Scheduler] Failed to update weights from awex: {e}")
            return UpdateWeightsFromAwexReqOutput(success=False, message=str(e))

    # MetaServer P2P integration methods
    _ms_p2p_client = None
    _ms_p2p_listener_thread = None
    _ms_p2p_last_version = 0

    def _get_gpu_identity(self: Scheduler) -> str:
        """Get unique GPU identity: hostname_deviceid."""
        import socket
        hostname = socket.gethostname()
        device_id = torch.cuda.current_device()
        return f"{hostname}_{device_id}"

    def _init_metaserver_p2p_listener(self: Scheduler):
        """
        Initialize MetaServer P2P listener if enabled.

        Creates a background thread that polls MetaServer for sync signals
        and fetches weights by GPU identity.
        """
        if self._ms_p2p_client is not None:
            return

        if not getattr(self.server_args, "enable_metaserver_p2p", False):
            return

        meta_server_addr = getattr(self.server_args, "meta_server_addr", None)
        if not meta_server_addr:
            logger.error("[MetaServer P2P] meta_server_addr is required")
            return

        # Parse address
        if ":" in meta_server_addr:
            host, port = meta_server_addr.rsplit(":", 1)
            port = int(port)
        else:
            logger.error(f"[MetaServer P2P] Invalid meta_server_addr format: {meta_server_addr}")
            return

        # Import and create MetaServer client
        import sys
        sys.path.insert(0, "/mnt/hisys-data/yqzhao/asystem-awex")
        from awex.meta.meta_server import MetaServerClient

        self._ms_p2p_client = MetaServerClient(host, port)
        self._ms_p2p_gpu_identity = self._get_gpu_identity()

        logger.info(
            f"[MetaServer P2P] Initialized listener: addr={meta_server_addr}, "
            f"gpu_identity={self._ms_p2p_gpu_identity}"
        )

        # Start background listener thread
        import threading
        self._ms_p2p_listener_thread = threading.Thread(
            target=self._metaserver_p2p_listener_loop,
            daemon=True,
            name="MetaServerP2PListener",
        )
        self._ms_p2p_listener_thread.start()
        logger.info("[MetaServer P2P] Background listener thread started")

    def _metaserver_p2p_listener_loop(self: Scheduler):
        """
        Background thread that polls MetaServer for new weight sync signals.

        Flow:
        1. Wait for sync signal (sync_v{version})
        2. GET all chunks by GPU identity (weights_{gpu_id}_v{version}_c{chunk_id})
        3. Load weights for each chunk
        4. Send single ACK (all_done_{gpu_id}_v{version})
        """
        import time
        poll_interval = 0.1  # Start with 100ms polling interval

        while True:
            try:
                # Check for new sync signal
                next_version = self._ms_p2p_last_version + 1
                sync_key = f"sync_v{next_version}"

                try:
                    sync_info = self._ms_p2p_client.get_object(sync_key, timeout=poll_interval)
                except Exception:
                    sync_info = None

                if sync_info is None:
                    time.sleep(poll_interval)
                    continue

                version = sync_info.get("version", next_version)
                total_chunks = sync_info.get("total_chunks", 0)

                logger.info(
                    f"[MetaServer P2P] Received sync signal: version={version}, "
                    f"chunks={total_chunks}"
                )

                # Process all chunks
                for chunk_id in range(total_chunks):
                    self._process_metaserver_p2p_chunk(version, chunk_id)

                # Send single ACK
                done_key = f"all_done_{self._ms_p2p_gpu_identity}_v{version}"
                self._ms_p2p_client.put_object(done_key, True)

                self._ms_p2p_last_version = version
                logger.info(
                    f"[MetaServer P2P] Weight update complete: version={version}, "
                    f"chunks={total_chunks}"
                )

            except Exception as e:
                logger.error(f"[MetaServer P2P] Listener error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)  # Back off on error

    def _process_metaserver_p2p_chunk(self: Scheduler, version: int, chunk_id: int):
        """Process a single chunk from MetaServer."""
        import time

        key = f"weights_{self._ms_p2p_gpu_identity}_v{version}_c{chunk_id}"

        # Wait for chunk to be available
        max_wait = 60  # 60 seconds timeout
        start_time = time.time()
        while True:
            try:
                chunk_data = self._ms_p2p_client.get_object(key, timeout=1)
                if chunk_data is not None:
                    break
            except Exception:
                pass

            if time.time() - start_time > max_wait:
                raise TimeoutError(f"Timeout waiting for chunk: {key}")
            time.sleep(0.1)

        # Extract data
        serialized_tensors = chunk_data.get("serialized_tensors", [])
        load_format = chunk_data.get("load_format", "flattened_bucket")
        weight_version = chunk_data.get("weight_version", str(version))

        # In P2P mode, serialized_tensors is a list of serialized data (one per dtype)
        # tp_worker expects serialized_named_tensors[tp_rank] to return the data
        # Since this is P2P mode, we only have this worker's data
        # Create a format compatible with tp_worker by processing each dtype separately
        for dtype_idx, serialized_data in enumerate(serialized_tensors):
            # Wrap single item in a list so [item][0] returns the item
            # But tp_worker uses self.tp_rank as index, so we need to handle this differently
            # Solution: call model_runner directly instead of going through tp_worker
            import pickle
            # Use standard pickle for CPU tensors (matches sender which uses pickle.dumps)
            named_tensors = pickle.loads(serialized_data)

            # Load weights directly to model_runner
            success, message = self.tp_worker.model_runner.update_weights_from_tensor(
                named_tensors=named_tensors,
                load_format=load_format,
            )
            if not success:
                logger.error(f"[MetaServer P2P] Failed to load chunk {chunk_id} dtype {dtype_idx}: {message}")


def _export_static_state(model):
    return dict(
        buffers=[
            (name, buffer.detach().clone()) for name, buffer in model.named_buffers()
        ]
    )


def _import_static_state(model, static_params):
    self_named_buffers = dict(model.named_buffers())
    for name, tensor in static_params["buffers"]:
        self_named_buffers[name][...] = tensor

