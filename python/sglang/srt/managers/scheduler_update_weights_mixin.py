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
    UpdateWeightsFromMetaserverReqInput,
    UpdateWeightsFromMetaserverReqOutput,
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

    def update_weights_from_metaserver(
        self: Scheduler, recv_req: UpdateWeightsFromMetaserverReqInput
    ) -> UpdateWeightsFromMetaserverReqOutput:
        """
        Receive and update weights via MetaServer P2P path.

        Each TpWorker:
        1. Gets its own gpu_identity (hostname_deviceid) - NOT scheduler's!
        2. Fetches IPC handles from MetaServer using that identity
        3. Deserializes and loads weights

        This ensures CUDA IPC handles are used on the correct physical GPU.
        """
        try:
            # Flush cache if requested
            if recv_req.flush_cache:
                self.flush_cache()

            # Delegate to TpWorker - it has the correct gpu_identity
            worker = self.draft_worker or self.tp_worker
            success, message = worker.update_weights_from_metaserver(recv_req)

            if not success:
                logger.error(f"[MetaServer P2P] Worker failed: {message}")

            # TP barrier to ensure all workers have received weights
            torch.distributed.barrier(group=self.tp_cpu_group)

            return UpdateWeightsFromMetaserverReqOutput(
                success=success,
                message=message,
            )
        except Exception as e:
            logger.error(f"[MetaServer P2P] Failed to update weights: {e}")
            import traceback
            traceback.print_exc()
            return UpdateWeightsFromMetaserverReqOutput(success=False, message=str(e))


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

