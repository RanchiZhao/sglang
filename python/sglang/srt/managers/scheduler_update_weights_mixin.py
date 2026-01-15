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
    UpdateWeightsFromAwexReqInput,
    UpdateWeightsFromAwexReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


def _log_gpu_memory(label: str, rank: int = None):
    """Log GPU memory usage for debugging memory issues."""
    try:
        if rank is None:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        if not torch.cuda.is_available():
            return

        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)

        try:
            max_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            free = max_memory - reserved
        except:
            max_memory = 0
            free = 0

        logger.info(
            f"[GPU_MEM] {label} | rank={rank} device={device} | "
            f"allocated={allocated:.2f}GB reserved={reserved:.2f}GB free={free:.2f}GB total={max_memory:.2f}GB"
        )
    except Exception as e:
        logger.warning(f"[GPU_MEM] Failed to log memory for {label}: {e}")


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
        worker = self.draft_worker or self.tp_worker
        success, message = worker.update_weights_from_tensor(recv_req)
        # TODO extract common code b/t update_weights_from_distributed and update_weights_from_tensor later
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
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

    def update_weights_from_awex(self: Scheduler, recv_req: UpdateWeightsFromAwexReqInput):
        """Update weights via AWEX colocate mode.

        This method is called by training side via Ray to trigger AWEX weight reception.
        The actual weight transfer happens through MetaServer + NCCL P2P.

        Flow:
        1. Training side sends Ray call to trigger this method
        2. This method calls AwexWeightReceiver.receive_weights()
        3. receive_weights() blocks waiting for training data from MetaServer
        4. Once data is available, P2P transfer happens
        5. Training side gets notified via MetaServer when done
        """
        try:
            # Check if AWEX receiver is initialized
            if not hasattr(self, "_awex_receiver") or self._awex_receiver is None:
                error_msg = "AWEX receiver not initialized. Enable AWEX with --enable-awex flag."
                logger.error(error_msg)
                return UpdateWeightsFromAwexReqOutput(success=False, message=error_msg)

            step_id = recv_req.step_id
            logger.info(f"[AWEX] Starting weight update for step {step_id}")

            # Call AWEX receiver - this blocks until weights are received
            success = self._awex_receiver.receive_weights(step_id=step_id)

            if success:
                # Flush cache if requested
                if recv_req.flush_cache:
                    flush_cache_success = self.flush_cache()
                    if not flush_cache_success:
                        logger.warning("Cache flush failed after AWEX weight update")

                # [GOLDEN_CHECK] Compare with golden stats after AWEX sync
                # Golden stats are saved in release_memory_occupation before weights are offloaded
                if hasattr(self, "_awex_golden_saved") and self._awex_golden_saved and hasattr(self, "tp_worker"):
                    try:
                        logger.info("[GOLDEN_CHECK] Comparing weights with golden standard after AWEX sync...")
                        self.tp_worker.model_runner.check_weights(action="compare_golden")
                    except Exception as e:
                        logger.warning(f"[GOLDEN_CHECK] Failed to compare with golden: {e}")

                message = f"AWEX weight update completed for step {step_id}"
                logger.info(f"[AWEX] {message}")
            else:
                message = f"AWEX weight update failed for step {step_id}"
                logger.error(f"[AWEX] {message}")

            return UpdateWeightsFromAwexReqOutput(success=success, message=message)

        except Exception as e:
            error_msg = f"AWEX weight update failed: {str(e)}"
            logger.error(f"[AWEX] {error_msg}")
            import traceback
            traceback.print_exc()
            return UpdateWeightsFromAwexReqOutput(success=False, message=error_msg)

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

        _log_gpu_memory(f"release_memory_occupation START tags={tags} offload_tags={self.offload_tags} memory_saver_enabled={self.memory_saver_adapter.enabled}")

        # [GOLDEN_CHECK] Save golden stats before weights are released
        # This captures the correct weights loaded from disk BEFORE offloading
        if GPU_MEMORY_TYPE_WEIGHTS in tags and GPU_MEMORY_TYPE_WEIGHTS not in self.offload_tags:
            if not hasattr(self, "_awex_golden_saved"):
                self._awex_golden_saved = False
            if not self._awex_golden_saved and hasattr(self, "tp_worker"):
                try:
                    logger.info("[GOLDEN_CHECK] Saving golden weight stats before memory release...")
                    self.tp_worker.model_runner.check_weights(action="save_golden")
                    self._awex_golden_saved = True
                    logger.info("[GOLDEN_CHECK] Golden weight stats saved successfully")
                except Exception as e:
                    logger.warning(f"[GOLDEN_CHECK] Failed to save golden stats: {e}")

        # Idempotent check: if all requested tags are already offloaded, skip entirely
        # This prevents issues when release_memory_occupation is called twice
        # (e.g., AWEX init + Slime offload) which could cause pause() to hang
        if all(tag in self.offload_tags for tag in tags):
            logger.info(f"[release_memory_occupation] All tags already offloaded, skipping: {tags}")
            return ReleaseMemoryOccupationReqOutput()

        # Filter out tags that are already offloaded to avoid re-pausing
        # This handles partial overlap (e.g., AWEX released kv_cache+weights,
        # then Slime requests kv_cache+weights+cuda_graph)
        tags_to_release = [tag for tag in tags if tag not in self.offload_tags]
        logger.info(f"[release_memory_occupation] tags_to_release={tags_to_release}")

        for tag in tags_to_release:
            self.offload_tags.add(tag)

        if GPU_MEMORY_TYPE_KV_CACHE in tags_to_release:
            _log_gpu_memory("release_memory_occupation BEFORE pause(kv_cache)")
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
            _log_gpu_memory("release_memory_occupation AFTER pause(kv_cache)")
            self.flush_cache()

            if self.disaggregation_mode == DisaggregationMode.DECODE:
                if hasattr(self, "disagg_decode_prealloc_queue"):
                    self.disagg_decode_prealloc_queue.release_memory_occupation()
            elif self.disaggregation_mode == DisaggregationMode.PREFILL:
                if hasattr(self, "disagg_prefill_bootstrap_queue"):
                    self.disagg_prefill_bootstrap_queue.release_memory_occupation()

        if GPU_MEMORY_TYPE_WEIGHTS in tags_to_release:
            _log_gpu_memory("release_memory_occupation BEFORE export_static_state")
            self.stashed_model_static_state = _export_static_state(
                self.tp_worker.model_runner.model
            )
            _log_gpu_memory("release_memory_occupation AFTER export_static_state")
            torch.distributed.barrier(self.tp_cpu_group)
            _log_gpu_memory("release_memory_occupation BEFORE pause(weights)")
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)
            _log_gpu_memory("release_memory_occupation AFTER pause(weights)")

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags_to_release:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_CUDA_GRAPH)

        torch.get_device_module().synchronize()

        _log_gpu_memory(f"release_memory_occupation END offload_tags={self.offload_tags}")
        return ReleaseMemoryOccupationReqOutput()

    def resume_memory_occupation(
        self: Scheduler, recv_req: ResumeMemoryOccupationReqInput
    ):
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        _log_gpu_memory(f"resume_memory_occupation START tags={tags} offload_tags={self.offload_tags} memory_saver_enabled={self.memory_saver_adapter.enabled}")

        # Only resume tags that were actually in offload_tags (i.e., paused)
        # This handles concurrent AWEX and HTTP release/resume calls where
        # a tag might have already been resumed by another caller
        tags_to_resume = [tag for tag in tags if tag in self.offload_tags]
        logger.info(f"[resume_memory_occupation] tags_to_resume={tags_to_resume}")

        for tag in tags_to_resume:
            self.offload_tags.remove(tag)

        # Only perform resume operations for tags that were actually paused
        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags_to_resume:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)

        if GPU_MEMORY_TYPE_WEIGHTS in tags_to_resume:
            _log_gpu_memory("resume_memory_occupation BEFORE resume(weights)")
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)
            _log_gpu_memory("resume_memory_occupation AFTER resume(weights)")
            torch.distributed.barrier(self.tp_cpu_group)
            _log_gpu_memory("resume_memory_occupation BEFORE import_static_state")
            _import_static_state(
                self.tp_worker.model_runner.model,
                self.stashed_model_static_state,
            )
            _log_gpu_memory("resume_memory_occupation AFTER import_static_state")
            del self.stashed_model_static_state

        if GPU_MEMORY_TYPE_KV_CACHE in tags_to_resume:
            _log_gpu_memory("resume_memory_occupation BEFORE resume(kv_cache)")
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)
            _log_gpu_memory("resume_memory_occupation AFTER resume(kv_cache)")

            if self.disaggregation_mode == DisaggregationMode.DECODE:
                if hasattr(self, "disagg_decode_prealloc_queue"):
                    self.disagg_decode_prealloc_queue.resume_memory_occupation()
            elif self.disaggregation_mode == DisaggregationMode.PREFILL:
                if hasattr(self, "disagg_prefill_bootstrap_queue"):
                    self.disagg_prefill_bootstrap_queue.resume_memory_occupation()

        _log_gpu_memory(f"resume_memory_occupation END offload_tags={self.offload_tags}")
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
