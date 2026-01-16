"""
Awex integration for SGLang weight synchronization.

This module provides adapter classes to integrate awex's optimized weight
synchronization into SGLang's inference framework.

Key optimizations over baseline:
- Receives weights in single batch instead of 317 chunks
- Uses NCCL P2P for efficient transfer in colocate mode
- Reduces IPC handle overhead through tensor grouping

Architecture:
- AwexWeightReceiver: Main entry point, wraps awex SGLangEngine
- SGLangSchedulerAdapter: Adapts SGLang Scheduler to sgl_engine interface
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


def is_awex_enabled(server_args) -> bool:
    """Check if awex integration is enabled."""
    return getattr(server_args, "enable_awex", False)


def get_awex_meta_server_addr(server_args) -> Optional[str]:
    """Get MetaServer address from server args or environment."""
    addr = getattr(server_args, "meta_server_addr", None)
    if addr is None:
        addr = os.environ.get("AWEX_META_SERVER_ADDR")
    return addr


class SGLangSchedulerAdapter:
    """
    Adapter that makes SGLang Scheduler look like an sgl_engine for awex.

    awex's SGLangEngine expects an sgl_engine with specific attributes.
    This adapter provides those attributes from the Scheduler.

    IMPORTANT: awex expects `tokenizer_manager.model_config` to have a `hf_config`
    attribute that is the actual HuggingFace PretrainedConfig. SGLang's ModelConfig
    stores this in `model_config.hf_config`, so we create a wrapper that exposes
    the HF config correctly.
    """

    def __init__(self, scheduler: "Scheduler", server_args: Any):
        self._scheduler = scheduler
        self._server_args = server_args

        # Build tokenizer_manager-like object for awex
        self.tokenizer_manager = self._create_tokenizer_manager_adapter()

    def _create_tokenizer_manager_adapter(self):
        """
        Create adapter for tokenizer_manager.model_config.

        awex expects:
        - sgl_engine.tokenizer_manager.model_config
        - This is passed to Engine.__init__ as hf_config
        - Later, code accesses hf_config.architectures, hf_config.to_dict(), etc.

        SGLang's ModelConfig is NOT the HF config - it wraps it in .hf_config.
        So we need to return the actual HF config, not the SGLang ModelConfig.
        """
        scheduler = self._scheduler

        class TokenizerManagerAdapter:
            def __init__(inner_self):
                inner_self.model_config = None

                try:
                    # Get model_config from scheduler
                    sglang_model_config = None
                    if hasattr(scheduler, "model_config"):
                        sglang_model_config = scheduler.model_config
                    elif hasattr(scheduler, "tp_worker") and scheduler.tp_worker is not None:
                        model_runner = getattr(scheduler.tp_worker, "model_runner", None)
                        if model_runner is not None:
                            sglang_model_config = getattr(model_runner, "model_config", None)

                    if sglang_model_config is not None:
                        # IMPORTANT: awex expects the actual HF config, not SGLang's ModelConfig
                        # SGLang stores HF config in model_config.hf_config
                        hf_config = getattr(sglang_model_config, "hf_config", None)
                        if hf_config is not None:
                            inner_self.model_config = hf_config
                            logger.info(f"[TokenizerManagerAdapter] Using HF config: {type(hf_config)}")
                        else:
                            # Fallback: use SGLang's model_config directly
                            inner_self.model_config = sglang_model_config
                            logger.warning("[TokenizerManagerAdapter] hf_config not found, using model_config directly")
                    else:
                        logger.warning("[TokenizerManagerAdapter] No model_config found")

                except Exception as e:
                    logger.warning(f"[TokenizerManagerAdapter] Failed to get model_config: {e}")

        return TokenizerManagerAdapter()

    def execute_task_in_model_worker(self, fn, **kwargs):
        """
        Execute a function in the model worker context.

        awex uses this to extract model parameter information. The function
        receives 'model' and 'model_context' as kwargs.

        Metadata collection strategy depends on awex_per_node_mode:

        1. awex_per_node_mode=True (recommended for multi-node colocate):
           - All workers initialize WeightsReader
           - All workers call this function
           - All workers participate in all_gather → get global metadata
           - TransferPlan can be built with full information

        2. awex_per_node_mode=False (default, legacy):
           - Only node_rank=0 workers initialize WeightsReader
           - Only those workers call this function
           - Cannot do all_gather (other workers don't participate → deadlock)
           - Return local metadata only, TransferPlan is partial
        """
        import torch.distributed as dist

        try:
            model_runner = self._scheduler.tp_worker.model_runner
            model = model_runner.model

            # Build model_context with required info for awex
            model_context = self._build_model_context()

            rank = dist.get_rank() if dist.is_initialized() else 0
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            logger.info(f"[SGLangSchedulerAdapter] rank={rank} world_size={world_size} executing task function...")

            # Execute the function with model and context
            result = fn(model=model, model_context=model_context, **kwargs)

            logger.info(f"[SGLangSchedulerAdapter] rank={rank} task function completed")

            # awex expects a list of results (one per rank)
            if isinstance(result, dict) and "rank_info" in result:
                enable_colocate = getattr(self._server_args, "enable_colocate_mode", False)
                awex_per_node_mode = getattr(self._server_args, "awex_per_node_mode", False)

                if enable_colocate and awex_per_node_mode:
                    # Per-node mode: ALL workers participate, can do all_gather
                    # This gives us global metadata for proper TransferPlan construction
                    if dist.is_initialized() and world_size > 1:
                        logger.info(
                            f"[SGLangSchedulerAdapter] rank={rank} starting all_gather_object "
                            f"(colocate + per_node_mode, world_size={world_size})..."
                        )
                        all_results = [None] * world_size
                        dist.all_gather_object(all_results, result)
                        logger.info(
                            f"[SGLangSchedulerAdapter] rank={rank} gathered metadata from {len(all_results)} ranks"
                        )
                        return all_results
                    else:
                        return [result]
                elif enable_colocate:
                    # Legacy colocate mode (per_node_mode=False):
                    # Only node_rank=0 workers call this, cannot do all_gather
                    # Return local metadata only - TransferPlan will be partial
                    logger.info(
                        f"[SGLangSchedulerAdapter] rank={rank} returning local metadata only "
                        f"(colocate mode, per_node_mode=False)"
                    )
                    return [result]
                elif dist.is_initialized() and world_size > 1:
                    # Non-colocate mode: all workers participate, do all_gather
                    logger.info(f"[SGLangSchedulerAdapter] rank={rank} starting all_gather_object (world_size={world_size})...")
                    all_results = [None] * world_size
                    dist.all_gather_object(all_results, result)
                    logger.info(
                        f"[SGLangSchedulerAdapter] rank={rank} gathered metadata from {len(all_results)} ranks"
                    )
                    return all_results
                else:
                    # Single rank: just return local result
                    return [result]
            return result

        except Exception as e:
            logger.error(f"[SGLangSchedulerAdapter] execute_task_in_model_worker failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _build_model_context(self):
        """Build model context dict for awex's get_sglang_rank_info."""
        import torch.distributed as dist
        from sglang.srt.distributed import get_tp_group

        scheduler = self._scheduler
        server_args = self._server_args

        # Get distributed info
        tp_group = get_tp_group()
        tp_rank = tp_group.rank_in_group if tp_group else 0
        tp_size = server_args.tp_size if server_args else 1
        pp_rank = getattr(scheduler, "pp_rank", 0)
        pp_size = getattr(server_args, "pp_size", 1)
        ep_size = getattr(server_args, "ep_size", 1)
        dp_size = getattr(server_args, "dp_size", 1)
        dp_rank = getattr(scheduler, "dp_rank", 0) or 0

        # Attention TP info (for DP attention)
        attn_tp_rank = getattr(scheduler, "attn_tp_rank", tp_rank)
        attn_tp_size = getattr(scheduler, "attn_tp_size", tp_size)
        attn_dp_rank = getattr(scheduler, "attn_dp_rank", 0)

        # Global and local rank
        global_rank = dist.get_rank() if dist.is_initialized() else 0
        local_rank = getattr(scheduler, "gpu_id", 0)

        # Build context with all fields needed by awex's get_sglang_rank_info
        model_context = {
            "scheduler": scheduler,
            "infer_engine_config": {
                "tp_size": tp_size,
                "pp_size": pp_size,
                "ep_size": ep_size,
                "dp_size": dp_size,
                "enable_dp_attention": getattr(server_args, "enable_dp_attention", False),
                "enable_dp_lm_head": getattr(server_args, "enable_dp_lm_head", False),
                "moe_dense_tp_size": getattr(server_args, "moe_dense_tp_size", None),
            },
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "pp_rank": pp_rank,
            "pp_size": pp_size,
            "ep_rank": getattr(scheduler, "moe_ep_rank", 0),
            "ep_size": ep_size,
            "dp_rank": dp_rank,
            "dp_size": dp_size,
            "attn_tp_rank": attn_tp_rank,
            "attn_tp_size": attn_tp_size,
            "attn_dp_rank": attn_dp_rank,
            "world_size": tp_size * pp_size,
            "global_rank": global_rank,
            "local_rank": local_rank,
        }

        return model_context

    def update_weights_from_disk(
        self, model_path: str, load_format: Optional[str] = None
    ):
        """Delegate to tp_worker."""
        from sglang.srt.managers.io_struct import UpdateWeightFromDiskReqInput

        req = UpdateWeightFromDiskReqInput(
            model_path=model_path, load_format=load_format, flush_cache=False
        )
        return self._scheduler.tp_worker.update_weights_from_disk(req)

    def release_memory_occupation(self, tags=None):
        """Delegate to scheduler."""
        from sglang.srt.managers.io_struct import ReleaseMemoryOccupationReqInput

        req = ReleaseMemoryOccupationReqInput(tags=tags)
        return self._scheduler.release_memory_occupation(req)

    def resume_memory_occupation(self, tags=None):
        """Delegate to scheduler."""
        from sglang.srt.managers.io_struct import ResumeMemoryOccupationReqInput

        req = ResumeMemoryOccupationReqInput(tags=tags)
        return self._scheduler.resume_memory_occupation(req)


class AwexWeightReceiver:
    """
    SGLang-side weight receiver that wraps awex SGLangEngine.

    This class handles receiving weights from the training side via awex's
    optimized colocate mode transfer.

    Args:
        scheduler: SGLang Scheduler instance
        server_args: Server arguments containing awex configuration
    """

    def __init__(self, scheduler: "Scheduler", server_args: Any):
        if scheduler is None:
            raise ValueError("scheduler cannot be None")
        if server_args is None:
            raise ValueError("server_args cannot be None")

        self._scheduler = scheduler
        self._server_args = server_args
        self._awex_engine = None  # awex SGLangEngine
        self._initialized = False

    def _create_awex_config(self):
        """Create awex InferenceConfig from server args."""
        try:
            from awex.config import InferenceConfig

            # Build config dict from server_args
            # Include all fields that awex WeightsReader needs
            config_dict = {
                "meta_server_addr": get_awex_meta_server_addr(self._server_args),
                "enable_colocate_mode": getattr(
                    self._server_args, "enable_colocate_mode", True
                ),
                "awex_per_node_mode": getattr(
                    self._server_args, "awex_per_node_mode", False
                ),
                "num_engines": getattr(self._server_args, "num_engines", 1),
                "engine_rank": getattr(self._server_args, "engine_rank", 0),
                "tp_size": getattr(self._server_args, "tp_size", 1),
                "pp_size": getattr(self._server_args, "pp_size", 1),
                "dp_size": getattr(self._server_args, "dp_size", 1),
                "ep_size": getattr(self._server_args, "ep_size", 1),
                "node_rank": getattr(self._server_args, "node_rank", 0),
                "local_rank": getattr(self._scheduler, "gpu_id", 0),
                "comm_backend": "nccl",  # Use NCCL for optimized transfer
                # DP attention and LM head config (critical for sharding strategy!)
                "enable_dp_attention": getattr(self._server_args, "enable_dp_attention", False),
                "enable_dp_lm_head": getattr(self._server_args, "enable_dp_lm_head", False),
                "moe_dense_tp_size": getattr(self._server_args, "moe_dense_tp_size", 1),
                # MoE A2A backend for shared_experts sharding (deepep/mooncake = NO_SHARDING)
                "moe_a2a_backend": getattr(self._server_args, "moe_a2a_backend", "none"),
                # Additional fields needed by WeightsReader
                "weights_exchange_ipc_backend": "cuda",
                "weights_validation_steps": 0,
                "validate_weights_every_n_steps": 1,
                "dump_weights_list_for_validation": [],
                "dump_weights_dir_for_validation": None,
                "weights_comm_nccl_group_size": 1,
                "enable_debug_mode": False,
            }

            logger.info(f"[AwexWeightReceiver] Creating config: num_engines={config_dict['num_engines']}, "
                       f"engine_rank={config_dict['engine_rank']}, node_rank={config_dict['node_rank']}, "
                       f"awex_per_node_mode={config_dict['awex_per_node_mode']}, "
                       f"enable_dp_attention={config_dict['enable_dp_attention']}, "
                       f"enable_dp_lm_head={config_dict['enable_dp_lm_head']}")
            return InferenceConfig(**config_dict)

        except ImportError as e:
            raise ImportError(
                "awex is not installed. Please install it with: "
                "cd /mnt/hisys-data/yqzhao/asystem-awex && pip install -e ."
            ) from e

    def initialize(self):
        """
        Initialize awex engine and register with MetaServer.

        IMPORTANT: This must be called during Scheduler.__init__ to register
        num_infer_engines with MetaServer before training side starts.
        """
        if self._initialized:
            return

        rank = (
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        )
        logger.info(f"[Rank {rank}] Initializing AwexWeightReceiver...")

        # Create adapter for the scheduler
        sgl_adapter = SGLangSchedulerAdapter(self._scheduler, self._server_args)

        # Create awex config
        config = self._create_awex_config()
        logger.info(f"[Rank {rank}] Created awex config: num_engines={config.num_engines}, engine_rank={config.engine_rank}")

        try:
            from awex.engine.sglang import SGLangEngine

            # Create awex SGLangEngine
            self._awex_engine = SGLangEngine(config, sgl_adapter)

            # Initialize - this registers num_infer_engines with MetaServer
            self._awex_engine.initialize()

            self._initialized = True
            logger.info(
                f"[Rank {rank}] AwexWeightReceiver initialized and registered with MetaServer"
            )

        except ImportError as e:
            logger.error(f"[Rank {rank}] awex not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"[Rank {rank}] Failed to initialize awex engine: {e}")
            import traceback
            traceback.print_exc()
            raise

    def receive_weights(self, step_id: int) -> bool:
        """
        Receive weights from training side.

        Args:
            step_id: Training step ID

        Returns:
            True if weights were received successfully
        """
        if not self._initialized:
            self.initialize()

        rank = (
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        )

        if rank == 0:
            logger.info(f"[AwexWeightReceiver] Receiving weights for step {step_id}...")

        try:
            # Use awex engine to receive weights
            self._awex_engine.update_weights(step_id=step_id)

            if rank == 0:
                logger.info(
                    f"[AwexWeightReceiver] Weights for step {step_id} received successfully"
                )
            return True

        except Exception as e:
            logger.error(f"[AwexWeightReceiver] Failed to receive weights: {e}")
            raise
