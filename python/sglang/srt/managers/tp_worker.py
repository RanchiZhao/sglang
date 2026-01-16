# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A tensor parallel worker."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.managers.io_struct import (
    DestroyWeightsUpdateGroupReqInput,
    GetWeightsByNameReqInput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterReqInput,
    SendWeightsToRemoteInstanceReqInput,
    UnloadLoRAAdapterReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import MultiprocessingSerializer, broadcast_pyobj, set_random_seed
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter

logger = logging.getLogger(__name__)


class BaseTpWorker(ABC):
    @abstractmethod
    def forward_batch_generation(self, forward_batch: ForwardBatch):
        pass

    @property
    @abstractmethod
    def model_runner(self) -> ModelRunner:
        pass

    @property
    def sliding_window_size(self) -> Optional[int]:
        return self.model_runner.sliding_window_size

    @property
    def is_hybrid_swa(self) -> bool:
        return self.model_runner.is_hybrid_swa is not None

    def get_tokens_per_layer_info(self):
        return (
            self.model_runner.full_max_total_num_tokens,
            self.model_runner.swa_max_total_num_tokens,
        )

    def get_pad_input_ids_func(self):
        return getattr(self.model_runner.model, "pad_input_ids", None)

    def get_tp_group(self):
        return self.model_runner.tp_group

    def get_attention_tp_group(self):
        return self.model_runner.attention_tp_group

    def get_attention_tp_cpu_group(self):
        return getattr(self.model_runner.attention_tp_group, "cpu_group", None)

    def get_memory_pool(self):
        return (
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool_allocator,
        )

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        success, message = self.model_runner.update_weights_from_disk(
            recv_req.model_path,
            recv_req.load_format,
            recapture_cuda_graph=recv_req.recapture_cuda_graph,
        )
        return success, message

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        success, message = self.model_runner.init_weights_update_group(
            recv_req.master_address,
            recv_req.master_port,
            recv_req.rank_offset,
            recv_req.world_size,
            recv_req.group_name,
            recv_req.backend,
        )
        return success, message

    def destroy_weights_update_group(self, recv_req: DestroyWeightsUpdateGroupReqInput):
        success, message = self.model_runner.destroy_weights_update_group(
            recv_req.group_name,
        )
        return success, message

    def init_weights_send_group_for_remote_instance(
        self, recv_req: InitWeightsSendGroupForRemoteInstanceReqInput
    ):
        success, message = (
            self.model_runner.init_weights_send_group_for_remote_instance(
                recv_req.master_address,
                recv_req.ports,
                recv_req.group_rank,
                recv_req.world_size,
                recv_req.group_name,
                recv_req.backend,
            )
        )
        return success, message

    def send_weights_to_remote_instance(
        self, recv_req: SendWeightsToRemoteInstanceReqInput
    ):
        success, message = self.model_runner.send_weights_to_remote_instance(
            recv_req.master_address,
            recv_req.ports,
            recv_req.group_name,
        )
        return success, message

    def update_weights_from_distributed(
        self, recv_req: UpdateWeightsFromDistributedReqInput
    ):
        success, message = self.model_runner.update_weights_from_distributed(
            recv_req.names,
            recv_req.dtypes,
            recv_req.shapes,
            recv_req.group_name,
            recv_req.load_format,
        )
        return success, message

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):

        monkey_patch_torch_reductions()
        success, message = self.model_runner.update_weights_from_tensor(
            named_tensors=MultiprocessingSerializer.deserialize(
                recv_req.serialized_named_tensors[self.tp_rank]
            ),
            load_format=recv_req.load_format,
        )
        return success, message

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update weights from IPC for checkpoint-engine integration."""
        success, message = self.model_runner.update_weights_from_ipc(recv_req)
        return success, message

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.model_runner.get_weights_by_name(
            recv_req.name, recv_req.truncate_size
        )
        return parameter

    def load_lora_adapter(self, recv_req: LoadLoRAAdapterReqInput):
        result = self.model_runner.load_lora_adapter(recv_req.to_ref())
        return result

    def unload_lora_adapter(self, recv_req: UnloadLoRAAdapterReqInput):
        result = self.model_runner.unload_lora_adapter(recv_req.to_ref())
        return result

    def can_run_lora_batch(self, lora_ids: list[str]) -> bool:
        return self.model_runner.lora_manager.validate_lora_batch(lora_ids)

    def forward_batch_embedding(self, model_worker_batch: ModelWorkerBatch):
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output, _ = self.model_runner.forward(forward_batch)
        embeddings = logits_output.embeddings
        return embeddings


class TpModelWorker(BaseTpWorker):
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
    ):
        # Parse args
        self.tp_size = server_args.tp_size
        self.tp_rank = tp_rank
        self.moe_ep_rank = moe_ep_rank
        self.pp_rank = pp_rank

        # Init model and tokenizer
        self.model_config = ModelConfig.from_server_args(
            server_args,
            model_path=(
                server_args.model_path
                if not is_draft_worker
                else server_args.speculative_draft_model_path
            ),
            model_revision=(
                server_args.revision
                if not is_draft_worker
                else server_args.speculative_draft_model_revision
            ),
            is_draft_model=is_draft_worker,
        )

        if server_args.dllm_algorithm is not None:
            self.dllm_algorithm = DllmAlgorithm.from_server_args(server_args)

        self._model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=server_args.ep_size,
            pp_rank=pp_rank,
            pp_size=server_args.pp_size,
            nccl_port=nccl_port,
            dp_rank=dp_rank,
            server_args=server_args,
            is_draft_worker=is_draft_worker,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
        self.device = self.model_runner.device

        # Init nccl groups
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()

        # Profile number of tokens
        self.max_total_num_tokens = self.model_runner.max_total_num_tokens
        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.max_running_requests = min(
            (
                self.max_total_num_tokens // 2
                if server_args.max_running_requests is None
                else server_args.max_running_requests
                // (server_args.dp_size if server_args.enable_dp_attention else 1)
            ),
            self.model_runner.req_to_token_pool.size,
        )
        assert self.max_running_requests > 0, "max_running_request is zero"
        self.max_queued_requests = server_args.max_queued_requests
        assert (
            self.max_queued_requests is None or self.max_queued_requests >= 1
        ), "If configured, max_queued_requests must be at least 1 for any work to be scheduled."
        self.max_req_len = min(
            self.model_config.context_len - 1,
            self.max_total_num_tokens - 1,
        )
        self.max_req_input_len = self.max_req_len - 5
        assert (
            self.max_req_len > 0 and self.max_req_input_len > 0
        ), "Memory pool size is too small"

        # Sync random seed across TP workers
        self.random_seed = broadcast_pyobj(
            [server_args.random_seed],
            self.tp_size * self.pp_rank + tp_rank,
            self.world_group.cpu_group,
            src=self.world_group.ranks[0],
        )[0]
        set_random_seed(self.random_seed)

        # Pre-collect AWEX metadata if colocate mode is enabled
        # This must happen when ALL workers are synchronized (right after broadcast_pyobj)
        # All 64 workers participate in all_gather, avoiding the deadlock issue
        self._awex_global_params_meta = None
        if getattr(server_args, "enable_colocate_mode", False):
            self._pre_collect_awex_metadata(server_args)

        self.enable_overlap = not server_args.disable_overlap_schedule
        self.enable_spec = server_args.speculative_algorithm is not None
        self.hicache_layer_transfer_counter = None

    @property
    def model_runner(self) -> ModelRunner:
        return self._model_runner

    def _pre_collect_awex_metadata(self, server_args):
        """
        Pre-collect AWEX metadata from ALL workers during initialization.

        This method is called when all workers are synchronized (right after broadcast_pyobj).
        It collects parameter metadata from each worker and does an all_gather to get
        the global metadata from all 64 workers.

        This solves the deadlock issue in execute_task_in_model_worker where only
        some workers would call all_gather.

        Args:
            server_args: Server arguments containing configuration
        """
        import torch.distributed as dist
        from sglang.srt.distributed import get_tp_group

        try:
            rank = dist.get_rank() if dist.is_initialized() else 0
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            logger.info(
                f"[AWEX_PRE_COLLECT] rank={rank} world_size={world_size} "
                f"Starting AWEX metadata pre-collection..."
            )

            # Try to import AWEX and build proper metadata
            try:
                from awex.sharding.rank_info import RankInfo
                from awex.models.registry import get_infer_weights_converter

                # Get model and rank info
                model = self.model_runner.model
                tp_group = get_tp_group()
                tp_rank = tp_group.rank_in_group if tp_group else 0
                tp_size = server_args.tp_size if server_args else 1
                pp_rank = self.pp_rank
                pp_size = getattr(server_args, "pp_size", 1)
                ep_size = getattr(server_args, "ep_size", 1)
                dp_size = getattr(server_args, "dp_size", 1)

                # Use the correct EP rank from TpModelWorker constructor
                # self.moe_ep_rank is set from the constructor parameter (line 223)
                ep_rank = self.moe_ep_rank if self.moe_ep_rank is not None else 0

                # Calculate EP-related TP sizes
                if ep_size > 1:
                    ep_tp_size = tp_size // ep_size
                    ep_tp_rank = tp_rank % ep_tp_size
                else:
                    ep_tp_size = 1
                    ep_tp_rank = 0

                # Create RankInfo object directly (avoid needing scheduler)
                # engine_rank is available from server_args (set per Ray Actor engine)
                engine_rank = getattr(server_args, "engine_rank", 0)
                rank_info = RankInfo(
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    pp_rank=pp_rank,
                    pp_size=pp_size,
                    dp_size=dp_size,
                    dp_rank=0,
                    ep_rank=ep_rank,
                    ep_size=ep_size,
                    ep_tp_rank=ep_tp_rank,
                    ep_tp_size=ep_tp_size,
                    attn_tp_rank=tp_rank,  # Same as tp_rank if no dp_attention
                    attn_tp_size=tp_size,
                    attn_dp_rank=0,
                    world_size=tp_size * pp_size,
                    global_rank=rank,
                    engine_rank=engine_rank,
                    local_rank=self.model_runner.gpu_id,
                    is_infer=True,
                )

                logger.info(
                    f"[AWEX_PRE_COLLECT] rank={rank} created RankInfo: "
                    f"tp_rank={rank_info.tp_rank}, ep_rank={rank_info.ep_rank}, "
                    f"engine_rank={rank_info.engine_rank}, global_rank={rank_info.global_rank}"
                )

                # Build infer_engine_config dict for converter
                infer_engine_config = {
                    "tp_size": tp_size,
                    "pp_size": pp_size,
                    "ep_size": ep_size,
                    "dp_size": dp_size,
                    "enable_dp_attention": getattr(server_args, "enable_dp_attention", False),
                    "enable_dp_lm_head": getattr(server_args, "enable_dp_lm_head", False),
                    "moe_dense_tp_size": getattr(server_args, "moe_dense_tp_size", None),
                }

                # Get the converter for parameter name conversion
                # Note: get_infer_weights_converter signature is:
                #   (engine_name, model_name, hf_config, rank_info, infer_engine_config)
                model_arch_name = type(model).__name__
                sglang_to_hf_converter = get_infer_weights_converter(
                    engine_name="sglang",
                    model_name=model_arch_name,  # model_name is the arch name like "DeepseekV3ForCausalLM"
                    hf_config=model.config,
                    rank_info=rank_info,
                    infer_engine_config=infer_engine_config,
                )

                # Collect parameter metadata with conversion
                # Note: dtype is stored as torch.dtype object (pickle-able via all_gather_object)
                params_meta = []
                for name, param in model.named_parameters():
                    try:
                        converted = sglang_to_hf_converter.convert_param(name, param)
                        for hf_name, hf_param in converted:
                            params_meta.append({
                                "name": hf_name,
                                "shape": list(hf_param.shape),
                                "dtype": hf_param.dtype,  # torch.dtype object, not string
                                "numel": hf_param.numel(),
                            })
                    except Exception as e:
                        logger.warning(f"[AWEX_PRE_COLLECT] Failed to convert {name}: {e}")
                        # Fall back to original name
                        params_meta.append({
                            "name": name,
                            "shape": list(param.shape),
                            "dtype": param.dtype,  # torch.dtype object, not string
                            "numel": param.numel(),
                        })

                local_meta = {
                    "rank_info": rank_info,
                    "params_meta": params_meta,
                    "model_arch_name": model_arch_name,
                }
                logger.info(
                    f"[AWEX_PRE_COLLECT] rank={rank} collected {len(params_meta)} parameters"
                )

            except ImportError as e:
                logger.warning(
                    f"[AWEX_PRE_COLLECT] rank={rank} AWEX not available, "
                    f"falling back to basic metadata collection: {e}"
                )
                # Fall back to basic metadata collection
                local_meta = self._collect_local_param_meta_basic(server_args)

            logger.info(
                f"[AWEX_PRE_COLLECT] rank={rank} starting all_gather_object (world_size={world_size})..."
            )

            # All 64 workers do all_gather together
            # This is safe because all workers are synchronized at this point
            if dist.is_initialized() and world_size > 1:
                all_results = [None] * world_size
                dist.all_gather_object(all_results, local_meta, group=self.world_group.cpu_group)
                self._awex_global_params_meta = all_results
                logger.info(
                    f"[AWEX_PRE_COLLECT] rank={rank} all_gather completed, "
                    f"got metadata from {len(all_results)} workers"
                )
            else:
                self._awex_global_params_meta = [local_meta]
                logger.info(f"[AWEX_PRE_COLLECT] rank={rank} single worker, using local metadata only")

        except Exception as e:
            logger.error(f"[AWEX_PRE_COLLECT] failed: {e}")
            import traceback
            traceback.print_exc()
            # Don't fail initialization, just set to None
            self._awex_global_params_meta = None

    def _collect_local_param_meta_basic(self, server_args):
        """
        Basic metadata collection fallback when AWEX is not available.

        This collects minimal parameter information without AWEX's converter.
        """
        import torch.distributed as dist
        from sglang.srt.distributed import get_tp_group

        model = self.model_runner.model
        tp_group = get_tp_group()
        tp_rank = tp_group.rank_in_group if tp_group else 0
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Create a simple rank_info dict (not RankInfo object)
        rank_info = {
            "tp_rank": tp_rank,
            "tp_size": server_args.tp_size,
            "pp_rank": self.pp_rank,
            "pp_size": getattr(server_args, "pp_size", 1),
            "ep_rank": self.moe_ep_rank,
            "ep_size": getattr(server_args, "ep_size", 1),
            "dp_rank": 0,
            "dp_size": getattr(server_args, "dp_size", 1),
            "global_rank": rank,
            "world_size": server_args.tp_size * getattr(server_args, "pp_size", 1),
        }

        # Collect basic parameter metadata
        params_meta = []
        for name, param in model.named_parameters():
            params_meta.append({
                "name": name,
                "shape": list(param.shape),
                "dtype": str(param.dtype),
                "numel": param.numel(),
            })

        return {
            "rank_info": rank_info,
            "params_meta": params_meta,
            "model_arch_name": type(model).__name__,
            "_is_basic_meta": True,  # Mark this as basic metadata
        }

    def get_awex_global_params_meta(self):
        """Return pre-collected AWEX global params metadata."""
        return self._awex_global_params_meta

    def register_hicache_layer_transfer_counter(self, counter: LayerDoneCounter):
        self.hicache_layer_transfer_counter = counter

    def set_hicache_consumer(self, consumer_index: int):
        if self.hicache_layer_transfer_counter is not None:
            self.hicache_layer_transfer_counter.set_consumer(consumer_index)

    def get_worker_info(self):
        return (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_queued_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            self.model_runner.req_to_token_pool.size,
            self.model_runner.req_to_token_pool.max_context_len,
            self.model_runner.token_to_kv_pool.size,
        )

    def is_dllm(self):
        return hasattr(self, "dllm_algorithm")

    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        forward_batch: Optional[ForwardBatch] = None,
        is_verify: bool = False,
        skip_attn_backend_init=False,
    ) -> GenerationBatchResult:
        # FIXME(lsyin): maybe remove skip_attn_backend_init in forward_batch_generation,
        #               which requires preparing replay to always be in this function

        if model_worker_batch is not None:
            # update the consumer index of hicache to the running batch
            self.set_hicache_consumer(model_worker_batch.hicache_consumer_index)

            forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        else:
            # FIXME(lsyin): unify the interface of forward_batch
            assert forward_batch is not None

        pp_proxy_tensors = None
        if not self.pp_group.is_first_rank:
            pp_proxy_tensors = PPProxyTensors(
                self.pp_group.recv_tensor_dict(
                    all_gather_group=self.get_attention_tp_group()
                )
            )

        if self.pp_group.is_last_rank:
            if self.is_dllm():
                logits_output, next_token_ids, can_run_cuda_graph = (
                    self.dllm_algorithm.run(self.model_runner, forward_batch)
                )
                return GenerationBatchResult(
                    logits_output=logits_output,
                    next_token_ids=next_token_ids,
                    can_run_cuda_graph=can_run_cuda_graph,
                )

            logits_output, can_run_cuda_graph = self.model_runner.forward(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
                skip_attn_backend_init=skip_attn_backend_init,
            )
            batch_result = GenerationBatchResult(
                logits_output=logits_output,
                can_run_cuda_graph=can_run_cuda_graph,
            )

            if is_verify:
                # Skip sampling and return logits for target forward
                return batch_result

            if (
                self.enable_overlap
                and not self.enable_spec
                and model_worker_batch.sampling_info.grammars is not None
            ):

                def sample_batch_func():
                    batch_result.next_token_ids = self.model_runner.sample(
                        logits_output, forward_batch
                    )
                    return batch_result

                batch_result.delay_sample_func = sample_batch_func
                return batch_result

            if model_worker_batch.is_prefill_only:
                # For prefill-only requests, create dummy token IDs on CPU
                # The size should match the batch size (number of sequences), not total tokens
                batch_result.next_token_ids = torch.zeros(
                    len(model_worker_batch.seq_lens),
                    dtype=torch.long,
                    device=model_worker_batch.input_ids.device,
                )
                if (
                    model_worker_batch.return_logprob
                    and logits_output.next_token_logits is not None
                ):
                    # NOTE: Compute logprobs without full sampling
                    self.model_runner.compute_logprobs_only(
                        logits_output, model_worker_batch
                    )
            else:
                batch_result.next_token_ids = self.model_runner.sample(
                    logits_output, forward_batch
                )

            return batch_result
        else:
            pp_proxy_tensors, can_run_cuda_graph = self.model_runner.forward(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
                skip_attn_backend_init=skip_attn_backend_init,
            )
            return GenerationBatchResult(
                pp_hidden_states_proxy_tensors=pp_proxy_tensors,
                can_run_cuda_graph=can_run_cuda_graph,
            )

    def forward_batch_split_prefill(self, batch: ScheduleBatch):
        if batch.split_index == 0:
            model_worker_batch = batch.get_model_worker_batch()
            forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
            batch.split_forward_batch = forward_batch
            batch.seq_lens_cpu_cache = model_worker_batch.seq_lens_cpu
        else:
            model_worker_batch = batch.get_model_worker_batch(batch.seq_lens_cpu_cache)

        logits_output, can_run_cuda_graph = self.model_runner.forward(
            batch.split_forward_batch, split_forward_count=batch.split_forward_count
        )
        if logits_output:
            next_token_ids = self.model_runner.sample(logits_output, model_worker_batch)
        else:
            next_token_ids = None
        batch_result = GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=can_run_cuda_graph,
        )
        batch_result.next_token_ids = next_token_ids
        return batch_result
