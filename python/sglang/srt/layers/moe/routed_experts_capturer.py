import logging
from abc import ABC
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather_into_tensor,
    get_attention_dp_rank,
    get_attention_tp_size,
    get_dp_local_info,
    is_dp_attention_enabled,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.server_args import get_global_server_args
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
)

logger = logging.getLogger(__name__)

_GB = 1024 * 1024 * 1024
_MB = 1024 * 1024


def get_tensor_size_bytes(t: torch.Tensor):
    return np.prod(t.shape) * t.dtype.itemsize


class _RoutedExpertsDeviceCache:
    def __init__(
        self,
        max_running_requests: int,
        num_hidden_layers: int,
        num_experts_per_tok: int,
        num_fused_shared_experts: int,
        device: str,
    ) -> None:
        self.buffer = torch.zeros(
            (
                max(
                    get_global_server_args().chunked_prefill_size
                    * get_global_server_args().dp_size,
                    max_running_requests,
                ),
                num_hidden_layers,
                num_experts_per_tok + num_fused_shared_experts,
            ),
            dtype=torch.int32,
            device=device,
        )
        self._finalize_allocation_log()

    def get_buffer_size_bytes(self):
        assert hasattr(self, "buffer")
        return get_tensor_size_bytes(self.buffer)

    def capture_fwd_routed_experts(self, layer_id: int, topk_ids: torch.Tensor):
        assert layer_id is not None, "capturing routing experts but get layer_id None"
        batch, _ = topk_ids.shape
        self.buffer[:batch, layer_id, :] = topk_ids

    def _finalize_allocation_log(self):
        """Common logging and memory usage computation for captured experts buffers."""
        buffer_size_MB = self.get_buffer_size_bytes() / _MB
        logger.info(
            f"Routing experts device buffer allocated. #shape: {tuple(self.buffer.shape)}, size: {buffer_size_MB:.2f} MB"
        )


class _RoutedExpertsHostCache:
    def __init__(
        self,
        num_tokens: int,
        num_hidden_layers: int,
        num_experts_per_tok: int,
    ) -> None:
        self.num_tokens = num_tokens
        self.buffer = torch.zeros(
            (
                num_tokens,
                num_hidden_layers,
                num_experts_per_tok,
            ),
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self._finalize_allocation_log()

    def get_buffer_size_bytes(self):
        assert hasattr(self, "buffer")
        return get_tensor_size_bytes(self.buffer)

    def _finalize_allocation_log(self):
        """Common logging and memory usage computation for captured experts buffers."""
        buffer_size_GB = self.get_buffer_size_bytes() / _GB
        logger.info(
            f"Routing experts host buffer allocated. #tokens: {self.num_tokens}, size: {buffer_size_GB:.2f} GB"
        )


class RoutedExpertsCapturer(ABC):
    @staticmethod
    def create(
        enable: bool,
        model_config: ModelConfig,
        num_fused_shared_experts: int,
        num_tokens: int,
        max_running_requests: int,
        device: str,
    ):
        if enable:
            return _RoutedExpertsCapturerReal(
                model_config,
                num_tokens=num_tokens,
                max_running_requests=max_running_requests,
                num_fused_shared_experts=num_fused_shared_experts,
                device=device,
            )
        else:
            return _RoutedExpertsCapturerNoop()

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        raise NotImplementedError

    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        raise NotImplementedError

    def sync_fwd_experts_buffer_DtoH(
        self,
        device_loc: torch.Tensor,
        cpu_loc: torch.Tensor,
        can_run_graph: bool,
        cuda_graph_batch: int,
    ):
        raise NotImplementedError

    @contextmanager
    def with_forward(self, forward_batch):
        yield

    def get_host_cache(self):
        raise NotImplementedError

    def get_device_cache(self):
        raise NotImplementedError


class _RoutedExpertsCapturerReal(RoutedExpertsCapturer):
    """Capturer for routed experts with host buffer"""

    def __init__(
        self,
        model_config: ModelConfig,
        num_tokens: int,
        max_running_requests: int,
        num_fused_shared_experts: int,
        device: str,
    ):
        self.forward_batch = None
        self.num_fused_shared_experts = num_fused_shared_experts
        self.num_hidden_layers = model_config.hf_text_config.num_hidden_layers
        self.num_experts_per_tok = model_config.hf_text_config.num_experts_per_tok

        self.host_cache = _RoutedExpertsHostCache(
            num_tokens=num_tokens,
            num_hidden_layers=self.num_hidden_layers,
            num_experts_per_tok=self.num_experts_per_tok,
        )

        self.device_cache = _RoutedExpertsDeviceCache(
            max_running_requests=max_running_requests,
            num_hidden_layers=self.num_hidden_layers,
            num_experts_per_tok=self.num_experts_per_tok,
            num_fused_shared_experts=self.num_fused_shared_experts,
            device=device,
        )

        if get_moe_a2a_backend().is_deepep():
            attn_tp_size = get_attention_tp_size() if is_dp_attention_enabled() else 1
            self.gather_buffer = torch.empty(
                (
                    self.device_cache.buffer.shape[0] * attn_tp_size,
                    self.device_cache.buffer.shape[2],
                ),
                dtype=torch.int32,
                device=device,
            )

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        if get_moe_a2a_backend().is_deepep():
            local_topk_ids = topk_ids
            topk_ids = self.gather_buffer[
                : local_topk_ids.size(0) * get_attention_tp_size()
            ]
            attn_tp_all_gather_into_tensor(topk_ids, local_topk_ids)
        self.device_cache.capture_fwd_routed_experts(layer_id, topk_ids)

    def sync_fwd_experts_buffer_DtoH(
        self,
        device_loc: torch.Tensor,
        cpu_loc: torch.Tensor,
        can_run_graph: bool,
        cuda_graph_batch: int,
    ):
        # When DeepEP is enabled, capture() already does all_gather, so device_cache.buffer
        # contains data from all DP ranks. We should not slice by DP rank in this case.
        if is_dp_attention_enabled() and not get_moe_a2a_backend().is_deepep():
            local_start_pos, local_num_tokens = get_dp_local_info(self.forward_batch)
            # handle with cuda graph padding
            if can_run_graph:
                local_start_pos = get_attention_dp_rank() * cuda_graph_batch
                local_end_pos = local_start_pos + local_num_tokens
            else:
                local_end_pos = local_start_pos + local_num_tokens
        else:
            local_start_pos = 0
            local_end_pos = device_loc.shape[0]

        if self.forward_batch.num_token_non_padded is not None:
            assert local_end_pos - local_start_pos >= self.forward_batch.num_token_non_padded
            local_end_pos = local_start_pos + self.forward_batch.num_token_non_padded
            cpu_loc = cpu_loc[: self.forward_batch.num_token_non_padded]

        self.host_cache.buffer[cpu_loc] = self.device_cache.buffer[
            local_start_pos:local_end_pos, :, : self.num_experts_per_tok
        ].cpu()

    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        cache_pool_idx = (
            req_to_token_pool.req_to_token[req_pool_idx][: seqlen - 1].cpu().clone()
        )
        return self.get_host_cache().buffer[cache_pool_idx]

    @contextmanager
    def with_forward(self, forward_batch):
        self.forward_batch = forward_batch
        yield

    def get_host_cache(self):
        return self.host_cache

    def get_device_cache(self):
        return self.device_cache


class _RoutedExpertsCapturerNoop(RoutedExpertsCapturer):
    def __init__(self):
        pass

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        pass

    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        pass

    def sync_fwd_experts_buffer_DtoH(
        self,
        device_loc: torch.Tensor,
        cpu_loc: torch.Tensor,
        can_run_graph: bool,
        cuda_graph_batch: int,
    ):
        pass

    @contextmanager
    def with_forward(self, forward_batch):
        yield

    def get_host_cache(self):
        pass

    def get_device_cache(self):
        pass


_global_expert_capturer: Optional[RoutedExpertsCapturer] = _RoutedExpertsCapturerNoop()


def get_global_experts_capturer():
    return _global_expert_capturer


def set_global_experts_capturer(capturer: RoutedExpertsCapturer):
    global _global_expert_capturer
    _global_expert_capturer = capturer