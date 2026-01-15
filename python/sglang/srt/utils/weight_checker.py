import logging
from typing import Dict, Iterable, Tuple

import torch

from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    inverse_transform_scale_ue8m0,
)

logger = logging.getLogger(__name__)


class WeightChecker:
    def __init__(self, model_runner):
        self._model_runner = model_runner
        self._snapshot_tensors = None
        self._golden_stats = None  # For golden standard comparison

    def handle(self, action: str):
        logger.info(f"[WeightChecker] handle action={action}")
        if action == "snapshot":
            self._snapshot()
        elif action == "reset_tensors":
            self._reset_tensors()
        elif action == "compare":
            self._compare()
        elif action == "dump_stats":
            self._dump_stats()
        elif action == "save_golden":
            self._save_golden_stats()
        elif action == "compare_golden":
            self._compare_with_golden()
        else:
            raise Exception(f"Unsupported {action=}")

    def _snapshot(self):
        named_tensors = [
            (name, param.data.detach().cpu()) for name, param in self._model_state()
        ]
        self._snapshot_tensors = dict(named_tensors)
        assert len(self._snapshot_tensors) == len(
            named_tensors
        ), f"should not have duplicated tensor name"

    def _reset_tensors(self):
        for name, param in self._model_state():
            param.copy_(_random_like(param))

    def _compare(self):
        assert self._snapshot_tensors is not None

        _check_tensors(
            expect_tensors=_postprocess_tensors(self._snapshot_tensors),
            actual_tensors=_postprocess_tensors(dict(self._model_state())),
        )

    def _dump_stats(self):
        """Dump statistics for all model parameters for manual comparison."""
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0

        logger.info(f"[WeightChecker] Dumping weight statistics for rank {rank}")
        for name, param in self._model_state():
            t = param.data.float()
            stats = {
                "sum": t.sum().item(),
                "mean": t.mean().item(),
                "shape": list(param.shape),
            }
            # Log with clear format for easy grep and comparison
            logger.info(
                f"[WEIGHT_STATS] rank={rank} {name}: "
                f"sum={stats['sum']:.6f} mean={stats['mean']:.9f} shape={stats['shape']}"
            )

    def _save_golden_stats(self):
        """Save current weight statistics as golden standard for later comparison."""
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0

        logger.info(f"[WeightChecker] Saving golden weight statistics for rank {rank}")
        self._golden_stats = {}
        for name, param in self._model_state():
            t = param.data.float()
            self._golden_stats[name] = {
                "sum": t.sum().item(),
                "mean": t.mean().item(),
                "shape": list(param.shape),
            }
        logger.info(f"[WeightChecker] Saved {len(self._golden_stats)} parameters as golden standard")

    def _compare_with_golden(self):
        """Compare current weights with saved golden standard.

        This is the key method for TransferPlan verification:
        1. Before AWEX sync: call save_golden to capture correct weights
        2. After AWEX sync: call compare_golden to verify transfer correctness

        If TransferPlan is correct, the weights should be nearly identical
        (small differences due to training updates are expected).
        """
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0

        if self._golden_stats is None:
            logger.error("[WeightChecker] No golden stats saved. Call save_golden first.")
            return

        logger.info(f"[WeightChecker] Comparing with golden standard for rank {rank}")

        mismatched_params = []
        matched_params = []

        for name, param in self._model_state():
            if name not in self._golden_stats:
                logger.warning(f"[GOLDEN_DIFF] rank={rank} {name}: NOT IN GOLDEN (new parameter)")
                continue

            t = param.data.float()
            current_sum = t.sum().item()
            current_mean = t.mean().item()

            golden = self._golden_stats[name]
            golden_sum = golden["sum"]
            golden_mean = golden["mean"]

            diff_sum = abs(current_sum - golden_sum)
            diff_mean = abs(current_mean - golden_mean)

            # Relative difference threshold (1% relative error or small absolute error)
            rel_threshold = 0.01
            abs_threshold = 1e-4

            is_match = (
                diff_sum < abs_threshold or
                (golden_sum != 0 and diff_sum / abs(golden_sum) < rel_threshold)
            )

            if is_match:
                matched_params.append(name)
            else:
                mismatched_params.append(name)
                # Log detailed info for mismatched parameters
                logger.error(
                    f"[GOLDEN_DIFF] rank={rank} {name}: MISMATCH "
                    f"golden_sum={golden_sum:.6f} current_sum={current_sum:.6f} diff_sum={diff_sum:.6f} | "
                    f"golden_mean={golden_mean:.9f} current_mean={current_mean:.9f} diff_mean={diff_mean:.9f}"
                )

        # Summary
        total = len(matched_params) + len(mismatched_params)
        logger.info(
            f"[WeightChecker] Golden comparison result for rank {rank}: "
            f"{len(matched_params)}/{total} matched, {len(mismatched_params)} mismatched"
        )

        if len(mismatched_params) > 0:
            logger.error(f"[WeightChecker] Mismatched parameters: {mismatched_params[:10]}...")  # Show first 10

    def _model_state(self):
        # TODO: support EAGLE etc (e.g. yield from both main model and draft model)
        yield from self._model_runner.model.named_parameters()
        yield from self._model_runner.model.named_buffers()


def _check_tensors(
    expect_tensors: Iterable[Tuple[str, bool, torch.Tensor]],
    actual_tensors: Iterable[Tuple[str, bool, torch.Tensor]],
):
    from sglang.srt.debug_utils.dumper import get_tensor_info

    good_names = []
    error_messages = []
    info_messages = []

    for (expect_name, expect_should_compare, expect), (
        actual_name,
        actual_should_compare,
        actual,
    ) in zip(expect_tensors, actual_tensors, strict=True):
        assert expect_name == actual_name, f"{expect_name=} {actual_name=}"
        assert (
            expect_should_compare == actual_should_compare
        ), f"{expect_should_compare=} {actual_should_compare=}"
        name = expect_name
        should_compare = expect_should_compare

        expect = expect.cuda()
        actual = actual.cuda()

        if torch.all(expect == actual):
            good_names.append(name)
        else:
            abs_diff = (actual.float() - expect.float()).abs()
            msg = (
                f"name={name} "
                f"max_abs_err={abs_diff.max()} "
                f"mean_abs_err={abs_diff.mean()} "
                f"{get_tensor_info(expect)=} "
                f"{get_tensor_info(actual)=} "
            )
            (error_messages if should_compare else info_messages).append(msg)

    logger.info(f"[check_tensors] equal tensors: {good_names}")
    if len(info_messages) > 0:
        logger.info(f"[check_tensors] info: {info_messages}")
    if len(error_messages) > 0:
        raise Exception(f"check tensor equality failed:\n" + "\n".join(error_messages))


def _random_like(t: torch.Tensor):
    device = t.device
    shape = t.shape
    dtype = t.dtype

    if dtype.is_floating_point:
        return torch.rand(shape, device=device, dtype=torch.float32).to(dtype)

    if dtype == torch.bool:
        return torch.rand(shape, device=device) > 0.5

    info = torch.iinfo(dtype)
    return torch.randint(
        low=int(info.min), high=int(info.max), size=shape, device=device, dtype=dtype
    )


def _postprocess_tensors(
    raw: Dict[str, torch.Tensor]
) -> Iterable[Tuple[str, bool, torch.Tensor]]:
    from sglang.srt.debug_utils.dumper import get_tensor_info

    skip_compare_names = []

    # dequant fp8
    quant_names = [
        name
        for name in raw
        # Match: `something.weight`, `something.experts.w2_weight`
        if name.endswith("weight") and name.replace("weight", "weight_scale_inv") in raw
    ]
    skip_compare_names += quant_names
    for name in quant_names:
        w_q = raw[name]
        w_s = raw[name.replace("weight", "weight_scale_inv")]

        try:
            # TODO this is only needed for Blackwell
            w_s_inverse_transformed = inverse_transform_scale_ue8m0(
                w_s, mn=w_q.shape[-2]
            )
            w_dequant = block_quant_dequant(
                w_q,
                w_s_inverse_transformed,
                # TODO do not hardcode
                block_size=[128, 128],
                dtype=torch.bfloat16,
            )
            yield name, True, w_dequant
        except Exception as e:
            e.add_note(
                f"when handling {name=} {get_tensor_info(w_q)=} {get_tensor_info(w_s)=}"
            )
            raise

    for name in raw:
        should_compare = name not in skip_compare_names
        yield name, should_compare, raw[name]
