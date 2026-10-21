# Copyright 2026 kaiko.ai
#
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
"""Sampling-mask replay: normalize trainer log-probs over the rollout's top-k/top-p support.

With ``top_k > 0`` / ``top_p < 1`` and ``logprobs_mode=processed_logprobs``, vLLM's rollout
log-probs are normalized over the truncated sampling support, while the trainer normalizes over the
full vocabulary. Replaying the support on the trainer removes that bias (the "keep sampling mask"
approach of DeepSeek-V3.2, also used by TRL, prime-rl and miles).

The support comes from the sampler's top-k log-prob head recorded with ``rollout.topk_log_probs``
(see ``verl.workers.rollout.utils.extract_response_topk_logprobs_with_coverage``): in a head that
proves complete support, the support is the set of ids with a finite log-prob. Rows without that
proof fall back to full-vocabulary log-probs.

Encoding: support slots hold ``token_id + 1`` and ``0`` marks an empty slot, so every zero padding
on the way (sequence packing, TP/CP alignment) means "no support". A row with no nonzero slot
(prompt, tool and padding tokens, unproven rows) uses the full vocabulary.

Storage: supports are small (often 2-3 ids for top_p < 1) next to the recorded head (k ids), so
each trajectory carries them in CSR form: ``rollout_support_counts`` (uint8, one count per response
token) and ``rollout_support_flat_ids`` (int32, the ids concatenated). Multi-turn agent loops put
the CSR of their whole response into ``extra_fields`` (``response_support_counts`` /
``response_support_flat_ids``); single-turn loops leave the sampler head there and it is converted
on postprocess. The Megatron forward expands the CSR into ``rollout_support_ids``, a nested
``[bsz, seq, k_batch]`` int32 tensor over the real tokens in prediction-row layout (row ``t`` holds
the support of token ``t + 1``, so it lines up with the logits row that predicts that token) with
``k_batch`` the largest support in the micro-batch. The logits processor splits it like the logits,
without rolling.
"""

import time
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch

SUPPORT_IDS_KEY = "rollout_support_ids"
SUPPORT_COUNTS_KEY = "rollout_support_counts"
SUPPORT_FLAT_IDS_KEY = "rollout_support_flat_ids"
EMPTY_SLOT = 0
ID_OFFSET = 1

# Per-request sampler head fields written by the vLLM server into TokenOutput.extra_fields.
TOPK_EXTRA_KEYS = ("response_topk_ids", "response_topk_log_probs", "response_support_complete")
# Response-aligned CSR support assembled by multi-turn agent loops, consumed by postprocess.
RESPONSE_SUPPORT_EXTRA_KEYS = ("response_support_counts", "response_support_flat_ids")

# Per-update measurements, drained into the actor metrics by pop_support_stats.
_STATS: dict[str, list] = defaultdict(list)


def support_rows_from_topk(
    topk_ids: np.ndarray,
    topk_log_probs: np.ndarray,
    complete: Optional[np.ndarray],
) -> np.ndarray:
    """Convert the sampler top-k head of each generated token into its encoded sampling support.

    Args:
        topk_ids: Top-k ids per generated token, shape (T, k).
        topk_log_probs: Matching processed log-probs, shape (T, k); -inf outside the support.
        complete: Per-token proof that the head covers the whole support, shape (T,). None means
            no proof for any token.

    Returns:
        Encoded support, shape (T, k), dtype int32: ``id + 1`` inside the support, ``0`` outside it
        and on every slot of rows without proof of complete support.
    """
    ids = np.asarray(topk_ids, dtype=np.int64)
    log_probs = np.asarray(topk_log_probs, dtype=np.float32)
    if ids.ndim != 2 or ids.shape != log_probs.shape:
        raise ValueError(f"expected matching (T, k) top-k ids and log-probs, got {ids.shape} and {log_probs.shape}")
    if complete is None:
        complete = np.zeros(ids.shape[0], dtype=np.bool_)
    complete = np.asarray(complete)
    if complete.dtype != np.bool_ or complete.shape != (ids.shape[0],):
        raise ValueError("sampler support coverage must have one bool per generated token")
    in_support = np.isfinite(log_probs) & complete[:, None]
    return np.where(in_support, ids + ID_OFFSET, EMPTY_SLOT).astype(np.int32)


def pop_support_rows(extra_fields: dict) -> Optional[np.ndarray]:
    """Pop the sampler head of one generation request and return its support rows, if recorded."""
    topk_ids, topk_log_probs, complete = (extra_fields.pop(key, None) for key in TOPK_EXTRA_KEYS)
    if topk_ids is None or topk_log_probs is None:
        return None
    return support_rows_from_topk(topk_ids, topk_log_probs, complete)


def compact_support_rows(support_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CSR form of encoded support rows: per-row counts (uint8) and the token ids concatenated (int32)."""
    rows = np.asarray(support_rows)
    if rows.ndim != 2:
        raise ValueError(f"expected (T, k) support rows, got shape {rows.shape}")
    if rows.shape[1] > np.iinfo(np.uint8).max:
        raise ValueError(f"support width {rows.shape[1]} does not fit uint8 counts")
    nonempty = rows != EMPTY_SLOT
    counts = nonempty.sum(-1).astype(np.uint8)
    flat_ids = (rows[nonempty] - ID_OFFSET).astype(np.int32)
    return counts, flat_ids


def pop_response_support(extra_fields: dict) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Pop the response-aligned CSR support of one trajectory from its agent loop extra fields.

    Multi-turn agent loops provide the CSR of their whole response; for single-turn loops the
    sampler head of the one generation request is converted. The head is popped either way so it
    never reaches TransferQueue as per-token object arrays.
    """
    support_rows = pop_support_rows(extra_fields)
    counts, flat_ids = (extra_fields.pop(key, None) for key in RESPONSE_SUPPORT_EXTRA_KEYS)
    if counts is not None and flat_ids is not None:
        counts = np.asarray(counts, dtype=np.uint8)
        flat_ids = np.asarray(flat_ids, dtype=np.int32)
        if int(counts.sum()) != flat_ids.size:
            raise ValueError(f"support counts sum to {int(counts.sum())}, but {flat_ids.size} ids were given")
        return counts, flat_ids
    if support_rows is None:
        return None
    return compact_support_rows(support_rows)


def prediction_rows_from_compact(
    counts: list[np.ndarray],
    flat_ids: list[np.ndarray],
    prompt_lens: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Expand per-trajectory CSR supports into packed encoded rows in prediction-row layout.

    Response token ``r`` of trajectory ``i`` is predicted by packed row
    ``cu_seqlens[i] + prompt_lens[i] - 1 + r``, so its support goes there; all other rows are empty.

    Args:
        counts: Per trajectory, the support size of each response token, shape (T_i,).
        flat_ids: Per trajectory, the support ids concatenated, shape (sum(counts[i]),).
        prompt_lens: Real prompt length per trajectory, shape (bsz,).
        cu_seqlens: Packed sequence offsets of the real tokens, shape (bsz + 1,).

    Returns:
        Encoded support rows, shape (cu_seqlens[-1], k_batch), int32, with ``k_batch`` the largest
        support in the batch (at least 1).
    """
    prompt_lens = prompt_lens.cpu().long()
    cu_seqlens = cu_seqlens.cpu().long()
    seq_lens = cu_seqlens.diff()
    num_rows = torch.tensor([len(c) for c in counts], dtype=torch.long)
    if len(counts) != len(prompt_lens) or (prompt_lens < 1).any() or (prompt_lens + num_rows > seq_lens).any():
        raise ValueError("support rows do not fit the packed sequences")
    all_counts = torch.from_numpy(np.concatenate(counts).astype(np.int64)) if counts else torch.zeros(0)
    all_ids = torch.from_numpy(np.concatenate(flat_ids).astype(np.int32)) if flat_ids else torch.zeros(0)
    if int(all_counts.sum()) != all_ids.numel():
        raise ValueError(f"support counts sum to {int(all_counts.sum())}, but {all_ids.numel()} ids were given")

    k_batch = max(int(all_counts.max()) if all_counts.numel() else 0, 1)
    rows = torch.full((int(cu_seqlens[-1]), k_batch), EMPTY_SLOT, dtype=torch.int32)
    if all_ids.numel() == 0:
        return rows
    # packed row of every response token, then of every support id, and its slot within the row
    first_row = cu_seqlens[:-1] + prompt_lens - 1
    token_rows = torch.repeat_interleave(first_row, num_rows) + (
        torch.arange(int(num_rows.sum())) - torch.repeat_interleave(num_rows.cumsum(0) - num_rows, num_rows)
    )
    id_rows = torch.repeat_interleave(token_rows, all_counts)
    id_slots = torch.arange(all_ids.numel()) - torch.repeat_interleave(all_counts.cumsum(0) - all_counts, all_counts)
    rows[id_rows, id_slots] = all_ids + ID_OFFSET
    return rows


def nested_support_ids(batch: Any) -> Optional[torch.Tensor]:
    """Expand a micro-batch's CSR supports into nested prediction rows aligned with ``input_ids``.

    Returns None when the batch carries no support (replay off). Also records the coverage of the
    trained tokens and the cost of the expansion for pop_support_stats.
    """
    counts = batch.get(SUPPORT_COUNTS_KEY, None)
    if counts is None:
        return None
    start = time.perf_counter()
    input_ids = batch["input_ids"]
    cu_seqlens = input_ids.offsets()
    counts_list = [c.cpu().numpy() for c in counts.unbind()]
    flat_list = [f.cpu().numpy() for f in batch[SUPPORT_FLAT_IDS_KEY].unbind()]
    rows = prediction_rows_from_compact(counts_list, flat_list, batch["prompts"].offsets().diff(), cu_seqlens)
    support = torch.nested.nested_tensor_from_jagged(rows.to(input_ids.device), offsets=cu_seqlens)

    trained = [m.cpu().bool().numpy() for m in batch["loss_mask"].unbind()]
    sizes = np.concatenate([c[: len(m)][m[: len(c)]] for c, m in zip(counts_list, trained, strict=True)])
    _STATS["expand_ms"].append((time.perf_counter() - start) * 1e3)
    _STATS["trained_tokens"].append(int(sum(m.sum() for m in trained)))
    _STATS["covered_tokens"].append(int((sizes > 0).sum()))
    _STATS["covered_ids"].append(int(sizes.sum()))
    _STATS["support_size_max"].append(int(sizes.max()) if sizes.size else 0)
    _STATS["csr_bytes"].append(sum(c.nbytes for c in counts_list) + sum(f.nbytes for f in flat_list))
    return support


def _start_timer() -> Optional[torch.cuda.Event]:
    if not torch.cuda.is_available():
        return None
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def _stop_timer(name: str, start: Optional[torch.cuda.Event]) -> None:
    if start is None:
        return
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    _STATS[name].append((start, end))


def pop_support_stats() -> dict[str, float]:
    """Sampling-mask metrics recorded since the last call (one actor update), then reset.

    - ``coverage``: fraction of trained tokens normalized over their sampling support; the rest
      used the full vocabulary.
    - ``support_size_{mean,max}``: support size of the covered tokens.
    - ``csr_mb``: CSR support bytes this rank received.
    - ``expand_ms``: CPU time expanding the CSR into prediction rows.
    - ``logprob_{fwd,bwd}_ms``: GPU time of the support log-prob forward / backward.
    - ``logits_gb`` / ``allocated_at_logprob_gb``: largest logits buffer and device memory in use
      when the support log-prob ran (the logits spike of the unfused path).
    """
    stats = dict(_STATS)
    _STATS.clear()
    if not stats:
        return {}
    metrics: dict[str, float] = {}
    trained, covered = sum(stats.get("trained_tokens", [])), sum(stats.get("covered_tokens", []))
    if trained:
        metrics["sampling_mask/coverage"] = covered / trained
        metrics["sampling_mask/support_size_mean"] = sum(stats["covered_ids"]) / covered if covered else 0.0
        metrics["sampling_mask/support_size_max"] = float(max(stats["support_size_max"]))
    if "csr_bytes" in stats:
        metrics["sampling_mask/csr_mb"] = sum(stats["csr_bytes"]) / 1e6
    if "expand_ms" in stats:
        metrics["sampling_mask/expand_ms"] = sum(stats["expand_ms"])
    for name in ("logprob_fwd", "logprob_bwd"):
        events = stats.get(name)
        if events:
            events[-1][1].synchronize()
            metrics[f"sampling_mask/{name}_ms"] = sum(start.elapsed_time(end) for start, end in events)
    for name in ("logits_gb", "allocated_at_logprob_gb"):
        if name in stats:
            metrics[f"sampling_mask/{name}"] = max(stats[name])
    return metrics


class _VocabParallelLogProbsOverSupport(torch.autograd.Function):
    """Support-normalized log-probs with a full-vocabulary fallback, in one pass over the logits.

    Like Megatron's vocab-parallel cross-entropy, it works on the logits buffer in place (shift,
    exp) and reuses it for the gradient: nothing saves the logits for backward (a plain
    ``torch.gather`` would, and the in-place ops would then break autograd), and no second
    vocabulary-sized gradient is allocated.
    """

    @staticmethod
    def forward(ctx, logits, label, support):
        import torch.distributed as dist
        from megatron.core import parallel_state as mpu
        from megatron.core.tensor_parallel.utils import VocabUtility

        timer = _start_timer()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_group = mpu.get_tensor_model_parallel_group() if tp_size > 1 else None

        def all_reduce(tensor, op):
            if tp_size > 1:
                dist.all_reduce(tensor, op=op, group=tp_group)
            return tensor

        partition_vocab_size = logits.size(-1)
        vocab_start, vocab_end = VocabUtility.vocab_range_from_per_partition_vocab_size(
            partition_vocab_size, mpu.get_tensor_model_parallel_rank(), tp_size
        )
        # In place when the logits are already fp32, as in Megatron's cross-entropy.
        logits_fp32 = logits.float()
        _STATS["logits_gb"].append((logits.nbytes + (logits_fp32.nbytes if logits_fp32 is not logits else 0)) / 1024**3)
        if torch.cuda.is_available():
            _STATS["allocated_at_logprob_gb"].append(torch.cuda.memory_allocated() / 1024**3)
        buffer = logits_fp32.view(-1, partition_vocab_size)
        label = label.reshape(-1).long()
        support = support.reshape(-1, support.size(-1)).to(device=logits.device, dtype=torch.long)
        rows = torch.arange(label.numel(), device=logits.device)

        valid = support != EMPTY_SLOT
        support_ids = support - ID_OFFSET
        in_shard = valid & (support_ids >= vocab_start) & (support_ids < vocab_end)
        local_ids = torch.where(in_shard, support_ids - vocab_start, torch.zeros_like(support_ids))
        label_in_shard = (label >= vocab_start) & (label < vocab_end)
        local_label = torch.where(label_in_shard, label - vocab_start, torch.zeros_like(label))

        is_label = valid & (support_ids == label.unsqueeze(-1))
        use_support = is_label.any(-1)
        if bool((valid.any(-1) & ~use_support).any()):
            raise RuntimeError(
                "sampling-mask replay: a target token lies outside its recorded sampling support; "
                "the support rows are misaligned with the labels."
            )

        logits_max = all_reduce(buffer.max(dim=-1).values, dist.ReduceOp.MAX)
        buffer.sub_(logits_max.unsqueeze(-1))
        # shifted logits at the support ids and the label, owned by exactly one TP rank each
        support_logits = all_reduce(buffer.gather(-1, local_ids) * in_shard, dist.ReduceOp.SUM)
        label_logits = all_reduce(buffer[rows, local_label] * label_in_shard, dist.ReduceOp.SUM)
        buffer.exp_()
        sum_exp = all_reduce(buffer.sum(dim=-1), dist.ReduceOp.SUM)
        full_log_probs = label_logits - sum_exp.log()

        # Rows without support get finite dummy logits so no branch of the select is NaN.
        support_logits = support_logits.masked_fill(~valid, float("-inf"))
        support_logits = torch.where(use_support.unsqueeze(-1), support_logits, torch.zeros_like(support_logits))
        support_lse = torch.logsumexp(support_logits, dim=-1)
        support_probs = torch.exp(support_logits - support_lse.unsqueeze(-1))
        # the label is in its support, so its support logit is label_logits
        out = torch.where(use_support, label_logits - support_lse, full_log_probs)

        ctx.save_for_backward(
            logits_fp32, sum_exp, support_probs, local_ids, in_shard, is_label, use_support, local_label
        )
        ctx.label_in_shard = label_in_shard
        ctx.logits_shape = logits.shape
        _stop_timer("logprob_fwd", timer)
        return out.view(ctx.logits_shape[:-1])

    @staticmethod
    def backward(ctx, grad_output):
        timer = _start_timer()
        exp_logits, sum_exp, support_probs, local_ids, in_shard, is_label, use_support, local_label = ctx.saved_tensors
        with torch.no_grad():
            grad = grad_output.reshape(-1).float()
            rows = torch.arange(grad.numel(), device=grad.device)
            full_grad = torch.where(use_support, torch.zeros_like(grad), grad)
            support_grad = grad - full_grad

            # d log p(y) / dz = onehot(y) - softmax(z), over the vocabulary or over the support;
            # the gradient overwrites the exp buffer in place
            grad_input = exp_logits.view(-1, exp_logits.size(-1))
            grad_input.div_(sum_exp.unsqueeze(-1)).mul_(-full_grad.unsqueeze(-1))
            grad_input[rows, local_label] += full_grad * ctx.label_in_shard
            support_term = support_grad.unsqueeze(-1) * (is_label.float() - support_probs) * in_shard
            grad_input.scatter_add_(-1, local_ids, support_term)
        _stop_timer("logprob_bwd", timer)
        return exp_logits, None, None


def vocab_parallel_log_probs_over_support(
    logits: torch.Tensor, label: torch.Tensor, support: torch.Tensor
) -> torch.Tensor:
    """Log-probs of ``label`` normalized over each row's sampling support, for TP-sharded logits.

    Each TP rank gathers the logits of the support ids in its vocabulary shard, and one all-reduce
    gives every rank the (rows, k) support logits. That is O(rows * k) memory, never a dense
    vocabulary mask. The gradient reaches only support ids, as in a masked softmax. Rows without
    support use the full-vocabulary log-prob, computed in the same pass.

    The logits are modified in place, as by ``vocab_parallel_log_probs_from_logits``.

    Args:
        logits: Temperature-scaled vocab-parallel logits, shape (..., vocab_size / tp).
        label: Target ids, shape (...).
        support: Encoded support per row (``id + 1``, ``0`` = empty), shape (..., k).

    Returns:
        Log-probs, shape (...), float32.
    """
    if support.shape[:-1] != label.shape or logits.shape[:-1] != label.shape:
        raise ValueError(
            f"support {tuple(support.shape)}, logits {tuple(logits.shape)} and label {tuple(label.shape)} "
            "must share their leading dims"
        )
    return _VocabParallelLogProbsOverSupport.apply(logits, label, support)
