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

With ``top_k > 0`` / ``top_p < 1`` and ``logprobs_mode=processed_logprobs``, vLLM's rollout log-probs
are normalized over the truncated sampling support, while the trainer normalizes over the full
vocabulary. Replaying the support on the trainer removes that bias (DeepSeek-V3.2's "keep sampling mask").

The support is read from the sampler's top-k head (``rollout.topk_log_probs``): when the head proves
complete support, the ids with a finite log-prob. Each trajectory carries it in CSR form
(``rollout_support_counts``, uint8 per response token, and ``rollout_support_flat_ids``, int32). The
Megatron forward expands it into ``rollout_support_ids``, nested ``[bsz, seq, k_batch]`` rows in
prediction-row layout, encoded ``id + 1`` with ``0`` for an empty slot so every zero padding means
"no support". Rows without support keep the full-vocabulary log-prob.
"""

import time
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch

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
    """Encoded (T, k) support per generated token: finite-log-prob ids of rows proven complete, else empty."""
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
    """Expand per-trajectory CSR supports into packed (tokens, k_batch) encoded rows.

    Response token ``r`` of trajectory ``i`` is predicted by packed row
    ``cu_seqlens[i] + prompt_lens[i] - 1 + r``, so its support goes there; all other rows are empty.
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
    _STATS["support_size_hist"].append(np.bincount(sizes, minlength=256))
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
    - ``support_size_{mean,max,p50,p90,p99,p999}``: support size of the covered tokens; a token whose
      support does not fit the recorded head is uncovered, so ``topk_log_probs`` should exceed p99.9.
    - ``csr_mb``: CSR support bytes this rank received.
    - ``expand_ms``: CPU time expanding the CSR into prediction rows.
    - ``logprob_{fwd,bwd}_ms``: GPU time of the support pair logits forward / backward.
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
        cumulative = np.cumsum(np.sum(stats["support_size_hist"], axis=0)[1:])
        if cumulative[-1]:
            for name, q in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("p999", 0.999)):
                metrics[f"sampling_mask/support_size_{name}"] = float(
                    1 + np.searchsorted(cumulative, q * cumulative[-1])
                )
    if "csr_bytes" in stats:
        metrics["sampling_mask/csr_mb"] = sum(stats["csr_bytes"]) / 1e6
    if "expand_ms" in stats:
        metrics["sampling_mask/expand_ms"] = sum(stats["expand_ms"])
    for name in ("logprob_fwd", "logprob_bwd"):
        events = stats.get(name)
        if events:
            events[-1][1].synchronize()
            metrics[f"sampling_mask/{name}_ms"] = sum(start.elapsed_time(end) for start, end in events)
    return metrics


_PAIR_CHUNK = 1 << 14


class _PairLogits(torch.autograd.Function):
    """``hidden[rows[p]] . weight[ids[p]]`` per (token, support id) pair.

    Forward and backward run in fixed-size chunks and save only the inputs, so no (pairs, hidden)
    buffer outlives a chunk and nothing vocabulary-sized is materialized beyond the weight gradient.
    """

    @staticmethod
    def forward(ctx, hidden, weight, rows, ids):
        timer = _start_timer()
        out = torch.empty(rows.numel(), dtype=torch.float32, device=hidden.device)
        for start in range(0, rows.numel(), _PAIR_CHUNK):
            chunk = slice(start, start + _PAIR_CHUNK)
            out[chunk] = torch.einsum("ph,ph->p", hidden[rows[chunk]].float(), weight[ids[chunk]].float())
        ctx.save_for_backward(hidden, weight, rows, ids)
        _stop_timer("logprob_fwd", timer)
        return out

    @staticmethod
    def backward(ctx, grad):
        timer = _start_timer()
        hidden, weight, rows, ids = ctx.saved_tensors
        d_hidden = torch.zeros(hidden.shape, dtype=torch.float32, device=hidden.device)
        d_weight = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
        for start in range(0, rows.numel(), _PAIR_CHUNK):
            chunk = slice(start, start + _PAIR_CHUNK)
            g = grad[chunk].unsqueeze(-1)
            d_hidden.index_add_(0, rows[chunk], g * weight[ids[chunk]].float())
            d_weight.index_add_(0, ids[chunk], g * hidden[rows[chunk]].float())
        _stop_timer("logprob_bwd", timer)
        return d_hidden.to(hidden.dtype), d_weight.to(weight.dtype), None, None


def log_probs_over_support_from_hidden(
    hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, support: torch.Tensor, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Label log-probs normalized over each row's support, as ``(log_probs, has_support)`` shaped like labels.

    Companion to the fused full-vocabulary kernel: each (token, support id) logit is recomputed from the
    gathered hidden state and this rank's vocabulary shard of the output weight, then summed across TP.
    Like the kernel's, the hidden-state gradient stays partial per TP rank.
    """
    from megatron.core import parallel_state as mpu
    from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region

    hidden = hidden.reshape(-1, hidden.size(-1))
    flat_labels = labels.reshape(-1).long()
    support = support.reshape(flat_labels.numel(), -1).to(device=hidden.device, dtype=torch.long)
    rows, slots = (support != EMPTY_SLOT).nonzero(as_tuple=True)
    ids = support[rows, slots] - ID_OFFSET
    vocab_start = mpu.get_tensor_model_parallel_rank() * weight.size(0)
    local = (ids >= vocab_start) & (ids < vocab_start + weight.size(0))
    logits = _PairLogits.apply(hidden, weight, rows, torch.where(local, ids - vocab_start, 0)) * local
    logits = reduce_from_tensor_model_parallel_region(logits) / temperature

    num_rows = flat_labels.numel()
    zeros = torch.zeros(num_rows, dtype=torch.float32, device=hidden.device)
    has_support = torch.zeros(num_rows, dtype=torch.bool, device=hidden.device).index_fill_(0, rows, True)
    is_label = ids == flat_labels[rows]
    has_label = torch.zeros_like(has_support).index_fill_(0, rows[is_label], True)
    if bool((has_support & ~has_label).any()):
        raise RuntimeError(
            "sampling-mask replay: a target token lies outside its recorded sampling support; "
            "the support rows are misaligned with the labels."
        )
    row_max = zeros.scatter_reduce(0, rows, logits.detach(), "amax", include_self=False)
    sum_exp = zeros.index_add(0, rows, (logits - row_max[rows]).exp())
    # 1 on rows without support keeps their log finite, so the caller's select back-propagates no NaN.
    log_norm = row_max + torch.where(has_support, sum_exp, 1.0).log()
    log_probs = zeros.index_add(0, rows[is_label], logits[is_label]) - log_norm
    return log_probs.view_as(labels), has_support.view_as(labels)
