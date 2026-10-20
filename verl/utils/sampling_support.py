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
on the way (response padding, sequence packing, TP/CP alignment) means "no support". A row with no
nonzero slot (prompt, tool and padding tokens, unproven rows) uses the full vocabulary.

Layout: ``rollout_support_ids`` is ``[bsz, response_length, K]`` int32, response-aligned (row ``r``
is the support of response token ``r``) and right-padded. ``left_right_2_no_padding`` turns it into
a nested full-sequence tensor in prediction-row layout (row ``t`` holds the support of token
``t + 1``, so it lines up with the logits row that predicts that token), which the logits processor
can split like the logits without rolling.
"""

from typing import Optional

import numpy as np
import torch

SUPPORT_IDS_KEY = "rollout_support_ids"
EMPTY_SLOT = 0
ID_OFFSET = 1

# Per-request sampler head fields written by the vLLM server into TokenOutput.extra_fields.
TOPK_EXTRA_KEYS = ("response_topk_ids", "response_topk_log_probs", "response_support_complete")
# Response-aligned support rows assembled by multi-turn agent loops, consumed by postprocess.
SUPPORT_ROWS_EXTRA_KEY = "response_support_rows"


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


def empty_support_rows(num_rows: int, k: int) -> np.ndarray:
    """Support rows for tokens the sampler did not generate (tool / interstitial tokens)."""
    return np.full((num_rows, k), EMPTY_SLOT, dtype=np.int32)


def pad_response_support(support_rows: np.ndarray, response_length: int) -> torch.Tensor:
    """Right-pad response-aligned support rows to ``(1, response_length, k)``."""
    rows = torch.as_tensor(np.asarray(support_rows), dtype=torch.int32)
    num_rows, k = rows.shape
    if num_rows > response_length:
        raise ValueError(f"support covers {num_rows} tokens, more than response_length={response_length}")
    padded = torch.full((1, response_length, k), EMPTY_SLOT, dtype=torch.int32)
    padded[0, :num_rows] = rows
    return padded


def response_support_to_prediction_rows(support: torch.Tensor, prompt_width: int, seq_width: int) -> torch.Tensor:
    """Lay response-aligned support over the left/right-padded prompt+response sequence.

    Response token ``r`` sits at column ``prompt_width + r`` and is predicted by the logits row at
    column ``prompt_width + r - 1``, so its support goes there. The row at ``prompt_width - 1`` is
    the last prompt token, which is real because prompts are left-padded and non-empty.

    Args:
        support: Response-aligned support, shape (bsz, response_length, k).
        prompt_width: Width of the left-padded prompt block.
        seq_width: Width of the padded prompt+response sequence.

    Returns:
        Support in prediction-row layout, shape (bsz, seq_width, k).
    """
    bsz, response_length, k = support.shape
    if prompt_width < 1 or prompt_width + response_length != seq_width:
        raise ValueError(
            f"support of width {response_length} does not fit prompt_width={prompt_width}, seq_width={seq_width}"
        )
    full = torch.full((bsz, seq_width, k), EMPTY_SLOT, dtype=support.dtype, device=support.device)
    full[:, prompt_width - 1 : prompt_width - 1 + response_length] = support
    return full


def compute_support_metrics(support: torch.Tensor, response_mask: torch.Tensor) -> dict[str, float]:
    """Coverage of the replayed support over generated tokens.

    ``sampling_mask/coverage`` is the fraction of generated tokens trained over their sampling
    support; the rest fall back to the full vocabulary (head without proof of complete support).
    """
    response_mask = response_mask.bool()
    num_tokens = response_mask.sum()
    if num_tokens == 0:
        return {}
    support_size = (support != EMPTY_SLOT).sum(-1)
    covered = (support_size > 0) & response_mask
    num_covered = covered.sum()
    return {
        "sampling_mask/coverage": (num_covered / num_tokens).item(),
        "sampling_mask/support_size_mean": (support_size[covered].float().mean().item() if num_covered > 0 else 0.0),
        "sampling_mask/support_size_max": (support_size[covered].max().item() if num_covered > 0 else 0),
    }


def vocab_parallel_log_probs_over_support(
    logits: torch.Tensor, label: torch.Tensor, support: torch.Tensor
) -> torch.Tensor:
    """Log-probs of ``label`` normalized over each row's sampling support, for TP-sharded logits.

    Each TP rank gathers the logits of the support ids in its vocabulary shard, and one all-reduce
    gives every rank the (rows, k) support logits. That is O(rows * k) memory, never a dense
    vocabulary mask. The gradient reaches only support ids, as in a masked softmax. Rows without
    support use the full-vocabulary log-prob.

    Args:
        logits: Temperature-scaled vocab-parallel logits, shape (..., vocab_size / tp).
        label: Target ids, shape (...).
        support: Encoded support per row (``id + 1``, ``0`` = empty), shape (..., k).

    Returns:
        Log-probs, shape (...), float32.
    """
    from megatron.core import parallel_state as mpu
    from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region
    from megatron.core.tensor_parallel.utils import VocabUtility

    from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits

    if support.shape[:-1] != label.shape or logits.shape[:-1] != label.shape:
        raise ValueError(
            f"support {tuple(support.shape)}, logits {tuple(logits.shape)} and label {tuple(label.shape)} "
            "must share their leading dims"
        )
    support = support.to(device=logits.device, dtype=torch.long)
    valid = support != EMPTY_SLOT
    support_ids = support - ID_OFFSET
    partition_vocab_size = logits.size(-1)
    vocab_start, vocab_end = VocabUtility.vocab_range_from_per_partition_vocab_size(
        partition_vocab_size, mpu.get_tensor_model_parallel_rank(), mpu.get_tensor_model_parallel_world_size()
    )

    # Gather before the full-vocabulary cross-entropy, which shifts the logits in place.
    in_shard = valid & (support_ids >= vocab_start) & (support_ids < vocab_end)
    local_ids = torch.where(in_shard, support_ids - vocab_start, torch.zeros_like(support_ids))
    shard_logits = torch.gather(logits, -1, local_ids).float() * in_shard
    support_logits = reduce_from_tensor_model_parallel_region(shard_logits)

    is_label = valid & (support_ids == label.unsqueeze(-1).long())
    use_support = is_label.any(-1)
    if bool((valid.any(-1) & ~use_support).any()):
        raise RuntimeError(
            "sampling-mask replay: a target token lies outside its recorded sampling support; "
            "the support rows are misaligned with the labels."
        )

    # Rows without support get finite dummy logits so neither branch of the final select is NaN.
    support_logits = support_logits.masked_fill(~valid, float("-inf"))
    support_logits = torch.where(use_support.unsqueeze(-1), support_logits, torch.zeros_like(support_logits))
    label_logits = support_logits.gather(-1, is_label.float().argmax(-1, keepdim=True)).squeeze(-1)
    support_log_probs = label_logits - torch.logsumexp(support_logits, dim=-1)

    full_log_probs = vocab_parallel_log_probs_from_logits(logits, label)
    return torch.where(use_support, support_log_probs, full_log_probs.float())
