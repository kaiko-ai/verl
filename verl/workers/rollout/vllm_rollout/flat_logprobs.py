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
"""Numpy-backed sample log-probs for token-in-token-out requests with a large top-k head.

vLLM 0.20 stores flat sample log-probs (``SamplingParams(flat_logprobs=True)``) as Python lists
with one object per entry, filled by a Python loop for every generated token. With a top-k head
of a hundred entries that loop and the resulting heap dominate the server's event loop. Requests
that skip detokenization only need the raw arrays, so this keeps them in growing numpy buffers.

The patch replaces vLLM internals and is applied only to the vLLM version it was written for.
Revisit it on every vLLM upgrade: newer releases can return the sampling support directly
(``return_sampling_mask``), which removes the need for the top-k head altogether.
"""

import logging

import numpy as np
from packaging.version import Version

logger = logging.getLogger(__file__)

_PATCHED_VLLM_RELEASE = "0.20.2"


class ArrayFlatLogprobs:
    """Sample log-probs of one request, ``width`` entries per position, in numpy buffers.

    Exposes the read side of vLLM's ``FlatLogprobs`` that token-in-token-out consumers use
    (``len``, flat ``token_ids`` / ``logprobs`` / ``ranks``, ``start_indices`` / ``end_indices``).
    """

    def __init__(self, width: int):
        self.width = width
        self._size = 0
        self._ids = np.empty((256, width), dtype=np.int32)
        self._log_probs = np.empty((256, width), dtype=np.float32)
        self._sampled_ranks = np.empty(256, dtype=np.int64)

    def append_block(self, token_ids: np.ndarray, log_probs: np.ndarray, sampled_ranks: np.ndarray) -> None:
        end = self._size + len(sampled_ranks)
        if end > len(self._sampled_ranks):
            capacity = max(end, 2 * len(self._sampled_ranks))
            self._ids = np.resize(self._ids, (capacity, self.width))
            self._log_probs = np.resize(self._log_probs, (capacity, self.width))
            self._sampled_ranks = np.resize(self._sampled_ranks, capacity)
        self._ids[self._size : end] = token_ids
        self._log_probs[self._size : end] = log_probs
        self._sampled_ranks[self._size : end] = sampled_ranks
        self._size = end

    def __len__(self) -> int:
        return self._size

    @property
    def token_ids(self) -> np.ndarray:
        return self._ids[: self._size].reshape(-1)

    @property
    def logprobs(self) -> np.ndarray:
        return self._log_probs[: self._size].reshape(-1)

    @property
    def ranks(self) -> np.ndarray:
        ranks = np.empty((self._size, self.width), dtype=np.int64)
        ranks[:, 0] = self._sampled_ranks[: self._size]
        ranks[:, 1:] = np.arange(1, self.width)
        return ranks.reshape(-1)

    @property
    def start_indices(self) -> list[int]:
        return list(range(0, self._size * self.width, self.width))

    @property
    def end_indices(self) -> list[int]:
        return list(range(self.width, (self._size + 1) * self.width, self.width))


def patch_vllm_flat_logprobs() -> bool:
    """Route flat, non-detokenized sample log-probs into ``ArrayFlatLogprobs``; returns whether it applied."""
    import vllm
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.logprobs import LogprobsProcessor

    if Version(vllm.__version__).base_version != _PATCHED_VLLM_RELEASE:
        logger.warning(
            "vLLM %s: flat log-prob patch written for %s, leaving vLLM unchanged",
            vllm.__version__,
            _PATCHED_VLLM_RELEASE,
        )
        return False
    if getattr(LogprobsProcessor, "_array_flat_logprobs_patched", False):
        return True

    from_new_request = LogprobsProcessor.from_new_request.__func__
    update_sample_logprobs = LogprobsProcessor._update_sample_logprobs

    def patched_from_new_request(cls, tokenizer, request):
        processor = from_new_request(cls, tokenizer, request)
        params = request.sampling_params
        if (
            tokenizer is None
            and params.flat_logprobs
            and params.logprobs is not None
            and params.logprobs > 0
            and params.output_kind != RequestOutputKind.DELTA
        ):
            processor.logprobs = ArrayFlatLogprobs(params.logprobs + 1)
        return processor

    def patched_update_sample_logprobs(self, logprobs_lists) -> None:
        if not isinstance(self.logprobs, ArrayFlatLogprobs):
            return update_sample_logprobs(self, logprobs_lists)
        token_ids, log_probs, sampled_ranks, _ = logprobs_lists
        width = self.logprobs.width
        # the batch arrays are as wide as the largest head requested in the step
        self.cumulative_logprob += float(log_probs[:, 0].sum())
        self.logprobs.append_block(token_ids[:, :width], log_probs[:, :width], sampled_ranks)

    LogprobsProcessor.from_new_request = classmethod(patched_from_new_request)
    LogprobsProcessor._update_sample_logprobs = patched_update_sample_logprobs
    LogprobsProcessor._array_flat_logprobs_patched = True
    return True
