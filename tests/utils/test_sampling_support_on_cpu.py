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
"""CPU tests for sampling-mask replay (verl.utils.sampling_support).

Megatron is replaced by a minimal gloo-backed fake so the vocab-parallel path runs at TP 1/2/4.
"""

import os
import sys
import types

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verl.trainer.ppo.padding_utils import construct_minimal_padding_template
from verl.trainer.ppo.v1.agent_loop_tq import _support_fields
from verl.utils.sampling_support import (
    EMPTY_SLOT,
    SUPPORT_COUNTS_KEY,
    SUPPORT_FLAT_IDS_KEY,
    compact_support_rows,
    nested_support_ids,
    pop_response_support,
    pop_support_rows,
    pop_support_stats,
    prediction_rows_from_compact,
    support_rows_from_topk,
    vocab_parallel_log_probs_over_support,
)
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

NEG_INF = float("-inf")


# --------------------------------------------------------------------------- rollout-side helpers


def test_support_rows_from_topk_encodes_finite_ids_of_complete_rows():
    ids = np.array([[5, 2, 9], [7, 1, 3], [4, 0, 8]])
    log_probs = np.array([[-0.1, -2.5, NEG_INF], [-0.3, -1.4, -2.0], [-0.2, NEG_INF, NEG_INF]], dtype=np.float32)
    complete = np.array([True, False, True])

    rows = support_rows_from_topk(ids, log_probs, complete)

    assert rows.dtype == np.int32
    # id + 1 inside the support, 0 outside it; unproven rows are entirely empty
    np.testing.assert_array_equal(rows, [[6, 3, 0], [0, 0, 0], [5, 0, 0]])


def test_support_rows_without_coverage_are_empty():
    rows = support_rows_from_topk(np.array([[1, 2]]), np.array([[-0.1, NEG_INF]]), None)
    np.testing.assert_array_equal(rows, [[0, 0]])


def test_support_rows_reject_malformed_coverage():
    with pytest.raises(ValueError):
        support_rows_from_topk(np.array([[1, 2]]), np.array([[-0.1, NEG_INF]]), np.array([1]))


def test_pop_support_rows_removes_head_fields():
    extra = {
        "response_topk_ids": np.array([[3, 4]]),
        "response_topk_log_probs": np.array([[-0.5, NEG_INF]], dtype=np.float32),
        "response_support_complete": np.array([True]),
        "global_steps": 7,
    }
    rows = pop_support_rows(extra)
    np.testing.assert_array_equal(rows, [[4, 0]])
    assert extra == {"global_steps": 7}
    assert pop_support_rows({"global_steps": 7}) is None


def test_compact_support_rows_is_csr_of_nonempty_slots():
    rows = np.array([[11, 12, 0], [0, 0, 0], [13, 0, 0], [0, 0, 0]], dtype=np.int32)
    counts, flat_ids = compact_support_rows(rows)
    assert counts.dtype == np.uint8 and flat_ids.dtype == np.int32
    np.testing.assert_array_equal(counts, [2, 0, 1, 0])
    np.testing.assert_array_equal(flat_ids, [10, 11, 12])  # raw token ids


def test_pop_response_support_converts_a_single_turn_head():
    extra = {
        "response_topk_ids": np.array([[4, 1, 2], [5, 2, 3]]),
        "response_topk_log_probs": np.array([[-0.1, -2.0, NEG_INF], [-0.2, -1.0, -3.0]], dtype=np.float32),
        "response_support_complete": np.array([True, False]),
        "global_steps": 3,
    }
    counts, flat_ids = pop_response_support(extra)
    # the second token's head is unproven, so it has no support
    np.testing.assert_array_equal(counts, [2, 0])
    np.testing.assert_array_equal(flat_ids, [4, 1])
    assert extra == {"global_steps": 3}


def test_pop_response_support_prefers_the_agent_csr_and_drops_heads():
    extra = {
        "response_support_counts": np.array([1, 0, 2], dtype=np.uint8),
        "response_support_flat_ids": np.array([4, 6, 7], dtype=np.int32),
        "response_topk_ids": np.array([[9, 1]]),
        "response_topk_log_probs": np.array([[-0.1, NEG_INF]], dtype=np.float32),
    }
    counts, flat_ids = pop_response_support(extra)
    np.testing.assert_array_equal(counts, [1, 0, 2])
    np.testing.assert_array_equal(flat_ids, [4, 6, 7])
    assert extra == {}
    assert pop_response_support({}) is None
    with pytest.raises(ValueError, match="sum to"):
        pop_response_support({"response_support_counts": [2], "response_support_flat_ids": [1]})


def test_support_fields_check_the_response_length():
    fields = _support_fields((np.array([1, 0], dtype=np.uint8), np.array([5], dtype=np.int32)), num_response_tokens=2)
    assert fields[SUPPORT_COUNTS_KEY].tolist() == [1, 0] and fields[SUPPORT_FLAT_IDS_KEY].tolist() == [5]
    with pytest.raises(ValueError, match="covers 2 tokens"):
        _support_fields((np.array([1, 0], dtype=np.uint8), np.array([5], dtype=np.int32)), num_response_tokens=3)
    with pytest.raises(ValueError, match="no sampler support"):
        _support_fields(None, num_response_tokens=1)
    assert _support_fields(None, num_response_tokens=0)[SUPPORT_FLAT_IDS_KEY].numel() == 0


def test_prediction_rows_from_compact():
    # trajectory 0: prompt 3, response 2 (+1 padding token in the packed sequence); trajectory 1: prompt 1, response 3
    counts = [np.array([2, 1], dtype=np.uint8), np.array([0, 3, 1], dtype=np.uint8)]
    flat_ids = [np.array([7, 8, 9], dtype=np.int32), np.array([4, 5, 6, 3], dtype=np.int32)]
    cu_seqlens = torch.tensor([0, 6, 10], dtype=torch.int32)

    rows = prediction_rows_from_compact(counts, flat_ids, torch.tensor([3, 1]), cu_seqlens)

    assert rows.shape == (10, 3) and rows.dtype == torch.int32  # k_batch = largest support
    expected = torch.zeros(10, 3, dtype=torch.int32)
    expected[2, :2] = torch.tensor([8, 9])  # response token r is predicted by row prompt_len - 1 + r
    expected[3, :1] = 10
    expected[6 + 1, :3] = torch.tensor([5, 6, 7])
    expected[6 + 2, :1] = 4
    torch.testing.assert_close(rows, expected)


def test_prediction_rows_from_compact_rejects_mismatched_ids():
    with pytest.raises(ValueError, match="sum to"):
        prediction_rows_from_compact(
            [np.array([2], dtype=np.uint8)], [np.array([1], dtype=np.int32)], torch.tensor([1]), torch.tensor([0, 2])
        )


def test_nested_support_ids_aligns_rows_with_the_packed_sequence_and_records_coverage():
    pop_support_stats()
    # trajectory 0: prompt 3, response 2; trajectory 1: prompt 1, response 3, no support at all
    rows = [
        {
            "prompts": torch.tensor([1, 2, 3]),
            "input_ids": torch.tensor([1, 2, 3, 7, 9]),
            "loss_mask": torch.tensor([1, 1]),
            SUPPORT_COUNTS_KEY: torch.tensor([2, 1], dtype=torch.uint8),
            SUPPORT_FLAT_IDS_KEY: torch.tensor([7, 8, 9], dtype=torch.int32),
        },
        {
            "prompts": torch.tensor([4]),
            "input_ids": torch.tensor([4, 5, 6, 3]),
            "loss_mask": torch.tensor([1, 0, 1]),
            SUPPORT_COUNTS_KEY: torch.zeros(3, dtype=torch.uint8),
            SUPPORT_FLAT_IDS_KEY: torch.zeros(0, dtype=torch.int32),
        },
    ]
    support = nested_support_ids(list_of_dict_to_tensordict(rows))

    assert support.is_nested and support.offsets().tolist() == [0, 5, 9]
    expected = torch.zeros(9, 2, dtype=torch.int32)
    expected[2] = torch.tensor([8, 9])  # response token 0 of trajectory 0 is predicted by row prompt_len - 1
    expected[3, 0] = 10
    torch.testing.assert_close(support.values(), expected)

    stats = pop_support_stats()
    assert stats["sampling_mask/coverage"] == pytest.approx(2 / 4)
    assert stats["sampling_mask/support_size_mean"] == pytest.approx(1.5)
    assert stats["sampling_mask/support_size_max"] == 2
    assert stats["sampling_mask/csr_mb"] == pytest.approx((5 + 12) / 1e6)
    assert pop_support_stats() == {}
    assert nested_support_ids(list_of_dict_to_tensordict([{"input_ids": torch.tensor([1, 2])}])) is None


def test_padding_template_carries_empty_support():
    source = {
        "prompts": torch.tensor([1, 2, 3]),
        "responses": torch.tensor([4, 5]),
        "input_ids": torch.tensor([1, 2, 3, 4, 5]),
        "position_ids": torch.arange(5),
        SUPPORT_COUNTS_KEY: torch.tensor([1, 1], dtype=torch.uint8),
        SUPPORT_FLAT_IDS_KEY: torch.tensor([4, 5], dtype=torch.int32),
    }
    template, _ = construct_minimal_padding_template(source, {"status": "success"}, eos_token_id=0)
    assert template[SUPPORT_COUNTS_KEY].tolist() == [0] * template["responses"].numel()
    assert template[SUPPORT_FLAT_IDS_KEY].numel() == 0


# --------------------------------------------------------------------------- vocab-parallel log-probs


class _ReduceFromTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = x.clone()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad


class _GatherFromTP(torch.autograd.Function):
    """All-gather along the vocab dim; every rank computes the same loss, so backward keeps the local slice."""

    @staticmethod
    def forward(ctx, x):
        ctx.rank, ctx.size = dist.get_rank(), x.size(-1)
        parts = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(parts, x.contiguous())
        return torch.cat(parts, dim=-1)

    @staticmethod
    def backward(ctx, grad):
        return grad[..., ctx.rank * ctx.size : (ctx.rank + 1) * ctx.size]


def _fake_vocab_parallel_cross_entropy(vocab_parallel_logits, target):
    if dist.is_initialized() and dist.get_world_size() > 1:
        full = _GatherFromTP.apply(vocab_parallel_logits)
    else:
        full = vocab_parallel_logits
    log_probs = torch.log_softmax(full.float(), dim=-1)
    return -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)


def _install_fake_megatron():
    def rank():
        return dist.get_rank() if dist.is_initialized() else 0

    def world_size():
        return dist.get_world_size() if dist.is_initialized() else 1

    parallel_state = types.SimpleNamespace(
        get_tensor_model_parallel_rank=rank,
        get_tensor_model_parallel_world_size=world_size,
        get_tensor_model_parallel_group=lambda: None,  # default group
    )
    vocab_utility = types.SimpleNamespace(
        vocab_range_from_per_partition_vocab_size=lambda size, r, ws: (r * size, (r + 1) * size)
    )
    modules = {
        "megatron": types.ModuleType("megatron"),
        "megatron.core": types.ModuleType("megatron.core"),
        "megatron.core.tensor_parallel": types.ModuleType("megatron.core.tensor_parallel"),
        "megatron.core.tensor_parallel.mappings": types.ModuleType("megatron.core.tensor_parallel.mappings"),
        "megatron.core.tensor_parallel.utils": types.ModuleType("megatron.core.tensor_parallel.utils"),
        "verl.utils.megatron.tensor_parallel": types.ModuleType("verl.utils.megatron.tensor_parallel"),
    }
    modules["megatron.core"].parallel_state = parallel_state
    modules["megatron.core.tensor_parallel.mappings"].reduce_from_tensor_model_parallel_region = _ReduceFromTP.apply
    modules["megatron.core.tensor_parallel.utils"].VocabUtility = vocab_utility
    modules["verl.utils.megatron.tensor_parallel"].vocab_parallel_log_probs_from_logits = (
        lambda logits, labels: -_fake_vocab_parallel_cross_entropy(logits, labels)
    )
    saved = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    return saved


def _restore_modules(saved):
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _make_case(seed=0, rows=12, vocab=16, k=4):
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(1, rows, vocab, generator=gen)
    label = torch.randint(0, vocab, (1, rows), generator=gen)
    support = torch.zeros(1, rows, k, dtype=torch.int32)
    for t in range(rows):
        if t % 4 == 3:
            continue  # rows without support: full-vocabulary fallback
        size = 1 + t % k
        others = [v for v in torch.randperm(vocab, generator=gen).tolist() if v != int(label[0, t])]
        ids = [int(label[0, t])] + others[: size - 1]
        support[0, t, :size] = torch.tensor(ids) + 1
    return logits, label, support


def _reference(logits, label, support):
    """Dense masked softmax over the support; full vocabulary where a row has no support."""
    valid = support != EMPTY_SLOT
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, (support.long() - 1).clamp(min=0), valid)
    has_support = valid.any(-1, keepdim=True)
    masked = torch.where(mask | ~has_support, logits, torch.full_like(logits, NEG_INF))
    return torch.log_softmax(masked, dim=-1).gather(-1, label.unsqueeze(-1)).squeeze(-1)


def _check_against_reference(world_size, rank):
    logits, label, support = _make_case()
    temperature = 0.7
    shard = logits.size(-1) // world_size
    local = logits[..., rank * shard : (rank + 1) * shard].clone().requires_grad_(True)

    # As in the Megatron engine: the model output is scaled in place by the temperature, then the
    # log-prob op shifts/exponentiates it in place. Autograd must still reach the model output.
    scaled = local * 1.0
    scaled.div_(temperature)
    out = vocab_parallel_log_probs_over_support(scaled, label, support)
    weights = torch.linspace(0.5, 1.5, out.numel()).view_as(out)
    (out * weights).sum().backward()

    full = logits.clone().requires_grad_(True)
    ref = _reference(full / temperature, label, support)
    (ref * weights).sum().backward()

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    assert torch.isfinite(local.grad).all()
    torch.testing.assert_close(local.grad, full.grad[..., rank * shard : (rank + 1) * shard], rtol=1e-5, atol=1e-5)
    # rows with support receive no gradient outside it
    grad_mask = full.grad.ne(0)
    in_support = torch.zeros_like(grad_mask)
    in_support.scatter_(-1, (support.long() - 1).clamp(min=0), support != EMPTY_SLOT)
    supported = (support != EMPTY_SLOT).any(-1)
    assert not (grad_mask & ~in_support)[supported].any()


def _tp_worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    saved = _install_fake_megatron()
    try:
        _check_against_reference(world_size, rank)
    finally:
        _restore_modules(saved)
        dist.destroy_process_group()


def test_kernel_matches_masked_softmax_tp1():
    saved = _install_fake_megatron()
    try:
        _check_against_reference(world_size=1, rank=0)
    finally:
        _restore_modules(saved)


@pytest.mark.parametrize("world_size", [2, 4])
def test_kernel_matches_masked_softmax_tp(world_size):
    port = 29500 + world_size + os.getpid() % 1000
    mp.spawn(_tp_worker, args=(world_size, port), nprocs=world_size, join=True)


def test_kernel_accepts_bf16_logits():
    saved = _install_fake_megatron()
    try:
        logits, label, support = _make_case()
        local = logits.to(torch.bfloat16).requires_grad_(True)
        out = vocab_parallel_log_probs_over_support(local, label, support)
        out.sum().backward()
        assert out.dtype == torch.float32 and local.grad.dtype == torch.bfloat16
        torch.testing.assert_close(out, _reference(local.detach().float(), label, support), rtol=1e-5, atol=1e-5)
    finally:
        _restore_modules(saved)


def test_kernel_rejects_label_outside_support():
    saved = _install_fake_megatron()
    try:
        logits, label, support = _make_case()
        support[0, 0] = 0
        support[0, 0, 0] = (int(label[0, 0]) + 1) % logits.size(-1) + 1  # a different token
        with pytest.raises(RuntimeError, match="outside its recorded sampling support"):
            vocab_parallel_log_probs_over_support(logits, label, support)
    finally:
        _restore_modules(saved)


def test_kernel_zero_padded_rows_use_full_vocabulary():
    saved = _install_fake_megatron()
    try:
        logits, label, _ = _make_case()
        support = torch.zeros(1, logits.size(1), 4, dtype=torch.int32)
        expected = torch.log_softmax(logits, -1).gather(-1, label.unsqueeze(-1)).squeeze(-1)
        out = vocab_parallel_log_probs_over_support(logits.clone(), label, support)
        torch.testing.assert_close(out, expected)
    finally:
        _restore_modules(saved)


# --------------------------------------------------------------------------- config


def _rollout_config(**kwargs):
    from verl.workers.config.rollout import RolloutConfig

    defaults = dict(name="vllm", top_k=20, top_p=0.95, calculate_log_probs=True, topk_log_probs=21)
    defaults.update(kwargs)
    return RolloutConfig(**defaults, replay_sampling_mask=True)


def test_config_accepts_replay_with_head_larger_than_top_k():
    cfg = _rollout_config()
    assert cfg.engine_kwargs["vllm"]["max_logprobs"] == 21


@pytest.mark.parametrize(
    "overrides",
    [
        dict(topk_log_probs=0),
        dict(topk_log_probs=20),  # no room for the -inf slot that proves complete support
        dict(temperature=0.0),
        dict(repetition_penalty=1.1),
    ],
)
def test_config_rejects_unreplayable_sampling(overrides):
    with pytest.raises(ValueError):
        _rollout_config(**overrides)
