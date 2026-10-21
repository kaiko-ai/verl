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
"""CPU tests for sampling-mask replay; Megatron's TP primitives are faked over gloo."""

import os
import sys
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verl.utils import tensordict_utils as tu
from verl.utils.sampling_support import (
    EMPTY_SLOT,
    SUPPORT_COUNTS_KEY,
    SUPPORT_FLAT_IDS_KEY,
    log_probs_over_support_from_hidden,
    nested_support_ids,
    pop_support_stats,
)
from verl.utils.tensordict_utils import list_of_dict_to_tensordict


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
    assert stats["sampling_mask/support_size_p50"] == 1
    assert stats["sampling_mask/support_size_p999"] == 2
    assert pop_support_stats() == {}

    # with replay on, a fetch that lost the support fields counts its trained tokens as uncovered
    support_keys = (SUPPORT_COUNTS_KEY, SUPPORT_FLAT_IDS_KEY)
    dropped = list_of_dict_to_tensordict([{k: v for k, v in row.items() if k not in support_keys} for row in rows])
    assert nested_support_ids(dropped) is None and pop_support_stats() == {}
    tu.assign_non_tensor(dropped, replay_sampling_mask=True)
    assert nested_support_ids(dropped) is None
    stats = pop_support_stats()
    assert stats == {"sampling_mask/missing_micro_batches": 1.0, "sampling_mask/coverage": 0.0}


class _ReduceFromTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if dist.is_initialized():
            x = x.clone()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad


def _install_fake_megatron():
    modules = {
        "megatron": types.ModuleType("megatron"),
        "megatron.core": types.ModuleType("megatron.core"),
        "megatron.core.tensor_parallel": types.ModuleType("megatron.core.tensor_parallel"),
        "megatron.core.tensor_parallel.mappings": types.ModuleType("megatron.core.tensor_parallel.mappings"),
    }
    modules["megatron.core"].parallel_state = types.SimpleNamespace(
        get_tensor_model_parallel_rank=lambda: dist.get_rank() if dist.is_initialized() else 0
    )
    modules["megatron.core.tensor_parallel.mappings"].reduce_from_tensor_model_parallel_region = _ReduceFromTP.apply
    saved = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    return saved


def _restore_modules(saved):
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _check_against_masked_softmax(world_size, rank):
    tokens, hidden_size, vocab, k, temperature = 12, 8, 16, 4, 0.7
    gen = torch.Generator().manual_seed(0)
    hidden = torch.randn(tokens, 1, hidden_size, generator=gen)
    weight = torch.randn(vocab, hidden_size, generator=gen)
    labels = torch.randint(0, vocab, (1, tokens), generator=gen)
    support = torch.zeros(1, tokens, k, dtype=torch.int32)
    for t in range(tokens):
        if t % 4 == 3:
            continue
        others = [v for v in torch.randperm(vocab, generator=gen).tolist() if v != int(labels[0, t])]
        support[0, t, : 1 + t % k] = torch.tensor([int(labels[0, t])] + others[: t % k]) + 1
    shard = slice(rank * vocab // world_size, (rank + 1) * vocab // world_size)
    loss_weights = torch.linspace(0.5, 1.5, tokens).view(1, tokens)

    local_hidden = hidden.clone().requires_grad_(True)
    local_weight = weight[shard].clone().requires_grad_(True)
    log_probs, has_support = log_probs_over_support_from_hidden(
        local_hidden, local_weight, labels, support, temperature
    )
    torch.where(has_support, log_probs * loss_weights, 0.0).sum().backward()
    d_hidden = local_hidden.grad.clone()
    if dist.is_initialized():
        dist.all_reduce(d_hidden)  # the sequence-parallel gather's backward sums the partial gradients

    ref_hidden = hidden.clone().requires_grad_(True)
    ref_weight = weight.clone().requires_grad_(True)
    logits = (ref_hidden.view(tokens, hidden_size) @ ref_weight.T / temperature).unsqueeze(0)
    valid = support != EMPTY_SLOT
    supported = valid.any(-1)
    ids = (support.long() - 1).clamp(min=0)
    mask = torch.zeros(logits.shape, dtype=torch.long).scatter_add_(-1, ids, valid.long()) > 0
    masked = logits.masked_fill(~(mask | ~supported.unsqueeze(-1)), float("-inf"))
    ref = torch.log_softmax(masked, -1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    torch.where(supported, ref * loss_weights, 0.0).sum().backward()

    assert torch.equal(has_support, supported)
    torch.testing.assert_close(log_probs[supported], ref[supported])
    torch.testing.assert_close(d_hidden, ref_hidden.grad)
    torch.testing.assert_close(local_weight.grad, ref_weight.grad[shard])


def _tp_worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    saved = _install_fake_megatron()
    try:
        _check_against_masked_softmax(world_size, rank)
    finally:
        _restore_modules(saved)
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2])
def test_support_log_probs_from_hidden_match_masked_softmax(world_size):
    if world_size == 1:
        saved = _install_fake_megatron()
        try:
            _check_against_masked_softmax(world_size, rank=0)
        finally:
            _restore_modules(saved)
    else:
        mp.spawn(_tp_worker, args=(world_size, 29500 + os.getpid() % 1000), nprocs=world_size, join=True)
