"""Regression test for the R3 nopad `routed_experts_compact` object-array construction.

Bug: `np.array([per_traj_array, ...], dtype=object)` collapses to a rectangular 4-D array
when every trajectory in a rollout group shares one response length (uniform shape), instead
of the intended 1-D object array of 3-D `[length, layer, topk]` slices. A uniform (4-D) sample
colliding with a ragged (1-D) sample inside `DataProto.concat` then fails at
`np.concatenate(val, axis=0)` with mismatched ndims. See agent_loop.py `_postprocess`.
"""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl.protocol import DataProto

LAYER, TOPK, SEQ = 48, 8, 200


def _make_sample(lengths, compact_builder):
    """One rollout sample (group of len(lengths) trajectories) as a DataProto.

    compact_builder maps a list of per-trajectory [length, LAYER, TOPK] uint8 arrays to the
    non_tensor value actually stored — this is the construction under test.
    """
    n = len(lengths)
    slices = [np.zeros((L, LAYER, TOPK), dtype=np.uint8) for L in lengths]
    batch = TensorDict({"input_ids": torch.zeros(n, SEQ, dtype=torch.long)}, batch_size=[n])
    non_tensor = {
        "routed_experts_compact": compact_builder(slices),
        "routed_experts_start_pos": np.array([SEQ - L for L in lengths], dtype=np.int32),
    }
    return DataProto(batch=batch, non_tensor_batch=non_tensor)


def _old_builder(slices):
    # The buggy construction (pre-fix).
    return np.array([s for s in slices], dtype=object)


def _new_builder(slices):
    # The fixed construction, mirroring agent_loop.py `_postprocess`.
    arr = np.empty(len(slices), dtype=object)
    for i, s in enumerate(slices):
        arr[i] = s
    return arr


def test_old_construction_collapses_and_breaks_concat():
    # A uniform-length group is the rare production trigger.
    uniform = _old_builder([np.zeros((100, LAYER, TOPK), dtype=np.uint8) for _ in range(4)])
    assert uniform.ndim == 4, "expected the buggy collapse to a rectangular 4-D array"

    ragged_sample = _make_sample([50, 60, 70, 80], _new_builder)  # normal ragged sample: 1-D
    uniform_sample = _make_sample([100, 100, 100, 100], _old_builder)  # collapsed: 4-D

    with pytest.raises(ValueError, match="same number of dimensions"):
        DataProto.concat([ragged_sample, uniform_sample])


def test_fixed_construction_stays_1d_and_concats():
    for lengths in ([50, 60, 70, 80], [100, 100, 100, 100], [7], [5, 5]):
        compact = _new_builder([np.zeros((L, LAYER, TOPK), dtype=np.uint8) for L in lengths])
        assert compact.ndim == 1 and compact.shape == (len(lengths),)
        assert compact[0].shape == (lengths[0], LAYER, TOPK)

    ragged_sample = _make_sample([50, 60, 70, 80], _new_builder)
    uniform_sample = _make_sample([100, 100, 100, 100], _new_builder)  # the ex-trigger, now safe

    merged = DataProto.concat([ragged_sample, uniform_sample])
    compacts = merged.non_tensor_batch["routed_experts_compact"]
    starts = merged.non_tensor_batch["routed_experts_start_pos"]
    assert compacts.shape == (8,)
    assert [c.shape[0] for c in compacts] == [50, 60, 70, 80, 100, 100, 100, 100]

    # Mirror the detach_utils JIT reconstruction to prove the merged batch is still usable.
    dense = torch.zeros(len(compacts), SEQ, LAYER, TOPK, dtype=torch.uint8)
    for i, c in enumerate(compacts):
        sp = int(starts[i])
        dense[i, sp : sp + c.shape[0]] = torch.from_numpy(c)
    assert dense.shape == (8, SEQ, LAYER, TOPK)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
