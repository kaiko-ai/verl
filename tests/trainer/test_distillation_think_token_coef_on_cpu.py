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

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from verl.trainer.distillation import losses

LOSSES = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]])
RESPONSE_MASK = [torch.tensor([1, 1, 1]), torch.tensor([1, 1])]
THINK_MASK = [torch.tensor([1, 0, 0]), torch.tensor([0, 0])]


def _configs(think_token_coef: float):
    actor = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={}, loss_scale_factor=None)
    loss_config = SimpleNamespace(
        loss_mode="stub", think_token_coef=think_token_coef, loss_max_clamp=None, use_policy_gradient=False
    )
    return actor, SimpleNamespace(distillation_loss=loss_config)


def _data(nested: bool) -> dict:
    if nested:
        response_mask = torch.nested.nested_tensor(RESPONSE_MASK, layout=torch.jagged)
        think_mask = torch.nested.nested_tensor(THINK_MASK, layout=torch.jagged)
    else:
        response_mask = torch.nn.utils.rnn.pad_sequence(RESPONSE_MASK, batch_first=True)
        think_mask = torch.nn.utils.rnn.pad_sequence(THINK_MASK, batch_first=True)
    return {
        "response_mask": response_mask,
        "think_token_mask": think_mask,
        "dp_size": 1,
        "batch_num_tokens": 5,
        "global_batch_size": 2,
    }


def _run(think_token_coef: float, nested: bool):
    actor, distillation = _configs(think_token_coef)
    with patch.object(losses, "get_distillation_loss_fn", return_value=lambda **_: (LOSSES.clone(), {})):
        return losses.distillation_loss(actor, distillation, model_output={}, data=_data(nested))


@pytest.mark.parametrize("nested", [False, True])
def test_think_tokens_excluded_at_zero_coef(nested):
    loss, metrics = _run(think_token_coef=0.0, nested=nested)
    # Only the first token is a think token; the remaining four answer tokens sum to 2+3+4+5.
    assert loss.item() == pytest.approx((2.0 + 3.0 + 4.0 + 5.0) / 5)
    assert metrics["distillation/think_ratio"].aggregate() == pytest.approx(1 / 5)


@pytest.mark.parametrize("nested", [False, True])
def test_coef_one_leaves_loss_unweighted(nested):
    loss, metrics = _run(think_token_coef=1.0, nested=nested)
    assert loss.item() == pytest.approx((1.0 + 2.0 + 3.0 + 4.0 + 5.0) / 5)
    assert "distillation/think_ratio" not in metrics
