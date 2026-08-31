# Copyright 2026 Boyuan Deng.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for :mod:`deepgp.eval.metrics` (``rmse`` and ``gaussian_nll``)."""

import math

import torch

import deepgp  # noqa: F401  (sets torch default dtype to float64)
from deepgp.eval.metrics import gaussian_nll, rmse


def test_rmse_hand_computed():
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 5.0])
    # diffs = [0, 0, -2] -> mean squared = 4/3 -> sqrt(4/3)
    out = rmse(pred, target)
    assert out.item() == math.sqrt(4.0 / 3.0)


def test_rmse_reshapes_mismatched_shapes():
    pred = torch.tensor([[1.0], [2.0], [3.0]])  # shape (3, 1)
    target = torch.tensor([1.0, 2.0, 5.0])  # shape (3,)
    out = rmse(pred, target)
    assert out.item() == math.sqrt(4.0 / 3.0)


def test_rmse_identical_is_exactly_zero():
    v = torch.tensor([1.5, -2.0, 3.25, 0.0])
    out = rmse(v, v.clone())
    assert out.item() == 0.0


def test_gaussian_nll_mean_equals_target():
    target = torch.tensor([0.5, -1.0, 2.0])
    var = torch.tensor([0.25, 1.0, 4.0])
    # mean == target -> squared error term vanishes:
    # nll_i = 0.5 * (log(2*pi) + log(var_i))
    expected = torch.mean(0.5 * (math.log(2.0 * math.pi) + torch.log(var))).item()
    out = gaussian_nll(target.clone(), var, target)
    assert out.item() == expected


def test_gaussian_nll_general_formula():
    mean = torch.tensor([0.0, 1.0, -2.0])
    var = torch.tensor([0.5, 2.0, 0.1])
    target = torch.tensor([0.3, 0.5, -1.5])
    per_point = 0.5 * (
        math.log(2.0 * math.pi) + torch.log(var) + (target - mean) ** 2 / var
    )
    expected = torch.mean(per_point).item()
    out = gaussian_nll(mean, var, target)
    assert out.item() == expected


def test_gaussian_nll_eps_clamp_prevents_inf():
    mean = torch.tensor([1.0, 2.0])
    var = torch.tensor([0.0, 1.0])  # zero variance would give -inf/inf
    target = torch.tensor([1.5, 2.0])
    out = gaussian_nll(mean, var, target)
    assert torch.isfinite(out)


def test_metrics_return_scalar_tensors():
    pred = torch.tensor([1.0, 2.0, 3.0])
    var = torch.tensor([1.0, 1.0, 1.0])
    target = torch.tensor([1.0, 2.0, 4.0])
    r = rmse(pred, target)
    n = gaussian_nll(pred, var, target)
    assert isinstance(r, torch.Tensor) and r.shape == torch.Size([])
    assert isinstance(n, torch.Tensor) and n.shape == torch.Size([])
