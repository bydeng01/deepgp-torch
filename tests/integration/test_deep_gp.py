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

"""Integration smoke test: SVGP and a 2-layer DeepGP on snelson1d.

Asserts that:

* training loss (negative ELBO) decreases,
* ``predict`` returns a finite mean and strictly-positive variance of the right
  shape (via the total-variance mixture reduction),
* test RMSE is below a loose threshold.
"""

import math

import torch

import deepgp  # noqa: F401  (import sets float64 default dtype)
from deepgp import SVGP, DeepGP, fit, predict, seed_everything
from deepgp.data import load_snelson1d
from deepgp.eval import gaussian_nll, rmse

RMSE_THRESHOLD = 0.5  # loose threshold on standardised targets (y std ~= 1)


def _assert_loss_decreases(losses: list) -> None:
    assert len(losses) > 40
    assert all(math.isfinite(v) for v in losses), "loss became non-finite"
    start = sum(losses[:20]) / 20.0
    end = sum(losses[-20:]) / 20.0
    assert end < start, f"loss did not decrease: start={start:.4f} end={end:.4f}"


def _assert_predictions_ok(mean, var, target) -> None:
    assert mean.shape == target.shape
    assert var.shape == target.shape
    assert torch.isfinite(mean).all(), "predictive mean has non-finite values"
    assert (var > 0).all(), "predictive variance must be strictly positive"
    assert torch.isfinite(gaussian_nll(mean, var, target)), "NLL is non-finite"


def test_params_are_float64() -> None:
    assert torch.get_default_dtype() == torch.float64
    model = DeepGP([1, 1], num_inducing=8)
    dtypes = {p.dtype for p in model.parameters()}
    assert dtypes == {torch.float64}, f"expected all float64 params, got {dtypes}"


def test_deep_gp_smoke_snelson1d() -> None:
    seed_everything(0)
    data = load_snelson1d(test_fraction=0.2, seed=0)

    model = DeepGP([1, 1], num_inducing=32)
    losses = fit(
        model,
        data.X_train,
        data.Y_train,
        epochs=300,
        lr=0.01,
        num_samples=10,
    )
    _assert_loss_decreases(losses)

    mean, var = predict(model, data.X_test, k=32)
    _assert_predictions_ok(mean, var, data.Y_test)

    test_rmse = rmse(mean, data.Y_test).item()
    assert test_rmse < RMSE_THRESHOLD, f"test RMSE too high: {test_rmse:.4f}"


def test_svgp_smoke_snelson1d() -> None:
    seed_everything(0)
    data = load_snelson1d(test_fraction=0.2, seed=0)

    inducing = data.X_train[:32].clone()
    model = SVGP(inducing, input_dims=data.input_dims)
    losses = fit(model, data.X_train, data.Y_train, epochs=300, lr=0.01)
    _assert_loss_decreases(losses)

    mean, var = predict(model, data.X_test)
    _assert_predictions_ok(mean, var, data.Y_test)

    test_rmse = rmse(mean, data.Y_test).item()
    assert test_rmse < RMSE_THRESHOLD, f"test RMSE too high: {test_rmse:.4f}"
