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

"""Integration test: multi-output SVGP, DeepGP and builder on synthetic data.

Two outputs, ``sin(2x)`` and ``cos(2x)``, observed with noise standard
deviations 0.05 and 0.3.  For each model:

* training loss (negative ELBO) decreases,
* ``predict`` returns finite ``(N, 2)`` means and strictly positive variances,
* each output's RMSE against the noise-free function is small,
* the noisier output learns the larger noise variance, which a single shared
  noise could not do.

The deep GP uses a width-2 hidden layer.  Every output is a function of the
last hidden layer's output, and at width 1 training from some random
initialisations folds that scalar so that ``sin`` and ``cos`` cannot both be
read off it.
"""

import math

import torch

import deepgp  # noqa: F401  (import sets float64 default dtype)
from deepgp import (
    SVGP,
    DeepGP,
    DeepGPConfig,
    build_deep_gp,
    fit,
    predict,
    seed_everything,
)
from deepgp.eval import gaussian_nll, rmse

NOISE_STD = torch.tensor([0.05, 0.3])
RMSE_THRESHOLD = 0.2  # against the noise-free functions


def _two_output_data():
    generator = torch.Generator().manual_seed(0)
    x = 4.0 * torch.rand(200, 1, generator=generator) - 2.0
    f = torch.cat([torch.sin(2.0 * x), torch.cos(2.0 * x)], dim=-1)
    y = f + NOISE_STD * torch.randn(200, 2, generator=generator)
    return x[:160], y[:160], x[160:], y[160:], f[160:]


def _check(model, losses, X_test, Y_test, F_test) -> None:
    assert all(math.isfinite(v) for v in losses), "loss became non-finite"
    assert sum(losses[-20:]) / 20.0 < sum(losses[:20]) / 20.0

    mean, var = predict(model, X_test, k=32)
    assert mean.shape == var.shape == Y_test.shape
    assert torch.isfinite(mean).all()
    assert (var > 0).all()
    assert torch.isfinite(gaussian_nll(mean, var, Y_test))
    for t in range(2):
        test_rmse = rmse(mean[:, t], F_test[:, t]).item()
        assert test_rmse < RMSE_THRESHOLD, f"output {t} RMSE {test_rmse:.4f}"

    noise = model.likelihood.task_noises.detach()
    assert noise[1] > noise[0], f"noise variances not ordered: {noise.tolist()}"


def test_deep_gp_two_outputs() -> None:
    X, Y, X_test, Y_test, F_test = _two_output_data()
    seed_everything(0)
    model = DeepGP([1, 2], num_inducing=32, num_outputs=2)
    losses = fit(model, X, Y, epochs=300, lr=0.01, num_samples=10)
    _check(model, losses, X_test, Y_test, F_test)


def test_svgp_two_outputs_recovers_each_noise_variance() -> None:
    X, Y, X_test, Y_test, F_test = _two_output_data()
    seed_everything(0)
    model = SVGP(X[:32].clone(), num_outputs=2)
    losses = fit(model, X, Y, epochs=300, lr=0.02)
    _check(model, losses, X_test, Y_test, F_test)

    # The outputs are independent GPs, so each noise variance is estimated from
    # its own column: both land within a factor of 2 of the truth.
    ratio = model.likelihood.task_noises.detach() / NOISE_STD**2
    assert ((ratio > 0.5) & (ratio < 2.0)).all(), f"noise / truth = {ratio}"


def test_build_deep_gp_two_outputs() -> None:
    X, Y, X_test, Y_test, F_test = _two_output_data()
    seed_everything(0)
    cfg = DeepGPConfig(
        num_inducing=20, inner_layer_qsqrt_factor=1e-5, likelihood_noise=1e-2
    )
    model = build_deep_gp(X, num_layers=2, config=cfg, num_outputs=2)
    losses = fit(model, X, Y, epochs=300, lr=0.02, num_samples=10)
    _check(model, losses, X_test, Y_test, F_test)
