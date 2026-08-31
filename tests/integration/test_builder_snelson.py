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

"""Integration test: the architecture builder trains on snelson1d.

Standard builder setting (dim-preserving 2-layer DGP, KMeans inducing, identity
hidden mean, shrunk hidden q_sqrt, Gaussian noise init) — built with
``build_deep_gp`` and trained with the standard ELBO loop.
"""

import math

import torch

import deepgp  # noqa: F401  (import sets float64 default dtype)
from deepgp import DeepGPConfig, build_deep_gp, fit, predict, seed_everything
from deepgp.data import load_snelson1d
from deepgp.eval import gaussian_nll, rmse

RMSE_THRESHOLD = 0.5


def test_build_deep_gp_trains_on_snelson1d() -> None:
    seed_everything(0)
    data = load_snelson1d(test_fraction=0.2, seed=0)

    cfg = DeepGPConfig(
        num_inducing=20,
        inner_layer_qsqrt_factor=1e-5,
        likelihood_noise=1e-2,
    )
    model = build_deep_gp(data.X_train, num_layers=2, config=cfg, seed=0)

    losses = fit(
        model,
        data.X_train,
        data.Y_train,
        epochs=300,
        lr=0.01,
        num_samples=10,
    )
    assert all(math.isfinite(v) for v in losses), "loss became non-finite"
    assert sum(losses[-20:]) / 20.0 < sum(losses[:20]) / 20.0

    mean, var = predict(model, data.X_test, k=32)
    assert mean.shape == data.Y_test.shape
    assert torch.isfinite(mean).all()
    assert (var > 0).all()
    assert torch.isfinite(gaussian_nll(mean, var, data.Y_test))

    test_rmse = rmse(mean, data.Y_test).item()
    assert test_rmse < RMSE_THRESHOLD, f"test RMSE too high: {test_rmse:.4f}"
