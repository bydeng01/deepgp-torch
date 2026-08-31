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

"""Tests for :func:`deepgp.predict` predictive mean/variance reduction.

Covers the deep-GP mixture reduction (law of total variance), the
observation-noise inflation from ``add_noise``, the ``k == 1`` NaN guard
(``unbiased=False``), the shallow ``SVGP`` (non-deep) branch, and the
train/eval mode restoration that keeps GPyTorch's variational caches from going
stale between calls.  No training is required: an untrained model already
yields finite means and strictly positive predictive variances.
"""

import pytest
import torch

import deepgp  # noqa: F401  (import sets float64 default dtype)
from deepgp import SVGP, DeepGP, predict


def _tiny_deep_gp() -> DeepGP:
    """A 2-layer single-output deep GP (1 hidden + 1 output layer)."""
    deepgp.seed_everything(0)
    return DeepGP(dims=[1, 1], num_inducing=8)


def _tiny_svgp() -> SVGP:
    """A shallow single-output SVGP with 8 inducing points."""
    deepgp.seed_everything(0)
    inducing = torch.linspace(-1.0, 1.0, 8).unsqueeze(-1)
    return SVGP(inducing_points=inducing)


def _test_inputs(n: int = 6) -> torch.Tensor:
    return torch.linspace(-1.5, 1.5, n).unsqueeze(-1)


def test_deep_gp_predict_shapes_and_positive_variance() -> None:
    model = _tiny_deep_gp()
    X = _test_inputs(6)

    torch.manual_seed(1)
    mean, var = predict(model, X, k=16)

    assert mean.shape == (6,)
    assert var.shape == (6,)
    assert torch.isfinite(mean).all()
    assert (var > 0).all()


def test_add_noise_inflates_variance_by_observation_noise() -> None:
    model = _tiny_deep_gp()
    X = _test_inputs(6)
    k = 16

    # Seed each call so the test is reproducible; the mixture sampling still
    # consumes RNG differently between the two branches, but the observation
    # noise (~0.7) dominates the tiny Monte-Carlo jitter, so the elementwise
    # inequality holds robustly.
    torch.manual_seed(7)
    _, var_latent = predict(model, X, k=k, add_noise=False)
    torch.manual_seed(7)
    _, var_noisy = predict(model, X, k=k, add_noise=True)

    # Observation noise can only add variance, elementwise.
    assert (var_noisy >= var_latent - 1e-9).all()
    assert var_noisy.mean() > var_latent.mean()

    # The average inflation is on the order of the observation noise.
    noise = model.likelihood.noise.item()
    assert noise > 0.0
    diff = (var_noisy - var_latent).mean().item()
    assert 0.5 * noise < diff < 1.5 * noise


def test_k1_single_sample_is_finite_and_positive() -> None:
    # k == 1 => Var_s[m_s] with unbiased=False must be 0, not 0/0 -> NaN.
    model = _tiny_deep_gp()
    X = _test_inputs(5)

    torch.manual_seed(3)
    mean, var = predict(model, X, k=1)

    assert mean.shape == (5,)
    assert var.shape == (5,)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(var).all()
    assert (var > 0).all()


def test_svgp_shallow_branch_shapes_finite_positive() -> None:
    model = _tiny_svgp()
    X = _test_inputs(6)

    # k is irrelevant for the shallow (non-deep) branch.
    mean, var = predict(model, X, k=5)

    assert mean.shape == (6,)
    assert var.shape == (6,)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(var).all()
    assert (var > 0).all()


def test_svgp_add_noise_increases_variance() -> None:
    model = _tiny_svgp()
    X = _test_inputs(6)

    _, var_latent = predict(model, X, add_noise=False)
    _, var_noisy = predict(model, X, add_noise=True)

    # Shallow branch is deterministic (no sampling); noise strictly inflates.
    assert (var_noisy > var_latent).all()
    noise = model.likelihood.noise.item()
    expected = torch.full_like(var_latent, noise)
    assert torch.allclose(var_noisy - var_latent, expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# Train/eval mode restoration (and the stale-cache regression it prevents)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_factory", [_tiny_svgp, _tiny_deep_gp])
@pytest.mark.parametrize("start_training", [True, False])
def test_predict_restores_training_mode(model_factory, start_training) -> None:
    """``predict`` must hand the model back in the mode it received it in.

    ``gpytorch.Module.train`` is where the memoised eval-mode quantities are
    dropped, so leaking eval mode pins them.
    """
    model = model_factory()
    model.train(start_training)

    predict(model, _test_inputs(4), k=2)

    assert model.training is start_training


def test_predict_restores_training_mode_even_when_the_forward_raises() -> None:
    model = _tiny_svgp()
    model.train()

    with pytest.raises(RuntimeError):
        predict(model, torch.randn(4, 7))  # input_dims mismatch (ARD expects 1)

    assert model.training is True


def test_predict_reflects_parameter_changes_made_between_calls() -> None:
    """A hyper-parameter change between two ``predict`` calls must take effect.

    Regression test.  ``predict`` used to leave the model in eval mode, which
    pinned GPyTorch's cached ``K_ZZ`` Cholesky factor; the second call then
    combined a freshly-computed ``K_xZ`` with that stale factor and silently
    returned a wrong answer (observed error ~1.06 on standardised targets).

    The shallow branch is fully deterministic, so the two paths must agree
    exactly -- not merely to a tolerance.
    """
    X = _test_inputs(6)
    new_lengthscale = 0.05

    # A: predict, change a hyper-parameter, predict again.
    model_a = _tiny_svgp()
    predict(model_a, X)
    model_a.covar_module.base_kernel.lengthscale = new_lengthscale
    mean_a, var_a = predict(model_a, X)

    # B: identical model, same change, predicted once -- no cache to go stale.
    model_b = _tiny_svgp()
    model_b.covar_module.base_kernel.lengthscale = new_lengthscale
    mean_b, var_b = predict(model_b, X)

    assert torch.equal(mean_a, mean_b)
    assert torch.equal(var_a, var_b)
