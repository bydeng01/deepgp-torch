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

"""Unit tests for the model containers (``SVGP`` and ``DeepGP``).

Covers construction-time validation of inducing/``dims`` shapes, the attached
Gaussian likelihood, ``DeepGP``'s hidden-layer stacking and doubly-stochastic
forward pass under ``num_likelihood_samples``, the ``from_layers`` assembly
path, and ``is_deep_gp`` discrimination between shallow and deep models.
"""

import gpytorch
import pytest
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood

import deepgp  # noqa: F401  (import sets float64 default dtype)
from deepgp import SVGP, DeepGP, DeepGPHiddenLayer, fit
from deepgp.training.elbo import is_deep_gp


@pytest.mark.parametrize("bad_shape", [(6,), (6, 1, 1)])
def test_svgp_rejects_non_2d_inducing_points(bad_shape) -> None:
    with pytest.raises(ValueError):
        SVGP(torch.randn(*bad_shape))


def test_svgp_valid_inducing_builds_and_forwards() -> None:
    deepgp.seed_everything(0)
    model = SVGP(torch.randn(6, 2))
    x = torch.randn(4, 2)
    out = model.forward(x)
    assert isinstance(out, MultivariateNormal)
    # Event ranges over the N=4 evaluation points.
    assert out.event_shape == torch.Size([4])
    assert out.mean.shape == torch.Size([4])


def test_svgp_has_gaussian_likelihood() -> None:
    model = SVGP(torch.randn(5, 1))
    assert isinstance(model.likelihood, GaussianLikelihood)


def test_svgp_casts_float32_inducing_points_to_float64() -> None:
    """A float32 ``Z`` must not leave the model in mixed precision.

    The strategy registers the caller's tensor as a *learnable* parameter, so an
    un-cast float32 input would train a float32 ``Z`` alongside float64 kernel /
    mean / noise parameters — silently, because torch promotes on every op.
    """
    deepgp.seed_everything(0)
    z32 = torch.randn(6, 1, dtype=torch.float32)
    model = SVGP(z32, input_dims=1)

    inducing = model.variational_strategy.inducing_points
    assert inducing.dtype == torch.float64
    # Only the dtype changes; float32 -> float64 is exact, so values survive.
    assert torch.equal(inducing.detach(), z32.to(torch.float64))
    assert {p.dtype for p in model.parameters()} == {torch.float64}

    # ...and a training step keeps every parameter in double precision.
    fit(model, torch.randn(20, 1), torch.randn(20), epochs=2, lr=0.05)
    assert {p.dtype for p in model.parameters()} == {torch.float64}


def test_deep_gp_empty_dims_raises() -> None:
    with pytest.raises(ValueError):
        DeepGP([])


def test_deep_gp_stacks_two_hidden_layers_and_forwards() -> None:
    deepgp.seed_everything(0)
    model = DeepGP([1, 1, 1], num_inducing=4)
    # dims=[1,1,1] -> two hidden layers + one output layer.
    assert len(model.hidden_layers) == 2
    assert isinstance(model.last_layer, DeepGPHiddenLayer)

    x = torch.randn(5, 1)
    with gpytorch.settings.num_likelihood_samples(3):
        out = model(x)
    # Doubly-stochastic: leading sample dim of size 3 over the N=5 points.
    assert out.mean.shape == torch.Size([3, 5])
    assert torch.isfinite(out.mean).all()


def test_is_deep_gp_discriminates_shallow_and_deep() -> None:
    deepgp.seed_everything(0)
    shallow = SVGP(torch.randn(5, 1))
    deep = DeepGP([1, 1], num_inducing=4)
    assert is_deep_gp(shallow) is False
    assert is_deep_gp(deep) is True


def test_deep_gp_from_layers_assembles_working_model() -> None:
    deepgp.seed_everything(0)
    h1 = DeepGPHiddenLayer(
        input_dims=1, output_dims=1, num_inducing=4, mean_type="linear"
    )
    h2 = DeepGPHiddenLayer(
        input_dims=1, output_dims=1, num_inducing=4, mean_type="linear"
    )
    last = DeepGPHiddenLayer(
        input_dims=1, output_dims=None, num_inducing=4, mean_type="constant"
    )
    model = DeepGP.from_layers([h1, h2], last)

    assert is_deep_gp(model) is True
    assert isinstance(model.likelihood, GaussianLikelihood)
    assert len(model.hidden_layers) == 2
    assert model.last_layer is last

    x = torch.randn(5, 1)
    with gpytorch.settings.num_likelihood_samples(3):
        out = model(x)
    assert out.mean.shape == torch.Size([3, 5])
    assert torch.isfinite(out.mean).all()
