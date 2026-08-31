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

"""Unit tests for ``DeepGPHiddenLayer`` (deepgp.layers.gp_layer).

Exercise the layer directly: the per-layer GP prior returned by ``forward``,
the ``DeepGPLayer`` sampling machinery reached via ``layer(x)``, mean-type
selection (and its error branch), the whitened/unwhitened variational-strategy
switch, and the persisting ``q_sqrt`` shrink.
"""

import gpytorch
import pytest
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean, LinearMean, ZeroMean
from gpytorch.variational import (
    UnwhitenedVariationalStrategy,
    VariationalStrategy,
)

import deepgp  # noqa: F401  (import sets float64 default dtype)
from deepgp.layers.gp_layer import DeepGPHiddenLayer


def _chol_diag(layer) -> torch.Tensor:
    strategy = layer.variational_strategy
    chol = strategy._variational_distribution.chol_variational_covar
    return chol.diagonal(dim1=-2, dim2=-1)


def test_forward_single_output_returns_mvn() -> None:
    torch.manual_seed(0)
    layer = DeepGPHiddenLayer(
        input_dims=1, output_dims=None, num_inducing=5, mean_type="constant"
    )
    x = torch.randn(7, 1)
    dist = layer.forward(x)

    assert isinstance(dist, MultivariateNormal)
    # Single-output layer: no batch dimension, event covers all N points.
    assert dist.batch_shape == torch.Size([])
    assert dist.event_shape == torch.Size([7])
    assert dist.mean.shape == torch.Size([7])
    assert torch.isfinite(dist.mean).all()
    # Covariance is a genuine N x N PSD matrix (diagonal strictly positive).
    covar = dist.covariance_matrix
    assert covar.shape == torch.Size([7, 7])
    assert (covar.diagonal() > 0).all()


def test_layer_call_multi_output_has_leading_sample_dim() -> None:
    torch.manual_seed(0)
    layer = DeepGPHiddenLayer(
        input_dims=1, output_dims=2, num_inducing=5, mean_type="linear"
    )
    x = torch.randn(7, 1)
    with gpytorch.settings.num_likelihood_samples(3):
        out = layer(x)

    mean = out.mean
    assert torch.isfinite(mean).all()
    # DeepGPLayer draws S samples -> leading sample dim; last dim = output_dims.
    assert mean.dim() == 3
    assert mean.shape[0] == 3
    assert mean.shape[-1] == 2
    # The middle dimension indexes the N input points.
    assert mean.shape[1] == 7


@pytest.mark.parametrize(
    "mean_type, expected",
    [
        ("constant", ConstantMean),
        ("zero", ZeroMean),
        ("linear", LinearMean),
    ],
)
def test_mean_type_selects_mean_module(mean_type, expected) -> None:
    layer = DeepGPHiddenLayer(
        input_dims=1, output_dims=None, num_inducing=4, mean_type=mean_type
    )
    assert type(layer.mean_module) is expected


def test_invalid_mean_type_raises_value_error() -> None:
    with pytest.raises(ValueError):
        DeepGPHiddenLayer(
            input_dims=1, output_dims=None, num_inducing=4, mean_type="bogus"
        )


def test_whiten_false_builds_unwhitened_strategy_and_forwards() -> None:
    torch.manual_seed(0)
    layer = DeepGPHiddenLayer(
        input_dims=1, output_dims=None, num_inducing=4, whiten=False
    )
    assert isinstance(layer.variational_strategy, UnwhitenedVariationalStrategy)
    # The default (whiten=True) path yields the plain whitened strategy...
    whitened = DeepGPHiddenLayer(
        input_dims=1, output_dims=None, num_inducing=4, whiten=True
    )
    assert isinstance(whitened.variational_strategy, VariationalStrategy)
    assert not isinstance(whitened.variational_strategy, UnwhitenedVariationalStrategy)

    # An unwhitened layer still produces a valid prior at x.
    x = torch.randn(6, 1)
    dist = layer.forward(x)
    assert isinstance(dist, MultivariateNormal)
    assert dist.mean.shape == torch.Size([6])
    assert torch.isfinite(dist.mean).all()


@pytest.mark.parametrize("output_dims", [None, 2])
def test_layer_casts_float32_inducing_points_to_float64(output_dims) -> None:
    """Explicit inducing points are normalised to the default dtype.

    Covers both the single-output ``(M, D)`` and batched ``(output_dims, M, D)``
    shapes, before and after a forward pass.
    """
    torch.manual_seed(0)
    shape = (5, 1) if output_dims is None else (output_dims, 5, 1)
    z32 = torch.randn(*shape, dtype=torch.float32)

    layer = DeepGPHiddenLayer(
        input_dims=1,
        output_dims=output_dims,
        num_inducing=5,
        inducing_points=z32,
    )

    inducing = layer.variational_strategy.inducing_points
    assert inducing.dtype == torch.float64
    assert torch.equal(inducing.detach(), z32.to(torch.float64))
    assert {p.dtype for p in layer.parameters()} == {torch.float64}

    with gpytorch.settings.num_likelihood_samples(2):
        out = layer(torch.randn(4, 1))
    assert out.mean.dtype == torch.float64


def test_q_sqrt_factor_shrinks_and_persists_past_forward() -> None:
    torch.manual_seed(0)
    factor = 1e-4
    layer = DeepGPHiddenLayer(
        input_dims=1,
        output_dims=1,
        num_inducing=5,
        mean_type="linear",
        q_sqrt_factor=factor,
    )

    diag_before = _chol_diag(layer).abs()
    # Default Cholesky init is the identity, so the shrunk diagonal ~ factor.
    assert torch.allclose(
        diag_before,
        torch.full_like(diag_before, factor),
        rtol=1e-3,
        atol=1e-8,
    )

    # The strategy lazily resets q(u) to the prior on first forward; the
    # initialised flag must keep the shrink in place.
    x = torch.randn(6, 1)
    with gpytorch.settings.num_likelihood_samples(2):
        _ = layer(x)

    diag_after = _chol_diag(layer).abs()
    assert torch.allclose(
        diag_after,
        torch.full_like(diag_after, factor),
        rtol=1e-3,
        atol=1e-8,
    )
    # Sanity: without the shrink the diagonal would be ~1.0, far from factor.
    assert diag_after.max().item() < 1e-2
