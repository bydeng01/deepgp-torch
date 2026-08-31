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

"""Unit tests for the architecture builder and its factories.

Verify that ``build_deep_gp`` applies the configured stable-start
initialisation: dim-preserving hidden layers, KMeans inducing init,
identity/PCA hidden means, hidden ``q_sqrt`` shrunk by
``inner_layer_qsqrt_factor`` (and persisting past the lazy first-forward reset),
and a Gaussian likelihood initialised to ``likelihood_noise``.
"""

import torch

import deepgp  # noqa: F401  (import sets float64 default dtype)
from deepgp import DeepGPConfig, build_deep_gp
from deepgp.data import init_inducing
from deepgp.kernels import make_kernel
from deepgp.means import make_mean


def _cfg(**kw) -> DeepGPConfig:
    base = dict(num_inducing=16, inner_layer_qsqrt_factor=1e-5, likelihood_noise=1e-2)
    base.update(kw)
    return DeepGPConfig(**base)


def _chol_diag(layer) -> torch.Tensor:
    chol = layer.variational_strategy._variational_distribution.chol_variational_covar
    return chol.diagonal(dim1=-2, dim2=-1)


def test_make_mean_identity_weights() -> None:
    mean = make_mean(3, 3, kind="identity")
    assert torch.allclose(mean.weights.squeeze(-1), torch.eye(3))
    assert torch.allclose(mean.bias, torch.zeros_like(mean.bias))


def test_make_mean_pca_directions_orthonormal() -> None:
    torch.manual_seed(0)
    x = torch.randn(50, 4)
    mean = make_mean(4, 2, kind="pca", X=x)
    weights = mean.weights.squeeze(-1)  # (output_dims=2, input_dims=4)
    assert weights.shape == (2, 4)
    # Principal directions are orthonormal rows.
    assert torch.allclose(weights @ weights.transpose(-1, -2), torch.eye(2), atol=1e-6)


def test_make_kernel_batch_shape() -> None:
    kernel = make_kernel(3, output_dims=2)
    assert kernel.batch_shape == torch.Size([2])


def test_init_inducing_shapes_and_determinism() -> None:
    torch.manual_seed(0)
    x = torch.randn(40, 2)
    z1 = init_inducing(x, 10, method="kmeans", seed=0)
    z2 = init_inducing(x, 10, method="kmeans", seed=0)
    assert z1.shape == (10, 2)
    assert torch.allclose(z1, z2)  # deterministic given the seed

    z_subset = init_inducing(x, 10, method="subset", seed=1)
    assert z_subset.shape == (10, 2)

    # Cannot place more centres than there are points -> capped to N.
    assert init_inducing(x, 1000, method="subset", seed=0).shape[0] == 40

    # Explicit override is used verbatim.
    assert torch.allclose(init_inducing(x, 5, z_init=x[:7]), x[:7])


def test_build_deep_gp_requires_config() -> None:
    try:
        build_deep_gp(torch.randn(20, 1), num_layers=2, config=None)
    except ValueError:
        return
    raise AssertionError("expected ValueError when config is None")


def test_build_deep_gp_applies_configured_initialisation() -> None:
    torch.manual_seed(0)
    x = torch.randn(60, 1)
    cfg = _cfg(num_inducing=12, inner_layer_qsqrt_factor=1e-5, likelihood_noise=0.03)
    model = build_deep_gp(x, num_layers=2, config=cfg, seed=0)

    # Structure: one dim-preserving hidden layer + a single-output layer.
    assert len(model.hidden_layers) == 1
    hidden = model.hidden_layers[0]
    assert hidden.input_dims == 1 and hidden.output_dims == 1
    assert model.last_layer.output_dims is None

    # KMeans inducing init, tiled across the output dim for the hidden layer.
    assert tuple(hidden.variational_strategy.inducing_points.shape) == (1, 12, 1)
    assert tuple(model.last_layer.variational_strategy.inducing_points.shape) == (
        12,
        1,
    )

    # Identity hidden mean.
    assert torch.allclose(hidden.mean_module.weights.squeeze(-1), torch.eye(1))
    assert torch.allclose(
        hidden.mean_module.bias, torch.zeros_like(hidden.mean_module.bias)
    )

    # Hidden q_sqrt shrunk to ~1e-5 AND persisting past the first forward pass
    # (the strategy's lazy init must not reset it back to identity).
    before = _chol_diag(hidden).abs().mean().item()
    _ = model(x[:5])
    after = _chol_diag(hidden).abs().mean().item()
    assert abs(before - 1e-5) < 1e-7
    assert abs(after - 1e-5) < 1e-7

    # Output layer q_sqrt is NOT shrunk (default ~ identity).
    assert _chol_diag(model.last_layer).abs().mean().item() > 0.5

    # Likelihood noise initialised via the constrained property.
    assert abs(float(model.likelihood.noise.detach()) - 0.03) < 1e-6
