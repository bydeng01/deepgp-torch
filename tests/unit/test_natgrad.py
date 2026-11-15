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

"""Unit tests for natural-parameter ``q(u)`` and natural-gradient training.

Covers the ``variational`` option of every model, the hidden ``q_sqrt`` shrink
in the natural parameterisation, how :func:`make_optimizers` splits the
parameters between NGD and Adam (and the misuse it refuses), and that NGD
trains SVGP, multi-output SVGP and deep GPs, converging faster than Adam on an
SVGP.
"""

import math

import pytest
import torch
from gpytorch.optim import NGD
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    NaturalVariationalDistribution,
    TrilNaturalVariationalDistribution,
)

import deepgp  # noqa: F401  (import sets float64 default dtype)
from deepgp import (
    SVGP,
    DeepGP,
    DeepGPConfig,
    DeepGPHiddenLayer,
    ExactGP,
    build_deep_gp,
    fit,
)
from deepgp.data import init_inducing, load_snelson1d
from deepgp.training import make_optimizers, natural_variational_parameters
from deepgp.variational import (
    make_variational_distribution,
    scale_variational_covariance,
)


def _q(strategy):
    return strategy._variational_distribution


def _data(n=40, d=1):
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(n, d, generator=gen) * 4 - 2
    y = torch.sin(2 * x[:, 0]) + 0.1 * torch.randn(n, generator=gen)
    return x, y


# --------------------------------------------------------------------------
# The variational option
# --------------------------------------------------------------------------
def test_make_variational_distribution_kinds() -> None:
    chol = make_variational_distribution("cholesky", 5, torch.Size([2]))
    nat = make_variational_distribution("Natural", 5)  # case-insensitive
    assert isinstance(chol, CholeskyVariationalDistribution)
    assert chol.batch_shape == torch.Size([2])
    assert isinstance(nat, TrilNaturalVariationalDistribution)
    with pytest.raises(ValueError, match="cholesky"):
        make_variational_distribution("meanfield", 5)


def test_every_model_takes_the_variational_option() -> None:
    z = torch.randn(6, 2)
    assert isinstance(_q(SVGP(z).variational_strategy), CholeskyVariationalDistribution)
    svgp = SVGP(z, variational="natural")
    assert isinstance(_q(svgp.variational_strategy), TrilNaturalVariationalDistribution)

    multi = SVGP(z, num_outputs=3, variational="natural")
    q = _q(multi.variational_strategy.base_variational_strategy)
    assert isinstance(q, TrilNaturalVariationalDistribution)
    assert q.batch_shape == torch.Size([3])

    deep = DeepGP([2, 2], num_inducing=6, output_variational="natural")
    assert isinstance(
        _q(deep.hidden_layers[0].variational_strategy), CholeskyVariationalDistribution
    )
    assert isinstance(
        _q(deep.last_layer.variational_strategy), TrilNaturalVariationalDistribution
    )
    deep = DeepGP([2, 2], num_inducing=6, hidden_variational="natural")
    assert isinstance(
        _q(deep.hidden_layers[0].variational_strategy),
        TrilNaturalVariationalDistribution,
    )

    x, _ = _data(d=2)
    cfg = DeepGPConfig(
        num_inducing=6,
        inner_layer_qsqrt_factor=1e-5,
        likelihood_noise=1e-2,
        hidden_variational="natural",
        output_variational="natural",
    )
    built = build_deep_gp(x, num_layers=3, config=cfg)
    assert len(natural_variational_parameters(built)) == 2 * 3  # 2 per layer


def test_natural_q_sqrt_shrink_matches_cholesky_shrink() -> None:
    """The shrunk hidden q(u) is the same distribution in both parameterisations."""
    kwargs = dict(input_dims=2, output_dims=3, num_inducing=5, q_sqrt_factor=1e-3)
    torch.manual_seed(0)
    chol = DeepGPHiddenLayer(**kwargs)
    torch.manual_seed(0)
    nat = DeepGPHiddenLayer(**kwargs, variational="natural")

    q_chol = _q(chol.variational_strategy)()
    q_nat = _q(nat.variational_strategy)()
    assert torch.allclose(q_nat.mean, torch.zeros(3, 5))
    assert torch.allclose(q_nat.mean, q_chol.mean)
    expected = 1e-6 * torch.eye(5).expand(3, 5, 5)  # S = factor**2 I
    assert torch.allclose(q_nat.covariance_matrix, expected, rtol=1e-8, atol=0)
    assert torch.allclose(q_chol.covariance_matrix, expected, rtol=1e-8, atol=0)

    # Marked initialised, so the first forward pass does not reset it.
    _ = nat(torch.randn(4, 2))
    assert torch.allclose(_q(nat.variational_strategy)().covariance_matrix, expected)


def test_scale_variational_covariance_rejects_other_distributions() -> None:
    with pytest.raises(TypeError):
        scale_variational_covariance(NaturalVariationalDistribution(4), 0.5)


# --------------------------------------------------------------------------
# make_optimizers
# --------------------------------------------------------------------------
def test_make_optimizers_splits_parameters_between_adam_and_ngd() -> None:
    model = DeepGP([1, 1], num_inducing=5, output_variational="natural")
    adam, ngd = make_optimizers(model, num_data=40, lr=0.01, natgrad_lr=0.1)

    assert isinstance(adam, torch.optim.Adam) and isinstance(ngd, NGD)
    adam_ids = {id(p) for g in adam.param_groups for p in g["params"]}
    ngd_ids = {id(p) for g in ngd.param_groups for p in g["params"]}
    natural = natural_variational_parameters(model)
    assert ngd_ids == {id(p) for p in natural}
    assert len(natural) == 2  # natural_vec and natural_tril_mat
    # Disjoint, and together they cover every parameter exactly once.
    assert not adam_ids & ngd_ids
    assert adam_ids | ngd_ids == {id(p) for p in model.parameters()}
    assert ngd.num_data == 40 and ngd.param_groups[0]["lr"] == 0.1


def test_make_optimizers_without_natgrad_is_plain_adam() -> None:
    model = SVGP(torch.randn(5, 1))
    (adam,) = make_optimizers(model, num_data=10, lr=0.05)
    assert isinstance(adam, torch.optim.Adam)
    assert adam.param_groups[0]["lr"] == 0.05
    assert natural_variational_parameters(model) == []


def test_natural_model_requires_natgrad_lr() -> None:
    x, y = _data()
    model = SVGP(x[:5].clone(), variational="natural")
    with pytest.raises(ValueError, match="natgrad_lr"):
        fit(model, x, y, epochs=1)


@pytest.mark.parametrize("make", [lambda x, y: SVGP(x[:5].clone()), ExactGP])
def test_natgrad_lr_requires_a_natural_model(make) -> None:
    x, y = _data()
    with pytest.raises(ValueError, match="variational='natural'"):
        fit(make(x, y), x, y, epochs=1, natgrad_lr=0.1)


def test_natgrad_lr_must_be_positive() -> None:
    x, y = _data()
    model = SVGP(x[:5].clone(), variational="natural")
    with pytest.raises(ValueError, match="> 0"):
        fit(model, x, y, epochs=1, natgrad_lr=0.0)


def test_given_optimizer_must_leave_natural_parameters_to_ngd() -> None:
    x, y = _data()
    model = SVGP(x[:5].clone(), variational="natural")
    with pytest.raises(ValueError, match="NGD steps"):
        fit(
            model,
            x,
            y,
            epochs=1,
            natgrad_lr=0.1,
            optimizer=torch.optim.Adam(model.parameters()),
        )

    natural = {id(p) for p in natural_variational_parameters(model)}
    rest = [p for p in model.parameters() if id(p) not in natural]
    losses = fit(
        model, x, y, epochs=3, natgrad_lr=0.1, optimizer=torch.optim.SGD(rest, lr=1e-3)
    )
    assert all(math.isfinite(v) for v in losses)


# --------------------------------------------------------------------------
# Training with natural gradients
# --------------------------------------------------------------------------
def test_ngd_trains_svgp_faster_than_adam() -> None:
    """On Snelson, 20 NGD steps on q(u) beat 20 Adam steps by a wide margin.

    (Measured: about 0.98 against 1.50 for the negative ELBO per point.)
    """
    data = load_snelson1d(test_fraction=0.2, seed=0)
    z = init_inducing(data.X_train, 20, seed=0)

    torch.manual_seed(0)
    adam = fit(SVGP(z.clone()), data.X_train, data.Y_train, epochs=20, lr=0.01)
    torch.manual_seed(0)
    nat = SVGP(z.clone(), variational="natural")
    ngd = fit(nat, data.X_train, data.Y_train, epochs=20, lr=0.01, natgrad_lr=0.5)

    assert all(math.isfinite(v) for v in ngd)
    assert ngd[-1] < adam[-1] - 0.3


def test_ngd_trains_multi_output_svgp() -> None:
    x, y = _data()
    y2 = torch.stack([y, -y], dim=-1)
    model = SVGP(x[:8].clone(), num_outputs=2, variational="natural")
    losses = fit(model, x, y2, epochs=30, lr=0.01, natgrad_lr=0.1)
    assert all(math.isfinite(v) for v in losses)
    assert losses[-1] < losses[0]


@pytest.mark.parametrize("whiten", [True, False])
def test_ngd_trains_a_builder_deep_gp(whiten) -> None:
    x, y = _data(d=2)
    cfg = DeepGPConfig(
        num_inducing=8,
        inner_layer_qsqrt_factor=1e-5,
        likelihood_noise=1e-2,
        whiten=whiten,
        output_variational="natural",
    )
    model = build_deep_gp(x, num_layers=2, config=cfg)
    losses = fit(model, x, y, epochs=30, lr=0.01, natgrad_lr=0.1, num_samples=4)
    assert all(math.isfinite(v) for v in losses)
    assert losses[-1] < losses[0]
