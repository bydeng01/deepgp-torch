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

"""Unit tests for ``ExactGP``: forward shapes, the exact MLL branch, ``fit``
and ``predict`` against the closed-form posterior."""

import torch
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood

import deepgp  # noqa: F401  (import sets float64 default dtype)
from deepgp import SVGP, DeepGP, ExactGP, fit, predict
from deepgp.training.elbo import is_exact_gp, make_mll


def _data(n: int = 20, t: int = 0):
    torch.manual_seed(0)
    X = torch.rand(n, 1)
    Y = torch.sin(6 * X).squeeze(-1) + 0.1 * torch.randn(n)
    if t:
        Y = torch.stack([Y * (i + 1) for i in range(t)], -1)
    return X, Y


def test_forward_single_and_multi_output() -> None:
    X, Y = _data()
    out = ExactGP(X, Y).forward(X)
    assert type(out) is MultivariateNormal
    assert out.event_shape == torch.Size([20])

    X, Y2 = _data(t=3)
    out = ExactGP(X, Y2, num_outputs=3).forward(X)
    assert isinstance(out, MultitaskMultivariateNormal)
    assert out.event_shape == torch.Size([20, 3])


def test_train_data_cast_to_default_dtype() -> None:
    X, Y = _data()
    model = ExactGP(X.float(), Y.float())
    assert model.train_inputs[0].dtype == torch.float64
    assert model.train_targets.dtype == torch.float64


def test_make_mll_returns_exact_mll() -> None:
    X, Y = _data()
    model = ExactGP(X, Y)
    assert is_exact_gp(model)
    assert not is_exact_gp(SVGP(torch.randn(8, 1)))
    assert not is_exact_gp(DeepGP([1, 1], num_inducing=8))
    assert isinstance(make_mll(model, num_data=20), ExactMarginalLogLikelihood)


def test_fit_decreases_loss_and_replaces_train_data() -> None:
    X, Y = _data()
    model = ExactGP(X[:5], Y[:5])
    losses = fit(model, X, Y, epochs=50, lr=0.1)
    assert losses[-1] < losses[0]
    assert torch.equal(model.train_inputs[0], X)
    assert torch.equal(model.train_targets, Y)


def test_predict_matches_closed_form_posterior() -> None:
    X, Y = _data()
    model = ExactGP(X, Y)
    fit(model, X, Y, epochs=20, lr=0.1)
    Xs = torch.linspace(-0.2, 1.2, 9).unsqueeze(-1)
    mean, var = predict(model, Xs, add_noise=False)

    with torch.no_grad():
        k = model.covar_module
        A = k(X).to_dense() + model.likelihood.noise * torch.eye(20)
        Ks = k(Xs, X).to_dense()
        m = model.mean_module.constant
        ref_mean = m + Ks @ torch.linalg.solve(A, Y - m)
        ref_var = k(Xs, diag=True) - (Ks * torch.linalg.solve(A, Ks.T).T).sum(-1)

    torch.testing.assert_close(mean, ref_mean)
    torch.testing.assert_close(var, ref_var)
    assert model.training


def test_predict_add_noise_adds_noise_variance() -> None:
    X, Y = _data()
    model = ExactGP(X, Y)
    Xs = torch.rand(6, 1)
    _, var_f = predict(model, Xs, add_noise=False)
    _, var_y = predict(model, Xs)
    noise = model.likelihood.noise.detach().expand_as(var_f)
    torch.testing.assert_close(var_y - var_f, noise)


def test_multi_output_fit_and_predict() -> None:
    X, Y = _data(t=2)
    model = ExactGP(X, Y, num_outputs=2)
    losses = fit(model, X, Y, epochs=50, lr=0.1)
    mean, var = predict(model, X[:7])
    assert losses[-1] < losses[0]
    assert mean.shape == var.shape == torch.Size([7, 2])
    assert (var > 0).all()
    assert model.likelihood.task_noises.shape == torch.Size([2])
