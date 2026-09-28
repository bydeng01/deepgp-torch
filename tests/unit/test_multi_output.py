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

"""Unit tests for multi-output models.

A multi-output model is ``T`` independent GPs at the output (for ``DeepGP``, on
top of hidden layers shared by all outputs), so the core tests compare it with
``T`` single-output models that carry the same parameters and check exact
identities rather than tolerances on a fit:

* ELBO.  For ``SVGP`` the multi-output ELBO is the sum of the ``T``
  single-output ELBOs.  For ``DeepGP`` that sum pays the shared hidden layers'
  KL ``T`` times while the multi-output model pays it once, so
  ``ELBO = sum_t ELBO_t + (T - 1) * (beta / N) * KL_hidden``.
* ``predict``.  Column ``t`` of the multi-output prediction is the prediction
  of single-output model ``t``, and observation noise adds exactly
  ``task_noises[t]`` to column ``t``.
* ``T = 1`` is the single-output model, up to the trailing output axis.

Reseeding torch before each forward pass gives every deep GP the same
hidden-layer samples: those samples are the only random draws in a forward
pass, and their shape does not depend on the number of outputs.
"""

import gpytorch
import pytest
import torch
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.variational import (
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)

import deepgp  # noqa: F401  (import sets float64 default dtype)
from deepgp import (
    SVGP,
    DeepGP,
    DeepGPConfig,
    DeepGPHiddenLayer,
    build_deep_gp,
    fit,
    predict,
)
from deepgp.likelihoods import make_likelihood
from deepgp.training.elbo import make_mll

N, D, M, BETA = 12, 2, 5, 0.5

# kind -> (multi-output factory taking T, single-output factory)
MODELS = {
    "svgp": (
        lambda T: SVGP(torch.randn(M, D), num_outputs=T),
        lambda: SVGP(torch.zeros(M, D)),
    ),
    "deep_gp": (
        lambda T: DeepGP([D, 2], num_inducing=M, num_outputs=T),
        lambda: DeepGP([D, 2], num_inducing=M),
    ),
}


def _randomise(model: torch.nn.Module) -> None:
    """Perturb every parameter so outputs differ and ``q(u)`` is off the prior.

    Must run before the first forward pass: GPyTorch memoises ``q(u)`` there,
    and an in-place change afterwards leaves ``kl_divergence()`` reading the old
    covariance until the next training-mode forward.  Marking the strategies
    initialised stops the lazy first-forward reset of ``q(u)`` to the prior.
    """
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn_like(p))
    for module in model.modules():
        if isinstance(module, VariationalStrategy):
            module.variational_params_initialized.fill_(1)


def _output_state(multi, single, t: int) -> dict:
    """``multi``'s state for output ``t``, keyed like ``single.state_dict()``."""
    ref = single.state_dict()
    num_outputs = multi.num_outputs
    state = {}
    for key, value in multi.state_dict().items():
        if key.startswith("likelihood."):
            continue
        key = key.replace("base_variational_strategy.", "")  # SVGP wrapper
        batched = value.shape == (num_outputs, *ref[key].shape)
        state[key] = value[t] if batched else value
    # Both noises sit behind the same GreaterThan(1e-4) softplus, so copying
    # the raw value copies the variance exactly.
    raw_noise = multi.likelihood.raw_task_noises[t : t + 1]
    state["likelihood.noise_covar.raw_noise"] = raw_noise
    for key in ref:
        if key.startswith("likelihood.noise_covar.raw_noise_constraint."):
            state[key] = ref[key]
    return state


def _per_output_models(multi, make_single) -> list:
    singles = []
    for t in range(multi.num_outputs):
        single = make_single()
        single.load_state_dict(_output_state(multi, single, t))
        singles.append(single)
    return singles


def _elbo(model, X, y) -> torch.Tensor:
    torch.manual_seed(1)  # same hidden-layer samples for every model
    with gpytorch.settings.num_likelihood_samples(4):
        return make_mll(model, num_data=N, beta=BETA)(model(X), y)


# --------------------------------------------------------------------------- #
# Likelihood
# --------------------------------------------------------------------------- #
def test_make_likelihood_single_output_is_gaussian() -> None:
    likelihood = make_likelihood(noise=0.03)
    assert isinstance(likelihood, GaussianLikelihood)
    assert torch.allclose(likelihood.noise, torch.tensor([0.03]))


def test_make_likelihood_has_exactly_one_noise_variance_per_output() -> None:
    likelihood = make_likelihood(3, noise=0.03)
    assert isinstance(likelihood, MultitaskGaussianLikelihood)
    assert likelihood.rank == 0
    assert not likelihood.has_global_noise
    # T free parameters and no shared term added on top of them.
    shapes = {name: tuple(p.shape) for name, p in likelihood.named_parameters()}
    assert shapes == {"raw_task_noises": (3,)}
    assert torch.allclose(likelihood.task_noises, torch.full((3,), 0.03))
    # The default starting noise is the single-output default.
    assert torch.allclose(make_likelihood(3).task_noises, make_likelihood().noise)


def test_make_likelihood_rejects_zero_outputs() -> None:
    with pytest.raises(ValueError):
        make_likelihood(0)


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #
def test_svgp_multi_output_has_per_output_parameters() -> None:
    torch.manual_seed(0)
    z = torch.randn(M, D)
    model = SVGP(z, num_outputs=3)

    assert model.num_outputs == 3
    assert isinstance(
        model.variational_strategy, IndependentMultitaskVariationalStrategy
    )
    inducing = model.variational_strategy.base_variational_strategy.inducing_points
    # Every output starts from its own copy of Z.
    assert inducing.shape == (3, M, D)
    assert torch.equal(inducing.detach(), z.expand(3, M, D))
    assert model.mean_module.constant.shape == (3,)
    assert model.covar_module.outputscale.shape == (3,)
    assert model.covar_module.base_kernel.lengthscale.shape == (3, 1, D)
    assert {p.dtype for p in model.parameters()} == {torch.float64}

    assert model(torch.randn(N, D)).event_shape == (N, 3)


def test_deep_gp_multi_output_shapes() -> None:
    torch.manual_seed(0)
    model = DeepGP([D, 2], num_inducing=M, num_outputs=3)

    assert model.num_outputs == 3
    assert model.last_layer.output_dims == 3
    assert model.last_layer.variational_strategy.inducing_points.shape == (3, M, 2)
    assert model.likelihood.task_noises.shape == (3,)

    with gpytorch.settings.num_likelihood_samples(4):
        out = model(torch.randn(N, D))
    # One mixture component per sample, one column per output.
    assert out.batch_shape == (4,)
    assert out.event_shape == (N, 3)


# --------------------------------------------------------------------------- #
# Exact identities against per-output models
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("num_outputs", [1, 3])
def test_svgp_elbo_is_the_sum_of_per_output_elbos(num_outputs) -> None:
    torch.manual_seed(0)
    X, Y = torch.randn(N, D), torch.randn(N, num_outputs)
    multi = SVGP(torch.randn(M, D), num_outputs=num_outputs)
    _randomise(multi)
    singles = _per_output_models(multi, lambda: SVGP(torch.zeros(M, D)))

    per_output = torch.stack([_elbo(s, X, Y[:, t]) for t, s in enumerate(singles)])
    assert torch.allclose(_elbo(multi, X, Y), per_output.sum(), rtol=0, atol=1e-12)


@pytest.mark.parametrize("dims", [[D], [D, 2], [D, 3, 2]])
@pytest.mark.parametrize("num_outputs", [1, 3])
def test_deep_gp_elbo_pays_the_shared_hidden_kl_once(dims, num_outputs) -> None:
    torch.manual_seed(0)
    X, Y = torch.randn(N, D), torch.randn(N, num_outputs)
    multi = DeepGP(dims, num_inducing=M, num_outputs=num_outputs)
    _randomise(multi)
    singles = _per_output_models(multi, lambda: DeepGP(dims, num_inducing=M))

    kl_hidden = sum(
        layer.variational_strategy.kl_divergence().sum()
        for layer in multi.hidden_layers
    )
    if multi.hidden_layers:
        assert kl_hidden > 1e-2, "test needs a non-trivial hidden KL"

    per_output = torch.stack([_elbo(s, X, Y[:, t]) for t, s in enumerate(singles)])
    expected = per_output.sum() + (num_outputs - 1) * BETA / N * kl_hidden
    assert torch.allclose(_elbo(multi, X, Y), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("add_noise", [False, True])
@pytest.mark.parametrize("num_outputs", [1, 3])
@pytest.mark.parametrize("kind", ["svgp", "deep_gp"])
def test_predict_columns_are_the_per_output_predictions(
    kind, num_outputs, add_noise
) -> None:
    make_multi, make_single = MODELS[kind]
    torch.manual_seed(0)
    X = torch.randn(N, D)
    multi = make_multi(num_outputs)
    _randomise(multi)

    torch.manual_seed(1)
    mean, var = predict(multi, X, k=7, add_noise=add_noise)
    assert mean.shape == var.shape == (N, num_outputs)

    for t, single in enumerate(_per_output_models(multi, make_single)):
        torch.manual_seed(1)
        mean_t, var_t = predict(single, X, k=7, add_noise=add_noise)
        assert torch.allclose(mean[:, t], mean_t, rtol=0, atol=1e-12)
        assert torch.allclose(var[:, t], var_t, rtol=0, atol=1e-12)


@pytest.mark.parametrize("kind", ["svgp", "deep_gp"])
def test_observation_noise_adds_the_noise_of_each_output(kind) -> None:
    torch.manual_seed(0)
    X = torch.randn(N, D)
    model = MODELS[kind][0](3)
    _randomise(model)

    torch.manual_seed(1)
    mean_f, var_f = predict(model, X, k=7, add_noise=False)
    torch.manual_seed(1)
    mean_y, var_y = predict(model, X, k=7, add_noise=True)

    assert torch.equal(mean_f, mean_y)
    noise = model.likelihood.task_noises.detach()
    assert len(set(noise.tolist())) == 3, "outputs need distinct noises"
    assert torch.allclose(var_y - var_f, noise.expand(N, 3), rtol=0, atol=1e-12)


# --------------------------------------------------------------------------- #
# Training targets
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "make_model, y_shape",
    [
        # (N,) targets against (N, T) outputs broadcast silently when N == T.
        (lambda: SVGP(torch.randn(3, 1), num_outputs=2), (2,)),
        (lambda: DeepGP([1, 1], num_inducing=3, num_outputs=2), (2,)),
        (lambda: SVGP(torch.randn(3, 1), num_outputs=3), (4, 2)),
        (lambda: SVGP(torch.randn(3, 1)), (4, 1)),
    ],
    ids=["svgp-n-eq-t", "deep-gp-n-eq-t", "wrong-num-outputs", "trailing-axis"],
)
def test_fit_rejects_targets_not_shaped_like_the_output(make_model, y_shape) -> None:
    torch.manual_seed(0)
    model = make_model()
    X = torch.randn(y_shape[0], 1)
    with pytest.raises(ValueError, match="event shape"):
        fit(model, X, torch.randn(*y_shape), epochs=1)


# --------------------------------------------------------------------------- #
# Assembly from layers and the builder
# --------------------------------------------------------------------------- #
def test_from_layers_takes_num_outputs_from_the_last_layer() -> None:
    torch.manual_seed(0)
    hidden = DeepGPHiddenLayer(input_dims=1, output_dims=1, num_inducing=4)
    last = DeepGPHiddenLayer(
        input_dims=1, output_dims=3, num_inducing=4, mean_type="constant"
    )

    model = DeepGP.from_layers([hidden], last)
    assert model.num_outputs == 3
    assert isinstance(model.likelihood, MultitaskGaussianLikelihood)
    assert model.likelihood.num_tasks == 3

    # A matching explicit value is accepted; a conflicting one is not.
    assert DeepGP.from_layers([hidden], last, num_outputs=3).num_outputs == 3
    with pytest.raises(ValueError):
        DeepGP.from_layers([hidden], last, num_outputs=2)


def test_build_deep_gp_multi_output_initialisation() -> None:
    torch.manual_seed(0)
    x = torch.randn(40, 2)
    cfg = DeepGPConfig(
        num_inducing=10, inner_layer_qsqrt_factor=1e-5, likelihood_noise=0.03
    )
    model = build_deep_gp(x, num_layers=2, config=cfg, num_outputs=3)
    hidden, last = model.hidden_layers[0], model.last_layer

    assert model.num_outputs == 3
    assert last.output_dims == 3
    # Each output GP starts from the KMeans centres the hidden layer uses.
    z_hidden = hidden.variational_strategy.inducing_points.detach()  # (D, M, D)
    z_last = last.variational_strategy.inducing_points.detach()
    assert z_last.shape == (3, 10, 2)
    assert torch.equal(z_last, z_hidden[:1].expand(3, -1, -1))
    # Per-output mean and kernel; the output q_sqrt is left at the identity.
    assert last.mean_module.constant.shape == (3,)
    assert last.covar_module.base_kernel.lengthscale.shape == (3, 1, 2)
    chol = last.variational_strategy._variational_distribution.chol_variational_covar
    assert torch.equal(chol.diagonal(dim1=-2, dim2=-1), torch.ones(3, 10))
    # Every output's noise starts at likelihood_noise.
    assert torch.allclose(model.likelihood.task_noises, torch.full((3,), 0.03))
