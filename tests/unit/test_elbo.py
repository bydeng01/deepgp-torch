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

"""Unit tests for ELBO construction and the KL weight."""

import torch
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO

import deepgp  # noqa: F401  (import sets float64 default dtype)
from deepgp import SVGP
from deepgp.models.deep_gp import DeepGP
from deepgp.training.elbo import make_mll


def _svgp() -> SVGP:
    torch.manual_seed(0)
    return SVGP(torch.randn(8, 1), input_dims=1)


def test_make_mll_shallow_returns_plain_variational_elbo() -> None:
    # Use non-default beta / num_data so the assertions would fail if the shallow
    # branch dropped or hardcoded either argument.
    mll = make_mll(_svgp(), num_data=137, beta=0.5)
    assert isinstance(mll, VariationalELBO)
    assert not isinstance(mll, DeepApproximateMLL)
    assert mll.num_data == 137
    assert mll.beta == 0.5


def test_make_mll_deep_wraps_in_deep_approximate_mll() -> None:
    model = DeepGP([1, 1], num_inducing=8)
    mll = make_mll(model, num_data=100, beta=1.0)
    assert isinstance(mll, DeepApproximateMLL)


def test_kl_weight_attributes_equal_beta_over_num_data() -> None:
    n_data, beta = 137, 0.5
    mll = make_mll(DeepGP([1, 1], num_inducing=8), num_data=n_data, beta=beta)
    assert mll.num_data == n_data
    assert mll.beta == beta
    # The ELBO's KL weight is exactly beta / num_data.
    assert abs(mll.beta / mll.num_data - beta / n_data) < 1e-12


def test_kl_term_scales_as_beta_over_num_data_numerically() -> None:
    """ELBO = E[log p(y|f)] - (beta/N) * KL, so d(ELBO)/d(beta) = -KL/N.

    For an SVGP (single variational strategy, Gaussian likelihood, no priors)
    everything is closed-form, so comparing the ELBO at two betas isolates the
    KL weight exactly.
    """
    n_data = 50
    model = _svgp()
    model.train()
    x = torch.randn(20, 1)
    y = torch.randn(20)

    # A freshly-initialised q(u) equals the (whitened) prior, so KL == 0 and the
    # scaling would be untestable. Trigger the strategy's lazy variational-param
    # initialisation with a first forward pass, THEN move the variational mean off
    # the prior (a perturbation done before the first forward would be overwritten
    # by that lazy init).
    _ = model(x)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "variational_mean" in name:
                param.add_(torch.randn_like(param))

    raw_kl = model.variational_strategy.kl_divergence()
    assert raw_kl > 1e-3, "test needs a non-trivial KL to be meaningful"

    out = model(x)
    elbo_beta1 = VariationalELBO(model.likelihood, model, num_data=n_data, beta=1.0)(
        out, y
    )
    elbo_beta2 = VariationalELBO(model.likelihood, model, num_data=n_data, beta=2.0)(
        out, y
    )

    # (elbo@1 - elbo@2) = raw_kl * (2 - 1) / N = raw_kl / N
    observed = elbo_beta1 - elbo_beta2
    expected = raw_kl / n_data
    assert torch.allclose(observed, expected, rtol=1e-5, atol=1e-7)
