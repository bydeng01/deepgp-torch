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

"""Unit tests for the top-level package surface and the ELBO trainer.

Two groups:

* **Package** — every documented top-level export is present and non-``None``,
  importing ``deepgp`` flips the torch default dtype to ``float64``, and the
  core must remain a pure PyTorch library (no TensorFlow/GPflow/GPflux/tf_keras
  pulled into ``sys.modules`` as a side effect of ``import deepgp``).
* **Trainer** — :func:`deepgp.fit` returns one finite loss per epoch, trains the
  model in place, and honours a caller-provided optimizer; and
  :func:`deepgp.builders.factory.build_gp_layer` builds a single configured
  :class:`DeepGPHiddenLayer` (identity mean for a dim-preserving layer, and a
  ``shrink_q_sqrt`` that scales the ``q(u)`` Cholesky).
"""

import math
import sys

import torch

import deepgp  # noqa: F401  (import sets float64 default dtype)
from deepgp import SVGP, DeepGPConfig, fit, seed_everything
from deepgp.builders.factory import build_gp_layer
from deepgp.layers.gp_layer import DeepGPHiddenLayer

# The documented top-level API (mirrors deepgp.__all__).
DOCUMENTED_EXPORTS = [
    "build_deep_gp",
    "DeepGP",
    "DeepGPHiddenLayer",
    "SVGP",
    "fit",
    "predict",
    "DeepGPConfig",
    "set_default_float64",
    "seed_everything",
    "__version__",
]

FORBIDDEN_BACKENDS = ("tensorflow", "gpflow", "gpflux", "tf_keras")


# --------------------------------------------------------------------------- #
# Package surface
# --------------------------------------------------------------------------- #
def test_documented_exports_exist_and_not_none() -> None:
    for name in DOCUMENTED_EXPORTS:
        assert hasattr(deepgp, name), f"missing top-level export: {name}"
        assert getattr(deepgp, name) is not None, f"export is None: {name}"


def test_import_sets_float64_default() -> None:
    assert torch.get_default_dtype() == torch.float64


def test_core_does_not_import_tf_backends() -> None:
    leaked = [
        key
        for key in sys.modules
        for name in FORBIDDEN_BACKENDS
        if key == name or key.startswith(name + ".")
    ]
    assert not leaked, f"core unexpectedly imported heavy backends: {leaked}"


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
def _toy_svgp():
    """A tiny SVGP on ~20 synthetic 1-D points (num_inducing=8)."""
    x = torch.linspace(-3.0, 3.0, 20).unsqueeze(-1)
    y = torch.sin(x).squeeze(-1)
    model = SVGP(x[:8].clone(), input_dims=1)
    return model, x, y


def test_fit_returns_finite_losses_and_trains_in_place() -> None:
    seed_everything(0)
    model, x, y = _toy_svgp()

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    losses = fit(model, x, y, epochs=5, lr=0.05)

    assert isinstance(losses, list)
    assert len(losses) == 5
    assert all(isinstance(v, float) and math.isfinite(v) for v in losses)

    # Trained in place: at least one parameter moved.
    changed = any(
        not torch.equal(before[n], p.detach()) for n, p in model.named_parameters()
    )
    assert changed, "no parameter changed; fit did not train in place"


def test_fit_uses_caller_provided_optimizer() -> None:
    seed_everything(0)
    model, x, y = _toy_svgp()

    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    assert len(opt.state) == 0  # no steps taken yet

    losses = fit(model, x, y, epochs=4, optimizer=opt)

    assert len(losses) == 4
    # Adam populates per-parameter state only when *this* optimizer is stepped;
    # a non-empty state proves fit used the passed optimizer, not a fresh one.
    assert len(opt.state) > 0, "caller-provided optimizer was not stepped"


def _cfg(**kw) -> DeepGPConfig:
    base = dict(num_inducing=8, inner_layer_qsqrt_factor=1e-5, likelihood_noise=1e-2)
    base.update(kw)
    return DeepGPConfig(**base)


def _chol_diag(layer) -> torch.Tensor:
    chol = layer.variational_strategy._variational_distribution.chol_variational_covar
    return chol.diagonal(dim1=-2, dim2=-1)


def test_build_gp_layer_identity_mean_and_shrunk_qsqrt() -> None:
    torch.manual_seed(0)
    x = torch.randn(20, 2)
    cfg = _cfg(num_inducing=8)

    layer = build_gp_layer(2, 2, cfg, X=x, shrink_q_sqrt=True)
    assert isinstance(layer, DeepGPHiddenLayer)
    assert layer.input_dims == 2 and layer.output_dims == 2

    # Dim-preserving hidden layer -> identity (eye) linear mean.
    assert torch.allclose(layer.mean_module.weights.squeeze(-1), torch.eye(2))

    # shrink_q_sqrt=True scales the q(u) Cholesky by inner_layer_qsqrt_factor.
    assert abs(_chol_diag(layer).abs().mean().item() - 1e-5) < 1e-7

    # Without the shrink the q_sqrt starts near identity (unchanged).
    layer_noshrink = build_gp_layer(2, 2, cfg, X=x, shrink_q_sqrt=False)
    assert _chol_diag(layer_noshrink).abs().mean().item() > 0.5
