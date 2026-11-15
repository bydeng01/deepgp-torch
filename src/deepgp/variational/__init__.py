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

"""Variational distributions of ``q(u)``.

Every GP here has a full-covariance Gaussian ``q(u) = N(m, S)`` over its
inducing outputs, in one of two parameterisations:

* ``'cholesky'`` — :class:`~gpytorch.variational.CholeskyVariationalDistribution`
  (``m`` and the Cholesky factor of ``S``), trained by Adam with the other
  parameters;
* ``'natural'`` — :class:`~gpytorch.variational.TrilNaturalVariationalDistribution`
  (the natural parameters ``S^{-1} m`` and a triangular factor of ``S^{-1}``),
  trained by natural gradients (:class:`gpytorch.optim.NGD`, through
  ``fit(..., natgrad_lr=...)``). GPyTorch's triangular form is the more
  numerically stable of its two natural parameterisations.

The whitened/unwhitened choice and the hidden ``q_sqrt`` shrink are set on
:class:`deepgp.layers.gp_layer.DeepGPHiddenLayer` (``whiten`` /
``q_sqrt_factor``).
"""

from __future__ import annotations

import torch
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    TrilNaturalVariationalDistribution,
    _VariationalDistribution,
)

__all__ = ["make_variational_distribution", "scale_variational_covariance"]

_KINDS = {
    "cholesky": CholeskyVariationalDistribution,
    "natural": TrilNaturalVariationalDistribution,
}


def make_variational_distribution(
    kind: str,
    num_inducing: int,
    batch_shape: torch.Size = torch.Size([]),
) -> _VariationalDistribution:
    """Construct ``q(u)`` over ``num_inducing`` inducing outputs.

    Parameters
    ----------
    kind:
        ``'cholesky'`` or ``'natural'`` (see the module docstring).
    num_inducing:
        Number of inducing points ``M``.
    batch_shape:
        One independent ``q(u)`` per batch entry (one per output GP).
    """
    try:
        cls = _KINDS[kind.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown variational={kind!r}; expected 'cholesky' or 'natural'."
        ) from None
    return cls(num_inducing, batch_shape=batch_shape)


def scale_variational_covariance(
    var_dist: _VariationalDistribution, factor: float
) -> None:
    """Scale the covariance ``S`` of ``q(u)`` by ``factor**2`` in place.

    The mean ``m`` is unchanged.  This is the hidden-layer ``q_sqrt`` shrink: with
    the Cholesky parameterisation the factor of ``S`` is multiplied by
    ``factor``; with the natural one, ``S = L L^T`` where ``L`` is the inverse
    of ``natural_tril_mat``, so that is divided by ``factor`` and ``natural_vec
    = S^{-1} m`` by ``factor**2``.
    """
    with torch.no_grad():
        if isinstance(var_dist, CholeskyVariationalDistribution):
            var_dist.chol_variational_covar.mul_(factor)
        elif isinstance(var_dist, TrilNaturalVariationalDistribution):
            var_dist.natural_tril_mat.div_(factor)
            var_dist.natural_vec.div_(factor**2)
        else:
            raise TypeError(
                f"Cannot scale the covariance of a {type(var_dist).__name__}."
            )
