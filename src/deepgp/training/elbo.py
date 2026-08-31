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

"""Marginal-log-likelihood (ELBO) construction.

The variational ELBO has the form::

    ELBO = E_q[ log p(y | f) ]  -  (beta / num_data) * KL[ q(u) || p(u) ]

so the effective KL weight is exactly ``beta / num_data``.  ``beta`` is
first-class KL tempering (GPflow's ``kl_temperature`` maps to
``beta = 1 / kl_temperature``).

For a :class:`gpytorch.models.deep_gps.DeepGP` the per-layer KLs are summed
automatically and the whole thing is wrapped in a
:class:`~gpytorch.mlls.DeepApproximateMLL`, which averages the objective over the
doubly-stochastic samples.  For a shallow model (e.g. ``SVGP``) a plain
:class:`~gpytorch.mlls.VariationalELBO` is returned.
"""

from __future__ import annotations

import gpytorch
from gpytorch.mlls import DeepApproximateMLL, MarginalLogLikelihood, VariationalELBO

__all__ = ["make_mll", "is_deep_gp"]


def is_deep_gp(model) -> bool:
    """Return ``True`` if ``model`` is a (multi-layer) deep GP."""
    return isinstance(model, gpytorch.models.deep_gps.DeepGP)


def make_mll(model, num_data: int, beta: float = 1.0) -> MarginalLogLikelihood:
    """Build the ELBO objective for ``model``.

    Parameters
    ----------
    model:
        A model exposing ``model.likelihood`` (both ``SVGP`` and ``DeepGP`` do).
    num_data:
        Total number of training points ``N`` (sets the KL weight
        ``beta / num_data``).
    beta:
        KL tempering coefficient (default ``1.0``).

    Returns
    -------
    gpytorch.mlls.MarginalLogLikelihood
        A ``DeepApproximateMLL(VariationalELBO(...))`` for deep GPs, or a plain
        ``VariationalELBO(...)`` for shallow models.  In both cases the returned
        object exposes ``.num_data`` and ``.beta``, so the KL weight is
        ``mll.beta / mll.num_data``.
    """
    base = VariationalELBO(model.likelihood, model, num_data=num_data, beta=beta)
    if is_deep_gp(model):
        return DeepApproximateMLL(base)
    return base
