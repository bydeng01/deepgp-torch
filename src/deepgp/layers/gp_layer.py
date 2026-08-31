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

"""The variational Deep GP layer.

``DeepGPHiddenLayer`` is a single doubly-stochastic variational GP layer
(Salimbeni & Deisenroth, 2017).  It owns:

* a set of inducing inputs ``Z`` (learnable),
* a variational distribution ``q(u)`` over the inducing outputs,
* a mean function, and
* a covariance (kernel).

It subclasses :class:`gpytorch.models.deep_gps.DeepGPLayer`, which provides the
sampling machinery that propagates a distribution (rather than a point) through
the layer and automatically contributes this layer's KL term to the model ELBO.

Defaults:

* ``CholeskyVariationalDistribution`` for a full-covariance ``q(u)``.
* ``VariationalStrategy(..., learn_inducing_locations=True)`` (whitened by
  default) so inducing locations are optimised.
* ``ScaleKernel(RBFKernel(ard_num_dims=input_dims))`` — signal variance lives on
  the outer ``ScaleKernel`` (``RBFKernel`` has no outputscale of its own).
* ``LinearMean`` on hidden layers (prevents doubly-stochastic hidden-field
  collapse), ``ConstantMean`` on the output layer, ``ZeroMean`` available behind
  ``mean_type='zero'``.

``batch_shape`` threads ``output_dims`` through every sub-module so that each of
the ``output_dims`` latent GPs gets its own hyper-parameters.  ``output_dims is
None`` denotes a single-output layer (the output layer of a regression DGP).
"""

from __future__ import annotations

from typing import Optional

import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean, LinearMean, Mean, ZeroMean
from gpytorch.models.deep_gps import DeepGPLayer
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    UnwhitenedVariationalStrategy,
    VariationalStrategy,
)

from deepgp.utils.dtype import as_default_dtype

__all__ = ["DeepGPHiddenLayer"]


def _make_mean(mean_type: str, input_dims: int, batch_shape: torch.Size):
    """Construct the requested mean function.

    ``linear`` is the recommended default for hidden layers; ``constant`` for
    the output layer; ``zero`` is available but risks hidden-field collapse.
    """
    mean_type = mean_type.lower()
    if mean_type == "constant":
        return ConstantMean(batch_shape=batch_shape)
    if mean_type == "zero":
        return ZeroMean(batch_shape=batch_shape)
    if mean_type == "linear":
        return LinearMean(input_dims, batch_shape=batch_shape)
    raise ValueError(
        f"Unknown mean_type={mean_type!r}; expected one of "
        "'linear', 'constant', 'zero'."
    )


class DeepGPHiddenLayer(DeepGPLayer):
    """A single variational GP layer for a doubly-stochastic deep GP.

    Parameters
    ----------
    input_dims:
        Dimensionality of the layer input.
    output_dims:
        Number of latent GP outputs.  ``None`` means a single output (no batch
        dimension), used for the final regression layer.
    num_inducing:
        Number of inducing points ``M``.
    mean_type:
        ``'linear'`` (default, hidden layers), ``'constant'`` (output layer) or
        ``'zero'``.  Ignored if ``mean_function`` is provided.
    inducing_points:
        Optional pre-computed inducing inputs.  If ``None`` they are initialised
        from a standard normal with the correct shape
        (``(num_inducing, input_dims)`` when ``output_dims is None`` else
        ``(output_dims, num_inducing, input_dims)``).  Supplied points are cast
        to the current default dtype (``float64``), device preserved, so they
        cannot leave the model in mixed precision.
    learn_inducing_locations:
        Whether inducing locations are optimised (default ``True``).
    mean_function:
        Optional pre-built mean module (e.g. an identity/PCA
        :class:`~gpytorch.means.LinearMean` from :func:`deepgp.means.make_mean`);
        overrides ``mean_type``.
    covar_module:
        Optional pre-built kernel (e.g. from :func:`deepgp.kernels.make_kernel`).
    whiten:
        If ``True`` (default) use the whitened
        :class:`~gpytorch.variational.VariationalStrategy`; if ``False`` use the
        :class:`~gpytorch.variational.UnwhitenedVariationalStrategy`.
    q_sqrt_factor:
        Optional multiplier applied to the Cholesky factor of ``q(u)`` at
        initialisation (the builder shrinks hidden-layer ``q_sqrt`` by ~1e-5
        for a stable, near-deterministic start).  When set, the variational
        distribution is marked initialised so the shrink is not overwritten by
        the strategy's lazy first-forward reset to the prior.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: Optional[int],
        num_inducing: int = 128,
        mean_type: str = "linear",
        inducing_points: Optional[torch.Tensor] = None,
        learn_inducing_locations: bool = True,
        mean_function: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        whiten: bool = True,
        q_sqrt_factor: Optional[float] = None,
    ) -> None:
        batch_shape = (
            torch.Size([]) if output_dims is None else torch.Size([output_dims])
        )
        if inducing_points is None:
            if output_dims is None:
                inducing_points = torch.randn(num_inducing, input_dims)
            else:
                inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        else:
            # Normalise at the boundary: the strategy registers this tensor as a
            # learnable parameter, so an un-cast float32 input would leave the
            # layer in mixed precision.
            inducing_points = as_default_dtype(inducing_points)

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing, batch_shape=batch_shape
        )
        strategy_cls = VariationalStrategy if whiten else UnwhitenedVariationalStrategy
        variational_strategy = strategy_cls(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy, input_dims, output_dims)

        self.mean_module = (
            mean_function
            if mean_function is not None
            else _make_mean(mean_type, input_dims, batch_shape)
        )
        self.covar_module = (
            covar_module
            if covar_module is not None
            else ScaleKernel(
                RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
                batch_shape=batch_shape,
            )
        )

        if q_sqrt_factor is not None:
            self._shrink_q_sqrt(float(q_sqrt_factor))

    def _shrink_q_sqrt(self, factor: float) -> None:
        """Scale the Cholesky factor of ``q(u)`` and mark it initialised.

        The strategy lazily resets ``q(u)`` to the prior on the first forward
        pass; marking it initialised keeps the shrink in place.
        """
        with torch.no_grad():
            var_dist = self.variational_strategy._variational_distribution
            var_dist.chol_variational_covar.mul_(factor)
        self.variational_strategy.variational_params_initialized.fill_(1)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """Return the layer's GP prior ``N(mean(x), K(x, x))`` at ``x``.

        The :class:`DeepGPLayer` base class handles expanding ``x`` across the
        output dimension and drawing the doubly-stochastic samples; here we only
        describe the per-layer GP.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
