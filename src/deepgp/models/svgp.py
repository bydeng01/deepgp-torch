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

"""Single-layer sparse variational GP (SVGP).

This is both the *base case* of the library (a shallow, one-layer GP) and the
building block the deep GP generalises.  It is a stochastic variational GP
(Hensman et al., 2013/2015) implemented as a
:class:`gpytorch.models.ApproximateGP` with a
:class:`~gpytorch.variational.VariationalStrategy` over a
:class:`~gpytorch.variational.CholeskyVariationalDistribution`.

A Gaussian likelihood is attached as ``self.likelihood`` so that the same
:func:`deepgp.fit` / :func:`deepgp.predict` helpers work for both ``SVGP`` and
:class:`deepgp.models.deep_gp.DeepGP`.

With ``num_outputs=T`` the model is ``T`` independent GPs.  The inducing
points, ``q(u)``, mean and kernel are batched over ``batch_shape=[T]``, an
:class:`~gpytorch.variational.IndependentMultitaskVariationalStrategy` turns
the batch into a multitask output, and the likelihood has one noise variance
per output.  No parameter is shared between outputs, so the ELBO is the sum of
the ``T`` single-output ELBOs.
"""

from __future__ import annotations

from typing import Optional

import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)

from deepgp.likelihoods.factory import make_likelihood
from deepgp.utils.dtype import as_default_dtype

__all__ = ["SVGP"]


class SVGP(ApproximateGP):
    """A single-layer sparse variational GP for regression.

    Parameters
    ----------
    inducing_points:
        Initial inducing inputs of shape ``(num_inducing, input_dims)``.  Cast
        to the current default dtype (``float64``), device preserved, so a
        lower-precision tensor cannot leave the model in mixed precision.
        With ``num_outputs=T`` each output GP starts from its own copy.
    input_dims:
        Input dimensionality; used for ARD.  Inferred from ``inducing_points``
        when ``None``.
    learn_inducing_locations:
        Whether to optimise the inducing locations (default ``True``).
    num_outputs:
        Number of regression outputs ``T``.  ``None`` (default) means a single
        output with a :class:`~gpytorch.likelihoods.GaussianLikelihood`; an
        integer gives ``T`` independent GPs with one noise variance each (see
        :func:`deepgp.likelihoods.make_likelihood`).
    """

    def __init__(
        self,
        inducing_points: torch.Tensor,
        input_dims: Optional[int] = None,
        learn_inducing_locations: bool = True,
        num_outputs: Optional[int] = None,
    ) -> None:
        if inducing_points.dim() != 2:
            raise ValueError(
                "SVGP expects inducing_points of shape (num_inducing, "
                f"input_dims); got shape {tuple(inducing_points.shape)}."
            )
        # Normalise at the boundary: the strategy registers this tensor as a
        # learnable parameter, so an un-cast float32 input would leave the model
        # in mixed precision (float32 Z alongside float64 kernel/mean/noise).
        inducing_points = as_default_dtype(inducing_points)
        num_inducing = inducing_points.size(-2)
        if input_dims is None:
            input_dims = inducing_points.size(-1)

        batch_shape = (
            torch.Size([]) if num_outputs is None else torch.Size([num_outputs])
        )
        if num_outputs is not None:
            inducing_points = inducing_points.unsqueeze(0).repeat(num_outputs, 1, 1)

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing, batch_shape=batch_shape
        )
        base_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        variational_strategy = (
            base_strategy
            if num_outputs is None
            else IndependentMultitaskVariationalStrategy(
                base_strategy, num_tasks=num_outputs
            )
        )
        super().__init__(variational_strategy)

        self.num_outputs = num_outputs
        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape,
        )
        self.likelihood = make_likelihood(num_outputs)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
