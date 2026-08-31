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

A :class:`~gpytorch.likelihoods.GaussianLikelihood` is attached as
``self.likelihood`` so that the same :func:`deepgp.fit` / :func:`deepgp.predict`
helpers work for both ``SVGP`` and :class:`deepgp.models.deep_gp.DeepGP`.
"""

from __future__ import annotations

from typing import Optional

import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)

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
    input_dims:
        Input dimensionality; used for ARD.  Inferred from ``inducing_points``
        when ``None``.
    learn_inducing_locations:
        Whether to optimise the inducing locations (default ``True``).
    """

    def __init__(
        self,
        inducing_points: torch.Tensor,
        input_dims: Optional[int] = None,
        learn_inducing_locations: bool = True,
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

        variational_distribution = CholeskyVariationalDistribution(num_inducing)
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)

        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=input_dims))
        self.likelihood = GaussianLikelihood()

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
