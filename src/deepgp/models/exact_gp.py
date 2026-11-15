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

"""Exact GP regression.

Same mean, kernel and likelihood as :class:`~deepgp.models.svgp.SVGP`, but the
posterior is computed exactly, at ``O(N^3)`` cost.  With ``num_outputs=T`` the
model is ``T`` independent GPs batched over ``batch_shape=[T]``.
"""

from __future__ import annotations

from typing import Optional

import torch
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP as _ExactGP

from deepgp.kernels.factory import make_kernel
from deepgp.likelihoods.factory import make_likelihood
from deepgp.utils.dtype import as_default_dtype

__all__ = ["ExactGP"]


class ExactGP(_ExactGP):
    """An exact GP for regression.

    Parameters
    ----------
    X:
        Training inputs, shape ``(N, input_dims)``.
    Y:
        Training targets, shape ``(N,)`` or ``(N, T)``.
    num_outputs:
        ``None`` for a single output, ``T`` for independent outputs.
    """

    def __init__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        num_outputs: Optional[int] = None,
    ) -> None:
        super().__init__(
            as_default_dtype(X), as_default_dtype(Y), make_likelihood(num_outputs)
        )
        batch = torch.Size([]) if num_outputs is None else torch.Size([num_outputs])
        self.num_outputs = num_outputs
        self.mean_module = ConstantMean(batch_shape=batch)
        self.covar_module = make_kernel(X.size(-1), num_outputs)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        f = MultivariateNormal(self.mean_module(x), self.covar_module(x))
        if self.num_outputs is None:
            return f
        return MultitaskMultivariateNormal.from_batch_mvn(f)
