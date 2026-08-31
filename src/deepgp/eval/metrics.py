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

"""Regression evaluation metrics.

* :func:`rmse` — root-mean-square error of a point prediction.
* :func:`gaussian_nll` — mean negative log-likelihood of the targets under a
  diagonal-Gaussian predictive distribution ``N(mean, variance)``.
"""

from __future__ import annotations

import math

import torch

__all__ = ["rmse", "gaussian_nll"]


def rmse(pred_mean: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root-mean-square error between ``pred_mean`` and ``target``.

    Returns a scalar tensor.
    """
    pred_mean = pred_mean.reshape_as(target)
    return torch.sqrt(torch.mean((pred_mean - target) ** 2))


def gaussian_nll(
    pred_mean: torch.Tensor,
    pred_var: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Mean Gaussian negative log-likelihood ``-log N(target | mean, var)``.

    Parameters
    ----------
    pred_mean, pred_var:
        Predictive mean and (strictly positive) variance.
    target:
        Observed targets.
    eps:
        Floor applied to the variance for numerical safety.

    Returns
    -------
    torch.Tensor
        Scalar mean NLL over all points.
    """
    pred_mean = pred_mean.reshape_as(target)
    pred_var = pred_var.reshape_as(target).clamp_min(eps)
    nll = 0.5 * (
        math.log(2.0 * math.pi)
        + torch.log(pred_var)
        + (target - pred_mean) ** 2 / pred_var
    )
    return torch.mean(nll)
