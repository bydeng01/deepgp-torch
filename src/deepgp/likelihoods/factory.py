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

"""Likelihood factory.

``make_likelihood`` builds the Gaussian observation model of every model here:

* ``num_outputs=None``: ``y_n = f(x_n) + e_n`` with ``e_n ~ N(0, s^2)``, a
  :class:`~gpytorch.likelihoods.GaussianLikelihood`.
* ``num_outputs=T``: ``y_nt = f_t(x_n) + e_nt`` with ``e_nt ~ N(0, s_t^2)``,
  independent over ``n`` and ``t``, a
  :class:`~gpytorch.likelihoods.MultitaskGaussianLikelihood` with ``rank=0``.

The multitask likelihood is built with ``has_global_noise=False``.  GPyTorch's
default also learns a shared ``s^2`` and gives output ``t`` the variance
``s_t^2 + s^2``: ``T + 1`` parameters for ``T`` variances, of which the data
identify only the sums.  Without it, ``likelihood.task_noises[t]`` is exactly
the noise variance of output ``t``, and ``T = 1`` is the single-output model.
"""

from __future__ import annotations

from typing import Optional

import torch
from gpytorch.likelihoods import (
    GaussianLikelihood,
    Likelihood,
    MultitaskGaussianLikelihood,
)

__all__ = ["make_likelihood"]


def make_likelihood(
    num_outputs: Optional[int] = None,
    noise: Optional[float] = None,
) -> Likelihood:
    """Construct the Gaussian likelihood for ``num_outputs`` outputs.

    Parameters
    ----------
    num_outputs:
        Number of outputs ``T``; ``None`` (default) for a single output.
    noise:
        Optional initial noise variance, applied to every output.  ``None``
        keeps GPyTorch's initial value, ``softplus(0) + 1e-4 ~= 0.693``.

    Returns
    -------
    gpytorch.likelihoods.Likelihood
        ``GaussianLikelihood()`` when ``num_outputs is None``, otherwise
        ``MultitaskGaussianLikelihood(num_outputs, rank=0,
        has_global_noise=False)``.
    """
    if num_outputs is None:
        likelihood = GaussianLikelihood()
        if noise is not None:
            likelihood.noise = torch.as_tensor(float(noise))
        return likelihood

    if num_outputs < 1:
        raise ValueError(f"num_outputs must be None or >= 1; got {num_outputs}.")
    multitask = MultitaskGaussianLikelihood(
        num_tasks=num_outputs, rank=0, has_global_noise=False
    )
    if noise is not None:
        multitask.task_noises = torch.as_tensor(float(noise))
    return multitask
