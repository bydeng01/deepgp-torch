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

"""Predictive mean/variance from the doubly-stochastic mixture.

A deep GP's predictive distribution at a test point is an ``S``-component
Gaussian *mixture* (one component per doubly-stochastic sample).  The correct
first two moments follow the **law of total variance**::

    mean = E_s[ m_s ]                       = out.mean.mean(0)
    var  = E_s[ v_s ] + Var_s[ m_s ]        = out.variance.mean(0)
                                              + out.mean.var(0, unbiased=False)

``unbiased=False`` is required so that ``k == 1`` gives ``Var_s[m_s] = 0``
instead of ``0/0 -> NaN``.  Using a single component's ``.variance`` alone
under-reports uncertainty.

For a shallow model (``SVGP``) there is no sample dimension, so the model's own
predictive mean/variance are returned directly.
"""

from __future__ import annotations

from typing import Tuple

import gpytorch
import torch

from deepgp.training.elbo import is_deep_gp
from deepgp.utils.dtype import as_default_dtype

__all__ = ["predict"]


def predict(
    model,
    X: torch.Tensor,
    k: int = 50,
    add_noise: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the predictive ``(mean, variance)`` at inputs ``X``.

    Parameters
    ----------
    model:
        A trained ``SVGP`` or ``DeepGP``.
    X:
        Test inputs, shape ``(N, input_dims)``.
    k:
        Number of doubly-stochastic samples for the mixture (deep GP only).
    add_noise:
        If ``True`` (default) return the predictive distribution of ``y``
        (observation noise included via ``model.likelihood``); if ``False``
        return the latent-function distribution of ``f``.

    Returns
    -------
    (mean, variance):
        Tensors of shape ``(N,)`` for single-output models (or ``(N, T)`` for
        multi-output).  ``variance`` is strictly positive.

    Notes
    -----
    The model's train/eval mode is restored before returning.  GPyTorch
    memoises the expensive per-model quantities (the ``K_ZZ`` Cholesky factor,
    ``q(u)``) on eval-mode forward passes and drops them only in
    ``gpytorch.Module.train``, so leaking eval mode would pin those caches and a
    later parameter change would be only partially reflected — ``K_xZ``
    recomputed against a stale ``K_ZZ``.
    """
    X = as_default_dtype(X)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(k):
            out = model.likelihood(model(X)) if add_noise else model(X)
            if is_deep_gp(model):
                # S-component Gaussian mixture -> law of total variance.
                mean = out.mean.mean(0)
                var = out.variance.mean(0) + out.mean.var(0, unbiased=False)
            else:
                mean = out.mean
                var = out.variance
    finally:
        model.train(was_training)
    return mean, var
