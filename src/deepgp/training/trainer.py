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

"""Full-batch ELBO training loop.

Implements :func:`fit`, a hand-written optimisation loop (GPyTorch has no
``model.fit``).  Both the training and evaluation forward passes are wrapped in
``gpytorch.settings.num_likelihood_samples(S)`` so the doubly-stochastic Monte
Carlo estimate of the ELBO uses ``S`` samples.

Minibatch / ``DataLoader`` training and LR schedulers are not implemented yet.
"""

from __future__ import annotations

from typing import List, Optional

import gpytorch
import torch

from deepgp.training.elbo import make_mll
from deepgp.utils.dtype import as_default_dtype

__all__ = ["fit"]


def fit(
    model,
    X: torch.Tensor,
    Y: torch.Tensor,
    epochs: int = 1000,
    lr: float = 1e-2,
    num_samples: int = 10,
    beta: float = 1.0,
    optimizer: Optional[torch.optim.Optimizer] = None,
    verbose: bool = False,
) -> List[float]:
    """Train ``model`` on ``(X, Y)`` by maximising the ELBO.

    The model is trained **in place**.

    Parameters
    ----------
    model:
        An ``SVGP`` or ``DeepGP`` (must expose ``model.likelihood``).
    X:
        Training inputs, shape ``(N, input_dims)``.
    Y:
        Training targets, shape ``(N,)`` (single output) or ``(N, T)``
        (multi-output).
    epochs:
        Number of full-batch optimisation steps.
    lr:
        Adam learning rate (ignored if ``optimizer`` is provided).
    num_samples:
        Number of doubly-stochastic samples ``S`` for the ELBO estimate.
    beta:
        KL tempering coefficient (KL weight ``= beta / N``).
    optimizer:
        Optional pre-constructed optimiser.  Defaults to
        ``torch.optim.Adam(model.parameters(), lr=lr)``.
    verbose:
        If ``True``, print the loss every ~10% of training.

    Returns
    -------
    list of float
        The per-epoch loss (negative ELBO).  Useful for asserting convergence.
    """
    X = as_default_dtype(X)
    Y = as_default_dtype(Y)

    model.train()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mll = make_mll(model, num_data=X.size(0), beta=beta)

    losses: List[float] = []
    log_every = max(1, epochs // 10)
    for epoch in range(epochs):
        with gpytorch.settings.num_likelihood_samples(num_samples):
            optimizer.zero_grad()
            output = model(X)
            loss = -mll(output, Y)
            loss.backward()
            optimizer.step()
        loss_value = float(loss.detach().item())
        losses.append(loss_value)
        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"epoch {epoch + 1:>5d}/{epochs}  loss={loss_value:.6f}")
    return losses
