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

"""Natural-gradient variational training.

A two-optimiser scheme (Salimbeni, Eleftheriadis & Hensman, 2018):
:class:`gpytorch.optim.NGD` steps every ``q(u)`` built with
``variational='natural'``, and Adam steps every other parameter (kernel, mean,
likelihood, inducing locations, and any Cholesky ``q(u)``).  :func:`deepgp.fit`
uses it when given ``natgrad_lr``.

NGD only applies to natural parameterisations: GPyTorch computes the natural
gradient as the ordinary gradient of those parameters, so another optimiser
(Adam's per-coordinate rescaling) would break it.  :func:`make_optimizers`
therefore refuses a natural ``q(u)`` without ``natgrad_lr``.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from gpytorch.optim import NGD
from gpytorch.variational import _NaturalVariationalDistribution

__all__ = ["natural_variational_parameters", "make_optimizers"]


def natural_variational_parameters(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    """Return the parameters of every natural-parameter ``q(u)`` in ``model``."""
    return [
        p
        for module in model.modules()
        if isinstance(module, _NaturalVariationalDistribution)
        for p in module.parameters()
    ]


def make_optimizers(
    model: torch.nn.Module,
    num_data: int,
    lr: float = 1e-2,
    natgrad_lr: Optional[float] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> List[torch.optim.Optimizer]:
    """Build the optimisers that :func:`deepgp.fit` steps together.

    Parameters
    ----------
    model:
        The model to train.
    num_data:
        Number of training points ``N``. NGD scales its step by ``N`` because
        the ELBO is divided by ``N``.
    lr:
        Adam learning rate (ignored if ``optimizer`` is given).
    natgrad_lr:
        NGD step size for the natural-parameter ``q(u)``; ``None`` for no
        natural gradients.
    optimizer:
        Optional optimiser for the other parameters.  With ``natgrad_lr`` it
        must not hold any natural ``q(u)`` parameter.

    Returns
    -------
    list of torch.optim.Optimizer
        ``[optimizer]``, or ``[optimizer, NGD]`` with ``natgrad_lr``.  The
        default ``optimizer`` is Adam over every parameter that NGD does not
        step.

    Raises
    ------
    ValueError
        If the model has a natural ``q(u)`` and ``natgrad_lr`` is ``None``, if
        ``natgrad_lr`` is given but the model has no natural ``q(u)`` (an
        ``ExactGP`` never has one), if ``natgrad_lr <= 0``, or if
        ``optimizer`` holds a natural ``q(u)`` parameter.
    """
    natural = natural_variational_parameters(model)
    if natgrad_lr is None:
        if natural:
            raise ValueError(
                "The model has a natural-parameter q(u) (variational='natural'), "
                "which only natural gradients can train; pass natgrad_lr, "
                "e.g. 0.1."
            )
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        return [optimizer]

    if natgrad_lr <= 0:
        raise ValueError(f"natgrad_lr must be > 0; got {natgrad_lr}.")
    if not natural:
        raise ValueError(
            "natgrad_lr needs a model with a natural-parameter q(u); build it "
            "with variational='natural' (or hidden_/output_variational)."
        )
    natural_ids = {id(p) for p in natural}
    if optimizer is None:
        rest = [p for p in model.parameters() if id(p) not in natural_ids]
        optimizer = torch.optim.Adam(rest, lr=lr)
    elif any(
        id(p) in natural_ids
        for group in optimizer.param_groups
        for p in group["params"]
    ):
        raise ValueError(
            "optimizer holds natural-parameter q(u) parameters, which NGD steps; "
            "give it only the other parameters."
        )
    return [optimizer, NGD(natural, num_data=num_data, lr=natgrad_lr)]
