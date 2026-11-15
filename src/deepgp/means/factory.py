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

"""Mean-function factory.

For a dim-preserving layer (``input_dims == output_dims``) the hidden mean is
the **identity** map. A layer wider than its input keeps the identity on its
first ``input_dims`` outputs and starts the others at zero; a narrower layer
gets a **PCA-initialised linear** map (top principal directions of the training
inputs). This follows the reference implementation of Salimbeni & Deisenroth
(2017).
Identity/linear hidden means prevent doubly-stochastic hidden-field collapse.
``ConstantMean`` / ``ZeroMean`` are also available; the output layer uses
``ConstantMean``.

The identity/PCA maps are threaded through ``batch_shape=[output_dims]`` so each
of the ``output_dims`` latent GPs gets its own linear functional:
``mean_d(x) = x @ W[d] (+ b[d])``.
"""

from __future__ import annotations

from typing import Optional

import torch
from gpytorch.means import ConstantMean, LinearMean, Mean, ZeroMean

__all__ = ["make_mean"]


def _pca_directions(X: torch.Tensor, output_dims: int) -> torch.Tensor:
    """Top ``output_dims`` principal directions of ``X`` as rows.

    Returns a tensor of shape ``(output_dims, input_dims)`` where row ``d`` is the
    ``d``-th principal direction, so ``x @ row_d`` is the ``d``-th PCA score.
    ``X`` has at most ``min(N, input_dims)`` directions; when ``output_dims`` asks
    for more, the remaining rows are zero, so those outputs start at zero.
    """
    x_centered = X - X.mean(dim=0, keepdim=True)
    # rows of Vh are the principal directions (V^T from the SVD of X_centered).
    _, _, vh = torch.linalg.svd(x_centered, full_matrices=False)
    directions = vh.new_zeros(output_dims, X.shape[-1])
    k = min(output_dims, vh.shape[0])
    directions[:k] = vh[:k]
    return directions


def make_mean(
    input_dims: int,
    output_dims: Optional[int],
    kind: str = "identity",
    X: Optional[torch.Tensor] = None,
) -> Mean:
    """Construct a mean function.

    Parameters
    ----------
    input_dims:
        Layer input dimensionality.
    output_dims:
        Number of latent GP outputs; ``None`` for a single-output layer.
    kind:
        ``'identity'`` (identity when ``input_dims == output_dims``, identity
        plus zero rows when ``output_dims > input_dims``, PCA otherwise),
        ``'linear'``/``'pca'`` (PCA-initialised linear, zero rows beyond the
        available directions), ``'constant'`` or ``'zero'``.
    X:
        Training inputs, required for the PCA initialisation.

    Returns
    -------
    gpytorch.means.Mean
    """
    kind = kind.lower()
    batch_shape = torch.Size([]) if output_dims is None else torch.Size([output_dims])

    if kind == "constant":
        return ConstantMean(batch_shape=batch_shape)
    if kind == "zero":
        return ZeroMean(batch_shape=batch_shape)
    if kind not in ("identity", "linear", "pca"):
        raise ValueError(
            f"Unknown mean kind={kind!r}; expected identity/linear/pca/"
            "constant/zero."
        )

    # Linear-family mean.
    mean = LinearMean(input_dims, batch_shape=batch_shape)

    if output_dims is None:
        # Single-output linear mean: identity only well-defined for 1-D input.
        with torch.no_grad():
            if kind == "identity" and input_dims == 1:
                mean.weights.copy_(torch.ones(input_dims, 1))
                if mean.bias is not None:  # pragma: no branch
                    mean.bias.zero_()
        return mean

    with torch.no_grad():
        if kind == "identity" and input_dims <= output_dims:
            # (output_dims, input_dims): the identity, then zero rows if wider.
            weights = torch.eye(output_dims, input_dims)
        else:
            if X is None:
                raise ValueError(
                    "PCA/linear mean init requires X (training inputs) unless "
                    "kind='identity' and input_dims <= output_dims."
                )
            weights = _pca_directions(X, output_dims)  # (output_dims, input_dims)
        # LinearMean weights have shape (output_dims, input_dims, 1); row d is W[d].
        mean.weights.copy_(weights.unsqueeze(-1))
        if mean.bias is not None:  # pragma: no branch
            mean.bias.zero_()
    return mean
