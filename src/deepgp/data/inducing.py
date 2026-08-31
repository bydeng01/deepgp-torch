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

"""Inducing-point initialisation strategies.

Supported strategies:

* ``kmeans``  — KMeans cluster centres of the training inputs (scikit-learn;
  falls back to a random subset if scikit-learn is unavailable),
* ``subset``  — a random subset of the training inputs,
* an explicit ``z_init`` override.

scikit-learn is an *optional* runtime dependency; the random-subset fallback
keeps the core install lightweight.
"""

from __future__ import annotations

from typing import Optional

import torch

from deepgp.utils.dtype import as_default_dtype

__all__ = ["init_inducing"]


def _random_subset(X: torch.Tensor, m: int, seed: int) -> torch.Tensor:
    n = X.shape[0]
    generator = torch.Generator().manual_seed(int(seed))
    idx = torch.randperm(n, generator=generator)[:m]
    return X.index_select(0, idx).clone()


def init_inducing(
    X: torch.Tensor,
    num_inducing: int,
    method: str = "kmeans",
    seed: int = 0,
    z_init: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Initialise inducing inputs from training data ``X``.

    Parameters
    ----------
    X:
        Training inputs, shape ``(N, input_dims)``.
    num_inducing:
        Desired number of inducing points ``M``. Capped to ``N`` for the
        ``kmeans``/``subset`` methods (you cannot place more distinct centres than
        there are points).
    method:
        ``'kmeans'`` (default) or ``'subset'``. Ignored if ``z_init`` is given.
    seed:
        Seed for KMeans / the random subset (reproducibility).
    z_init:
        Explicit inducing inputs to use verbatim (cast to the default dtype).

    Returns
    -------
    torch.Tensor
        Inducing inputs of shape ``(M, input_dims)``.
    """
    if z_init is not None:
        return as_default_dtype(torch.as_tensor(z_init))

    X = as_default_dtype(torch.as_tensor(X))
    if X.dim() != 2:
        raise ValueError(f"X must be 2-D (N, input_dims); got shape {tuple(X.shape)}.")
    n = X.shape[0]
    m = min(int(num_inducing), n)
    method = method.lower()

    if method == "kmeans":
        try:
            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=m, n_init=10, random_state=int(seed))
            km.fit(X.detach().cpu().numpy())
            return as_default_dtype(torch.as_tensor(km.cluster_centers_))
        except Exception:
            # scikit-learn absent or KMeans failed -> random-subset fallback.
            return _random_subset(X, m, seed)

    if method == "subset":
        return _random_subset(X, m, seed)

    raise ValueError(f"Unknown method={method!r}; expected 'kmeans' or 'subset'.")
