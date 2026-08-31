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

"""Standard regression datasets with normalisation.

``snelson1d`` (Snelson & Ghahramani, 2006) is a ~200-point 1-D regression
benchmark.  The data file (``snelson1d.npz`` with arrays ``X`` and ``Y``) is
vendored inside the package (``deepgp/data/snelson1d.npz``, originally from
GPflux's test fixtures) so loading needs no network access.  If it is missing,
:func:`load_snelson1d` falls back to a synthetic 1-D regression problem.

Inputs and targets are standardised (zero mean, unit variance) using the
*training* statistics; the fitted statistics are returned so predictions can be
mapped back to the original scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import numpy as np
import torch

from deepgp.utils.dtype import as_default_dtype

__all__ = ["RegressionData", "load_snelson1d"]


@dataclass
class RegressionData:
    """A standardised train/test regression split.

    All tensors use the current default dtype (``float64``).  ``X_*`` have shape
    ``(N, input_dims)`` and ``Y_*`` have shape ``(N,)``.  The ``*_mean`` /
    ``*_std`` arrays are the training statistics used for standardisation.
    """

    X_train: torch.Tensor
    Y_train: torch.Tensor
    X_test: torch.Tensor
    Y_test: torch.Tensor
    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor

    @property
    def input_dims(self) -> int:
        return int(self.X_train.size(-1))

    def unstandardize_y(self, y: torch.Tensor) -> torch.Tensor:
        """Map a standardised target/prediction back to the original scale."""
        return y * self.y_std + self.y_mean


def _load_snelson_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Load the raw ``(X, Y)`` arrays from the vendored npz, or synthesise them."""
    try:
        with resources.as_file(
            resources.files("deepgp.data").joinpath("snelson1d.npz")
        ) as path:
            with np.load(path) as npz:
                return np.asarray(npz["X"]), np.asarray(npz["Y"])
    except (FileNotFoundError, ModuleNotFoundError, KeyError, OSError):
        # Deterministic synthetic 1-D fallback (no network, no vendored file).
        rng = np.random.default_rng(0)
        x = np.sort(rng.uniform(0.0, 6.0, size=(200, 1)), axis=0)
        y = np.sin(x) + 0.3 * np.cos(3.0 * x) + 0.1 * rng.standard_normal((200, 1))
        return x, y


def load_snelson1d(
    test_fraction: float = 0.2,
    seed: int = 0,
    standardize: bool = True,
) -> RegressionData:
    """Load ``snelson1d`` as a standardised train/test split.

    Parameters
    ----------
    test_fraction:
        Fraction of points held out for testing (default ``0.2``).
    seed:
        Seed for the (reproducible) random train/test split.
    standardize:
        If ``True`` (default) standardise inputs and targets using training
        statistics.  If ``False`` the statistics are set to 0/1 (a no-op).

    Returns
    -------
    RegressionData
    """
    x_np, y_np = _load_snelson_arrays()
    x_np = np.asarray(x_np, dtype=np.float64).reshape(x_np.shape[0], -1)
    y_np = np.asarray(y_np, dtype=np.float64).reshape(-1)

    n = x_np.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(test_fraction * n)))
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])

    x_train_np, y_train_np = x_np[train_idx], y_np[train_idx]
    x_test_np, y_test_np = x_np[test_idx], y_np[test_idx]

    if standardize:
        x_mean = x_train_np.mean(axis=0, keepdims=True)
        x_std = x_train_np.std(axis=0, keepdims=True)
        x_std[x_std == 0.0] = 1.0
        y_mean = y_train_np.mean()
        y_std = y_train_np.std()
        y_std = y_std if y_std != 0.0 else 1.0
    else:
        x_mean = np.zeros((1, x_np.shape[1]))
        x_std = np.ones((1, x_np.shape[1]))
        y_mean = 0.0
        y_std = 1.0

    def _t(a: np.ndarray) -> torch.Tensor:
        return as_default_dtype(torch.from_numpy(np.ascontiguousarray(a)))

    return RegressionData(
        X_train=_t((x_train_np - x_mean) / x_std),
        Y_train=_t((y_train_np - y_mean) / y_std),
        X_test=_t((x_test_np - x_mean) / x_std),
        Y_test=_t((y_test_np - y_mean) / y_std),
        x_mean=_t(x_mean.reshape(-1)),
        x_std=_t(x_std.reshape(-1)),
        y_mean=_t(np.asarray(y_mean).reshape(())),
        y_std=_t(np.asarray(y_std).reshape(())),
    )
