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

"""Reproducibility helpers: seed Python, NumPy and torch RNGs together."""

from __future__ import annotations

import random

import numpy as np
import torch

__all__ = ["seed_everything"]


def seed_everything(seed: int = 0) -> int:
    """Seed the ``random``, ``numpy`` and ``torch`` RNGs.

    Parameters
    ----------
    seed:
        The seed to use for every RNG.

    Returns
    -------
    int
        The seed that was applied (for convenient logging).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - requires a CUDA device
        torch.cuda.manual_seed_all(seed)
    return seed
