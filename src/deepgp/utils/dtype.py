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

"""Floating-point precision utilities.

Deep GPs rely on Cholesky factorisations of kernel matrices, which are far more
numerically stable in double precision.  The rule is to set the default dtype
to ``float64`` *before* any module (kernel, inducing points, variational
distribution) is constructed so that every learnable parameter is created in
double precision.

``deepgp`` calls :func:`set_default_float64` at import time (see
``deepgp/__init__.py``) so that ``import deepgp`` is sufficient to guarantee
float64 everywhere.  The function is also exported for callers who want to be
explicit.
"""

from __future__ import annotations

import torch

__all__ = ["set_default_float64", "default_dtype", "as_default_dtype"]


def set_default_float64() -> None:
    """Set the global torch default dtype to ``torch.float64``.

    Call this before constructing any GP module.  Idempotent.
    """
    torch.set_default_dtype(torch.float64)


def default_dtype() -> torch.dtype:
    """Return the current torch default dtype."""
    return torch.get_default_dtype()


def as_default_dtype(x: torch.Tensor) -> torch.Tensor:
    """Cast ``x`` to the current default dtype (leaving the device unchanged)."""
    return x.to(dtype=torch.get_default_dtype())
