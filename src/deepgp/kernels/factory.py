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

"""Kernel factory.

``make_kernel`` builds ``ScaleKernel(RBFKernel(...))`` with the signal variance on
the outer ``ScaleKernel`` (``RBFKernel`` has no outputscale of its own).
``batch_shape=[output_dims]`` gives per-output hyper-parameters; ``[]`` shares
them. ARD is on by default (``ard_num_dims=input_dims``).
"""

from __future__ import annotations

from typing import Optional

import torch
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel

__all__ = ["make_kernel"]


def make_kernel(
    input_dims: int,
    output_dims: Optional[int] = None,
    ard: bool = True,
) -> Kernel:
    """Construct ``ScaleKernel(RBFKernel(...))`` for a (batched) GP layer.

    Parameters
    ----------
    input_dims:
        Input dimensionality (used for ARD).
    output_dims:
        Number of latent GP outputs; ``None`` for a single output (no batch dim).
    ard:
        If ``True`` (default) use a separate lengthscale per input dimension.
    """
    batch_shape = torch.Size([]) if output_dims is None else torch.Size([output_dims])
    ard_num_dims = input_dims if ard else None
    return ScaleKernel(
        RBFKernel(batch_shape=batch_shape, ard_num_dims=ard_num_dims),
        batch_shape=batch_shape,
    )
