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

"""Deep GP architecture configuration.

The configuration consumed by :func:`deepgp.builders.build_deep_gp`.
``num_inducing`` / ``inner_layer_qsqrt_factor`` / ``likelihood_noise`` are
required; only ``whiten`` carries a default. An ``inner_layer_qsqrt_factor`` of
~1e-5 gives a stable, near-deterministic hidden-layer start.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

__all__ = ["DeepGPConfig"]


@dataclass
class DeepGPConfig:
    """Configuration for :func:`deepgp.builders.build_deep_gp`.

    Attributes
    ----------
    num_inducing:
        Number of inducing points per layer (required). Capped to ``N`` for the
        KMeans/subset inducing init.
    inner_layer_qsqrt_factor:
        Multiplier (~1e-5) applied to hidden-layer ``q_sqrt`` (the Cholesky of
        ``q(u)``) for a stable, near-deterministic initialisation (required).
    likelihood_noise:
        Initial observation-noise variance (required).
    hidden_dims:
        Optional explicit hidden-layer widths; ``None`` (default) means
        dim-preserving hidden layers (width = input dim).
    whiten:
        Whether to use the whitened variational parameterisation (default
        ``True``).
    mean_type:
        Hidden-layer mean function (default ``'identity'`` — an identity map
        for dim-preserving layers, a PCA-linear map otherwise).
    inducing_init:
        Inducing initialisation strategy: ``'kmeans'`` (default) or ``'subset'``.
    kernel:
        Optional kernel spec/override (reserved; default builds an ARD RBF).
    """

    num_inducing: int
    inner_layer_qsqrt_factor: float
    likelihood_noise: float
    hidden_dims: Optional[Sequence[int]] = None
    whiten: bool = True
    mean_type: str = "identity"
    inducing_init: str = "kmeans"
    kernel: Optional[object] = field(default=None)
