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

"""One-call deep GP builder.

``build_deep_gp(X, num_layers, config)`` assembles a constant-input-dimension
deep GP and applies the builder's stable-start initialisation:

* **dim-preserving** hidden layers of width ``X.shape[-1]``;
* **KMeans** inducing initialisation from ``X`` (shared across layers, since the
  identity-mean hidden layers keep the representation ~ ``X``);
* hidden-layer **identity** (or PCA) linear mean functions;
* hidden-layer ``q_sqrt`` **shrunk** by ``config.inner_layer_qsqrt_factor`` (~1e-5);
* a ``ConstantMean`` output layer (single regression output);
* a ``GaussianLikelihood`` initialised to ``config.likelihood_noise``.
"""

from __future__ import annotations

from typing import Optional

import torch
from gpytorch.likelihoods import GaussianLikelihood

from deepgp.builders.config import DeepGPConfig
from deepgp.data.inducing import init_inducing
from deepgp.kernels.factory import make_kernel
from deepgp.layers.gp_layer import DeepGPHiddenLayer
from deepgp.means.factory import make_mean
from deepgp.models.deep_gp import DeepGP
from deepgp.utils.dtype import as_default_dtype

__all__ = ["build_deep_gp", "build_gp_layer"]


def build_gp_layer(
    input_dims: int,
    output_dims: Optional[int],
    config: DeepGPConfig,
    X: Optional[torch.Tensor] = None,
    inducing_points: Optional[torch.Tensor] = None,
    mean_kind: Optional[str] = None,
    shrink_q_sqrt: bool = False,
    seed: int = 0,
) -> DeepGPHiddenLayer:
    """Build a single configured :class:`DeepGPHiddenLayer`.

    Constructs the mean (identity/PCA/constant), kernel (ARD RBF) and inducing
    points (KMeans/subset, or the provided ``inducing_points``) from ``config``.
    ``shrink_q_sqrt`` applies ``config.inner_layer_qsqrt_factor`` (hidden layers).
    """
    kind = mean_kind or config.mean_type
    mean_function = make_mean(input_dims, output_dims, kind=kind, X=X)
    covar_module = make_kernel(input_dims, output_dims)

    if inducing_points is None:
        if X is None:
            raise ValueError("build_gp_layer needs X or explicit inducing_points.")
        z = init_inducing(
            X, config.num_inducing, method=config.inducing_init, seed=seed
        )
        m = z.shape[0]
        inducing_points = (
            z if output_dims is None else z.unsqueeze(0).repeat(output_dims, 1, 1)
        )
    else:
        m = inducing_points.shape[-2]

    return DeepGPHiddenLayer(
        input_dims=input_dims,
        output_dims=output_dims,
        num_inducing=m,
        inducing_points=inducing_points.clone(),
        mean_function=mean_function,
        covar_module=covar_module,
        whiten=config.whiten,
        q_sqrt_factor=(config.inner_layer_qsqrt_factor if shrink_q_sqrt else None),
    )


def build_deep_gp(
    X: torch.Tensor,
    num_layers: int = 2,
    config: Optional[DeepGPConfig] = None,
    seed: int = 0,
) -> DeepGP:
    """Build a stacked deep GP with the stable-start initialisation.

    Parameters
    ----------
    X:
        Training inputs, shape ``(N, input_dims)`` — used for inducing init and
        the PCA/identity hidden means.
    num_layers:
        Total number of GP layers (``num_layers - 1`` dim-preserving hidden
        layers + 1 output layer). Must be >= 1.
    config:
        A :class:`DeepGPConfig` (required).
    seed:
        Seed for the inducing initialisation.

    Returns
    -------
    DeepGP
        A single-output deep GP ready for :func:`deepgp.fit` / :func:`deepgp.predict`.
    """
    if config is None:
        raise ValueError("build_deep_gp requires a DeepGPConfig (config=...).")
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1.")

    X = as_default_dtype(torch.as_tensor(X))
    if X.dim() != 2:
        raise ValueError(f"X must be 2-D (N, input_dims); got {tuple(X.shape)}.")
    input_dims = X.shape[-1]

    # Shared KMeans inducing init (identity-mean hidden layers keep the
    # representation ~ X, so the same centres initialise every layer).
    z = init_inducing(X, config.num_inducing, method=config.inducing_init, seed=seed)
    num_inducing = z.shape[0]

    hidden_layers = []
    prev = input_dims
    for _ in range(num_layers - 1):
        z_hidden = z.unsqueeze(0).repeat(input_dims, 1, 1)  # (D, M, D)
        hidden_layers.append(
            DeepGPHiddenLayer(
                input_dims=prev,
                output_dims=input_dims,
                num_inducing=num_inducing,
                inducing_points=z_hidden,
                mean_function=make_mean(prev, input_dims, kind=config.mean_type, X=X),
                covar_module=make_kernel(prev, input_dims),
                whiten=config.whiten,
                q_sqrt_factor=config.inner_layer_qsqrt_factor,
            )
        )
        prev = input_dims

    last_layer = DeepGPHiddenLayer(
        input_dims=prev,
        output_dims=None,
        num_inducing=num_inducing,
        inducing_points=z.clone(),
        mean_function=make_mean(prev, None, kind="constant"),
        covar_module=make_kernel(prev, None),
        whiten=config.whiten,
    )

    likelihood = GaussianLikelihood()
    with torch.no_grad():
        likelihood.noise = torch.as_tensor(float(config.likelihood_noise))

    return DeepGP.from_layers(hidden_layers, last_layer, likelihood=likelihood)
