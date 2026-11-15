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

``build_deep_gp(X, num_layers, config)`` assembles a deep GP and applies the
builder's stable-start initialisation:

* hidden layers of the widths in ``config.hidden_dims``, by default
  **dim-preserving** (width ``X.shape[-1]``);
* **KMeans** inducing initialisation from ``X`` for the first layer; each later
  layer starts from those centres mapped through the hidden means before it
  (unchanged by identity means);
* hidden-layer **identity** (or PCA) linear mean functions;
* hidden-layer ``q_sqrt`` **shrunk** by ``config.inner_layer_qsqrt_factor`` (~1e-5);
* a ``ConstantMean`` output layer with one GP per output (``num_outputs``);
* a Gaussian likelihood whose noise variance (for every output) starts at
  ``config.likelihood_noise``.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from gpytorch.means import LinearMean

from deepgp.builders.config import DeepGPConfig
from deepgp.data.inducing import init_inducing
from deepgp.kernels.factory import make_kernel
from deepgp.layers.gp_layer import DeepGPHiddenLayer
from deepgp.likelihoods.factory import make_likelihood
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
    num_outputs: Optional[int] = None,
) -> DeepGP:
    """Build a stacked deep GP with the stable-start initialisation.

    Parameters
    ----------
    X:
        Training inputs, shape ``(N, input_dims)`` — used for inducing init and
        the PCA/identity hidden means.
    num_layers:
        Total number of GP layers (``num_layers - 1`` hidden layers + 1 output
        layer). Must be >= 1. The hidden layers have the widths in
        ``config.hidden_dims``, or the input width when it is ``None``.
    config:
        A :class:`DeepGPConfig` (required).
    seed:
        Seed for the inducing initialisation.
    num_outputs:
        Number of regression outputs ``T``; ``None`` (default) for one.  Each
        output GP starts from its own copy of the KMeans inducing points.

    Returns
    -------
    DeepGP
        A deep GP with ``num_outputs`` outputs, ready for :func:`deepgp.fit` /
        :func:`deepgp.predict`.
    """
    if config is None:
        raise ValueError("build_deep_gp requires a DeepGPConfig (config=...).")
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1.")

    X = as_default_dtype(torch.as_tensor(X))
    if X.dim() != 2:
        raise ValueError(f"X must be 2-D (N, input_dims); got {tuple(X.shape)}.")
    input_dims = X.shape[-1]
    widths = _hidden_widths(config, num_layers, input_dims)

    # KMeans inducing init in the input space of the first layer.
    z = init_inducing(X, config.num_inducing, method=config.inducing_init, seed=seed)
    num_inducing = z.shape[0]

    # ``h`` (the training inputs) and ``z`` (the inducing inputs) are carried in
    # the input space of the layer being built. At initialisation a hidden
    # layer's output is ~ its mean function (q(u) has zero mean and a shrunk
    # covariance), so both are mapped through each linear hidden mean. An
    # identity mean leaves them unchanged.
    h = X
    hidden_layers = []
    prev = input_dims
    for width in widths:
        mean_function = make_mean(prev, width, kind=config.mean_type, X=h)
        hidden_layers.append(
            DeepGPHiddenLayer(
                input_dims=prev,
                output_dims=width,
                num_inducing=num_inducing,
                inducing_points=z.unsqueeze(0).repeat(width, 1, 1),  # (W, M, D)
                mean_function=mean_function,
                covar_module=make_kernel(prev, width),
                whiten=config.whiten,
                q_sqrt_factor=config.inner_layer_qsqrt_factor,
                variational=config.hidden_variational,
            )
        )
        if isinstance(mean_function, LinearMean):
            h, z = _through_mean(mean_function, h), _through_mean(mean_function, z)
        elif width != prev:
            raise ValueError(
                f"A hidden layer that changes width ({prev} -> {width}) needs a "
                f"linear mean to place the next layer's inducing points; got "
                f"mean_type={config.mean_type!r}. Use 'identity' or 'linear'."
            )
        prev = width

    last_layer = DeepGPHiddenLayer(
        input_dims=prev,
        output_dims=num_outputs,
        num_inducing=num_inducing,
        inducing_points=(
            z.clone()
            if num_outputs is None
            else z.unsqueeze(0).repeat(num_outputs, 1, 1)  # (T, M, D)
        ),
        mean_function=make_mean(prev, num_outputs, kind="constant"),
        covar_module=make_kernel(prev, num_outputs),
        whiten=config.whiten,
        variational=config.output_variational,
    )

    likelihood = make_likelihood(num_outputs, noise=config.likelihood_noise)
    return DeepGP.from_layers(hidden_layers, last_layer, likelihood=likelihood)


def _hidden_widths(config: DeepGPConfig, num_layers: int, input_dims: int) -> List[int]:
    """The width of each of the ``num_layers - 1`` hidden layers."""
    if config.hidden_dims is None:
        return [input_dims] * (num_layers - 1)
    widths = list(config.hidden_dims)
    if len(widths) != num_layers - 1:
        raise ValueError(
            f"hidden_dims={widths} gives {len(widths)} hidden layers, but "
            f"num_layers={num_layers} has {num_layers - 1}."
        )
    return widths


def _through_mean(mean_function: LinearMean, x: torch.Tensor) -> torch.Tensor:
    """Map rows ``x`` of shape ``(N, D)`` through a batched linear mean.

    The mean has ``batch_shape=[W]`` and returns ``(W, N)``; the result is the
    ``(N, W)`` matrix of the layer's initial outputs.
    """
    with torch.no_grad():
        return mean_function(x).transpose(-1, -2).contiguous()
