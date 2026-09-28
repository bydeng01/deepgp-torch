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

"""Doubly-stochastic Deep GP container.

Composes a stack of :class:`~deepgp.layers.gp_layer.DeepGPHiddenLayer` objects
followed by a single output layer, and attaches a likelihood.  Subclasses
:class:`gpytorch.models.deep_gps.DeepGP`, whose variational machinery discovers
every layer through ``model.modules()`` and sums their KL terms automatically.

Layers must be stored so that ``model.modules()`` can find them; the hidden
layers therefore live in a :class:`torch.nn.ModuleList` (a plain Python list
would hide them from module discovery, and their KL contributions would be
silently dropped).

With ``num_outputs=T`` the output layer is ``T`` independent GPs on top of
hidden layers that all outputs share.  Given a sample of the hidden layers the
outputs are independent; integrating the hidden layers out makes them
dependent in general.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
from gpytorch.likelihoods import Likelihood
from gpytorch.models.deep_gps import DeepGP as _DeepGP

from deepgp.layers.gp_layer import DeepGPHiddenLayer
from deepgp.likelihoods.factory import make_likelihood

__all__ = ["DeepGP"]


class DeepGP(_DeepGP):
    """A stacked deep GP for regression.

    Parameters
    ----------
    dims:
        Widths of the input and hidden layers, e.g. ``[1, 1]`` for a
        1-D input with a single dim-preserving hidden layer.  ``dims[0]`` is the
        input dimensionality; each subsequent entry adds one hidden layer of
        that width.  A final output layer maps the last hidden width to
        ``num_outputs``.  With ``dims=[D]`` (no hidden entries) the model reduces
        to a single output layer.
    num_inducing:
        Number of inducing points per layer.
    num_outputs:
        Number of regression outputs ``T``.  ``None`` (default) means a single
        output with a :class:`~gpytorch.likelihoods.GaussianLikelihood`.  An
        integer makes the output layer ``T`` independent GPs, each with its own
        inducing points, ``q(u)``, mean and kernel hyper-parameters, and
        attaches a :class:`~gpytorch.likelihoods.MultitaskGaussianLikelihood`
        with one noise variance per output (see
        :func:`deepgp.likelihoods.make_likelihood`).
    hidden_mean_type:
        Mean function for hidden layers (default ``'linear'``).
    output_mean_type:
        Mean function for the output layer (default ``'constant'``).
    """

    def __init__(
        self,
        dims: Sequence[int],
        num_inducing: int = 128,
        num_outputs: Optional[int] = None,
        hidden_mean_type: str = "linear",
        output_mean_type: str = "constant",
    ) -> None:
        if len(dims) < 1:
            raise ValueError("`dims` must contain at least the input dimension.")
        super().__init__()

        hidden_layers = []
        prev = dims[0]
        for width in dims[1:]:
            hidden_layers.append(
                DeepGPHiddenLayer(
                    input_dims=prev,
                    output_dims=width,
                    num_inducing=num_inducing,
                    mean_type=hidden_mean_type,
                )
            )
            prev = width

        self.hidden_layers = torch.nn.ModuleList(hidden_layers)
        self.last_layer = DeepGPHiddenLayer(
            input_dims=prev,
            output_dims=num_outputs,
            num_inducing=num_inducing,
            mean_type=output_mean_type,
        )

        self.num_outputs = num_outputs
        self.likelihood = make_likelihood(num_outputs)

    @classmethod
    def from_layers(
        cls,
        hidden_layers: Iterable[DeepGPHiddenLayer],
        last_layer: DeepGPHiddenLayer,
        likelihood: Optional[Likelihood] = None,
        num_outputs: Optional[int] = None,
    ) -> "DeepGP":
        """Assemble a :class:`DeepGP` from pre-built layers.

        Used by :func:`deepgp.builders.build_deep_gp`, which constructs each layer
        with the stable-start initialisation (KMeans inducing, identity/PCA
        means, shrunk hidden ``q_sqrt``). The hidden layers are stored in a
        :class:`torch.nn.ModuleList` so the variational strategy discovers them.

        The number of outputs is ``last_layer.output_dims``; ``num_outputs``
        need not be given, and a value that disagrees raises ``ValueError``.
        """
        if num_outputs is None:
            num_outputs = last_layer.output_dims
        elif num_outputs != last_layer.output_dims:
            raise ValueError(
                f"num_outputs={num_outputs} does not match "
                f"last_layer.output_dims={last_layer.output_dims}."
            )
        self = cls.__new__(cls)
        _DeepGP.__init__(self)
        self.hidden_layers = torch.nn.ModuleList(list(hidden_layers))
        self.last_layer = last_layer
        self.num_outputs = num_outputs
        if likelihood is None:
            likelihood = make_likelihood(num_outputs)
        self.likelihood = likelihood
        return self

    def forward(self, x: torch.Tensor):
        for layer in self.hidden_layers:
            x = layer(x)
        return self.last_layer(x)
