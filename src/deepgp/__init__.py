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

"""deepgp — a GPyTorch library for doubly-stochastic deep Gaussian processes.

Top-level API::

    from deepgp import DeepGP, SVGP, fit, predict, build_deep_gp
    from deepgp.data import load_snelson1d

    data = load_snelson1d()
    model = DeepGP([1, 1], num_inducing=64)     # 1 hidden + 1 output layer
    fit(model, data.X_train, data.Y_train, epochs=500)
    mean, var = predict(model, data.X_test)

Importing ``deepgp`` sets the torch default dtype to ``float64`` so every GP
module is built in double precision for Cholesky stability.  Call
:func:`deepgp.set_default_float64` yourself if you reset the default dtype
elsewhere.
"""

from __future__ import annotations

from deepgp.builders.config import DeepGPConfig
from deepgp.builders.factory import build_deep_gp
from deepgp.inference.predict import predict
from deepgp.layers.gp_layer import DeepGPHiddenLayer
from deepgp.models.deep_gp import DeepGP
from deepgp.models.svgp import SVGP
from deepgp.training.trainer import fit
from deepgp.utils.dtype import set_default_float64
from deepgp.utils.seed import seed_everything
from deepgp.version import __version__

# float64 must be the default before any module is built. Safe to call again;
# users can override via torch.set_default_dtype.
set_default_float64()

__all__ = [
    "__version__",
    "DeepGP",
    "DeepGPHiddenLayer",
    "SVGP",
    "fit",
    "predict",
    "build_deep_gp",
    "DeepGPConfig",
    "set_default_float64",
    "seed_everything",
]
