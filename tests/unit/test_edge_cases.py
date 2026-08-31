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

"""Edge cases, error paths, and fallbacks that the happy-path tests skip."""

import importlib

import pytest
import torch
from gpytorch.means import LinearMean

import deepgp  # noqa: F401  (sets torch default dtype to float64)
from deepgp import SVGP, DeepGPConfig, build_deep_gp, fit
from deepgp.builders.factory import build_gp_layer
from deepgp.data import init_inducing
from deepgp.layers.gp_layer import DeepGPHiddenLayer
from deepgp.means import make_mean
from deepgp.utils.dtype import default_dtype


def _cfg(**kw) -> DeepGPConfig:
    base = dict(num_inducing=8, inner_layer_qsqrt_factor=1e-5, likelihood_noise=1e-2)
    base.update(kw)
    return DeepGPConfig(**base)


def test_fit_verbose_prints_progress(capsys) -> None:
    torch.manual_seed(0)
    x = torch.randn(20, 1)
    y = torch.randn(20)
    model = SVGP(x[:8].clone(), input_dims=1)
    losses = fit(model, x, y, epochs=5, lr=0.05, verbose=True)
    assert len(losses) == 5
    assert "loss=" in capsys.readouterr().out  # the verbose branch ran


def test_build_gp_layer_with_explicit_inducing_points() -> None:
    torch.manual_seed(0)
    z = torch.randn(6, 1)
    layer = build_gp_layer(1, None, _cfg(), inducing_points=z)
    assert isinstance(layer, DeepGPHiddenLayer)
    assert layer.variational_strategy.inducing_points.shape[-2] == 6


def test_build_gp_layer_casts_float32_inducing_points() -> None:
    """The builder entry point normalises explicit inducing points too."""
    torch.manual_seed(0)
    z32 = torch.randn(6, 1, dtype=torch.float32)
    layer = build_gp_layer(1, None, _cfg(), inducing_points=z32)

    assert layer.variational_strategy.inducing_points.dtype == torch.float64
    assert {p.dtype for p in layer.parameters()} == {torch.float64}


def test_build_gp_layer_requires_x_or_inducing_points() -> None:
    with pytest.raises(ValueError):
        build_gp_layer(1, None, _cfg())  # neither X nor inducing_points given


def test_build_deep_gp_rejects_bad_num_layers() -> None:
    with pytest.raises(ValueError):
        build_deep_gp(torch.randn(10, 1), num_layers=0, config=_cfg())


def test_build_deep_gp_rejects_non_2d_x() -> None:
    with pytest.raises(ValueError):
        build_deep_gp(torch.randn(10), num_layers=2, config=_cfg())


def test_init_inducing_falls_back_when_kmeans_raises(monkeypatch) -> None:
    """If KMeans errors (or scikit-learn is absent), fall back to a subset of X."""
    import sklearn.cluster as skc

    class _BoomKMeans:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            raise RuntimeError("simulated KMeans failure")

    monkeypatch.setattr(skc, "KMeans", _BoomKMeans)

    x = torch.randn(20, 2)
    z = init_inducing(x, 5, method="kmeans", seed=0)
    assert z.shape == (5, 2)
    # The fallback draws rows from X itself.
    x_rows = {tuple(round(float(v), 9) for v in row) for row in x}
    for row in z:
        assert tuple(round(float(v), 9) for v in row) in x_rows


def test_default_dtype_helper() -> None:
    assert default_dtype() == torch.float64


def test_make_mean_single_output_non_identity_linear() -> None:
    # output_dims=None with a non-identity linear kind -> a plain LinearMean
    # (exercises the else path of the single-output identity guard).
    mean = make_mean(2, None, kind="linear")
    assert isinstance(mean, LinearMean)
    assert mean.weights.shape == (2, 1)


def test_all_submodules_import() -> None:
    """The full package tree, placeholder modules included, imports cleanly."""
    for name in (
        "deepgp.layers.base",
        "deepgp.likelihoods",
        "deepgp.variational",
        "deepgp.training.natgrad",
        "deepgp.training.checkpoint",
        "deepgp.kernels.factory",
        "deepgp.means.factory",
    ):
        assert importlib.import_module(name) is not None
