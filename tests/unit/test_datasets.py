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

"""Tests for ``load_snelson1d`` / ``RegressionData`` in ``deepgp.data``."""

import numpy
import pytest
import torch

import deepgp  # noqa: F401  (sets torch default dtype to float64)
from deepgp.data.datasets import load_snelson1d

TOTAL = 200


def test_default_load_shapes_and_dtypes():
    data = load_snelson1d()
    for t in (data.X_train, data.Y_train, data.X_test, data.Y_test):
        assert t.dtype == torch.float64
    # X_* are 2-D (N, 1); Y_* are 1-D.
    assert data.X_train.ndim == 2 and data.X_train.size(-1) == 1
    assert data.X_test.ndim == 2 and data.X_test.size(-1) == 1
    assert data.Y_train.ndim == 1
    assert data.Y_test.ndim == 1
    assert data.input_dims == 1
    # Split sizes sum to the full 200-point dataset.
    n_train = data.X_train.size(0)
    n_test = data.X_test.size(0)
    assert n_train == data.Y_train.size(0)
    assert n_test == data.Y_test.size(0)
    assert n_train + n_test == TOTAL
    # test_fraction=0.2 -> 40 test / 160 train.
    assert n_test == 40
    assert n_train == 160


def _recover_x_rows(data):
    """Recover raw X rows from a standardised split as a set of rounded floats."""
    x_train_raw = data.X_train * data.x_std + data.x_mean
    x_test_raw = data.X_test * data.x_std + data.x_mean
    train_rows = {round(float(v), 9) for v in x_train_raw.reshape(-1)}
    test_rows = {round(float(v), 9) for v in x_test_raw.reshape(-1)}
    return train_rows, test_rows


def test_train_test_indices_are_disjoint():
    data = load_snelson1d()
    train_rows, test_rows = _recover_x_rows(data)
    # No leaked points: reconstructed raw X rows never appear in both splits.
    assert train_rows.isdisjoint(test_rows)
    # And every point is accounted for exactly once.
    assert len(train_rows) == data.X_train.size(0)
    assert len(test_rows) == data.X_test.size(0)
    assert len(train_rows | test_rows) == TOTAL


def test_training_statistics_are_standardised():
    data = load_snelson1d()
    # Training inputs: ~0 mean, ~1 (population) std using the fitted stats.
    x_mean = data.X_train.mean()
    x_std = data.X_train.std(unbiased=False)
    assert x_mean.abs().item() < 1e-6
    assert x_std.item() == pytest.approx(1.0, rel=1e-6)
    # Training targets: ~0 mean, ~1 (population) std.
    y_mean = data.Y_train.mean()
    y_std = data.Y_train.std(unbiased=False)
    assert y_mean.abs().item() < 1e-6
    assert y_std.item() == pytest.approx(1.0, rel=1e-6)


def test_same_seed_is_reproducible():
    a = load_snelson1d(seed=7)
    b = load_snelson1d(seed=7)
    assert torch.equal(a.X_train, b.X_train)
    assert torch.equal(a.Y_train, b.Y_train)
    assert torch.equal(a.X_test, b.X_test)
    assert torch.equal(a.Y_test, b.Y_test)


def test_different_seed_gives_different_split():
    a = load_snelson1d(seed=0)
    b = load_snelson1d(seed=1)
    # Same sizes, but the selected points (and hence values) differ.
    assert a.X_train.shape == b.X_train.shape
    assert not torch.equal(a.X_train, b.X_train)
    a_rows, _ = _recover_x_rows(a)
    b_rows, _ = _recover_x_rows(b)
    assert a_rows != b_rows


def test_standardize_false_is_a_noop():
    raw = load_snelson1d(seed=3, standardize=False)
    # Statistics collapse to the identity transform.
    assert raw.x_mean.abs().max().item() == 0.0
    assert torch.equal(raw.x_std, torch.ones_like(raw.x_std))
    assert raw.y_mean.item() == 0.0
    assert raw.y_std.item() == 1.0
    # The unstandardised data equals the raw split reconstructed from a
    # standardised load with the same seed.
    std = load_snelson1d(seed=3, standardize=True)
    x_train_raw = std.X_train * std.x_std + std.x_mean
    y_train_raw = std.Y_train * std.y_std + std.y_mean
    assert torch.allclose(raw.X_train, x_train_raw, atol=1e-9)
    assert torch.allclose(raw.Y_train, y_train_raw, atol=1e-9)


def test_unstandardize_y_round_trips():
    std = load_snelson1d(seed=2, standardize=True)
    raw = load_snelson1d(seed=2, standardize=False)
    recovered = std.unstandardize_y(std.Y_train)
    # Round-trip recovers the raw-scale training targets.
    assert torch.allclose(recovered, raw.Y_train, atol=1e-9)
    # Recovered mean/std match the fitted training statistics.
    assert recovered.mean().item() == pytest.approx(std.y_mean.item(), abs=1e-9)
    assert recovered.std(unbiased=False).item() == pytest.approx(
        std.y_std.item(), rel=1e-9
    )
    # And the standardised targets were genuinely rescaled (std != 1 raw).
    assert std.y_std.item() > 0.1


def test_synthetic_fallback_when_npz_missing(monkeypatch):
    def _boom(*args, **kwargs):
        raise FileNotFoundError("vendored npz missing")

    monkeypatch.setattr(numpy, "load", _boom)
    data = load_snelson1d(seed=0, standardize=True)
    # Fallback still yields a valid standardised 200-point split.
    assert data.X_train.size(0) + data.X_test.size(0) == TOTAL
    assert data.X_train.ndim == 2 and data.X_train.size(-1) == 1
    assert data.Y_train.ndim == 1
    for t in (data.X_train, data.Y_train, data.X_test, data.Y_test):
        assert t.dtype == torch.float64
        assert torch.isfinite(t).all()
    # Standardisation still holds on the synthetic data.
    assert data.X_train.mean().abs().item() < 1e-6
    assert data.X_train.std(unbiased=False).item() == pytest.approx(1.0, rel=1e-6)
