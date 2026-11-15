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

"""Unit tests for the mean/kernel factory branches not covered by test_builder.

Covers the ``constant``/``zero``/invalid ``kind`` branches of ``make_mean``, the
single-output (``output_dims=None``) identity linear mean, the PCA-without-X
error path, and the ``make_kernel`` no-batch / no-ARD branches.
"""

import pytest
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean, LinearMean, ZeroMean

import deepgp  # noqa: F401  (import sets float64 default dtype)
from deepgp.kernels import make_kernel
from deepgp.means import make_mean


def test_make_mean_constant_is_constant_mean_batched() -> None:
    mean = make_mean(3, 2, kind="constant")
    assert isinstance(mean, ConstantMean)
    # ConstantMean carries one learnable constant per batch element.
    assert mean.constant.shape[0] == 2
    # A batched input is reduced to a per-batch constant mean vector.
    x = torch.randn(2, 5, 3)
    assert mean(x).shape == torch.Size([2, 5])


def test_make_mean_zero_is_zero_mean_with_batch_shape() -> None:
    mean = make_mean(3, 2, kind="zero")
    assert isinstance(mean, ZeroMean)
    assert mean.batch_shape == torch.Size([2])
    # ZeroMean maps a batched input to an all-zero mean of the right shape.
    x = torch.randn(2, 5, 3)
    out = mean(x)
    assert out.shape == torch.Size([2, 5])
    assert torch.count_nonzero(out) == 0


def test_make_mean_invalid_kind_raises() -> None:
    with pytest.raises(ValueError):
        make_mean(2, 2, kind="bogus")


def test_make_mean_pca_without_X_raises() -> None:
    with pytest.raises(ValueError):
        make_mean(4, 2, kind="pca", X=None)


def test_make_mean_single_output_identity_is_unit_linear() -> None:
    mean = make_mean(1, None, kind="identity")
    assert isinstance(mean, LinearMean)
    # No batch dim for a single-output layer.
    assert mean.weights.shape == torch.Size([1, 1])
    assert torch.allclose(mean.weights, torch.ones(1, 1))
    assert mean.bias is not None
    assert torch.allclose(mean.bias, torch.zeros_like(mean.bias))
    # Acts as the identity on 1-D inputs.
    x = torch.randn(6, 1)
    assert torch.allclose(mean(x), x.squeeze(-1))


def test_make_mean_identity_has_zero_bias() -> None:
    mean = make_mean(2, 2, kind="identity")
    assert isinstance(mean, LinearMean)
    assert mean.bias is not None
    assert torch.allclose(mean.bias, torch.zeros_like(mean.bias))
    assert torch.count_nonzero(mean.bias) == 0


def test_make_kernel_no_output_dims_has_empty_batch_shape() -> None:
    kernel = make_kernel(3, output_dims=None)
    assert isinstance(kernel, ScaleKernel)
    assert isinstance(kernel.base_kernel, RBFKernel)
    assert kernel.batch_shape == torch.Size([])


def test_make_kernel_ard_toggle() -> None:
    ard_kernel = make_kernel(3, ard=True)
    no_ard_kernel = make_kernel(3, ard=False)
    assert isinstance(no_ard_kernel, ScaleKernel)
    assert isinstance(no_ard_kernel.base_kernel, RBFKernel)
    # ARD off -> a single shared lengthscale (no per-dim ARD).
    assert no_ard_kernel.base_kernel.ard_num_dims is None
    assert no_ard_kernel.base_kernel.lengthscale.shape[-1] == 1
    # ARD on -> one lengthscale per input dimension.
    assert ard_kernel.base_kernel.ard_num_dims == 3
    assert ard_kernel.base_kernel.lengthscale.shape[-1] == 3


def test_make_mean_identity_wider_layer_pads_with_zero_rows() -> None:
    """A layer wider than its input: identity on the first outputs, then zeros."""
    mean = make_mean(2, 5, kind="identity")  # no X needed
    weights = mean.weights.squeeze(-1)  # (output_dims=5, input_dims=2)
    assert torch.equal(weights, torch.eye(5, 2))
    x = torch.randn(7, 2)
    out = mean(x)  # (5, 7)
    assert torch.allclose(out[:2], x.T)
    assert torch.count_nonzero(out[2:]) == 0


def test_make_mean_pca_wider_layer_pads_with_zero_rows() -> None:
    torch.manual_seed(0)
    x = torch.randn(30, 2)
    weights = make_mean(2, 5, kind="pca", X=x).weights.squeeze(-1)
    assert weights.shape == (5, 2)
    # The two available principal directions are orthonormal ...
    assert torch.allclose(weights[:2] @ weights[:2].T, torch.eye(2), atol=1e-10)
    # ... and the rest of the layer starts at zero.
    assert torch.count_nonzero(weights[2:]) == 0


def test_make_mean_one_dimensional_input_wider_layer_is_not_degenerate() -> None:
    # Before the zero padding, every output of a 1 -> 3 layer copied x.
    weights = make_mean(1, 3, kind="identity").weights.squeeze(-1)
    assert torch.equal(weights, torch.tensor([[1.0], [0.0], [0.0]]))
