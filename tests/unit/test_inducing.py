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

"""Unit tests for ``init_inducing`` branches beyond the kmeans path.

Covers the ``subset`` method (rows drawn from ``X``, deterministic), the
explicit ``z_init`` override (verbatim, dtype-cast, ignoring method/count),
capping ``num_inducing`` to ``N`` for both methods, and the ValueError guards
for non-2-D ``X`` and unknown methods.
"""

import pytest
import torch

import deepgp  # noqa: F401  (import sets float64 default dtype)
from deepgp.data import init_inducing


def test_subset_rows_come_from_X_and_are_deterministic() -> None:
    torch.manual_seed(0)
    x = torch.randn(20, 3)
    z1 = init_inducing(x, 5, method="subset", seed=7)
    z2 = init_inducing(x, 5, method="subset", seed=7)

    assert z1.shape == (5, 3)
    # Deterministic given the seed.
    assert torch.equal(z1, z2)

    # Every returned row is exactly one of the rows of X (a subset, not centres).
    for row in z1:
        assert (x == row).all(dim=1).any()

    # The subset picks distinct rows (randperm indices are unique).
    assert torch.unique(z1, dim=0).shape[0] == 5


def test_z_init_override_is_verbatim_and_cast_to_float64() -> None:
    torch.manual_seed(0)
    x = torch.randn(20, 3)
    z_init = torch.arange(6, dtype=torch.float32).reshape(3, 2)

    # method / num_inducing are ignored when z_init is supplied.
    out = init_inducing(x, num_inducing=99, method="nope", z_init=z_init)

    assert out.dtype == torch.float64
    assert out.shape == (3, 2)
    assert torch.equal(out, z_init.to(torch.float64))


def test_num_inducing_capped_to_N_for_kmeans() -> None:
    torch.manual_seed(0)
    x = torch.randn(6, 2)
    z = init_inducing(x, num_inducing=1000, method="kmeans", seed=0)
    assert z.shape == (6, 2)


def test_num_inducing_capped_to_N_for_subset() -> None:
    torch.manual_seed(0)
    x = torch.randn(6, 2)
    z = init_inducing(x, num_inducing=1000, method="subset", seed=0)
    assert z.shape == (6, 2)
    # Cap uses every point exactly once -> a permutation of X's rows.
    assert torch.unique(z, dim=0).shape[0] == 6


def test_non_2d_X_raises_value_error() -> None:
    with pytest.raises(ValueError):
        init_inducing(torch.randn(10), num_inducing=3, method="subset")
    with pytest.raises(ValueError):
        init_inducing(torch.randn(4, 2, 2), num_inducing=3, method="kmeans")


def test_unknown_method_raises_value_error() -> None:
    x = torch.randn(10, 2)
    with pytest.raises(ValueError):
        init_inducing(x, num_inducing=3, method="nope")
