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

"""Utility helpers: seeding and default-dtype management."""

from deepgp.utils.dtype import as_default_dtype, default_dtype, set_default_float64
from deepgp.utils.seed import seed_everything

__all__ = [
    "set_default_float64",
    "default_dtype",
    "as_default_dtype",
    "seed_everything",
]
