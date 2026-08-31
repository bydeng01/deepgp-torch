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
#
# Reproducible CPU image for deepgp-torch.
#   docker build -t deepgp-torch .
#   docker run --rm deepgp-torch                 # -> prints the version
#   docker run --rm deepgp-torch pytest -q       # -> runs the test suite
#   docker run --rm -it deepgp-torch python       # -> interactive REPL

FROM python:3.11-slim

LABEL org.opencontainers.image.title="deepgp-torch" \
      org.opencontainers.image.description="GPyTorch library for doubly-stochastic deep Gaussian processes" \
      org.opencontainers.image.source="https://github.com/bydeng01/deepgp-torch" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install CPU-only torch first (the default wheel bundles large CUDA libraries),
# then the package with its dev tooling so the image can also run the test suite.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0" \
    && pip install ".[dev]"

# Copy tests last so source edits don't bust the dependency layer cache.
COPY tests ./tests

# Non-root user (least privilege).
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

CMD ["python", "-c", "import deepgp; print('deepgp', deepgp.__version__)"]
