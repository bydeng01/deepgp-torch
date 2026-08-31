# deepgp-torch

[![PyPI](https://img.shields.io/pypi/v/deepgp-torch.svg)](https://pypi.org/project/deepgp-torch/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/bydeng01/deepgp-torch/ci.yml?branch=main&label=CI)](https://github.com/bydeng01/deepgp-torch/actions/workflows/ci.yml)
[![Docker](https://img.shields.io/badge/Docker-supported-2496ED.svg?logo=docker&logoColor=white)](Dockerfile)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A GPyTorch library for doubly-stochastic deep Gaussian processes (Salimbeni &
Deisenroth, 2017). It provides modular SVGP layers, a deep-GP container,
variational (ELBO) training, and calibrated prediction.

The importable package is `deepgp`. Its API and engineering are modelled on
[GPflux](https://github.com/secondmind-labs/GPflux) but built on
PyTorch + GPyTorch + linear_operator.

> **Status:** early development. `SVGP`, `DeepGP` and `build_deep_gp` train and
> predict on `snelson1d`. Multi-output support, minibatch and natural-gradient
> training, and the benchmark harness are not implemented yet (see
> [Roadmap](#roadmap)).

## Install

```bash
pip install deepgp-torch
```

Or, from a checkout, for development:

```bash
pip install -e ".[dev]"     # editable install with dev tooling
```

Requires Python ≥ 3.10, `torch>=2.0`, `gpytorch>=1.15`,
`linear_operator>=0.6.1`, `numpy>=1.26`.

## Quickstart

Importing `deepgp` sets the torch default dtype to `float64` (Cholesky
stability), so every GP module is built in double precision.

```python
import deepgp
from deepgp import DeepGP, fit, predict
from deepgp.data import load_snelson1d

data = load_snelson1d(test_fraction=0.2, seed=0)      # standardised split

model = DeepGP([1, 1], num_inducing=32)               # 1 hidden + 1 output layer
losses = fit(model, data.X_train, data.Y_train,       # maximise the ELBO
             epochs=300, lr=0.01, num_samples=10)

mean, var = predict(model, data.X_test, k=32)         # total-variance mixture
print("first training loss:", losses[0], "-> last:", losses[-1])
```

A single-layer SVGP uses the same `fit` / `predict` helpers:

```python
from deepgp import SVGP

svgp = SVGP(inducing_points=data.X_train[:32].clone(), input_dims=1)
fit(svgp, data.X_train, data.Y_train, epochs=300, lr=0.01)
mean, var = predict(svgp, data.X_test)
```

Or build a constant-input-dimension deep GP with **stable-start
initialisation** in one call (KMeans inducing init, dim-preserving identity/PCA
hidden means, shrunk hidden `q_sqrt`, Gaussian noise init):

```python
from deepgp import build_deep_gp, DeepGPConfig

cfg = DeepGPConfig(num_inducing=20, inner_layer_qsqrt_factor=1e-5, likelihood_noise=1e-2)
model = build_deep_gp(data.X_train, num_layers=2, config=cfg)
fit(model, data.X_train, data.Y_train, epochs=300, lr=0.01)
mean, var = predict(model, data.X_test)
```

## Why a deep-GP-specific library over raw GPyTorch?

It encodes the non-obvious correctness details that raw GPyTorch leaves to you:

- **Layers own their variational structure** — `DeepGPHiddenLayer` bundles
  inducing points, `q(u)`, mean and kernel; the deep-GP variational strategy
  discovers layers via `model.modules()` and sums their KL terms automatically.
- **Linear hidden means** — `LinearMean` hidden layers prevent doubly-stochastic
  hidden-field collapse (`mean_type='zero'` is available but discouraged).
- **Law-of-total-variance prediction** — a deep GP's predictive distribution is
  an `S`-component Gaussian *mixture*; `predict` reduces it as
  `mean = out.mean.mean(0)` and
  `var = out.variance.mean(0) + out.mean.var(0, unbiased=False)`. A single
  component's `.variance` under-reports uncertainty.
- **First-class KL tempering** — the ELBO KL weight is exactly `beta / num_data`.

## Development

```bash
make install      # pip install -e ".[dev]"
make format       # black + isort
make check        # black --check, isort --check-only, flake8, mypy src
make test         # pytest
```

CI (`.github/workflows/ci.yml`) runs lint + type-check + tests on Python 3.10,
3.11 and 3.12, and builds the Docker image.

### Docker

A CPU-only image for a reproducible environment:

```bash
docker build -t deepgp-torch .
docker run --rm deepgp-torch                # prints the installed version
docker run --rm deepgp-torch pytest -q      # runs the test suite
docker run --rm -it deepgp-torch python     # interactive REPL
```

## Roadmap

Implemented:

- **Core stack:** `layers/gp_layer`, `models/svgp`, `models/deep_gp`,
  `training/elbo` + `trainer`, `inference/predict`, `data/datasets` (snelson1d),
  `eval/metrics`, `utils/seed` + `dtype`.
- **Architecture builder:** `DeepGPConfig` and `build_deep_gp` with stable-start
  init, `kernels/factory` + `means/factory` (identity/PCA `LinearMean`), and
  `data/inducing` (KMeans / subset / explicit `z_init`).

Not yet implemented — the modules exist as placeholders:

- Multi-output support (`MultitaskGaussianLikelihood`, `batch_shape` threading)
  and `likelihoods` / `variational` convenience helpers.
- Minibatch/`DataLoader` training with LR schedulers, checkpointing and logging,
  and natural-gradient training.
- Golden equivalence test against GPflux, a UCI benchmark harness, and Sphinx
  docs.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/bydeng01/deepgp-torch/blob/main/CONTRIBUTING.md) for the dev
setup and workflow, and please follow the
[Code of Conduct](https://github.com/bydeng01/deepgp-torch/blob/main/CODE_OF_CONDUCT.md). To report a security issue, see
[SECURITY.md](https://github.com/bydeng01/deepgp-torch/blob/main/SECURITY.md).

## Citing

If you use `deepgp-torch` in your research, please cite it — see
[CITATION.cff](https://github.com/bydeng01/deepgp-torch/blob/main/CITATION.cff) (GitHub renders a "Cite this repository" button).

## References

- Salimbeni & Deisenroth, *Doubly Stochastic Variational Inference for Deep
  Gaussian Processes*, NeurIPS 2017.
- Snelson & Ghahramani, *Sparse Gaussian Processes using Pseudo-inputs*, 2006
  (the `snelson1d` dataset).

## License

Apache-2.0 — see [LICENSE](https://github.com/bydeng01/deepgp-torch/blob/main/LICENSE).
