# deepgp-torch

[![PyPI](https://img.shields.io/pypi/v/deepgp-torch.svg)](https://pypi.org/project/deepgp-torch/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/bydeng01/deepgp-torch/blob/main/LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/bydeng01/deepgp-torch/ci.yml?branch=main&label=CI)](https://github.com/bydeng01/deepgp-torch/actions/workflows/ci.yml)
[![Docker](https://img.shields.io/badge/Docker-supported-2496ED.svg?logo=docker&logoColor=white)](https://github.com/bydeng01/deepgp-torch/blob/main/Dockerfile)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/1324955093.svg)](https://doi.org/10.5281/zenodo.22188724)

deepgp-torch is a GPyTorch library for Gaussian process (GP) regression. It
provides three models:

- `ExactGP`, a GP with exact inference (Rasmussen & Williams, 2006).
- `SVGP`, a sparse variational GP with inducing points (Hensman et al., 2013).
- `DeepGP`, a deep GP trained by doubly stochastic variational inference
  (Salimbeni & Deisenroth, 2017).

All three models are trained with `fit` and evaluated with `predict`. The
package is imported as `deepgp`. Its API follows
[GPflux](https://github.com/secondmind-labs/GPflux), and it is built on PyTorch,
GPyTorch and linear_operator.

> **Status:** early development. Minibatch training, natural-gradient training
> and the benchmark harness are not implemented yet (see [Roadmap](#roadmap)).

## Install

```bash
pip install deepgp-torch
```

For development, install from a checkout:

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.10, `torch>=2.0`, `gpytorch>=1.15`,
`linear_operator>=0.6.1` and `numpy>=1.26`.

## Models

| Model | Training objective | Cost per iteration | Predictive distribution |
|---|---|---|---|
| `ExactGP` | log marginal likelihood | O(N³) | Gaussian |
| `SVGP` | ELBO | O(NM²) | Gaussian |
| `DeepGP` | Monte Carlo estimate of the ELBO | O(SNM²) per GP | mixture of S Gaussians |

N is the number of training points, M the number of inducing points per GP and
S the number of Monte Carlo samples. A `DeepGP` layer contains one GP per output
dimension. Every GP uses an RBF kernel with an output scale and one lengthscale
per input dimension. The likelihood is Gaussian.

## Quickstart

Importing `deepgp` sets the torch default dtype to `float64`. Cholesky
factorisations of kernel matrices are more stable in double precision.

```python
from deepgp import DeepGP, ExactGP, SVGP, fit, predict
from deepgp.data import load_snelson1d

data = load_snelson1d(test_fraction=0.2, seed=0)  # standardised train/test split
X, Y = data.X_train, data.Y_train                 # shapes (N, 1) and (N,)

models = {
    "exact": ExactGP(X, Y),
    "svgp": SVGP(inducing_points=X[:32].clone()),
    "deep": DeepGP([1, 1], num_inducing=32),      # one hidden layer of width 1
}
for name, model in models.items():
    losses = fit(model, X, Y, epochs=300, lr=0.01)
    mean, var = predict(model, data.X_test)
```

`fit` trains the model in place and returns the loss at each epoch. `predict`
returns the predictive mean and variance of `y`. With `add_noise=False` it
returns the mean and variance of the latent function `f`.

An `ExactGP` stores its training data. `fit` replaces it with the `X` and `Y`
passed to `fit`.

For a `DeepGP`, `num_samples` in `fit` and `k` in `predict` set S. Their
defaults are 10 and 50.

### Deep GP builder

`build_deep_gp` builds a deep GP whose hidden layers have the same width as the
input, like `build_constant_input_dim_deep_gp` in GPflux. Here `q(u)` denotes
the variational distribution of a layer's inducing variables `u`. The builder
initialises the model from the training inputs:

- inducing points at the k-means centroids of `X`,
- identity mean functions in the hidden layers,
- the Cholesky factor of each hidden layer's `q(u)` covariance multiplied by
  `inner_layer_qsqrt_factor`,
- the observation noise variance set to `likelihood_noise`.

A small `inner_layer_qsqrt_factor`, such as `1e-5`, reduces the sampling noise
of the hidden layers at the start of training.

```python
from deepgp import DeepGPConfig, build_deep_gp

cfg = DeepGPConfig(num_inducing=20, inner_layer_qsqrt_factor=1e-5, likelihood_noise=1e-2)
model = build_deep_gp(X, num_layers=2, config=cfg)
fit(model, X, Y, epochs=300, lr=0.01)
mean, var = predict(model, data.X_test)
```

### Multiple outputs

`ExactGP`, `SVGP`, `DeepGP` and `build_deep_gp` take `num_outputs=T` and train
on targets of shape `(N, T)`:

```python
import torch

X = torch.linspace(-2, 2, 200).unsqueeze(-1)                 # (200, 1)
F = torch.cat([torch.sin(2 * X), torch.cos(2 * X)], dim=-1)  # (200, 2)
Y = F + torch.tensor([0.05, 0.3]) * torch.randn_like(F)      # noise std 0.05 and 0.3

model = DeepGP([1, 2], num_inducing=32, num_outputs=2)       # hidden width 2, 2 outputs
fit(model, X, Y, epochs=300, lr=0.01)
mean, var = predict(model, X)                                # each (200, 2)
print(model.likelihood.task_noises)                          # one noise variance per output
```

The `T` outputs are modelled by `T` independent GPs, which form the output
layer of a `DeepGP`. Each GP has its own mean and kernel hyperparameters. In
`SVGP` and `DeepGP`, each GP also has its own inducing points and `q(u)`. The
likelihood has one noise variance per output.

`ExactGP` and `SVGP` share no parameters between outputs. Their log marginal
likelihood and ELBO are sums of `T` single-output terms.

In a `DeepGP` all outputs share the hidden layers. Each output GP takes the
output of the last hidden layer as its input. A hidden layer of width 1
therefore passes every output through one scalar feature.

`predict` returns the marginal mean and variance of each output. Under a
`DeepGP` the outputs are dependent. Within each mixture component, all `T`
means are computed from the same hidden-layer sample. `predict` does not return
the covariance between outputs.

## Implementation notes

- **Layers.** `DeepGPHiddenLayer` holds the inducing points, `q(u)`, mean and
  kernel of one layer. GPyTorch's `DeepGP` base class finds the layers through
  `model.modules()` and sums their KL divergences.
- **Hidden-layer means.** Compositions of zero-mean GPs lose degrees of freedom
  as depth increases (Duvenaud et al., 2014). Hidden layers therefore use linear
  mean functions, as in Salimbeni & Deisenroth (2017). Each hidden layer is then
  a GP perturbation of a linear map of its input.
- **Training objective.** For `SVGP` and `DeepGP`, `fit` maximises the ELBO
  divided by N,
  `(1/N) Σ_n E_q[log p(y_n | f_n)] − (β/N) KL[q(u) ‖ p(u)]`.
  The `beta` argument of `fit` sets β and defaults to 1. For `ExactGP`, `fit`
  maximises the log marginal likelihood divided by the number of targets.
- **Deep GP prediction.** The predictive distribution of a `DeepGP` is a mixture
  of S Gaussians with means `m_s` and variances `v_s`. By the law of total
  variance, `predict` returns the mean `E_s[m_s]` and the variance
  `E_s[v_s] + Var_s[m_s]`. The variance of a single component omits
  `Var_s[m_s]` and underestimates the predictive variance.
- **Output noise.** By default, GPyTorch's `MultitaskGaussianLikelihood` adds a
  shared noise variance σ² to the noise variance σ_t² of each output. The data
  identify only the `T` sums σ_t² + σ², not all `T + 1` parameters. The
  multi-output likelihood here omits σ², so `likelihood.task_noises[t]` is σ_t².
- **Target shape.** `fit` raises `ValueError` when `Y.shape` differs from the
  event shape of `model(X)`. Without this check, an `(N,)` target broadcasts
  against `(N, T)` outputs when N = T, and `fit` optimises the wrong objective.

## Development

```bash
make install      # pip install -e ".[dev]"
make format       # black + isort
make check        # black --check, isort --check-only, flake8, mypy src
make test         # pytest
```

CI (`.github/workflows/ci.yml`) runs the format, lint, type and test checks on
Python 3.10, 3.11 and 3.12. It also builds and runs the Docker image.

### Docker

The Dockerfile builds a CPU-only image:

```bash
docker build -t deepgp-torch .
docker run --rm deepgp-torch                # prints the installed version
docker run --rm deepgp-torch pytest -q      # runs the test suite
docker run --rm -it deepgp-torch python     # interactive REPL
```

## Roadmap

Implemented:

- **Models:** `ExactGP`, `SVGP` and `DeepGP`, trained with `fit` and evaluated
  with `predict`.
- **Deep GP builder:** `DeepGPConfig` and `build_deep_gp`, with k-means or
  subset inducing initialisation and identity or PCA hidden means.
- **Multi-output regression:** `num_outputs` on all three models and on
  `build_deep_gp`.
- **Utilities:** the `snelson1d` dataset, evaluation metrics, seeding and dtype
  helpers.

Planned (the modules exist as placeholders):

- `variational` helpers.
- Minibatch training with `DataLoader`, learning-rate schedulers, checkpointing
  and logging.
- Natural-gradient training.
- An equivalence test against GPflux, a UCI benchmark harness and Sphinx docs.

## Contributing

See [CONTRIBUTING.md](https://github.com/bydeng01/deepgp-torch/blob/main/CONTRIBUTING.md)
for the development setup and workflow. Participants follow the
[Code of Conduct](https://github.com/bydeng01/deepgp-torch/blob/main/CODE_OF_CONDUCT.md).
Report security issues as described in
[SECURITY.md](https://github.com/bydeng01/deepgp-torch/blob/main/SECURITY.md).

## Citing

If you use deepgp-torch in your research, please cite it with the metadata in
[CITATION.cff](https://github.com/bydeng01/deepgp-torch/blob/main/CITATION.cff).
GitHub shows this file as a "Cite this repository" button.

## References

- Duvenaud, Rippel, Adams & Ghahramani, *Avoiding Pathologies in Very Deep
  Networks*, AISTATS 2014.
- Hensman, Fusi & Lawrence, *Gaussian Processes for Big Data*, UAI 2013.
- Rasmussen & Williams, *Gaussian Processes for Machine Learning*, MIT Press,
  2006.
- Salimbeni & Deisenroth, *Doubly Stochastic Variational Inference for Deep
  Gaussian Processes*, NeurIPS 2017.
- Snelson & Ghahramani, *Sparse Gaussian Processes using Pseudo-inputs*, 2006
  (the `snelson1d` dataset).

## License

Apache-2.0. See [LICENSE](https://github.com/bydeng01/deepgp-torch/blob/main/LICENSE).
