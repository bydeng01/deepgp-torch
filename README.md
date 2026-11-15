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
> and the benchmark harness are not implemented yet (see
> [planned work](https://github.com/bydeng01/deepgp-torch/blob/main/CONTRIBUTING.md#planned-work)).

## Install

```bash
pip install deepgp-torch
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

### Deep GP builder

`build_deep_gp` builds a deep GP whose hidden layers have the same width as the
input, like `build_constant_input_dim_deep_gp` in GPflux. It initialises the
inducing points, hidden-layer means and variational distributions from the
training inputs.

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

Each output has its own GP and noise variance. In a `DeepGP` the outputs share
the hidden layers.

## Documentation

- [User guide](https://github.com/bydeng01/deepgp-torch/blob/main/docs/user_guide.md):
  training objectives, deep GP prediction, builder initialisation and multiple
  outputs.
- API reference: the docstrings, for example `help(deepgp.fit)`.
- [Changelog](https://github.com/bydeng01/deepgp-torch/blob/main/CHANGELOG.md).

## Contributing

[CONTRIBUTING.md](https://github.com/bydeng01/deepgp-torch/blob/main/CONTRIBUTING.md)
covers the development setup, the checks CI runs, the Docker image and planned
work. Participants follow the
[Code of Conduct](https://github.com/bydeng01/deepgp-torch/blob/main/CODE_OF_CONDUCT.md).
Report security issues as described in
[SECURITY.md](https://github.com/bydeng01/deepgp-torch/blob/main/SECURITY.md).

## Citing

If you use deepgp-torch in your research, please cite it with the metadata in
[CITATION.cff](https://github.com/bydeng01/deepgp-torch/blob/main/CITATION.cff).
GitHub shows this file as a "Cite this repository" button.

## References

- Hensman, Fusi & Lawrence, *Gaussian Processes for Big Data*, UAI 2013.
- Rasmussen & Williams, *Gaussian Processes for Machine Learning*, MIT Press,
  2006.
- Salimbeni & Deisenroth, *Doubly Stochastic Variational Inference for Deep
  Gaussian Processes*, NeurIPS 2017.
- Snelson & Ghahramani, *Sparse Gaussian Processes using Pseudo-inputs*, 2006
  (the `snelson1d` dataset).

## License

Apache-2.0. See [LICENSE](https://github.com/bydeng01/deepgp-torch/blob/main/LICENSE).
