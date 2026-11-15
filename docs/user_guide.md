# User guide

This guide describes how deepgp-torch trains and evaluates its models. The API
reference is in the docstrings, for example `help(deepgp.fit)`.

| Symbol | Meaning |
|---|---|
| $N$ | number of training points |
| $T$ | number of outputs |
| $M$ | number of inducing points per GP |
| $S$ | number of Monte Carlo samples |
| $u$ | inducing variables of a GP |
| $q(u)$ | variational distribution of $u$ |

## Precision

Importing `deepgp` sets the torch default dtype to `float64`. Cholesky
factorisations of kernel matrices are more stable in double precision. The
models, `fit` and `predict` cast input tensors to the default dtype. Call
`deepgp.set_default_float64()` again if other code changes the default dtype.

## Training

`fit(model, X, Y)` runs full-batch Adam with learning rate `lr` for `epochs`
steps and returns the loss at each step. Pass `optimizer` to use a different
optimiser. The model is trained in place.

### Natural gradients

`SVGP`, `DeepGPHiddenLayer` and `DeepGP` take `variational="natural"` (for a
`DeepGP` or `DeepGPConfig`: `hidden_variational` and `output_variational`).
Such a $q(u)$ is parameterised by its natural parameters and trained by natural
gradient descent (Salimbeni, Eleftheriadis & Hensman, 2018). `fit(...,
natgrad_lr=0.1)` steps those parameters with GPyTorch's `NGD` and every other
parameter with Adam. `fit` raises `ValueError` for a natural $q(u)$ without
`natgrad_lr`, because other optimisers do not follow the natural gradient.

On Snelson with an SVGP, after 20 steps the negative ELBO per point was about
0.98 with NGD (step sizes from 0.1 to 1) and 1.50 with Adam alone. After 300
steps both were at about 0.48. In deep GPs the effect depends on the data. On
Snelson, after 500 steps, NGD in every layer with step sizes of 0.05 or more
gave the lowest loss. On a 4-D synthetic problem, NGD on the output layer or on
every layer ended with a higher loss and test error than Adam alone, after 500
and 1500 steps, for step sizes from 0.01 to 0.5.

### Objectives

For `SVGP` and `DeepGP`, `fit` maximises the evidence lower bound (ELBO)
divided by $N$:

```math
\mathcal{L} = \frac{1}{N} \sum_{n=1}^{N} \mathbb{E}_{q}\left[\log p(y_n \mid f_n)\right] - \frac{\beta}{N}\, \mathrm{KL}\left[q(u) \,\|\, p(u)\right]
```

The `beta` argument sets $\beta$ and defaults to 1. In a `DeepGP` the
expectation is taken over all layers and estimated with $S$ = `num_samples`
Monte Carlo samples (Salimbeni & Deisenroth, 2017). Its KL term is the sum over
all layers.

For `ExactGP`, `fit` maximises the log marginal likelihood divided by the
number of targets, $\frac{1}{NT} \log p(Y \mid X)$. The arguments `beta` and
`num_samples` have no effect.

### Training data of an ExactGP

The exact posterior depends on every training point, so an `ExactGP` stores its
training data. `fit(model, X, Y)` replaces the stored data with `X` and `Y`.

### Target shape

`fit` raises `ValueError` when `Y.shape` differs from the event shape of
`model(X)`. Targets have shape `(N,)` for one output and `(N, T)` for $T$
outputs. Without this check, an `(N,)` target broadcasts against `(N, T)`
outputs when $N = T$, and `fit` optimises the wrong objective.

## Prediction

`predict(model, X)` returns the predictive mean and variance of $y$ at `X`.
With `add_noise=False` it returns the mean and variance of the latent function
$f$. Both tensors have shape `(N,)` for one output and `(N, T)` for $T$
outputs. `predict` restores the train or eval mode the model had before the
call.

### Deep GP predictive distribution

The predictive distribution of a `DeepGP` is a mixture of $S$ Gaussians, one
per sample of the hidden layers. The argument `k` sets $S$ and defaults to 50.
Let $m_s$ and $v_s$ be the mean and variance of component $s$. By the law of
total variance, `predict` returns

```math
\mu = \frac{1}{S} \sum_{s=1}^{S} m_s, \qquad
\sigma^2 = \frac{1}{S} \sum_{s=1}^{S} v_s + \frac{1}{S} \sum_{s=1}^{S} (m_s - \mu)^2 .
```

The variance of a single component omits the second term and underestimates
$\sigma^2$.

## Deep GP architecture

### Layers

`DeepGP(dims)` takes the input dimension `dims[0]` followed by the widths of
the hidden layers. An output layer maps the last hidden layer to the outputs.
For example, `DeepGP([1, 1])` has one hidden layer of width 1 and one output
layer.

`DeepGPHiddenLayer` holds the inducing points, $q(u)$, mean and kernel of one
layer. A layer of width $D$ contains $D$ independent GPs. GPyTorch's `DeepGP`
base class finds the layers through `model.modules()` and adds their KL
divergences to the ELBO.

### Hidden-layer means

Compositions of zero-mean GPs lose degrees of freedom as depth increases
(Duvenaud et al., 2014). Hidden layers therefore use linear mean functions, as
in Salimbeni & Deisenroth (2017). Each hidden layer is then a GP perturbation of
a linear map of its input. `DeepGP` uses `hidden_mean_type="linear"` and
`output_mean_type="constant"` by default.

### Builder initialisation

`build_deep_gp(X, num_layers, config)` builds a `DeepGP` with `num_layers - 1`
hidden layers and one output layer. It initialises the model from the training
inputs `X`:

- inducing points of the first layer at the k-means centroids of `X`,
- identity mean functions in the hidden layers,
- the Cholesky factor of each hidden layer's $q(u)$ covariance multiplied by
  `inner_layer_qsqrt_factor`,
- the observation noise variance set to `likelihood_noise`.

At this initialisation a hidden layer's output is close to its mean function.
Each later layer's inducing points are therefore the previous layer's inducing
points mapped through the previous layer's mean. Identity means leave them
unchanged, so by default every layer starts at the centroids of `X`.

A small `inner_layer_qsqrt_factor`, such as `1e-5`, reduces the sampling noise
of the hidden layers at the start of training.

`DeepGPConfig` holds these settings:

| Field | Meaning | Default |
|---|---|---|
| `num_inducing` | inducing points per layer, at most $N$ | required |
| `inner_layer_qsqrt_factor` | factor for the Cholesky factor of hidden-layer $q(u)$ covariances | required |
| `likelihood_noise` | initial noise variance of every output | required |
| `hidden_dims` | one width per hidden layer, so `num_layers - 1` of them; `None` for the input width | `None` |
| `whiten` | whitened parameterisation of $q(u)$ | `True` |
| `mean_type` | hidden-layer mean, `"identity"`, `"linear"`, `"constant"` or `"zero"` | `"identity"` |
| `inducing_init` | `"kmeans"` or `"subset"` of the training inputs | `"kmeans"` |
| `hidden_variational`, `output_variational` | $q(u)$ parameterisation of the hidden and output layers, `"cholesky"` or `"natural"` (see [Natural gradients](#natural-gradients)) | `"cholesky"` |

With `mean_type="identity"`, a hidden layer narrower than its input uses a
linear mean initialised from the principal directions of its input. A layer
wider than its input keeps the identity on its first outputs and starts the
others at zero, as in the reference implementation of Salimbeni & Deisenroth
(2017). A hidden layer that changes width needs `mean_type="identity"` or
`"linear"`. With `"constant"` or `"zero"` its outputs start as constants, and
the builder raises `ValueError`.

## Multiple outputs

With `num_outputs=T`, the $T$ outputs are modelled by $T$ independent GPs. In a
`DeepGP` these GPs form the output layer. Each GP has its own mean and kernel
hyperparameters. In `SVGP` and `DeepGP`, each GP also has its own inducing
points and $q(u)$.

`ExactGP` and `SVGP` share no parameters between outputs. Their log marginal
likelihood and ELBO are sums of $T$ single-output terms.

In a `DeepGP` all outputs share the hidden layers. Each output GP takes the
output of the last hidden layer as its input. A hidden layer of width 1
therefore passes every output through one scalar feature.

`predict` returns the marginal mean and variance of each output. Under a
`DeepGP` the outputs are dependent. Within each mixture component, all $T$
means are computed from the same hidden-layer sample. `predict` does not return
the covariance between outputs.

### Noise variances

The likelihood has one noise variance $\sigma_t^2$ per output. By default,
GPyTorch's `MultitaskGaussianLikelihood` also adds a shared noise variance
$\sigma^2$, so output $t$ has noise variance $\sigma_t^2 + \sigma^2$. The data
identify only these $T$ sums, not all $T + 1$ parameters. The multi-output
likelihood in deepgp-torch omits $\sigma^2$, so `likelihood.task_noises[t]` is
$\sigma_t^2$.

## References

- Duvenaud, Rippel, Adams & Ghahramani, *Avoiding Pathologies in Very Deep
  Networks*, AISTATS 2014.
- Salimbeni & Deisenroth, *Doubly Stochastic Variational Inference for Deep
  Gaussian Processes*, NeurIPS 2017.
- Salimbeni, Eleftheriadis & Hensman, *Natural Gradients in Practice:
  Non-Conjugate Variational Inference in Gaussian Process Models*, AISTATS 2018.
