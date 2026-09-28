# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Multi-output regression:** `SVGP`, `DeepGP` and `build_deep_gp` take
  `num_outputs=T` and train on `(N, T)` targets. The output layer is `T`
  independent GPs, each with its own inducing points, `q(u)`, mean and kernel
  hyper-parameters; in a `DeepGP` all outputs share the hidden layers.
  `predict` returns `(N, T)` per-output marginal means and variances.
- `deepgp.likelihoods.make_likelihood(num_outputs, noise)`, used by every model
  and the builder: a `GaussianLikelihood`, or a `MultitaskGaussianLikelihood`
  with one noise variance per output.
- **Tests:** exact identities against `T` single-output models with the same
  parameters. The `SVGP` ELBO is the sum of the per-output ELBOs; the `DeepGP`
  ELBO is that sum plus `(T - 1) * (beta / N) * KL_hidden`, because the shared
  hidden layers' KL counts once, not `T` times; column `t` of `predict` is
  single-output model `t`; observation noise adds exactly `task_noises[t]` to
  column `t`; and `T = 1` reproduces the single-output model. Integration tests
  train all three entry points on two outputs with different noise levels.

### Changed

- The multitask likelihood no longer has GPyTorch's default shared noise term.
  With it, output `t` had variance `σ_t² + σ²` (`T + 1` parameters, of which
  the data identify only the `T` sums) and started at about 1.39 instead of the
  single-output 0.69. `likelihood.task_noises[t]` is now the noise variance of
  output `t`.

### Fixed

- `fit` raises `ValueError` when `Y.shape` differs from the event shape of
  `model(X)`. A `(N,)` target against a `T`-output model broadcast silently
  when `N == T` and trained on the wrong objective.
- `DeepGP.from_layers` takes the number of outputs from
  `last_layer.output_dims` and raises `ValueError` on a conflicting
  `num_outputs`. A multi-output last layer passed without `num_outputs` used to
  get a single-output `GaussianLikelihood`.

## [0.1.1] - 2026-08-31

### Fixed

- README links to `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`,
  `CITATION.cff` and `LICENSE` are now absolute. As relative paths they resolved
  against `pypi.org` rather than the repository, so all five were dead on the
  PyPI project page.

## [0.1.0] - 2026-08-31

Initial release: the core deep-GP stack plus the architecture builder.

### Added

- **Core deep-GP stack:** `DeepGPHiddenLayer` (Cholesky `q(u)`, whitened/
  unwhitened variational strategy, ARD RBF kernel, linear/constant/zero means),
  the `DeepGP` container, and the single-layer `SVGP`. Caller-supplied inducing
  points are cast to the default dtype, so a float32 `Z` cannot leave a model in
  mixed precision.
- **Training & inference:** `make_mll`
  (`DeepApproximateMLL(VariationalELBO(beta, num_data))`), a full-batch `fit`
  loop with `num_likelihood_samples`, and `predict` using the law-of-total-
  variance mixture reduction. `predict` restores the caller's train/eval mode,
  which is what drops GPyTorch's memoised `K_ZZ` Cholesky factor and `q(u)` — so
  a parameter change between two calls is reflected in full rather than combined
  with a stale cache.
- **Architecture builder:** `build_deep_gp(X, num_layers, config)` +
  `DeepGPConfig` with the stable-start initialisation — KMeans inducing init,
  dim-preserving identity/PCA hidden means, hidden `q_sqrt` shrink, Gaussian
  noise init. Kernel/mean factories and `init_inducing` (kmeans/subset/`z_init`).
- **Data & metrics:** vendored `snelson1d` loader with standardisation (and a
  synthetic fallback), `rmse` and `gaussian_nll`.
- **Utilities:** `set_default_float64` (called on import) and `seed_everything`.
- **Tooling & packaging:** `pyproject.toml` (PEP 621 metadata, PEP 639 license
  expression, `[dev]` and `[parity]` extras), a CI matrix (Python
  3.10/3.11/3.12: black, isort, flake8, mypy, pytest) with a coverage gate, a
  `Release` workflow publishing to PyPI via trusted publishing, a `make dist`
  target, `MANIFEST.in`, `Makefile`, `pre-commit`, `Dockerfile`, and open-source
  community files (contributing guide, code of conduct, security policy,
  issue/PR templates).
- **Tests:** 89 tests at 100% statement and branch coverage — the metric
  formulas, dataset standardisation/reproducibility/synthetic fallback, the
  inducing/mean/kernel factory branches, layer `mean_type` variants with
  `whiten=False` and the `q_sqrt` shrink, `add_noise` semantics and the `k=1`
  NaN guard, model input validation, and builder/trainer edge paths; integration
  smoke tests for SVGP, the 2-layer `DeepGP`, and the builder on `snelson1d`;
  and a guard that `import deepgp` pulls in no TensorFlow/GPflow/GPflux.

### Not yet implemented (placeholder modules)

- Multi-output support; minibatch, scheduler, checkpoint and natural-gradient
  training; the GPflux golden-equivalence test, the UCI benchmark harness, and
  Sphinx docs.

[Unreleased]: https://github.com/bydeng01/deepgp-torch/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/bydeng01/deepgp-torch/releases/tag/v0.1.1
[0.1.0]: https://github.com/bydeng01/deepgp-torch/releases/tag/v0.1.0
