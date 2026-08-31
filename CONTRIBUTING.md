# Contributing to deepgp-torch

Thanks for your interest in contributing! This project welcomes bug reports,
feature requests, documentation improvements, and code.

## Getting started

```bash
git clone https://github.com/bydeng01/deepgp-torch
cd deepgp-torch
python -m venv .venv && source .venv/bin/activate
make install            # editable install with dev tooling  (pip install -e ".[dev]")
pre-commit install      # run black/isort/flake8/mypy on every commit
```

## Development workflow

1. **Create a branch** off `main` (`git switch -c my-feature`). Do not commit
   directly to `main`.
2. **Make focused changes** with tests. New behaviour needs a test; bug fixes
   need a regression test.
3. **Run the checks locally** — CI runs the same ones on Python 3.10/3.11/3.12:

   ```bash
   make format         # auto-fix with black + isort
   make check          # black --check, isort --check-only, flake8, mypy src
   make test           # pytest
   ```

4. **Open a pull request** against `main` and fill in the PR template. Keep PRs
   small and single-purpose; describe the change and how you verified it.

## Coding standards

- **Style:** [black](https://github.com/psf/black) (line length 88) and
  [isort](https://pycqa.github.io/isort/) (`profile = black`). `flake8` must be
  clean; `mypy src` must pass.
- **Types:** add type hints to new public functions.
- **License header:** every source file starts with the full Apache-2.0 license
  header (see any file under `src/deepgp/`). Copy it verbatim into new files.
- **Scope:** `src/deepgp` is the only importable code and must **never** import
  TensorFlow / GPflow / GPflux — those belong to the optional `[parity]` extra
  used only by benchmarks and golden-equivalence tests. Keep the library
  application-agnostic.

## Reporting bugs / requesting features

Open an issue using the templates under
[`.github/ISSUE_TEMPLATE`](.github/ISSUE_TEMPLATE). For bugs, include a minimal
reproducer and your `python` / `torch` / `gpytorch` versions.

## Developer Certificate of Origin (DCO)

By contributing, you certify the [DCO](https://developercertificate.org/) — that
you wrote the change or otherwise have the right to submit it under the project
license. Sign off your commits:

```bash
git commit -s -m "Your message"
```

## License

By contributing, you agree that your contributions are licensed under the
project's [Apache-2.0](LICENSE) license.
