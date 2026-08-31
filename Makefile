.PHONY: install format check lint type test cov dist clean

install:  ## Editable install with dev tooling
	pip install -e ".[dev]"

format:  ## Auto-format with black + isort
	black src tests
	isort src tests

lint:  ## flake8 style checks
	flake8 src tests

type:  ## mypy type checks
	mypy src

check: ## All CI checks (no auto-fix): format-check + lint + type
	black --check src tests
	isort --check-only src tests
	flake8 src tests
	mypy src

test:  ## Run the test suite
	pytest

cov:  ## Run tests with coverage (fails under 90%)
	pytest --cov=deepgp --cov-report=term-missing --cov-fail-under=90

dist:  ## Build sdist + wheel into dist/ and validate the metadata
	rm -rf dist
	python -m build
	twine check --strict dist/*

clean:  ## Remove caches and build artifacts
	rm -rf build dist *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov
