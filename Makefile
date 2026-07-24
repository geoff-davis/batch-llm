.PHONY: help test test-ci coverage lint typecheck typecheck-ty format check-all pre-commit clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

test:  ## Run all tests
	uv run pytest tests/ -v

test-fast:  ## Run tests excluding slow tests (default)
	uv run pytest tests/ -v -m 'not slow'

test-all:  ## Run all tests including slow ones
	uv run pytest tests/ -v -m ''

test-ci:  ## Run CI test suite with coverage reporting
	uv run pytest tests/ -v --tb=short --cov --cov-report=term-missing --cov-report=xml

coverage:  ## Run tests with coverage report
	uv run pytest --cov=async_batch_llm --cov-report=term-missing --cov-report=html

coverage-report:  ## Generate and open HTML coverage report
	uv run pytest --cov=async_batch_llm --cov-report=html
	@echo "\n==> Opening coverage report..."
	open htmlcov/index.html || xdg-open htmlcov/index.html || echo "Please open htmlcov/index.html manually"

lint:  ## Run ruff linter
	uv run ruff check src/ tests/

lint-fix:  ## Run ruff linter with auto-fix
	uv run ruff check src/ tests/ --fix

format:  ## Format code with ruff
	uv run ruff format src/ tests/

typecheck:  ## Run mypy type checker
	uv run mypy src/async_batch_llm/ --ignore-missing-imports

typecheck-ty:  ## Run ty type checker
	uv run ty check src/

markdown-lint:  ## Check markdown files
	npx markdownlint-cli2 "README.md" "docs/*.md" "CLAUDE.md"

markdown-lint-fix:  ## Fix markdown issues
	npx markdownlint-cli2 "README.md" "docs/*.md" "CLAUDE.md" --fix

check-all:  ## Run all checks (lint + typecheck + test)
	@echo "==> Running linter..."
	@$(MAKE) lint
	@echo "\n==> Running mypy..."
	@$(MAKE) typecheck
	@echo "\n==> Running ty..."
	@$(MAKE) typecheck-ty
	@echo "\n==> Running tests..."
	@$(MAKE) test-fast
	@echo "\n==> All checks passed! ✓"

ci:  ## Run local CI checks (lint + mypy + ty + coverage tests + markdown)
	@echo "==> Running linter..."
	@$(MAKE) lint
	@echo "\n==> Running mypy..."
	@$(MAKE) typecheck
	@echo "\n==> Running ty..."
	@$(MAKE) typecheck-ty
	@echo "\n==> Running tests with coverage..."
	@$(MAKE) test-ci
	@echo "\n==> Running markdown linter..."
	@$(MAKE) markdown-lint
	@echo "\n==> CI checks passed! ✓"

pre-commit-install:  ## Install git hooks via prek
	uv run prek install
	@echo "\n==> Git hooks installed! They will run automatically on 'git commit'"

pre-commit-run:  ## Run hooks on all files via prek
	uv run prek run --all-files

pre-commit-update:  ## Update hooks to latest versions via prek
	uv run prek autoupdate

scale-smoke:  ## Run the reduced deterministic scale-soak profile (CI-sized)
	uv run python -m benchmarks.scale_soak --profile ci --output benchmark-results/scale-ci.json

scale-100k:  ## Run the 100k reference scale-soak profile (minutes)
	uv run python -m benchmarks.scale_soak --profile 100k --output benchmark-results/scale-100k.json

scale-1m:  ## Run the 1m scale-soak profile, large scenarios only (long)
	uv run python -m benchmarks.scale_soak --profile 1m --output benchmark-results/scale-1m.json

clean:  ## Clean up cache files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete

.DEFAULT_GOAL := help
