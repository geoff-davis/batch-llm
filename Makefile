.PHONY: help test lint typecheck format check-all clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

test:  ## Run all tests
	uv run pytest tests/ -v

test-fast:  ## Run tests excluding slow tests (default)
	uv run pytest tests/ -v -m 'not slow'

test-all:  ## Run all tests including slow ones
	uv run pytest tests/ -v -m ''

lint:  ## Run ruff linter
	uv run ruff check src/ tests/ examples/

lint-fix:  ## Run ruff linter with auto-fix
	uv run ruff check src/ tests/ examples/ --fix

format:  ## Format code with ruff
	uv run ruff format src/ tests/ examples/

typecheck:  ## Run mypy type checker
	uv run mypy src/batch_llm/ --ignore-missing-imports

markdown-lint:  ## Check markdown files
	npx markdownlint-cli2 "README.md" "docs/*.md" "CLAUDE.md"

markdown-lint-fix:  ## Fix markdown issues
	npx markdownlint-cli2 "README.md" "docs/*.md" "CLAUDE.md" --fix

check-all:  ## Run all checks (lint + typecheck + test)
	@echo "==> Running linter..."
	@$(MAKE) lint
	@echo "\n==> Running type checker..."
	@$(MAKE) typecheck
	@echo "\n==> Running tests..."
	@$(MAKE) test-fast
	@echo "\n==> All checks passed! ✓"

ci:  ## Run CI checks (what GitHub Actions runs)
	@echo "==> Running linter..."
	@$(MAKE) lint
	@echo "\n==> Running type checker..."
	@$(MAKE) typecheck
	@echo "\n==> Running tests..."
	@$(MAKE) test
	@echo "\n==> Running markdown linter..."
	@$(MAKE) markdown-lint
	@echo "\n==> CI checks passed! ✓"

clean:  ## Clean up cache files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

.DEFAULT_GOAL := help
