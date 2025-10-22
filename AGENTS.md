# Repository Guidelines

## Tooling Prerequisites
Python workflows run through `uv`; install it first, then sync the environment with `uv sync`. Markdown linting depends on Node tooling—install Node 18+ and add `markdownlint-cli2` as a dev dependency (`npm install --save-dev markdownlint-cli2`). Run the Make targets via `npx` so the locally pinned binary is used.

## Project Structure & Module Organization
Source lives in `src/batch_llm`, with core orchestration under `core/`, reusable strategy interfaces in `strategies/`, parallel scheduling in `parallel.py`, and middleware/observers grouped by folder. Shared test fixtures are in `src/batch_llm/testing`. End-to-end and regression suites sit in `tests/`, while runnable client snippets are in `examples/`. Reference material, including architecture notes, is in `docs/`. The `Makefile` and `pyproject.toml` define tooling defaults—review them before adjusting project-wide settings.

## Build, Test, and Development Commands
Use `uv` to ensure the pinned virtual environment: `uv sync` installs dependencies. Core workflows are wrapped in make targets: `make lint` (Ruff checks), `make format` (Ruff formatting), `make typecheck` (mypy), and `make test-fast` (pytest excluding `slow`). Run `make check-all` for the standard local gate or `make ci` to mirror GitHub Actions. When debugging a single test, call `uv run pytest tests/test_retry_logic.py -k partial_name`.

## Coding Style & Naming Conventions
Python code targets 3.10 with a Ruff-enforced 100-character soft limit. Prefer type-hinted, dataclass-friendly APIs and keep async flows explicit. Modules and packages use snake_case; concrete strategy classes use PascalCase suffixed with `Strategy` or `Classifier`. Observers and middleware should expose verbs describing side effects (e.g., `LoggingObserver`). Run `make format` and `make lint` before submitting to keep imports sorted and styles consistent.

## Testing Guidelines
Pytest is configured via `pyproject.toml`, discovering files matching `test_*.py` and skipping `@pytest.mark.slow` by default. Add new coverage under `tests/` mirroring the target module path, and prefer descriptive test names like `test_strategy_handles_token_limits`. Integration fixtures live in `src/batch_llm/testing`; reuse rather than duplicating helpers. For scenarios that hit remote APIs, guard them with `slow` or a dedicated marker so they stay opt-in.

## Commit & Pull Request Guidelines
Commits follow a concise, imperative summary (e.g., `Add on_error retry callback`) with focused scope; group related changes and document breaking behavior in `CHANGELOG.md` when relevant. Pull requests should link any tracked issues, outline behavioral changes, list new commands or flags, and include screenshots for UI- or docs-heavy updates where clarity helps reviewers. Confirm `make ci` succeeds locally before requesting review to reduce turnarounds.
