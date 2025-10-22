# Contributing to Batch LLM

Thank you for your interest in contributing to Batch LLM! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/batch-llm.git
cd batch-llm
```

2. **Install uv** (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. **Install dependencies**

```bash
uv sync --all-extras
```

4. **Install markdown lint tooling** (requires Node 18+)

```bash
npm install --save-dev markdownlint-cli2
```

5. **Run tests**

```bash
uv run pytest
```

## Development Workflow

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=batch_llm --cov-report=html

# Run specific test file
uv run pytest tests/test_basic.py

# Run with verbose output
uv run pytest -v
```

### Code Quality

We use several tools to maintain code quality:

```bash
# Format code
uv run ruff format src/ tests/ examples/

# Lint code
uv run ruff check src/ tests/ examples/

# Type check
uv run mypy src/batch_llm/

# Markdown lint (requires npm install --save-dev markdownlint-cli2)
npx markdownlint-cli2 "README.md" "docs/*.md" "CLAUDE.md"
```

### Running Examples

```bash
# Run the main example (requires API key)
uv run python examples/example.py

# Set API key first if needed
export GEMINI_API_KEY="your-api-key"
```

## Making Changes

1. **Create a branch** for your changes

```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes** following these guidelines:
   - Write clear, descriptive commit messages
   - Add tests for new functionality
   - Update documentation as needed
   - Follow existing code style

3. **Test your changes**

```bash
uv run pytest
uv run ruff check src/
uv run mypy src/batch_llm/
```

4. **Submit a pull request**
   - Describe what your changes do
   - Reference any related issues
   - Ensure all tests pass

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Write docstrings for public APIs
- Keep functions focused and concise
- Use descriptive variable names

## Testing Guidelines

- Write tests for all new features
- Aim for high test coverage
- Use `MockAgent` for tests that don't require API calls
- Test both success and failure cases
- Test edge cases and error handling

## Documentation

- Update README.md for user-facing changes
- Add docstrings to new classes and functions
- Update examples/ if adding new features
- Update CHANGELOG.md following Keep a Changelog format

## Release Process

(For maintainers)

1. Update version in `src/batch_llm/__init__.py`
2. Update CHANGELOG.md with release notes
3. Create a git tag: `git tag v2.0.1`
4. Build: `uv build`
5. Publish to PyPI: `uv publish`

## Questions?

Feel free to open an issue for:

- Bug reports
- Feature requests
- Questions about development
- Documentation improvements

Thank you for contributing!
