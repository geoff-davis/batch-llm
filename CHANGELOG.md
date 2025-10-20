# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.2] - 2025-10-20

### Fixed
- Fixed crash when middleware's `before_process()` returns `None` - now properly preserves original `item_id`
- Fixed stats race condition where re-queued rate-limited items inflated the total count
- Improved token usage extraction robustness with multiple fallback strategies for different LLM providers
- Fixed all linting issues (20 total: 2 in source code, 18 in tests)

### Added
- `max_queue_size` configuration option to prevent memory issues with large batches (default: 0 = unlimited)
- 3 new tests for edge cases and bug fixes
- `docs/internal/` directory for development documentation (gitignored)

### Changed
- Token extraction now uses a robust helper method with 3 fallback strategies
- Moved 16 internal documentation files to `docs/internal/` for cleaner repository
- Updated `CLAUDE.md` with ruff workflow reminder
- **BREAKING**: Removed unused `batch_size` parameter from `BatchProcessor` and `ParallelBatchProcessor`

### Removed
- `batch_size` parameter (was unused and ignored)

### Documentation
- Created comprehensive bug fix documentation in `docs/internal/BUG_FIXES_V2.0.2.md`
- Updated code review with completion status
- Cleaned up repository root (70% reduction in visible markdown files)

## [2.0.1] - 2025-10-19

### Added
- Optional dependencies: `pydantic-ai` and `google-genai` are now optional extras
- Comprehensive Gemini integration guide (`docs/GEMINI_INTEGRATION.md`)
- Working Gemini direct API example (`examples/example_gemini_direct.py`)
- Installation options: `[pydantic-ai]`, `[gemini]`, `[all]`, `[dev]`

### Fixed
- Direct call timeout enforcement - now properly wraps calls in `asyncio.wait_for()`
- Middleware `on_error` now called after retry exhaustion (not just for non-retryable errors)
- Middleware execution order - `after_process` now runs in reverse order (onion pattern)

### Changed
- Core dependency now only `pydantic>=2.0.0` (was also `pydantic-ai` and `google-genai`)
- String annotations for `Agent` type to work without `pydantic-ai` installed

### Documentation
- `OPTIONAL_DEPENDENCIES.md` - Complete installation guide
- Migration guide for v2.0.0 → v2.0.1
- Updated README with installation options

## [2.0.0] - 2025-10-19

### Added
- Initial PyPI package release
- Provider-agnostic error classification system
- Pluggable rate limit strategies (ExponentialBackoff, FixedDelay)
- Middleware pipeline for extensible processing
- Observer pattern for monitoring and metrics
- Configuration-based setup with `ProcessorConfig`
- `GeminiErrorClassifier` for Google Gemini API error handling
- `MetricsObserver` for tracking processing statistics
- `MockAgent` for testing without API calls
- Comprehensive test suite with pytest
- Support for Python 3.10+

### Changed
- Refactored to src-layout for better packaging
- Improved error handling with retryable error detection
- Enhanced documentation with installation instructions
- Updated examples to use new configuration system

### Features
- `ParallelBatchProcessor` - Async parallel LLM request processing
- Work queue management with context passing
- Post-processing hooks for custom logic
- Partial failure handling
- Token usage tracking
- Timeout support per item
- Type-safe with generics support

## [1.0.0] - Internal

### Added
- Initial implementation for internal use
- Basic parallel processing
- PydanticAI integration
- Work item and result models
