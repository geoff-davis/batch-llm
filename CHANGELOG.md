# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - TBD

### ⚠️ Breaking Changes

This major release introduces the **LLM Call Strategy Pattern**, providing a flexible, provider-agnostic architecture for batch LLM processing.

#### Removed Parameters

- **`agent=` parameter removed** from `LLMWorkItem` - Use `strategy=` instead
- **`client=` parameter removed** from `LLMWorkItem` - Use `strategy=` instead

#### Migration Required

All code using `LLMWorkItem` must be updated:

```python
# ❌ Old (v0.0.x) - No longer works
work_item = LLMWorkItem(
    item_id="item_1",
    agent=agent,  # or client=client
    prompt="Test prompt",
)

# ✅ New (v0.1) - Use strategy
from batch_llm import PydanticAIStrategy

strategy = PydanticAIStrategy(agent=agent)
work_item = LLMWorkItem(
    item_id="item_1",
    strategy=strategy,
    prompt="Test prompt",
)
```

See **[Migration Guide](docs/MIGRATION_V3.md)** for complete upgrade instructions.

---

### Added

#### Core Features

- **`LLMCallStrategy` abstract base class** - Universal interface for any LLM provider
  - `prepare()` - Initialize resources before processing
  - `execute()` - Execute LLM call with retry support
  - `on_error()` - Handle errors and adjust retry behavior (new in this release)
  - `cleanup()` - Clean up resources after processing
- **`on_error()` callback for intelligent retry strategies**
  - Called automatically when `execute()` raises an exception
  - Enables error-type-aware retry logic (validation vs. network vs. rate limit errors)
  - Allows state tracking across retry attempts
  - Use cases:
    - **Smart model escalation**: Only escalate to expensive models on validation errors, not network errors
    - **Smart retry prompts**: Build better retry prompts based on which fields failed validation
    - **Error tracking**: Distinguish and count different error types
  - Non-breaking: Default no-op implementation
  - Framework catches and logs exceptions in `on_error()` to prevent crashes

#### Built-in Strategies

- **`PydanticAIStrategy`** - Wraps PydanticAI agents for batch processing
- **`GeminiStrategy`** - Direct Google Gemini API calls without caching
- **`GeminiCachedStrategy`** - Gemini API calls with automatic context caching
  - Automatic cache creation on `prepare()`
  - Automatic cache TTL refresh during processing
  - Automatic cache deletion on `cleanup()`
  - Configurable cache TTL and refresh threshold

#### Documentation

- **`docs/API.md`** - Complete API reference documentation
  - Added `TokenUsage` TypedDict documentation
  - Added `FrameworkTimeoutError` exception documentation
  - Documented `LLMCallStrategy.dry_run()` method
  - **Documented `LLMCallStrategy.on_error()` callback** with 3 complete use case examples
    - Smart model escalation (validation errors only)
    - Smart retry with partial parsing
    - Error type tracking
  - Updated strategy lifecycle description to include `on_error` call sequence
  - Updated `ErrorInfo` field documentation (error_category, is_timeout)
  - Added missing `ProcessorConfig.progress_callback_timeout` field
  - Updated all code examples to use `TokenUsage`
- **`docs/MIGRATION_V3.md`** - Comprehensive v0.0.x → v0.1 migration guide
- **`README.md`** - Comprehensive improvements
  - Added complete table of contents with 40+ section links
  - Added **Configuration Reference** section (200+ lines)
  - Added **Best Practices** section (120+ lines)
  - Added **Troubleshooting** section (180+ lines)
  - Added **FAQ** section (180+ lines) with 15+ Q&A
  - **Updated Strategy Pattern section** to include `on_error` method
  - **Updated Smart Retry section** to demonstrate `on_error` callback usage
  - **Updated Model Escalation section** to show smart escalation with `on_error`
  - **Updated FAQ** with `on_error` callback approach for adaptive prompts
  - Enhanced Middleware & Observers documentation
  - Improved Testing section with 3 approaches
  - Updated all code examples to use `TokenUsage` TypedDict
  - Fixed mutable default argument in progressive temperature example
- **`examples/example_openai.py`** - OpenAI integration examples
- **`examples/example_anthropic.py`** - Anthropic Claude integration examples
- **`examples/example_langchain.py`** - LangChain integration examples (including RAG)
- **`examples/example_llm_strategies.py`** - All built-in strategies with examples
- **`examples/example_smart_model_escalation.py`** - Smart model escalation using `on_error` callback
  - Only escalates to expensive models on validation errors
  - Retries with same cheap model on network/rate limit errors
  - Demonstrates 60-80% cost savings vs. always using best model
  - Includes comparison with blind escalation strategy
- **`examples/example_gemini_smart_retry.py`** - Enhanced with `on_error` callback documentation
  - Shows how to use `on_error` to track validation errors cleanly
  - Demonstrates building targeted retry prompts based on which fields failed
- **All example files** - Updated to use `TokenUsage` TypedDict consistently

#### Testing

- **`batch_llm.testing.MockAgent`** - Mock agent for testing without API calls
- Comprehensive test coverage for all strategies
- Strategy lifecycle tests (prepare/execute/cleanup)
- **New `on_error` callback tests** (4 comprehensive tests):
  - `test_on_error_callback_called` - Verifies callback is invoked with correct parameters
  - `test_on_error_callback_with_state` - Tests state tracking across retries (validation vs. network errors)
  - `test_on_error_callback_exception_handling` - Ensures buggy callbacks don't crash processor
  - `test_on_error_not_called_on_success` - Confirms callback only runs on errors

---

### Changed

#### Architecture

- **Strategy pattern** replaces direct agent/client parameters
  - Cleaner separation of concerns
  - Framework handles timeout enforcement at top level
  - Strategies no longer need `asyncio.wait_for()` wrappers
- **Improved timeout enforcement** - Framework-level with `asyncio.wait_for()`
  - Consistent behavior across all strategies
  - Timeout parameter still passed to `execute()` for informational purposes
  - Removed redundant timeout wrappers from built-in strategies
- **Enhanced strategy execution lifecycle** to include error callback
  - Framework now calls `strategy.on_error(exception, attempt)` when `execute()` raises
  - Error callback invoked before retry logic, allowing strategies to adjust behavior
  - Exceptions in `on_error()` are caught and logged (won't crash processing)
  - Type guard added for mypy compliance

#### Type System

- **`LLMWorkItem` now accepts `strategy=`** instead of `agent=` or `client=`
- Generic type parameters preserved: `LLMWorkItem[TInput, TOutput, TContext]`
- Better type safety with strategy pattern

#### Internal

- Refactored `_process_work_item_direct()` to use strategy lifecycle
- Improved error handling in strategy execution
- Better resource cleanup with context managers

---

### Fixed

- **Timeout enforcement bug** - Custom strategies now respect timeouts consistently
  - Framework wraps all `strategy.execute()` calls in `asyncio.wait_for()`
  - Previously, custom strategies could ignore timeout parameter
  - All 61 tests now pass (was 60 passing, 1 skipped)
- **Test coverage** - Fixed `test_custom_strategy_timeout_handling` (previously skipped)
- **Error classification logic bug handling** - Error classifiers now properly distinguish logic bugs from transient failures
  - `DefaultErrorClassifier` and `GeminiErrorClassifier` now explicitly check for logic bug exceptions (`ValueError`, `TypeError`, `AttributeError`, etc.)
  - Logic bugs are marked as non-retryable to avoid wasting retry attempts and tokens on deterministic failures
  - Pydantic `ValidationError` is explicitly marked as retryable (LLM might generate valid output on retry)
  - Generic `Exception` instances remain retryable (allows custom transient errors and test mocks)
  - Prevents wasting `max_attempts` retries on programming errors that won't be fixed by retrying
  - Added regression test `test_logic_bugs_fail_fast` to ensure logic bugs fail after 1 attempt
  - Fixed `test_token_usage_tracked_across_retries` exception chain construction

---

### Migration Path

**Upgrading from v0.0.x?** Follow these steps:

1. **Read the Migration Guide**: `docs/MIGRATION_V3.md`
2. **Update imports**: Add `PydanticAIStrategy`, `GeminiStrategy`, or `GeminiCachedStrategy`
3. **Wrap your agents/clients**: Create strategy instances
4. **Update LLMWorkItem**: Replace `agent=` or `client=` with `strategy=`
5. **Test thoroughly**: Verify timeout and retry behavior

**Estimated migration time**: 15-60 minutes for most codebases

**Benefits**:

- ✅ Support for any LLM provider (OpenAI, Anthropic, LangChain, etc.)
- ✅ Better caching with automatic lifecycle management
- ✅ More reliable timeout enforcement
- ✅ Cleaner, more maintainable code
- ✅ Easy to create custom strategies

---

## [0.0.2] - 2025-10-20

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

## [0.0.1] - 2025-10-19

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
- Migration guide for v0.0.0 → v0.0.1
- Updated README with installation options

## [0.0.0] - 2025-10-19

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

## [0.0.0-alpha] - Internal

### Added

- Initial implementation for internal use
- Basic parallel processing
- PydanticAI integration
- Work item and result models
