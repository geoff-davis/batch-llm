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
  - `cleanup()` - Clean up resources after processing

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
  - Enhanced Middleware & Observers documentation
  - Improved Testing section with 3 approaches
  - Updated all code examples to use `TokenUsage` TypedDict
  - Fixed mutable default argument in progressive temperature example
- **`examples/example_openai.py`** - OpenAI integration examples
- **`examples/example_anthropic.py`** - Anthropic Claude integration examples
- **`examples/example_langchain.py`** - LangChain integration examples (including RAG)
- **`examples/example_llm_strategies.py`** - All built-in strategies with examples
- **All example files** - Updated to use `TokenUsage` TypedDict consistently

#### Testing
- **`batch_llm.testing.MockAgent`** - Mock agent for testing without API calls
- Comprehensive test coverage for all strategies
- Strategy lifecycle tests (prepare/execute/cleanup)

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
