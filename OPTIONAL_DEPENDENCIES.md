# Optional Dependencies

## Overview

As of v2.0.1, `batch-llm` has made both **PydanticAI** and **google-genai** optional dependencies. This allows users to:
- Use only the core framework with their own LLM integration
- Use direct Gemini API calls without PydanticAI
- Use PydanticAI agents without the Gemini SDK
- Mix and match based on their needs

**Core dependency**: Only `pydantic>=2.0.0` is required.

## Installation Options

### Minimal Installation (Core only)

```bash
uv add batch-llm
# or
pip install batch-llm
```

This installs only the core dependencies:
- `pydantic>=2.0.0`

**Use case**: Direct API calls with any LLM provider (OpenAI, Anthropic, Cohere, etc.)

**What works**:
- ✅ Direct call mode
- ✅ All retry/rate-limit logic
- ✅ Middleware and observers
- ✅ GeminiErrorClassifier (falls back to string-based detection)

**What doesn't work**:
- ❌ Agent mode (requires pydantic-ai)
- ❌ Agent factory mode (requires pydantic-ai)
- ❌ Gemini-specific error detection (requires google-genai)

### With PydanticAI Support

```bash
uv add batch-llm[pydantic-ai]
# or
pip install batch-llm[pydantic-ai]
```

**Use case**: Using PydanticAI agents (agent mode or agent_factory mode)

### With Gemini SDK

```bash
uv add batch-llm[gemini]
# or
pip install batch-llm[gemini]
```

**Use case**: Direct Gemini API calls (see `examples/example_gemini_direct.py`)

### With Everything

```bash
uv add batch-llm[all]
# or
pip install batch-llm[all]
```

Includes:
- PydanticAI support
- Gemini SDK support

### For Development

```bash
uv add batch-llm[dev]
# or
pip install batch-llm[dev]
```

Includes:
- All dependencies above
- pytest, pytest-asyncio
- ruff, mypy

## Usage by Mode

### 1. Agent Mode (requires `pydantic-ai`)

```python
from pydantic_ai import Agent
from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig

agent = Agent('openai:gpt-4', result_type=MyOutput)

work_item = LLMWorkItem(
    item_id="item_1",
    agent=agent,
    prompt="Your prompt here",
)
```

**Required**: `batch-llm[pydantic-ai]`

### 2. Agent Factory Mode (requires `pydantic-ai`)

```python
from pydantic_ai import Agent

def create_agent(attempt: int) -> Agent:
    temps = [0.0, 0.25, 0.5]
    temp = temps[min(attempt - 1, len(temps) - 1)]
    return Agent('openai:gpt-4', result_type=MyOutput, temperature=temp)

work_item = LLMWorkItem(
    item_id="item_1",
    agent_factory=create_agent,
    prompt="Your prompt here",
)
```

**Required**: `batch-llm[pydantic-ai]`

### 3. Direct Call Mode (no extra dependencies)

```python
from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig

async def my_llm_call(attempt: int, timeout: float) -> tuple[MyOutput, dict]:
    # Your custom LLM API call here
    output = await call_your_llm_api(...)
    token_usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    return output, token_usage

work_item = LLMWorkItem(
    item_id="item_1",
    direct_call=my_llm_call,
)
```

**Required**: Only `batch-llm` (no extras needed)

## Examples

### Direct Gemini API Call (no PydanticAI)

See `examples/example_gemini_direct.py` for a complete example of using the Gemini API directly with:
- Progressive temperature based on retry attempt
- Custom response schema validation with Pydantic
- Token usage tracking
- No PydanticAI dependency required

**Full documentation**: [docs/GEMINI_INTEGRATION.md](docs/GEMINI_INTEGRATION.md)

Install:
```bash
uv add batch-llm[gemini]
export GOOGLE_API_KEY=your_api_key
uv run python examples/example_gemini_direct.py
```

### PydanticAI Agent

See `examples/example.py` for using PydanticAI agents.

Install:
```bash
uv add batch-llm[pydantic-ai]
```

## Migration Guide

If you were using v2.0.0 and want to upgrade:

### No changes needed if using PydanticAI

If you're using agent or agent_factory modes, just update your dependencies:

```bash
# Before (v2.0.0)
uv add batch-llm

# After (v2.0.1)
uv add batch-llm[pydantic-ai]  # Now explicit
```

### Switch to direct_call mode

If you want to avoid the PydanticAI dependency:

```python
# Before (v2.0.0)
from pydantic_ai import Agent

agent = Agent('gemini-2.0-flash-exp', result_type=MyOutput)
work_item = LLMWorkItem(agent=agent, prompt="...")

# After (v2.0.1)
async def call_gemini(attempt: int, timeout: float):
    from google import genai
    client = genai.Client()
    response = await client.aio.models.generate_content(...)
    return parsed_output, token_usage

work_item = LLMWorkItem(direct_call=call_gemini)
```

## Benefits of Optional Dependencies

1. **Smaller install**: Core package is lighter without PydanticAI
2. **Flexibility**: Choose your LLM integration method
3. **Direct API control**: Full control over API calls, temperature, retries
4. **Provider agnostic**: Works with any LLM provider

## Technical Implementation

The `Agent` type is conditionally imported:

```python
# In src/batch_llm/base.py
try:
    from pydantic_ai import Agent
except ImportError:
    Agent = Any  # Fallback for type hints
```

This allows the package to:
- Work without PydanticAI installed
- Maintain type hints for those who do install it
- Fail gracefully with clear errors if you try to use agent mode without installing pydantic-ai

## Recommendation

- **For new projects**: Use `direct_call` mode for maximum control and minimal dependencies
- **For existing PydanticAI users**: Install with `[pydantic-ai]` extra
- **For quick prototypes**: Use `[all]` to get everything
