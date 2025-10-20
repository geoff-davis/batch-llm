# Batch LLM Examples

This directory contains example scripts demonstrating how to use the `batch-llm` package.

## Running Examples

### Setup

First, install the package:

```bash
# If developing locally
uv pip install -e ..

# If installed from PyPI
uv pip install batch-llm
```

### Set API Key

Most examples require a Google Gemini API key:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

Get your API key from: https://makersuite.google.com/app/apikey

### Run Examples

```bash
# Run the main example
python example.py
```

## What's Included

### `example.py`

Comprehensive examples demonstrating:

1. **Simple Batch Processing** - Basic parallel processing with multiple workers
2. **Context and Post-Processing** - Using context data and post-processing hooks
3. **Error Handling** - Handling timeouts and failures gracefully
4. **Testing with MockAgent** - Testing without making real API calls

Each example is self-contained and includes detailed comments.

## Example Output

```
================================================================================
EXAMPLE 1: Simple Batch Processing (New API)
================================================================================
Processed 5 items:
  Succeeded: 5
  Failed: 0
  Total tokens: 1,234
  ✓ Pride and Prejudice: Pride and Prejudice - Romance
  ✓ 1984: Nineteen Eighty-Four - Dystopian Fiction
  ...
```

## Tips

- Start with Example 4 (MockAgent) - it doesn't require an API key
- Examples 1-3 require a valid Gemini API key
- Adjust `max_workers` and `timeout_per_item` for your use case
- Check the metrics output to monitor performance
