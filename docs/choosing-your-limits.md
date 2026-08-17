# Choosing Your Limits

One page, one pass through every limit that matters, in the order you should
decide them. Each section says what the knob bounds, how to pick a value, and
when to leave it alone. The deep dives — the
[production checklist](production-checklist.md),
[high-throughput guide](openai-high-throughput.md),
[guardrails](guardrails.md), and
[bounded work](bounded-work.md) — expand on each step; you should be able to
size a real run without leaving this page.

## The short version

```python
from async_batch_llm import JsonlArtifactStore, ResumePolicy, llm, process_prompts

strategy = llm("deepseek:deepseek-v4-flash", thinking=False)

batch = await process_prompts(
    strategy,
    prompts,                       # 10,000 items
    concurrency=64,                # step 1 — sizes everything aligned below
    progress=True,
    artifact_store=JsonlArtifactStore("runs/tagging.jsonl"),
    resume=ResumePolicy.REUSE_SUCCESSES,
)
```

`concurrency=N` (v0.20) coherently sizes the worker pool, provider-capacity
admission, and — for built-in models created via `llm()` or `from_api_key()`
without an explicit `max_connections` — the httpx connection pool. If you set
nothing else, the rest of this page is the explanation of what you just got
and when to override it.

## The decision tree

```text
1. concurrency      how many requests in flight at once?
2. connection pool  can the HTTP client actually carry that many?
3. provider concurrency  how many calls may hold provider capacity?
4. quota scope      which strategies spend the same upstream budget?
5. RPM              how many physical attempts may start per minute?
6. TPM + estimator  how much estimated token load may start per minute?
7. startup ramp     should full concurrency arrive gradually?
8. cooldown         what happens when the provider says 429?
9. timeouts         how long may an attempt, item, and batch take?
```

### 1. Concurrency — `concurrency=N`

Pick the number of simultaneous provider requests. This is a property of the
**provider**, not your CPU — LLM calls are I/O-bound, so never use
`cpu_count()`.

- Rate-limited endpoints (OpenAI/Gemini tiers): start at **5–10**.
- High-concurrency providers (DeepSeek, self-hosted vLLM): **50–200**.
- If the provider publishes an RPM limit, a serviceable estimate is
  `concurrency ≈ RPM / 60 × typical_latency_seconds`, rounded down.

Prefer the single knob over setting `max_workers`,
`max_provider_concurrency`, and `max_connections` separately — misalignment
between those three is the most common performance bug in this package.
Explicit values for any individual knob still win when you set them.

### 2. Connection pool — `max_connections`

The openai SDK's default httpx pool holds **~100 connections**. Workers above
that number don't fail — they silently queue inside httpx where no timeout is
running, which caps throughput invisibly. **This is the classic DeepSeek
footgun**: DeepSeek happily accepts hundreds of concurrent requests, so
`max_workers=150` against a default pool gives you exactly 100-wide
throughput and 50 workers waiting in the transport.

- With `concurrency=N` and a factory-built model (`llm("...")` or
  `from_api_key()` without `max_connections`), the pool is resized to `N`
  before the first request — nothing to do.
- If you build your own `AsyncOpenAI`/httpx client, size its
  `httpx.Limits(max_connections=..., max_keepalive_connections=...)` to at
  least your concurrency yourself; the framework cannot introspect or resize
  a caller-supplied client.
- An explicit `max_connections` smaller than `concurrency` is treated as a
  real contradiction and warns.

Details: [high-throughput guide](openai-high-throughput.md).

### 3. Provider concurrency — `max_provider_concurrency`

Admission caps how many attempts may hold a provider "slot" at once, and the
wait happens **before** `attempt_timeout` starts — so a queued attempt can't
burn its execution timeout waiting for capacity. With `concurrency=N` this is
already `N`; set it explicitly only to run the framework wider than the
provider (e.g. many cheap local workers feeding a narrow paid API), or when a
strategy advertises its own `max_concurrency` (the lower limit applies).

### 4. Quota scope — `strategy.quota_scope`

Identify which strategies spend the same provider/account RPM and TPM budget.
The default follows `concurrency_scope`. Share one stable object when multiple
models use one account quota; use distinct objects only for genuinely
independent budgets. Quota scope also owns coordinated cooldown, while
concurrency scope owns provider/client capacity. See
[Token-Aware Admission](token-aware-admission.md#quota-scopes).

### 5. Requests per minute — `max_requests_per_minute`

Set the provider/account's physical-attempt budget, with safety margin for
other applications using the same account. Admission is FIFO within each quota
scope. Every retry is another physical request and gets a fresh reservation. A
429 still triggers reactive coordinated cooldown; RPM smoothing cannot
reproduce every provider window exactly.

### 6. Tokens per minute — `max_tokens_per_minute`

TPM requires an estimator. Reserve estimated input plus expected output, not
only prompt tokens:

```python
from async_batch_llm import CharacterTokenEstimator, ProcessorConfig

config = ProcessorConfig(
    concurrency=16,
    max_requests_per_minute=500,
    max_tokens_per_minute=200_000,
    token_estimator=CharacterTokenEstimator(expected_output_tokens=300),
)
```

Prefer a provider tokenizer over the character heuristic. Review actual
refunds, underestimation debt, and unknown-usage attempts, then tune the output
allowance. One estimate must fit inside the configured bucket. Increasing
workers cannot overcome RPM/TPM and can increase task and queue pressure while
work waits.

### 7. Startup ramp — `startup_ramp`

Opening at full concurrency against a cold endpoint can trip instant 429s.
`StartupRampConfig(initial_concurrency=4, concurrency_step=4,
ramp_interval_seconds=2.0)` walks up to full width instead. Skip it for small
runs or generous providers; reach for it when the first minute of a large run
keeps tripping cooldowns.

### 8. Cooldown — `RateLimitConfig`

What happens on 429: one worker triggers a shared cooldown, everyone pauses,
then traffic resumes through a slow-start ramp. Defaults are conservative
(300s cooldown). For providers with short quota windows, start with
`RateLimitConfig(cooldown_seconds=30.0, max_cooldown_seconds=300.0)` — the
classifier honors a server-sent `Retry-After` as a floor regardless. If you
see repeated consecutive cooldowns, your `concurrency` (step 1) is too high;
fix the cause, not the cooldown. Cooldown precedes estimation and reservation,
so paused work does not consume live quota state early.

### 9. Attempt, item, and batch timeouts

`attempt_timeout` bounds one `execute()` call (default 120s), not its quota or
provider-capacity wait. Size it for one slow response: p99 provider latency
plus margin. Long-output or reasoning models may need more than the common 30s
starting point. `timeout_per_item` is its deprecated pre-v0.20 alias.

`GuardrailConfig.total_timeout_per_item` bounds one logical item end to end:
cooldown, estimation, RPM/TPM wait, provider admission, every call, and retry
backoff. `GuardrailConfig.batch_timeout` bounds the whole run. On expiry,
accepted items receive terminal results and checkpoints clean up according to
the configured abort mode. Size the batch deadline below the job scheduler's
hard kill and pair it with fail-fast categories such as authentication and
insufficient balance. Details: [guardrails](guardrails.md).

## Worked example: 10k items against a rate-limited provider

Target: 10,000 classification prompts against an OpenAI-tier endpoint
(~500 RPM), ~2s per call, overnight job, must survive restarts.

- **Step 1:** `500 / 60 × 2 ≈ 16` → `concurrency=16`.
- **Steps 2–3:** covered by the knob (factory pool and admission both become 16).
- **Step 4:** this strategy owns one account quota, so its default scope is sufficient.
- **Step 5:** reserve 450 RPM, leaving shared-account headroom below the published 500.
- **Step 6:** reserve measured prompt tokens plus 300 expected output tokens.
- **Step 7:** skip the ramp at this width.
- **Step 8:** `cooldown_seconds=60.0` for reactive 429 recovery.
- **Step 9:** one-call p99 ≈ 10s gives `attempt_timeout=30.0`; allow the
  complete retry/cooldown chain 180s and stop the batch at 1h50m.

```python
from async_batch_llm import (
    CharacterTokenEstimator,
    GuardrailConfig,
    JsonlArtifactStore,
    ProcessorConfig,
    RateLimitConfig,
    ResumePolicy,
    llm,
    process_prompts,
)

config = ProcessorConfig(
    concurrency=16,
    attempt_timeout=30.0,
    max_requests_per_minute=450,
    max_tokens_per_minute=200_000,
    token_estimator=CharacterTokenEstimator(expected_output_tokens=300),
    rate_limit=RateLimitConfig(cooldown_seconds=60.0),
    guardrails=GuardrailConfig(
        total_timeout_per_item=180.0,
        batch_timeout=6600.0,
        abort_on_error_categories=frozenset({"authentication", "insufficient_balance"}),
    ),
)

batch = await process_prompts(
    llm("openai:gpt-4o-mini"),
    prompts,
    config=config,
    progress=True,
    artifact_store=JsonlArtifactStore("runs/overnight.jsonl"),
    resume=ResumePolicy.REUSE_SUCCESSES,
)
print(batch.summary())
```

Rerunning the same command after a crash replays completed successes from the
artifact and executes only the remainder.

## If memory is the constraint

For very large or unbounded inputs, add `max_queue_size` (bounded input
buffering) and switch to `process_stream()` so results don't accumulate.
That's a memory decision, not a throughput one — see
[bounded work and backpressure](bounded-work.md).
