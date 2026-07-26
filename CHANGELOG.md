# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Indexed SQLite artifact store ([#124])** — `SqliteArtifactStore` and
  `SqliteDurability` join the public API. Same logical version-1 records and
  replay-compatibility semantics as JSONL (one shared private codec), stored
  in standard-library SQLite with indexed replay lookup over the complete
  compatibility key — including a NULL-safe `context_fingerprint IS ?`
  predicate — and no history decode on reopen. Appends batch into
  transactions (`commit_batch_size` / `commit_interval_seconds`); a result is
  published only after its transaction commits, and a failed transaction
  rolls back without exposing partial rows. `BALANCED` durability maps to
  WAL + `synchronous=NORMAL` (process-crash safe; a power failure may lose
  recently committed transactions), `FULL` to WAL + `synchronous=FULL`. WAL
  auto-checkpointing bounds log growth; `close()` drains writes, attempts a
  truncating checkpoint, and terminates the store's worker thread.
  `iter_results()` is a bounded keyset iterator that never holds a read
  transaction across an async yield; `read_results()` is an async
  classmethod. Single-process, single-writer by design — documented, not
  worked around.
- **Bounded JSONL result iteration** — `JsonlArtifactStore.iter_results()`
  now streams a finite byte-snapshot of the artifact in line pages instead of
  materializing the complete record history, preserving the truncated-tail
  tolerance and malformed-record errors.
- **Exact queue high-water diagnostics** — `get_stats()` and the
  `batch_completed` observer payload report
  `input_queue_high_water_mark` / `result_queue_high_water_mark`, observed
  exactly at queue-ownership points (control messages excluded), for both
  bounded and unbounded configurations.
- **Deterministic scale-soak harness ([#127] phase one)** —
  `benchmarks/scale_soak`: eight credential-free scenarios driving the real
  streaming/artifact code paths with a seeded fake provider, order-independent
  digest accounting, bounded resource monitoring, and a versioned JSON report.
  Profiles: `ci` (runs in CI as the `scale-smoke` job), `100k`, `1m`, and
  `custom`; a manual **Scale Benchmark** workflow runs the large profiles.
- **Documentation** — new Large Runs guide; SQLite backend, durability, and
  WAL-lifecycle documentation in Results, Artifacts, and Resume; queue
  high-water documentation in Bounded Work; artifact-storage production
  checks; scale-harness methodology and dated-results policy in Benchmarks.

### Fixed

- **Artifact identity and context correctness** — zero-configuration JSONL and
  SQLite stores now reject a second inferred execution identity within one live
  store before lookup or provider work. Explicit identities remain the
  caller-owned override for deliberate mixed runs. Inspection restores raw
  context only when it was persisted, while replay always keeps the current
  work item's context and does not decode historical context.
- **Strict SQLite item-row validation** — replay and inspection validate every
  consumed row's logical artifact schema version before decoding result or
  context JSON. Unsupported newest rows fail loudly instead of falling back to
  older compatible records.
- **Read-only SQLite path inspection** —
  `SqliteArtifactStore.read_results()` now uses an operationally read-only,
  finite high-water snapshot with no schema, identity, writer, WAL-policy, or
  checkpoint side effects. It works without database/directory write
  permission and alongside an active writer.
- **Deterministic SQLite lifecycle errors** — idle writer crashes and detached
  operation failures are retained without making a fatal store usable again.
  `close()` completes checkpoint, connection, and executor cleanup before
  delivering the first unobserved operation error, then one distinct cleanup
  error on the next close; later closes are silent.
- **v0.20 context-free replay fingerprints** — context-free items now use a
  logical `None` context fingerprint (and a NULL column in SQLite) instead of
  hashing Python `None`; JSONL lookup retains a read-only fallback to the
  v0.20 spelling — including custom `context_fingerprinter` hashes — so
  existing artifacts still replay.

[#124]: https://github.com/geoff-davis/async-batch-llm/issues/124
[#127]: https://github.com/geoff-davis/async-batch-llm/issues/127

## [0.20.0] - 2026-07-20

> v0.19.0 was not published. Its planned onboarding and ergonomics
> improvements are included in v0.20.0.

### Fixed

- **Per-item smart-retry state ([#125])** — shared strategy instances no longer
  retain validation feedback or other item-specific recovery data. Validation
  feedback now lives in each work item's `RetryState`, remains isolated under
  deterministic concurrent interleaving, and survives an intervening transport
  failure without leaking to another item.

- **Streaming worker-crash ordering** — a failed worker is reported on the
  control plane before end-of-stream even when completed-result capacity is
  exhausted. This closes a callback-ordering race that could otherwise let a
  consumer observe normal stream termination before the crash signal.

- **Caller cancellation no longer finishes a shared cooldown early ([#88])**
  — the coordinated rate-limit cooldown now runs in a coordinator-owned task
  instead of the reporting caller's task. Previously, cancelling the caller
  that reported the 429 (a gateway `submit_timeout`, an item deadline)
  success-finalized the shared cooldown and woke every waiter — e.g. ~45s
  into a 300s pause. Now a cancelled caller cancels only itself; all other
  waiters stay paused until the real cooldown expires. Host teardown
  (`ParallelBatchProcessor.cleanup()` / gateway & single-call `aclose()`)
  cancels the owned task via the new `RateLimitCoordinator.shutdown()`,
  waking waiters so shutdown never hangs and the task is never leaked.

- **`WorkItemResult.exception` propagation confirmed fixed ([#101])** — the
  pre-0.13 backlog concern is closed with regression tests pinning the
  behavior on every surface: batch and streaming paths store the originating
  exception (last attempt on retry exhaustion, `FrameworkTimeoutError` on
  framework timeouts) with its traceback detached, post-processors see it,
  successes carry `None`, and serialization emits a safe descriptor that
  deliberately restores as `None` (no class loading from untrusted JSON).
  No code change was needed — the fix shipped across the executor rework and
  the v0.18 serialization design.

### Deprecated

- **`BatchResult.cache_hit_rate` is now a property ([#93])** — consistent
  with the sibling zero-arg scalar accessors (`total_cached_tokens`,
  `succeeded`, `.successes`, ...), so forgetting the `()` no longer yields a
  silently-truthy bound method in f-strings. The v0.18 call spelling
  `batch.cache_hit_rate()` still works via a transitional callable-float
  return value and emits a `DeprecationWarning`; removal with the next major
  release.

- **`ProcessorConfig.timeout_per_item` → `attempt_timeout` ([#98])** — the
  per-attempt execution timeout is now named `attempt_timeout`, ending the
  naming trap against `GuardrailConfig.total_timeout_per_item` (which bounds
  the whole logical item). `timeout_per_item` remains accepted — as a keyword,
  positionally, and as a config attribute — with a `DeprecationWarning`;
  passing both names raises `ValueError`. Removal targeted for the next major
  release. All docs and examples now use `attempt_timeout`.

### Added

- **Existing async client integration** — new `CallableStrategy` and immutable
  `CallOutcome` adapt an existing async SDK, third-party gateway client,
  PydanticAI agent, or internal service to the existing `ItemExecutor` pipeline.
  Invocation, lifecycle, error, dry-run, capacity, shared-scope, and resize
  callbacks may be supplied without subclassing. Usage and metadata are copied
  and validated, provider timing follows existing conventions, callbacks share
  per-item `RetryState`, and failed billed calls continue to use
  `TokenTrackingError`. Arbitrary callables require an explicit stable
  `ArtifactIdentity` before artifact execution; compatible replay bypasses the
  callback and an identity change invalidates reuse.

- **Bounded completed-result handoff** —
  `ProcessorConfig.max_result_queue_size` optionally limits terminal results
  waiting for a `process_stream()` consumer. The default `0` preserves the
  historical unbounded behavior. A result-only bounded semaphore leaves the
  mixed control queue unbounded, so worker failures and end-of-stream remain
  deliverable at full data capacity. Checkpointing and terminal bookkeeping
  still precede capacity acquisition and publication.

- **`LLMCallPool` preferred shared-call name** — exported as an exact alias of
  `LLMGateway`, preserving every existing import and behavior without warnings.
  The preferred name describes the queue-less, in-process shared executor and
  avoids implying a network gateway or provider-routing service.

- **Embedded application example and guide** — a credential-free flagship
  example streams paginated repository input through `CallableStrategy` into an
  async transactional sink with bounded input/output handoff, item-private
  validation recovery, failed-attempt usage, checkpoint-before-publication, and
  a second compatible replay run that makes no live calls. New documentation
  covers callable composition, retry ownership, lifecycle, identity, usage,
  cancellation, and output backpressure.

- **`progress=` on `process_prompts()`/`process_stream()` ([#100])** —
  `progress=True` installs a bundled reporter: a tqdm bar when tqdm is
  importable (new optional extra `[progress]`; also included in `[all]`),
  falling back to interval logging otherwise. Pass a callable
  `(completed, total, current_item_id)` for a custom reporter. Supported for
  both collected and streamed runs. The bundled reporter observes exact counts
  inline without a per-item thread-dispatch task, coalesces terminal rendering
  by `progress_refresh_interval_seconds`, supports growing lazy-source totals,
  and forces one exact final state during finalization. User-supplied progress
  callbacks still fire for **every** completed item with their existing
  async/sync timeout and thread behavior; `progress_interval` remains only the
  processor log-line cadence.

- **Zero-config checkpointing ([#99])** — `JsonlArtifactStore("run.jsonl")`
  now works without an `ArtifactIdentity`: `provider` and `model` are
  inferred from the strategy at run start (built-in models map to their
  provider names, custom models use their class name) and the remaining
  identity fields default to `"unversioned"`. Prompt — and, by default,
  context — still participate in the per-item compatibility fingerprint, so
  a changed prompt or model never silently replays a stale result. The full
  explicit `ArtifactIdentity` remains the recommended path for versioned
  production pipelines.

- **Unified `concurrency=` knob ([#97])** — `ProcessorConfig(concurrency=N)`
  (or the `process_prompts(..., concurrency=N)` /
  `process_stream(..., concurrency=N)` shorthand) coherently sizes every
  alignment-sensitive limit that is not explicitly configured: `max_workers`,
  `max_provider_concurrency`, and the httpx connection pool on built-in
  OpenAI-compatible models created via `llm()`/`from_api_key` without an
  explicit `max_connections` (the processor asks the model to right-size its
  pool before the first request via the new
  `request_concurrency()` hook on `OpenAICompatibleModel`/`ModelStrategy`).
  Explicit values for any individual knob override the derived ones; the
  capacity warning fires only on a real contradiction (an explicit client
  capacity smaller than the requested concurrency), never on an override.

- **Readable run reports ([#96])** — `BatchResult.summary()` returns a
  printable post-run report: item counts, termination (including guardrail
  metadata), retry totals, token totals with replayed work reported separately,
  admission/execution p50/p95/p99, wall time, and failures grouped by error
  category. `BatchResult.outputs()` iterates successful outputs (or
  `(item_id, output)` pairs with `with_ids=True`), complementing the existing
  `.successes`/`.failures` properties. New `BatchResult.wall_time_seconds`
  field is stamped by `process_all()`/`process_prompts()` and survives
  dict/JSON/JSONL round-trips (older records restore as `None`), so
  `summary()` works identically on deserialized batches.

- **String-based strategy factory ([#95])** — `llm("provider:model")` builds a
  ready-to-use built-in strategy in one line: `llm("openai:gpt-4o-mini")`,
  `llm("gemini:gemini-2.5-flash")`, `llm("openrouter:anthropic/claude-haiku-4-5")`,
  `llm("deepseek:deepseek-v4-flash", thinking=False, max_connections=150)`.
  Keyword arguments forward to the model constructor;
  `response_parser`/`temperature`/`generation_config` forward to the strategy.
  Unknown prefixes and missing optional dependencies raise errors naming the
  valid prefixes and the exact install extra. The explicit two-object form
  remains the documented path for custom clients, cached models, and custom
  strategies.

### Documentation

- **Complete onboarding path** — the README and documentation home now lead
  with `llm()`, `process_prompts()`, unified concurrency, coalesced progress,
  and `BatchResult.summary()`. A credential-free terminal animation,
  regeneration recipe, and no-key Colab notebook cover retry, terminal
  failures, summaries, checkpoint/replay, and cleanup.
- **Honest alternatives guide** — a dated scenario comparison covers
  `asyncio.gather`, Bespoke Curator, LiteLLM and other gateways, provider-native
  batch APIs, and durable workflow engines, including direct guidance about
  when not to use ABL.
- **Troubleshooting and migration** — new operational guidance covers pools,
  429s, deadlines, blocked responses, apparent hangs, artifacts, memory,
  progress, gateway composition, and deprecations. The primary migration guide
  targets v0.18.x directly; the old v0.19 URL remains as a historical pointer.

## [0.18.0] - 2026-07-13

### Added

- **Stable opt-in input ordering** — every accepted item receives a submission
  index independent of `item_id`; `BatchResult.in_input_order()` returns a new
  ordered batch, and `process_prompts(..., preserve_order=True)` opts into input
  order. Completion order remains the default and `process_stream()` remains
  completion ordered to avoid blocking and unbounded reorder buffers.
- **Strict versioned result serialization** — `WorkItemResult` supports safe
  mapping round trips, while `BatchResult` supports mapping, JSON, and one-record
  per-line JSONL round trips. Timing, token use, error category, replay state,
  submission index, and batch termination are preserved. Dataclasses, Pydantic
  models, enums, temporal values, UUIDs, paths, tuples, and sets normalize to
  JSON primitives; hooks handle application types. Unsupported values and
  future schemas raise `ResultSerializationError`; exceptions restore only as
  safe descriptors without tracebacks or runtime class loading.
- **Version-1 replayable JSONL artifacts ([#81])** — `ArtifactIdentity`,
  `ArtifactStore`, `JsonlArtifactStore`, and `ResumePolicy` provide append-only
  checkpoints with canonical SHA-256 compatibility matching, privacy-safe raw
  prompt/context defaults, checkpoint-before-publication durability, flush per
  item with optional `fsync`, truncated-tail recovery, read-only review, and
  `REUSE_SUCCESSES`/`REUSE_ALL` replay without duplicate provider work or
  checkpoint records. Optional costs are caller-calculated; no provider price
  table is bundled.
- **Opt-in execution guardrails** — `GuardrailConfig` adds a logical
  end-to-end item deadline spanning admission, cooldown, provider attempts, and
  retry backoff; a run-scoped batch deadline; configurable terminal-category
  fail-fast; and `drain_active`/`cancel_active` abort modes. Controlled batch
  stops preserve terminal results for accepted work and expose serializable
  `BatchTermination` metadata. New framework exceptions distinguish total item
  expiry, batch expiry, and collateral batch aborts.
- **Replay/deadline/abort observability** — observer events and metrics expose
  replayed items, item-deadline expiry, controlled batch abort, and replayed or
  aborted counts without counting historical replay tokens as newly consumed.

### Changed

- Built-in OpenAI and Gemini classifiers now distinguish reliable HTTP 401
  (`authentication`) and 403 (`permission_denied`) statuses from ordinary
  item-specific `client_error`; account/balance exhaustion remains
  `insufficient_balance` where the provider supplies that signal.
- High-level streaming now closes async inputs exposing `aclose()` on normal,
  error, cancellation, and controlled-abort exits so source resources are not
  leaked. Supplied async iterators are therefore owned by one stream run.

### Fixed

- Worker, producer, streaming, and artifact-failure cleanup now use one
  exactly-once queue-accounting path and surface unexpected worker or artifact
  failures instead of allowing `queue.join()` or a result stream to hang.
- Persisted framework-controlled errors redact labeled authorization, API-key,
  token, and secret values; exception objects and tracebacks are never written.
- Artifact resume builds a compatibility index instead of scanning the complete
  history for every item, and the read-only loader now rejects missing or empty
  paths instead of returning a misleading empty batch.
- Cancelled artifact operations retain exclusive ownership of their file handle
  until threaded writes finish, preventing a concurrent append or close from
  interleaving with an in-flight checkpoint.
- Total-item and batch deadlines now bound pre-execution middleware and strategy
  error hooks, and controlled guardrail failures bypass application recovery
  hooks so accepted work cannot strand batch or stream completion.
- Sensitive context and identity values remain redacted from persisted records
  but still participate in compatibility fingerprints, so changing a credential
  cannot accidentally reuse a stale result.
- Streaming admission has no cancellation point after queue ownership commits,
  keeping accepted-item accounting consistent during an abort race. Empty
  result JSONL exports now retain batch termination metadata.

### Documentation

- Reworked the README into a shorter production-oriented guide, corrected
  falsy-output rendering, documented artifact durability and process-safety
  boundaries, and replaced repository-relative links with portable URLs for
  the PyPI description. PyPI and MkDocs discovery metadata now highlight
  bounded streaming, resumable checkpoints, and deadlines.

## [0.17.0] - 2026-07-09

### Added

- **Model/transport concurrency diagnostics ([#74])** — OpenAI-compatible
  models created with `from_api_key(max_connections=N)` now expose
  `max_concurrency=N`; `ModelStrategy` forwards it, and
  `ParallelBatchProcessor`/`LLMGateway` warn when `max_workers` exceeds the
  known capacity. User-supplied clients remain capacity-unknown rather than
  relying on fragile httpx/OpenAI SDK introspection.
- **Provider-capacity admission outside execution timeout ([#79])** — the shared
  executor now limits each `strategy.execute()` attempt to the lower of
  `ProcessorConfig.max_provider_concurrency` and the strategy/model's advertised
  `max_concurrency`. Capacity is released between retries and shared by
  strategies wrapping the same model. Admission runs before
  `timeout_per_item`; cumulative wait is exposed on `WorkItemResult`, processor
  stats, `ITEM_ADMITTED` observer events, and `MetricsObserver` aggregates.
- **Structured attempt and item timing ([#76])** — `WorkItemResult.timing`
  exposes total wall time plus per-try admission, startup-ramp, execution,
  built-in provider-call, cooldown, retry-backoff, success/error classification,
  and timeout category. Processor stats retain bounded samples and expose
  admission/execution p50, p95, and p99 summaries.
- **Optional startup concurrency ramp ([#77])** — `StartupRampConfig` begins at
  `initial_concurrency`, adds `concurrency_step` on each interval, supports an
  optional maximum and jitter, composes with advertised/explicit provider
  capacity, and runs outside `timeout_per_item` across batch/gateway/single-call
  executor paths.
- **OpenAI-compatible and DeepSeek high-throughput guide ([#75])** — documents
  model-owned and custom httpx clients, capacity alignment, timeout/retry layers,
  startup ramp versus cooldown, gateway load shedding, timing diagnostics, and
  a troubleshooting matrix.
- **Opt-in conservative structured-output recovery ([#82])** —
  `pydantic_json_parser(..., recover_trailing_markdown=True)` can recover one
  complete top-level JSON object or array followed only by a recognized closing
  Markdown fence artifact. Recovered results expose typed metadata views, keep
  normal token/cost accounting, and increment processor and observer recovery
  and avoided-retry metrics. Malformed JSON, arbitrary prose, multiple values,
  scalar values, and schema-invalid data still follow normal retry behavior.
- **Bounded-work and backpressure guide ([#80])** — documents the distinct
  worker, provider-capacity, batch-queue, and gateway-admission limits; provides
  incremental async-ingestion and low-level streaming recipes; clarifies that
  `process_prompts()` retains results and `process_all()` cannot backpressure a
  preloaded batch; and shows how to avoid unbounded gateway task creation.

### Changed

- **Timeout and concurrency semantics documented ([#78])** — the production,
  OpenAI, gateway, README, and API docs now distinguish queue/admission wait,
  per-attempt execution timeout, transport-pool wait, cooldown/backoff, and the
  gateway's end-to-end `submit_timeout`. The docs also correct the old claim
  that post-rate-limit slow-start applies during initial batch startup.

### Fixed

- **Python 3.10 asyncio timeout handling** — capacity-ramp wakeups,
  post-processor timeouts, observer timeouts, and worker shutdown now catch both
  the pre-3.11 `asyncio.TimeoutError` class and built-in `TimeoutError`. The
  previous bare catches could turn normal ramp wakeups into item retries and
  misclassify callback timeouts on Python 3.10.
- **Synchronous post-processors no longer block the event loop or bypass
  `post_processor_timeout`** — sync callbacks now run through
  `asyncio.to_thread()` and share the same wait budget as async callbacks.
  Timing out stops waiting; Python cannot forcibly cancel callback code already
  running in a worker thread.

## [0.16.0] - 2026-07-02

### ⚠️ Breaking Changes

Details for each live under **Changed**/**Removed** below.

- **`MetricsObserver.reset()` is now async** — change `observer.reset()` to
  `await observer.reset()`. A leftover sync call silently no-ops (and emits
  a "coroutine was never awaited" `RuntimeWarning`).
- **`BatchResult` summary fields are `init=False`** — construct with
  `BatchResult(results=[...])`; passing `total_items=`, `succeeded=`,
  `failed=`, or the token totals to the constructor now raises `TypeError`
  (they were always recomputed and discarded anyway).
- **Gemini non-429 4xx errors fail fast** — `GeminiErrorClassifier` now
  classifies 400/401/403/404/… as non-retryable client errors instead of
  retrying them; only genuinely transient statuses are retried.
- **Dead protocols removed from `async_batch_llm.core`** — `AgentLike`,
  `ResultLike`, `UsageLike`, and `core.TOutput` are gone (they were unused
  by the framework). Import `LLMModel`/`ManagedLLMModel` instead.
- **Gemini grounding lands in `metadata['grounding']` by default** — only
  when the caller requested the `google_search` tool, so most payloads are
  unchanged; strict consumers of the metadata dict should expect the new
  key on grounded calls.

### Added

- **Typed auxiliary-output views** ([#52] Phase 2, **experimental**) —
  provider-specific structured output travels under four reserved,
  documented `metadata` keys (`grounding`, `reasoning`, `tool_calls`,
  `logprobs`; plain JSON-serializable dicts — the shapes and view APIs may
  change in a minor release while they stabilize), and both `LLMResponse`
  and `WorkItemResult` expose lazy read-only typed views over them:
  `.grounding` (`Grounding`/`GroundingSource`), `.reasoning` (`str`),
  `.tool_calls` (`list[ToolCall]`, visibility only — the framework never
  executes tools), and `.logprobs`. Views parse the dict on each access —
  nothing is stored twice, and the strategy `execute()` return contract is
  untouched. New top-level exports: `Grounding`, `GroundingSource`,
  `ToolCall`. OpenAI-compatible models now emit `reasoning` (DeepSeek
  `reasoning_content`, falling back to OpenRouter `reasoning`), `tool_calls`
  (arguments kept as the raw JSON string), and `logprobs` (via
  `model_dump()`) when present. New example:
  `examples/example_gemini_grounding.py`.
- **`RateLimitConfig.max_cooldown_seconds`** (default 600) — configurable cap
  on the exponentially-backed-off cooldown; previously the
  `ExponentialBackoffStrategy` cap existed but wasn't reachable through
  `RateLimitConfig`. The cap can never sit below `cooldown_seconds`: a
  config with a larger cooldown (e.g. 900s for daily-quota waits, including
  via the legacy `rate_limit_cooldown` kwarg or `dataclasses.replace`)
  lifts the cap to match — silently when the cap was left at its default,
  with a logged warning when it overrides an explicitly-set lower value.
  `slow_start_final_delay` is now validated non-negative.
- **`EmptyResponseError`** (exported at top level) — raised by the built-in
  models when the API call succeeded but produced no usable text (Gemini
  safety block, OpenAI `finish_reason` of `length`/`content_filter`/tool
  call). Subclasses `ValueError` (existing handlers keep working) and
  carries the tokens the provider already billed as `_failed_token_usage`,
  so failed-attempt accounting reflects real spend.
- **`ProviderResponseError`** (exported at top level) — OpenRouter reports
  upstream failures (no provider available, upstream 5xx, upstream rate
  limits) as HTTP 200 with an `error` object and no choices, so the openai
  SDK never raises. That path used to surface as a non-retryable "No choices
  returned" error. `OpenRouterModel` now raises `ProviderResponseError`
  (carrying the embedded code and raw payload) via a
  `_raise_on_response_error` hook on `OpenAICompatibleModel`;
  `OpenRouterErrorClassifier` retries it, treating embedded 429s as rate
  limits so the coordinated cooldown engages.
- **PEP 561 `py.typed` marker** — downstream mypy/pyright now consume the
  package's inline annotations instead of treating it as untyped. Also adds
  the `Typing :: Typed` trove classifier.

### Changed

- **Gemini grounding is emitted by default** ([#52] Phase 2) — a grounded
  response (`google_search` tool) now lands in `metadata['grounding']`
  without opting in; previously this required passing
  `grounding_metadata_extractor`, which remains exported (for custom models
  and explicit configurations) and is now redundant-but-harmless on the
  built-in Gemini models. Non-grounded calls are unaffected — the key only
  appears when the caller requested the tool.
- **Test suite runs in ~15s instead of ~80s** — shared
  `fast_retry`/`fast_rate_limit` fixtures (`tests/conftest.py`) replace
  real 1s+ retry waits and 5s observer timeouts with millisecond
  equivalents; `pytest-timeout` (60s cap) makes deadlock regressions fail
  instead of hanging CI. No coverage lost.
- **PEP 696 defaults on the framework type variables** —
  `TInput`/`TOutput`/`TContext` default to `str`/`Any`/`None` (via
  `typing_extensions.TypeVar`), so trailing parameters can be dropped:
  `ParallelBatchProcessor[str, MyOutput]`. `typing-extensions>=4.4` is now
  an explicit dependency (already present transitively via pydantic).
  `TInput` remains unused by the framework (prompts are always `str`) and is
  slated for removal at the next major.
- **`BatchResult` derived fields are `init=False`** — `total_items`,
  `succeeded`, `failed`, and the token totals were constructor arguments
  that `__post_init__` silently recomputed and discarded; the constructor
  signature no longer advertises them.
- **Reading `WorkItemResult.gemini_safety_ratings` emits a
  `DeprecationWarning`** — read `result.metadata['safety_ratings']` instead.
  Construction, `repr()`, comparisons, and `dataclasses.replace()`/`asdict()`
  (which read every field via `getattr`) stay silent, so copying a result
  or framework-internal operations don't warn — only direct reads do.
- **Pre-push guard against stale branches** — `scripts/check_branch_fresh.sh`
  runs as a `pre-push` hook (installed by `pre-commit install` via
  `default_install_hook_types`) and refuses pushes from branches missing
  commits that are on `origin/main`, printing the rebase fix. Fails open
  when offline; skip intentionally-behind pushes with
  `SKIP=check-branch-fresh git push`. Commit-time hooks are pinned to
  `default_stages: [pre-commit]` so pushes stay fast. See CLAUDE.md
  "Sync before working".
- **`GeminiErrorClassifier` dispatches on HTTP status codes** — genai SDK
  exceptions now classify on `APIError.code` instead of string-matching
  `str(exception)`. Deterministic 4xx client errors (invalid API key,
  malformed request, missing model) now **fail fast** instead of burning the
  whole retry budget; 429 parses `Retry-After` into
  `ErrorInfo.suggested_wait`; 503 keeps its dedicated `server_overload`
  category (per-item backoff, no coordinated cooldown). Without the
  `[gemini]` extra the classifier now falls through to the generic
  pattern/type chain instead of returning `unknown` immediately, so rate
  limits from mocks still engage the cooldown. `Retry-After` parsing is
  shared across classifiers (`strategies/errors.py`) and ignores
  non-positive values.
- **`MetricsObserver.reset()` is now async and lock-guarded** (breaking) —
  the unlocked sync version could lose an in-flight event's counts into the
  discarded pre-reset dict while advertising itself as thread-safe. Call
  `await observer.reset()`.
- **Observers receive independent event payloads** — `EventDispatcher.emit`
  passes each observer a shallow copy of the event data, so one observer
  mutating the dict can't corrupt what the next sees. Delivery order
  (registration order) and middleware onion semantics (before in order,
  after reversed, first non-None `on_error` wins, failures log and fail
  open) are now documented on the base classes.
- **CI hardening** — the Tests workflow now lints `tests/` (CI previously
  checked less than local tooling and pre-commit), verifies formatting with
  `ruff format --check`, drops the stale `continue-on-error` for Python 3.14
  (stable since Oct 2025 and advertised in classifiers), and gains a
  `docs-build` job running `mkdocs build --strict` on PRs so broken nav/links
  surface before merge. The docs deploy also builds `--strict` and queues
  per-ref instead of racing concurrent gh-pages pushes.

### Removed

- **Dead `AgentLike`/`ResultLike`/`UsageLike` protocols**
  (`async_batch_llm.core`) — never referenced by the framework, never
  exported at top level, and `UsageLike` documented the deprecated
  pydantic-ai 0.x field names. The module-private `TOutput` TypeVar in
  `core.protocols` went with them.
- **Conceptually-wrong config warnings** — `ProcessorConfig` warned when
  `timeout_per_item` was smaller than cumulative retry waits, but the
  timeout is a per-attempt limit enforced around each `execute()` call;
  between-attempt waits happen outside it, so the comparison was
  meaningless and confusing.

### Fixed

- **CHANGELOG release dates for 0.1.0–0.4.0** — corrected from placeholder
  2025-01-xx dates (and a `TBD`) to the actual November 2025 releases.
- **Cooldown log no longer mislabels a strategy-requested zero cooldown as
  an error** — the coordinator now distinguishes "strategy errored",
  "pausing for Ns", and "strategy requested no cooldown".
- **`TokenExtractor` checks the framework-stamped count first** — the exact
  per-attempt `_failed_token_usage` stamped by strategies was checked last,
  so an exception that also exposed a heuristic `.usage` attribute (or a
  cause chain) had its exact count shadowed. Float/str counts in the stamped
  dict are now coerced instead of silently dropped, pydantic-ai v1
  `cache_read_tokens` maps into `cached_input_tokens`, and property-style
  `result.usage` (pydantic-ai 1.x) is read directly on the `__cause__` path.
- **`PydanticAIStrategy` reads `result.usage` as a property** — pydantic-ai
  1.x exposes usage as a property; the old `result.usage()` call only worked
  through a deprecation shim slated for removal (and failed `ty` checks
  against current pydantic-ai). Method-style results (older versions, test
  doubles) still work.
- **Publish workflow could upload mislabeled code to PyPI** — it published on
  any `v*` tag with no check that the tag matched `pyproject.toml`'s version
  and no tests. A `test` job now verifies `tag == v<project.version>`, runs
  pytest/ruff/mypy, and gates the `publish` job; permissions declare
  `contents: read` explicitly.
- **`get_stats()['total_cached_tokens']` always reported 0** — the "preferred
  alias" was a second `ProcessingStats` field that nothing ever incremented;
  only `cached_input_tokens` was updated by the worker loop. The duplicate
  storage is gone and `copy()` now maps both dict keys to the single real
  counter.

## [0.15.0] - 2026-06-16

### Added

- **Pluggable metadata extraction** ([#52]) — every built-in model now accepts
  `metadata_extractors: list[MetadataExtractor]`, hooks that contribute extra
  keys to `LLMResponse.metadata` (and therefore `WorkItemResult.metadata`)
  without subclassing the model or overriding `execute()`. User extractors merge
  on top of the built-in payload (`safety_ratings`/`finish_reason` for Gemini,
  `finish_reason`/`model`/`provider` for OpenAI-compatible); user keys win, and
  a failing extractor is logged and skipped rather than breaking the call.
  Available as a constructor argument on `GeminiModel`, `GeminiCachedModel`,
  `OpenAICompatibleModel`, `OpenAIModel`, `OpenRouterModel`, and `DeepSeekModel`;
  the OpenAI-compatible models also accept it through `from_api_key(...)`
  (Gemini models have no `from_api_key`).
- **`grounding_metadata_extractor`** ([#52]) — an opt-in extractor that maps a
  grounded Gemini response (`google_search` tool) onto
  `metadata['grounding']` (`sources`, `queries`, `supports`) as plain dicts, so
  callers get web-search citations through the framework instead of reaching into
  `LLMResponse.raw`. Not registered by default — pass it explicitly via
  `metadata_extractors=[grounding_metadata_extractor]`. New public exports:
  `MetadataExtractor`, `grounding_metadata_extractor`. See
  `docs/GEMINI_INTEGRATION.md`.

### Changed

- **Dependency maintenance** — bumped `pyjwt` 2.12.1 → 2.13.0 ([#51]); pinned
  `js-yaml`/`markdown-it` to patched versions via npm `overrides` to clear newly
  published moderate-severity DoS advisories
  (GHSA-h67p-54hq-rp68, GHSA-6v5v-wf23-fmfq) in the markdownlint dev-tooling
  chain. No runtime impact.

## [0.14.0] - 2026-06-14

### Added

- **Single-call helper** — `call()` / `call_result()` run one prompt through the
  full resilience pipeline (error-aware retries, the coordinated rate-limit
  cooldown, token accounting) with no queue, workers, or result stream. `call()`
  returns the output, re-raising the provider's own exception on failure (or
  `LLMCallError` when none was preserved); `call_result()` returns the full
  `WorkItemResult`. See `examples/example_single_call.py`.
- **`LLMGateway`** — a long-lived, shared entry point for a web service's request
  path. Many concurrent callers `submit()` against one shared
  `RateLimitCoordinator` (so one caller's 429 throttles everyone, then
  slow-starts), with global concurrency bounded by a semaphore. `submit()` raises
  on failure; `submit_result()` returns the full `WorkItemResult`. No queue,
  worker pool, or per-request Future demux: each caller runs the request itself
  under the semaphore, so a cancelled caller simply frees its slot. Two opt-in
  load-shedding knobs (off by default): `max_pending` caps in-flight requests
  (running + waiting) and rejects over-cap submits instantly instead of growing
  an unbounded waiter list; `submit_timeout` bounds per-caller latency. See
  `examples/example_gateway.py` and `docs/api/single-gateway.md`.
- **`WorkItemResult.exception`** — failed results now carry the originating
  exception (when one was raised). `call()` / `LLMGateway.submit()` re-raise that
  exact exception, preserving the provider's type, instead of always wrapping it
  in `LLMCallError`. `None` for successes and non-error outcomes (e.g. a
  middleware filter-skip). Excluded from result equality.
- **`ModelStrategy(generation_config=...)`** — an optional default config
  forwarded to `model.generate(config=...)` on every call, so a built-in
  strategy (`GeminiStrategy`/`OpenAIStrategy`/`OpenRouterStrategy`/`DeepSeekStrategy`)
  can carry native structured output or grounding (Gemini `response_schema` /
  `response_mime_type` / `tools`, OpenAI-compatible `response_format` /
  `max_tokens`) without subclassing `execute()`. Defaults to `None` (unchanged
  behavior); a subclass overriding `execute()` can read `self.generation_config`
  to merge a per-attempt config.

### Changed

- **Internal: `ItemExecutor` extraction.** The per-item execution engine
  (retries, error classification, rate-limit coordination, token accounting) was
  factored out of `ParallelBatchProcessor` into `_internal/item_executor.py` so
  the batch worker, the single-call helper, and the gateway share one engine. The
  processor's public surface and behavior are unchanged; it now delegates its
  per-item methods to the executor.
- **`google-genai` upgraded to 2.8.0** — the pinned dependency moves from 1.73.1
  to 2.8.0. API-compatible for the Gemini integration; the existing `>=1.49.0`
  constraint already permitted it (lockfile-only change).
- **Deprecation fix** — replaced `asyncio.iscoroutinefunction` with
  `inspect.iscoroutinefunction` (the asyncio alias is slated for removal in
  Python 3.16); identical semantics for plain coroutine functions and a
  callable's `__call__`.

## [0.13.0] - 2026-06-10

### Added

- **First-class streaming mode** on `ParallelBatchProcessor` —
  `start()` / `add_work()` / `finish()` / `results()`. Workers run *while* you
  add work, so a bounded `max_queue_size` becomes backpressure (constant-memory
  processing of huge or unbounded inputs) instead of a deadlock risk. The
  high-level `process_stream` / `process_prompts` are built on it.
- **Per-item context in the high-level API** — `process_prompts` and
  `process_stream` now accept `(item_id, prompt, context)` triples alongside
  bare strings and `(item_id, prompt)` pairs; the context rides through to
  `WorkItemResult.context` and post-processors.
- **New docs**: a Production Checklist, a Testing guide, and a results-first
  Benchmarks page backed by a real GSM8K test-split run — committed charts +
  machine-readable `summary.json`/`throughput.json` under `docs/assets/`,
  regenerable via `examples/generate_benchmark_charts.py` (matplotlib added to
  the `[docs]` extra).

### Changed

- **Gemini `503` / "high demand" overload now retries with per-item exponential
  backoff instead of a coordinated cooldown.** A 503 is a transient server-side
  capacity blip (usually per-request — a retry often lands on a healthy
  backend), so `GeminiErrorClassifier` now treats it like any other 5xx
  (`is_rate_limit=False`), matching `OpenAIErrorClassifier`, rather than pausing
  *all* workers for the full rate-limit cooldown. The coordinated cooldown stays
  reserved for genuine quota exhaustion (429 / `RESOURCE_EXHAUSTED`). For
  *sustained* overload at high concurrency, lower `max_workers` or set
  `ProcessorConfig.max_requests_per_minute`.
- **README restructured around the value proposition** (~half the length):
  leads with benchmark-backed scale numbers, "vs. rolling your own", and "When
  NOT to use this"; reference-style sections moved to the docs site.
- **Benchmark example overhaul** (`example_batch_benchmark.py`): a
  `--throughput` mode racing chunked `gather` vs a semaphore pool vs the
  framework at equal concurrency; terse-vs-verbose sample capture and
  methodology metadata in `summary.json`; the bake-off + judge now use the
  high-level `process_prompts` API; a benchmark-appropriate 30 s cooldown
  override so a transient blip doesn't dominate the demo.

## [0.12.0] - 2026-06-09

### Added

- **Open-file-limit warning.** `ParallelBatchProcessor` now emits a
  `UserWarning` at construction when `max_workers` is close to the process's
  soft `RLIMIT_NOFILE`. Each in-flight request typically holds a socket (a file
  descriptor), so a high worker count near the OS limit (≈256 by default on
  macOS) would otherwise fail mid-run with `OSError: [Errno 24] Too many open
  files`. The library only *warns* — it does not mutate the process-global
  limit — and points to a new "Open file limits and high concurrency" section
  in the getting-started docs covering `ulimit`, in-process `resource.setrlimit`,
  and connection-pool sizing.

### Changed

- **`GeminiErrorClassifier` routes 503 / model-overload through the coordinated
  cooldown.** A 503 `UNAVAILABLE` / "overloaded" / "high demand" now classifies
  as `is_rate_limit=True` (category `server_overload`), so the
  `RateLimitCoordinator` pauses all workers and slow-starts — under high
  concurrency, per-item backoff alone can't relieve an overloaded model. Other
  5xx (500/502/504) stay one-off per-item retries. Refines the blanket 5xx retry
  added in v0.11.0.

### Documentation

- **Expanded the bulk-benchmark example and refreshed the README.** Added a
  `--throughput` mode to `examples/example_batch_benchmark.py` (worker-pool vs
  chunked-`asyncio.gather` throughput at matched concurrency, with per-leg
  rate-limit counts and an inter-leg quota-reset gap), and a new
  `examples/peek_outputs.py` for comparing raw model output verbosity across
  providers. Reworked the README value proposition (provider-agnostic worker
  pool, error-type-aware resilience, cost/observability) with a benchmark
  "sense of scale" callout, a minimal lead example, and pointers to deeper docs.

## [0.11.0] - 2026-06-08

### Added

- **`examples/example_batch_benchmark.py`** — flagship bulk-processing demo.
  Runs the GSM8K benchmark through DeepSeek Flash (`deepseek-v4-flash`) vs
  Gemini 3.1 Flash-Lite (`gemini-3.1-flash-lite`) vs Gemini 2.5 Flash-Lite
  (`gemini-2.5-flash-lite`) with no-thinking→thinking escalation, a per-provider
  3-way wall-time race (sequential vs `asyncio.gather` vs `ParallelBatchProcessor`),
  async gzip I/O via `aiogzip` (a single-consumer queue writer for results),
  token/cost reporting, and an OpenAI LLM-as-judge fallback grader.
  `examples/download_gsm8k.py`
  fetches the data; walkthrough at `docs/examples/bulk-benchmark.md`.

### Changed

- **`CachedTokenRates.DEEPSEEK` corrected to `0.02`** (was `0.10`). DeepSeek V4
  Flash bills cache hits at $0.0028/M vs $0.14/M cache-miss input (~2%), after
  a price drop that took effect April 2026. `effective_input_tokens(
  CachedTokenRates.DEEPSEEK)` now reflects the larger cache discount.

### Fixed

- **`GeminiErrorClassifier` now retries 5xx `ServerError`s** (503
  overload/UNAVAILABLE, 500, 502, 504), not just server *timeouts*. A transient
  503 ("high demand, try again later") was previously classified non-retryable
  and failed the item permanently — inconsistent with `OpenAIErrorClassifier`,
  which already retries all 5xx. Backoff + rate-limit cooldown still prevent
  hammering.

### Security

- **Bumped `starlette` 1.0.0 → 1.0.1** (#30), resolving a moderate Dependabot
  advisory. Transitive dependency pinned in the `uv` lockfile.

## [0.10.0] - 2026-06-07

### Added

- **DeepSeek `thinking` toggle** (#27). `DeepSeekModel(..., thinking=False)`
  and `DeepSeekModel.from_api_key(..., thinking=False)` force non-thinking
  mode (`extra_body={"thinking": {"type": "disabled"}}`); `thinking=True`
  forces it on; `None` (default) uses the model default. V4 models
  (`deepseek-v4-flash`/`-pro`) default to thinking, which is expensive for
  batch classification, so this avoids relying on the deprecating
  `deepseek-chat`/`deepseek-reasoner` aliases. New DeepSeek quickstart in the
  README ties together `thinking`, `json_mode`, `max_connections`, and the
  fence-tolerant parser.
- **`402 Insufficient Balance` is now classified as non-retryable** (#27).
  `OpenAIErrorClassifier` maps HTTP 402 (and the `insufficient balance` /
  `insufficient_quota` string patterns) to a new `insufficient_balance`
  category with `is_retryable=False` and an `ErrorInfo.hint`. Previously 402
  fell through to the "unknown status → retry" path and silently burned every
  retry attempt. The processor logs the hint at WARNING when a non-retryable
  error gives up. `ErrorInfo` gained an optional `hint` field.
- **First-class structured output for OpenAI-compatible strategies** (#26).
  `*Model.from_api_key(..., json_mode=True)` adds
  `response_format={"type": "json_object"}` to `extra_body` (an explicit
  caller-supplied `response_format` still wins). New
  `async_batch_llm.pydantic_json_parser(Model)` builds a `response_parser`
  that strips markdown code fences (```` ```json ... ``` ````) before
  validating with Pydantic — so providers like DeepSeek that wrap JSON in
  fences even in JSON mode no longer burn retries. Also exports the
  underlying `strip_code_fences` helper.
- **`max_connections` on `*Model.from_api_key`** (OpenAI-compatible models).
  Sizes the underlying `httpx.AsyncClient` connection pool (both
  `max_connections` and `max_keepalive_connections`) so it can keep up with
  `ProcessorConfig.max_workers`. The openai SDK otherwise uses httpx's ~100
  default pool, silently capping throughput when `max_workers` exceeds it —
  the biggest high-concurrency footgun, especially for DeepSeek. Mutually
  exclusive with passing your own `http_client` (#25).
- **First-class DeepSeek provider.** New `DeepSeekModel` (subclass of
  `OpenAICompatibleModel`, base URL `https://api.deepseek.com`, reads
  `DEEPSEEK_API_KEY`) and `DeepSeekStrategy`. `DeepSeekModel._extract_tokens`
  reads DeepSeek's native `prompt_cache_hit_tokens` into `cached_input_tokens`
  (DeepSeek doesn't use OpenAI's nested `prompt_tokens_details.cached_tokens`).
  Use with `CachedTokenRates.DEEPSEEK`. New `[deepseek]` optional dependency
  group.
- **`ModelStrategy` base class** — shared base for `GeminiStrategy`,
  `OpenAIStrategy`, `OpenRouterStrategy`, and `DeepSeekStrategy`, which are now
  thin subclasses. Centralizes `execute()`, lifecycle delegation, and the
  token-tracking-on-parse-failure path. Exported for custom model-backed
  strategies.
- **`temperature=None` support** across the `LLMModel` protocol, all built-in
  models, and `ModelStrategy` — omits the `temperature` parameter entirely so
  providers use their default. Required for OpenAI reasoning models (o1/o3)
  that reject an explicit temperature.
- **`Retry-After` parsing** in `OpenAIErrorClassifier`
  (`ErrorInfo.suggested_wait`), now honored by the `RateLimitCoordinator` as a
  *floor* on the cooldown duration.
- **First-class OpenAI and OpenRouter providers.** New `OpenAICompatibleModel`
  base class plus `OpenAIModel` and `OpenRouterModel` subclasses, each with
  a `from_api_key(...)` convenience constructor that accepts an optional
  `api_key` and falls back to `OPENAI_API_KEY` / `OPENROUTER_API_KEY` env
  vars. New `OpenAIStrategy` and `OpenRouterStrategy` (thin shells over the
  model). New `OpenAIErrorClassifier` and `OpenRouterErrorClassifier`
  (subclass with `no_provider_available` body-marker handling). New
  optional dependency groups `[openai]` and `[openrouter]`. Models
  implement `ManagedLLMModel`: `cleanup()` closes the underlying httpx
  client when constructed via `from_api_key`. New docs at
  `docs/OPENAI_INTEGRATION.md` and `docs/OPENROUTER_INTEGRATION.md`.
- **`CachedTokenRates` constants** (`GEMINI=0.10`, `OPENAI=0.50`,
  `ANTHROPIC_READ=0.10`, `DEEPSEEK=0.10`) for provider-aware billing
  arithmetic. `BatchResult.effective_input_tokens()` accepts a
  `cached_token_rate` parameter; default stays at the Gemini rate for
  backward compatibility.
- **Response metadata reaches `WorkItemResult`** ([#8]).
  `LLMCallStrategy.execute()` now supports a 3-tuple return shape
  `(output, tokens, metadata)`; legacy 2-tuple is still accepted via
  `_unpack_strategy_result`. All built-in strategies forward
  `LLMResponse.metadata` (provider, finish_reason, routed model, safety
  ratings) into `WorkItemResult.metadata`. Per-item provider-aware
  billing in mixed-provider OpenRouter batches no longer requires a
  custom `response_parser` (parser path still supported).

### Changed

- `OpenAICompatibleModel.from_api_key` is now generic over `cls` (a `TypeVar`
  bound to `OpenAICompatibleModel`), so it returns the calling subclass type
  (`OpenAIModel`/`OpenRouterModel`/`DeepSeekModel`) instead of the base. Subclass
  overrides no longer need a `cast()` — `OpenRouterModel.from_api_key` drops its
  workaround. (Closes #10.)
- `BatchResult.effective_input_tokens()` now emits a `UserWarning` when called
  without an explicit `cached_token_rate` while cached tokens are present
  (the implicit Gemini default is wrong for other providers). Pass an explicit
  `CachedTokenRates` constant to silence it. The default behavior is otherwise
  unchanged.
- `ErrorInfo.suggested_wait` now carries **only genuine server signals**
  (e.g. a parsed `Retry-After`); the previously-unused hardcoded
  `DEFAULT_RATE_LIMIT_WAIT` fallbacks were removed from the classifiers (the
  `RateLimitStrategy` owns the default cooldown).

### Deprecated

- `WorkItemResult.gemini_safety_ratings` — read
  `WorkItemResult.metadata['safety_ratings']` instead. The named field is
  still populated for backward compat; scheduled for removal alongside
  the 2-tuple compat shim.

### Security

- Bumped transitive (lockfile-only) dependencies to clear open Dependabot
  alerts. None are direct dependencies of async-batch-llm, and the affected
  code paths are not exercised by the library — they reach us via
  `pydantic-ai[fastmcp]` and the docs/HTTP toolchain — but the bumps clear
  the alerts and keep the dev/CI environment current:
  - `urllib3` 2.6.3 → 2.7.0 (GHSA-qccp-gfcp-xxvc cross-origin header leak on
    proxied redirects; GHSA-mf9v-mfxr-j63j decompression-bomb bypass — both
    high).
  - `python-multipart` 0.0.26 → 0.0.29 (GHSA-pp6c-gr5w-3c5g DoS via unbounded
    multipart part headers — high).
  - `pydantic-ai` / `pydantic-ai-slim` 1.82.0 → 1.103.0 (GHSA-cqp8-fcvh-x7r3
    SSRF cloud-metadata blocklist bypass).
  - `idna` 3.11 → 3.17 (GHSA-65pc-fj4g-8rjx `idna.encode()` bypass).
  - `authlib` 1.7.0 → 1.7.2 (GHSA-r95x-qfjj-fjj2 OIDC open redirect).
  - `pymdown-extensions` 10.21.2 → 10.21.3 (GHSA-62q4-447f-wv8h snippets path
    traversal — docs build only).

[#8]: https://github.com/geoff-davis/async-batch-llm/issues/8
[#51]: https://github.com/geoff-davis/async-batch-llm/pull/51
[#52]: https://github.com/geoff-davis/async-batch-llm/issues/52
[#81]: https://github.com/geoff-davis/async-batch-llm/issues/81
[#88]: https://github.com/geoff-davis/async-batch-llm/issues/88
[#93]: https://github.com/geoff-davis/async-batch-llm/issues/93
[#95]: https://github.com/geoff-davis/async-batch-llm/issues/95
[#96]: https://github.com/geoff-davis/async-batch-llm/issues/96
[#97]: https://github.com/geoff-davis/async-batch-llm/issues/97
[#98]: https://github.com/geoff-davis/async-batch-llm/issues/98
[#99]: https://github.com/geoff-davis/async-batch-llm/issues/99
[#100]: https://github.com/geoff-davis/async-batch-llm/issues/100
[#101]: https://github.com/geoff-davis/async-batch-llm/issues/101

## [0.8.0] - 2026-04-24

### Fixed

- Gemini rate-limit errors are now retryable. Previously `GeminiErrorClassifier`
  marked rate limits with `is_retryable=False`, so the coordinated cooldown in
  `_handle_rate_limit()` paused *other* workers but the work item that hit the
  rate limit itself failed permanently. Rate-limited items now retry after the
  cooldown completes.
- Rate-limit retries no longer add exponential backoff on top of the coordinated
  cooldown. `_handle_rate_limit()` already waits `rate_limit.cooldown_seconds`
  before re-raising; the retry loop previously sleep()ed again using
  `retry.initial_wait`, doubling the effective delay (default: 300s + 1s).

### Changed

- **Breaking for typed users:** `GeminiStrategy(model=...)` without a
  `response_parser` is now restricted via `@overload` to `GeminiStrategy[str]`.
  The default parser returns `LLMResponse.text`; using it with a non-str
  `TOutput` was previously a silent runtime footgun (`cast(TOutput, response.text)`
  returned a `str`). Code that was already passing a `response_parser` is
  unaffected; code relying on the default with a non-str type parameter must
  now pass a parser.
- Minimum `pydantic-ai` version raised from `>=0.0.1` to `>=1.0.0` in the
  `pydantic-ai`, `all`, and `dev` extras.

### Packaging

- `tests/` are now included in the source distribution (helpful for downstream
  packagers who verify builds).
- Node `package.json` marked `private: true` with the version/license aligned
  to the Python package. This Node manifest exists only to pin the
  markdownlint dev tool; it is not a publishable package.

## [0.7.2] - 2026-04-22

### Fixed

- `GeminiCachedModel.generate()` no longer emits misleading "Cache expired" log
  lines with a ~56-year age under concurrent workers. The log is now inside the
  cache lock and after the double-check, so only the worker that actually renews
  logs the message; losing-race workers stay silent. Age is rendered as
  "unknown (cache not yet initialized)" when `_cache_created_at` is `None`
  instead of being computed as `time.time() - 0`.

### Security

- Bumped transitive dependency `authlib` 1.6.10 → 1.7.0 to clear
  [GHSA-jj8c-mmj3-mmgv](https://github.com/authlib/authlib/security/advisories/GHSA-jj8c-mmj3-mmgv)
  (CSRF in Authlib's OAuth cache path, medium/CVSS 5.4). async-batch-llm does not
  use Authlib directly — it reaches us via `pydantic-ai[fastmcp]` → `fastmcp` —
  and the vulnerable code path is not exercised, but the bump clears the
  Dependabot alert.

## [0.7.1] - 2026-04-22

### Fixed

- `GeminiCachedModel.prepare()` no longer crashes with `CreateCachedContentConfig`'s
  `extra_forbidden` ValidationError when `cache_tags` is non-empty. google-genai's
  `CreateCachedContentConfig` has no `metadata` field in the 1.x line; tags are now
  encoded into the cache's `display_name` with a sentinel prefix (`abl-tags:<json>`)
  and decoded on lookup. Previously any `GeminiCachedModel` with a non-empty
  `cache_tags=` dict failed every worker's prepare() on current google-genai versions.

### Changed

- `cache_tags` are persisted in `CachedContent.display_name` instead of `metadata`.
  Tag values should stay short — Gemini's `display_name` has a 128-character limit.
  Caches created outside async-batch-llm (no `abl-tags:` prefix on display_name) are
  treated as untagged and won't match a `GeminiCachedModel` with `cache_tags` set.

## [0.7.0] - 2026-04-16

Internal refactor release. Public API (`async_batch_llm/__init__.py`) is unchanged —
all new code lives behind underscore-prefixed internal modules. Includes one real
bug fix in `GeminiCachedModel.delete_cache()`.

### Added

- `ProcessorConfig.post_processor_timeout` — previously hardcoded at 90s; now configurable
  (typical 30–120s).
- Input validation on `LLMWorkItem` (rejects `strategy=None`, non-string prompts) with
  actionable error messages.
- Warning when `GeminiCachedModel.cache_renewal_buffer_seconds < 60` — small buffers risk
  renewing on every call if generation takes longer than the buffer.
- `TokenExtractor` class for centralized token-usage extraction from exceptions
  (PydanticAI-style `__cause__.result.usage()`, direct `.usage` attribute, framework-attached
  `_failed_token_usage`).
- New `_internal/` package containing `EventDispatcher`, `StrategyLifecycle`,
  `RateLimitCoordinator`, and `error_logging` helpers. Underscore-prefixed — not public API.
- New test modules: `test_input_validation.py`, `test_cleanup_errors.py`,
  `test_token_extractor.py`, `test_shared_strategy_stress.py` (15 new tests total).

### Changed

- Error-message truncation now uses `ERROR_MESSAGE_MAX_LENGTH` (200) and
  `ERROR_MESSAGE_DETAILED_LENGTH` (500) constants instead of scattered magic numbers.
- Best-effort cleanup / metadata / delete paths now log with `exc_info=True` so tracebacks
  are visible for debugging.
- `GeminiCachedModel` docstring strengthens the "share one instance" guidance (10× cost
  impact if violated).
- DEBUG log emitted when `_extract_tokens()` receives a response without `usage_metadata`
  so users can diagnose empty token counts.

### Fixed

- **Race in `GeminiCachedModel.delete_cache()`**: concurrent callers could observe
  `self._cache` as `None` mid-operation and raise `AttributeError` from the logger. Now
  serialized under the cache lock with the cache name captured upfront; exactly one API
  delete is issued per cache, late callers return silently.

### Refactored

- `ParallelBatchProcessor` decomposed from 1,323 → 945 lines (-29%). Four collaborators
  extracted under `_internal/`: `EventDispatcher` (observer/middleware dispatch),
  `StrategyLifecycle` (prepare/cleanup with double-checked locking), `RateLimitCoordinator`
  (generation-counter state machine), and `error_logging` (validation error formatting).
  Public API unchanged; read-only property shims preserve back-compat for tests that
  introspect internal state.
- Token-usage extraction moved from inline processor method into `TokenExtractor` class.
- Removed 7 obsolete `# type: ignore` codes; mypy now clean under `--warn-unused-ignores`.

### Documentation

- Removed stale `RATE_LIMIT_FIX_PLAN.md`. A diagnostic run showed the three
  `test_worst_case_rate_limit.py` scenarios pass reliably (5/5 consecutive runs); the
  proposed re-architecture was unnecessary. The current generation-counter design is the
  shipped behavior.
- README: replaced removed `GeminiCachedStrategy` with
  `GeminiStrategy(model=GeminiCachedModel(...))` in the Cost Optimization and RAG examples.
- CLAUDE.md: extended Version History past v0.1.0 with v0.3.0, v0.6.0, v0.7.0 entries.
- PACKAGE_REVIEW_2025_01_10.md: added "superseded" banner referencing the April 2026
  follow-up.

## [0.6.0] - 2026-04-15

**BREAKING**: Separates model/client management from strategy logic via LLMModel protocol.

### Added

- **`LLMResponse`** dataclass — normalized response from any LLM provider with `.text`,
  `.input_tokens`, `.output_tokens`, `.total_tokens`, `.cached_input_tokens`, `.metadata`,
  `.raw`, and `.token_usage` property
- **`LLMModel`** protocol — minimal interface for LLM model instances (`generate()`)
- **`ManagedLLMModel`** protocol — extends `LLMModel` with lifecycle (`prepare()`, `cleanup()`)
- **`GeminiModel`** — concrete `LLMModel` wrapping a `genai.Client` + model name
- **`GeminiCachedModel`** — concrete `ManagedLLMModel` with Gemini context caching
  (cache find/create/renew/delete lifecycle, same behavior as old `GeminiCachedStrategy`)
- **`MockLLMModel`** in testing utilities for easy strategy testing

### Changed

- **`GeminiStrategy`** now accepts an `LLMModel` and `Callable[[LLMResponse], TOutput]`
  instead of `(model: str, client: genai.Client, response_parser: Callable[[Any], TOutput])`.
  It delegates lifecycle calls to the model if it implements `ManagedLLMModel`.
- **`response_parser`** functions now receive `LLMResponse` instead of raw provider response

### Removed

- **`GeminiCachedStrategy`** — use `GeminiStrategy(model=GeminiCachedModel(...))` instead
- **`GeminiResponse`** dataclass — metadata is now always available in `LLMResponse.metadata`
- **`include_metadata`** parameter — no longer needed
- **`_extract_safety_ratings()`** module-level helper — moved into `GeminiModel`

### Migration

```python
# Before (v0.5.0)
strategy = GeminiStrategy(
    model="gemini-2.5-flash", client=client,
    response_parser=lambda r: r.text,
)

# After (v0.6.0)
model = GeminiModel("gemini-2.5-flash", client)
strategy = GeminiStrategy(model, response_parser=lambda r: r.text)

# Before (cached)
strategy = GeminiCachedStrategy(
    model="gemini-2.5-flash", client=client,
    response_parser=lambda r: r.text,
    cached_content=[...],
)

# After (cached)
model = GeminiCachedModel("gemini-2.5-flash", client, cached_content=[...])
strategy = GeminiStrategy(model, response_parser=lambda r: r.text)
```

## [0.5.0] - 2025-11-25

Release focused on developer experience improvements and code quality.

### Added

- **`TokenTrackingError` export** - The `TokenTrackingError` class is now exported from the main
  `async_batch_llm` package for users who want to catch it explicitly when handling failed LLM calls
  with token usage tracking.
- **Type aliases** - Added convenience type aliases for the most common use case (string input,
  typed output, no context):
  - `SimpleBatchProcessor[T]` → `ParallelBatchProcessor[str, T, None]`
  - `SimpleWorkItem[T]` → `LLMWorkItem[str, T, None]`
  - `SimpleResult[T]` → `WorkItemResult[T, None]`

### Changed

- **Deprecation warnings** - Added `DeprecationWarning` for legacy constructor parameters on
  `ParallelBatchProcessor`:
  - `max_workers` → Use `ProcessorConfig(max_workers=...)`
  - `timeout_per_item` → Use `ProcessorConfig(timeout_per_item=...)`
  - `rate_limit_cooldown` → Use `ProcessorConfig(rate_limit=RateLimitConfig(cooldown_seconds=...))`
- **Module docstring** - Updated to accurately describe the v0.1+ strategy pattern API instead
  of the removed v0.0.x integration modes.
- **Python classifiers** - Added Python 3.13 and 3.14 to pyproject.toml classifiers (tests already
  run on these versions in CI).
- **Config auto-validation** - `ProcessorConfig`, `RetryConfig`, and `RateLimitConfig` now
  automatically validate on construction via `__post_init__`. Invalid values raise `ValueError`
  immediately instead of waiting for explicit `validate()` call. This is a **behavior change**
  for code that creates configs with invalid values and validates later.
- **Logging format** - Replaced emoji characters in log messages with ASCII text alternatives
  for better terminal compatibility:
  - `✓` → `[OK]`
  - `✗` → `[FAIL]`
  - `⚠️` → `[WARN]`
  - `ℹ️` → `[INFO]`

### Fixed

- **Code duplication** - Deduplicated the `TokenTrackingError` class which was defined inline
  3 times in `llm_strategies.py`. Now defined once in `strategies/errors.py`.
- **Code duplication** - Deduplicated `_extract_safety_ratings` method from `GeminiStrategy`
  and `GeminiCachedStrategy` into a module-level function.

## [0.4.0] - 2025-11-19

Major release adding strategy lifecycle management with context managers.

### Added

- **Strategy lifecycle management** - Hybrid approach using context managers for prepare/cleanup
  - Strategies are prepared once when first used (via existing `_ensure_strategy_prepared()`)
  - Strategies are cleaned up once on exit when using `async with` context manager
  - Backward compatible: without context manager, no cleanup is called
  - Prevents adding work after process_all() starts (raises `RuntimeError`)
  - Supports shared strategy instances (prepared once, cleaned up once)
  - Comprehensive test coverage in `test_strategy_lifecycle.py`

### Changed

- **BREAKING: Per-item cleanup removed** - Strategies are no longer cleaned up after each item
  - Previously: `strategy.cleanup()` called after each work item completed
  - Now: `strategy.cleanup()` only called in `__aexit__` when using context manager
  - Migration: Wrap processor in `async with` to enable automatic cleanup
  - For production caches that should persist, make `cleanup()` a no-op
- **Base class tracking** - `BatchProcessor` now tracks unique strategy instances and processing
  state

### Fixed

- **Cancellation propagation** - Added regression test and explicit `asyncio.CancelledError`
  handling so `ParallelBatchProcessor` no longer swallows cancellations inside retries,
  middlewares, or rate-limit coordination. Ensures shutdown behaves correctly on Python 3.10+.

## [0.3.6] - 2025-01-13

Minor release fixing mypy compatibility and improving code quality standards.

### Fixed

- **Mypy compatibility** - Changed `TokenTrackingError` from dynamic base class `type(e)` to static `Exception`
  base (mypy doesn't support dynamic base classes)
- **Linter warnings** - Replaced try/import/except pattern with `importlib.util.find_spec()` for optional
  dependency checks in test_cache_tag_matching.py
- **Documentation linting** - Fixed all 173 markdown linting issues across README.md and docs/
  - Line length violations (wrapped at 120 chars)
  - Missing code block language specifiers
  - Ordered list numbering consistency
  - Bold text used as headings
  - Broken internal links

### Changed

- **Pre-commit checklist** - Added to CLAUDE.md emphasizing quality checks (tests, linter, mypy, markdown-lint)
  before ANY commit
- **Development workflow** - Updated with common mypy issues and solutions

## [0.3.5] - 2025-01-13

Critical bug fix for token tracking when items fail validation.

### Fixed

- **Token tracking for failed items** - Fixed critical bug where failed items showed 0 tokens consumed
  - Previously caused 20-30% cost underestimation when failure rates were high
  - Now correctly tracks tokens even when validation fails
  - Applies to all three built-in strategies: GeminiStrategy, GeminiCachedStrategy, PydanticAIStrategy
  - Exceptions now carry `_failed_token_usage` in `__dict__` for framework to aggregate
  - Critical for accurate production cost tracking and budget planning

### Changed

- **Token extraction timing** - All strategies now extract token usage BEFORE parsing/validation
- **Exception handling** - Enhanced to preserve token usage through exception chain

### Added

- **Test coverage** - Added `test_token_tracking_on_failure.py` to verify fix works correctly
- **MockAgent enhancement** - Added `tokens_per_call` parameter for token tracking tests

## [0.3.4] - 2025-01-12

Compatibility release for google-genai v1.49.0+.

### Fixed

- **google-genai v1.49.0+ compatibility** - Updated GeminiCachedStrategy to handle API changes
  - Detection of three API versions: v1.45, v1.46-v1.48, v1.49+
  - Proper handling of `CreateCachedContentConfig` vs `CachedContent` parameter types
  - Automatic version detection during strategy initialization

### Added

- **API version tests** - Added `test_gemini_api_versions.py` to verify version detection works correctly

## [0.3.3] - 2025-01-11

Bug fix release for proactive rate limiting.

### Fixed

- **Proactive rate limiting** - Fixed bug where `check_rate_limit_proactively()` wasn't being called
  - Now properly prevents hitting API limits by checking before each batch
  - Reduces wasted API calls and improves throughput

### Added

- **Default error classifier** - Added rate limit detection to default classifier (not just Gemini-specific)

## [0.3.2] - 2025-01-11

Bug fix release for error classification.

### Fixed

- **Error classifiers** - Fixed classifiers to prevent wasting retries on logic bugs
  - ValidationError, TypeError, AttributeError, KeyError now marked as non-retryable
  - Saves API costs by not retrying programmer errors

## [0.3.1] - 2025-01-11

Bug fix release for process_all() state contamination.

### Fixed

- **State contamination** - Fixed bug where reusing processor with process_all() contaminated state
  - Each process_all() call now gets fresh state
  - Safe to call process_all() multiple times on same processor instance

## [0.3.0] - 2025-11-10

This release adds advanced retry patterns for multi-stage LLM strategies,
safety ratings access for content moderation, and precise cache tagging for production deployments.

### ⚠️ Breaking Changes

**None** - This release is 100% backward compatible. All new features are opt-in with default values that
preserve existing behavior.

---

### Added

#### Core Features

- **RetryState for multi-stage strategies** (#8, HIGH priority)
  - New `RetryState` class for per-work-item mutable state that persists across all retry attempts
  - Enables advanced retry patterns like partial recovery and progressive prompting
  - Dictionary-style API: `get()`, `set()`, `delete()`, `clear()`, `__contains__()`
  - Automatically created by framework and passed to `execute()` and `on_error()`
  - Each work item gets its own isolated `RetryState` instance
  - Use cases:
    - **Partial recovery**: Parse what succeeded, retry only failed parts (81% cost savings)
    - **Multi-stage strategies**: Track state across validation/formatting/final output stages
    - **Progressive prompting**: Build increasingly detailed prompts based on previous failures
    - **Error tracking**: Count different error types per work item
  - Example:

    ```python
    async def execute(self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None):
        if state and attempt > 1:
            # Use state from previous attempts
            partial_results = state.get("partial_results", {})
            # Retry with context from previous failures
    ```

- **GeminiResponse for safety ratings access** (#3, MEDIUM priority)
  - New `GeminiResponse[TOutput]` generic wrapper class for Gemini API responses
  - Access safety ratings, finish reasons, and raw response metadata
  - Opt-in via `include_metadata=True` parameter on `GeminiStrategy` and `GeminiCachedStrategy`
  - Fields:
    - `output: TOutput` - The parsed output
    - `safety_ratings: dict[str, str] | None` - Content safety ratings (HARM_CATEGORY_HATE_SPEECH, etc.)
    - `finish_reason: str | None` - Why generation stopped (STOP, MAX_TOKENS, SAFETY, etc.)
    - `token_usage: dict[str, int]` - Token counts
    - `raw_response: Any` - Full Google API response object
  - Use cases:
    - Content moderation and filtering
    - Safety rating logging for compliance
    - Debugging generation issues (finish_reason)
    - Accessing provider-specific metadata
  - Example:

    ```python
    strategy = GeminiCachedStrategy(
        model="gemini-2.0-flash",
        client=client,
        response_parser=parse_output,
        cached_content=content,
        include_metadata=True,  # Opt-in for safety ratings
    )
    result = await processor.process_all()
    if isinstance(result.results[0].output, GeminiResponse):
        ratings = result.results[0].output.safety_ratings
        if ratings.get("HARM_CATEGORY_HATE_SPEECH") == "HIGH":
            # Handle unsafe content
    ```

- **Cache tagging for precise cache matching** (#6, LOW priority)
  - New `cache_tags` parameter for `GeminiCachedStrategy`
  - Attach custom metadata tags to caches for precise identification
  - Prevents accidental cache reuse across experiments/versions
  - Tags are checked during cache matching in `_find_or_create_cache()`
  - Version-aware: Falls back gracefully on older google-genai versions
  - Use cases:
    - Multi-tenant applications (tag by customer_id)
    - A/B testing (tag by experiment variant)
    - Version control (tag by schema version or prompt version)
    - Environment isolation (tag by "production" vs "staging")
  - Example:

    ```python
    strategy = GeminiCachedStrategy(
        model="gemini-2.0-flash",
        client=client,
        response_parser=parse_output,
        cached_content=content,
        cache_tags={
            "customer_id": "acme-corp",
            "schema_version": "v2",
            "experiment": "new-prompt-A"
        }
    )
    ```

#### API Changes (Backward Compatible)

- **Updated `LLMCallStrategy` protocol**:
  - `execute()` now accepts optional `state: RetryState | None = None` parameter
  - `on_error()` now accepts optional `state: RetryState | None = None` parameter
  - Default `None` ensures backward compatibility with existing strategies

- **Updated `GeminiStrategy`**:
  - New parameter: `include_metadata: bool = False`
  - Return type: `TOutput | GeminiResponse[TOutput]`
  - Helper method: `_extract_safety_ratings()` for parsing response metadata

- **Updated `GeminiCachedStrategy`**:
  - New parameter: `include_metadata: bool = False`
  - New parameter: `cache_tags: dict[str, str] | None = None`
  - Return type: `TOutput | GeminiResponse[TOutput]`
  - Enhanced `_find_or_create_cache()` to check tags when matching caches
  - Enhanced `_create_new_cache()` to include metadata/tags when creating caches

#### Documentation

- **New implementation plan**: `docs/IMPLEMENTATION_PLAN_V0_3.md`
  - Detailed technical specifications for all three features
  - Priority ranking and use case analysis
  - Test requirements and success criteria
  - Timeline estimates

- **New migration guide**: `docs/MIGRATION_V0_3.md`
  - Complete v0.2 → v0.3 upgrade instructions
  - Emphasizes 100% backward compatibility
  - Code examples for all new features
  - Use case guides and best practices

#### Testing

- **12 new comprehensive tests** (141 tests total, all passing)
  - `tests/test_v0_3_features.py` - Comprehensive test coverage for all v0.3.0 features
  - **RetryState tests** (5 tests):
    - `test_retry_state_persistence` - State persists across retry attempts
    - `test_retry_state_isolation` - Each work item gets its own state
    - `test_retry_state_operations` - Dictionary operations (get/set/delete/clear)
    - `test_retry_state_none_backward_compatibility` - Works when state is None
    - `test_on_error_receives_state` - on_error receives the same state as execute
  - **GeminiResponse tests** (3 tests):
    - `test_gemini_response_with_metadata` - Safety ratings extraction
    - `test_gemini_response_without_metadata` - Backward compatibility (include_metadata=False)
    - `test_gemini_response_generic_type` - Works with different output types
  - **Cache tagging tests** (2 tests):
    - `test_cache_tags_isolation` - Different tags stored correctly
    - `test_cache_tags_none_default` - Tags default to empty dict
  - **Integration tests** (2 tests):
    - `test_retry_state_with_gemini_response` - Features work together
    - `test_shared_strategy_with_retry_state` - Shared strategies with per-item state

### Changed

- **ParallelBatchProcessor** now creates `RetryState` instance per work item
- **ParallelBatchProcessor** passes `retry_state` to `strategy.execute()` and `strategy.on_error()`
- All built-in strategies updated to accept optional `state` parameter

### Performance

- **No performance regression** - All additions are opt-in
- **RetryState is lightweight** - Simple dict wrapper, minimal memory overhead
- **GeminiResponse is zero-cost when disabled** - Only overhead when `include_metadata=True`
- **Cache tags are checked efficiently** - Simple dict comparison during cache matching

### Test Coverage

- **76% overall coverage** (meets requirement)
- **141 tests total** (129 existing + 12 new)
- **All tests passing** including edge cases and integration scenarios

---

## [0.2.0] - 2025-11-09

This release addresses critical production issues identified from real-world usage,
particularly around shared strategy instances for cost optimization with Gemini prompt caching.

### ⚠️ Breaking Changes

#### GeminiCachedStrategy cleanup() Behavior

`cleanup()` now **preserves** caches for reuse by default (previously deleted them).
This enables 70-90% cost savings when running multiple batches within the TTL window.

**Before (v0.1):**

```python
await strategy.cleanup()  # Deleted cache
```

**After (v0.2):**

```python
await strategy.cleanup()  # Preserves cache for reuse

# Explicitly delete if needed
await strategy.delete_cache()
```

**Migration:** If you relied on automatic cache deletion, call `await strategy.delete_cache()` explicitly.
For most production use cases, the new behavior is better (no code changes needed).

See **[Migration Guide](docs/MIGRATION_V0_2.md)** for complete upgrade instructions.

---

### Added

#### Core Features

- **Shared strategy optimization** (#1)
  - Framework now tracks unique strategy instances by `id()`
  - `prepare()` called only once per unique strategy instance, even with concurrent workers
  - Thread-safe via double-checked locking pattern
  - Enables cost optimization: single cache shared across all work items
  - New internal method: `_ensure_strategy_prepared()`

- **Cached token tracking** (#4)
  - `BatchResult.total_cached_tokens` - Sum of cached input tokens across all work items
  - `BatchResult.cache_hit_rate()` - Calculate cache hit rate as percentage
  - `BatchResult.effective_input_tokens()` - Calculate actual cost after caching discount (90% savings)
  - `ProcessingStats.total_cached_tokens` - Track cached tokens in stats

- **Automatic cache renewal** (#7)
  - New parameter: `GeminiCachedStrategy.cache_renewal_buffer_seconds` (default: 300s)
  - New parameter: `GeminiCachedStrategy.auto_renew` (default: True)
  - Proactive cache expiration detection via `_is_cache_expired()`
  - Automatic renewal before API calls to prevent expiration errors
  - Critical bug fix: Reused caches now use actual creation time (not current time)

- **Cache reuse across runs** (#5, #6)
  - `GeminiCachedStrategy._find_or_create_cache()` - Find existing caches or create new ones
  - Caches matched by model name suffix
  - Preserves caches between pipeline runs (within TTL window)
  - Enables 70-90% cost savings for recurring jobs

- **`GeminiCachedStrategy.delete_cache()` method** (#5)
  - Explicit cache deletion for tests and one-off jobs
  - Separated from `cleanup()` hook for better lifecycle control

#### API Compatibility

- **google-genai v1.46+ support** (#2)
  - Auto-detects API version via `_detect_google_genai_version()`
  - Uses `CreateCachedContentConfig` for v1.46+
  - Falls back to legacy API for v1.45 and earlier
  - Both versions fully supported

#### Documentation

- **Enhanced `LLMCallStrategy.cleanup()` docstring**
  - Clarified use cases (connections, locks, metrics)
  - Documented what NOT to use it for (cache deletion)
  - Added note on cache lifecycle best practices

- **Updated README.md**
  - New section: "Shared Strategies for Cost Optimization"
  - Enhanced "Token Tracking" section with cache metrics examples
  - Added cache hit rate and effective cost calculation examples

- **New migration guide**: `docs/MIGRATION_V0_2.md`
  - Complete v0.1 → v0.2 upgrade instructions
  - Breaking changes with code examples
  - New features with usage patterns
  - Troubleshooting guide

- **Implementation plan**: `docs/IMPLEMENTATION_PLAN_V0_2.md`
  - Detailed technical specifications
  - Issue analysis and solutions
  - Test strategy and coverage goals

### Changed

- `GeminiCachedStrategy.prepare()` now calls `_find_or_create_cache()` (reuses existing caches)
- `GeminiCachedStrategy.execute()` now includes automatic cache renewal logic
- `GeminiCachedStrategy.cleanup()` preserves caches by default (breaking change)

### Deprecated

- `GeminiCachedStrategy.cache_refresh_threshold` - Use `cache_renewal_buffer_seconds` instead
  - Percentage-based threshold less predictable than absolute time for long jobs
  - Will be removed in v0.3.0

### Fixed

- **Multiple `prepare()` calls on shared strategies** (#1)
  - Framework called `prepare()` once per work item, not once per unique strategy
  - With concurrent workers, created multiple caches instead of one
  - Fixed via `_ensure_strategy_prepared()` with double-checked locking

- **Cache expiration errors in long pipelines** (#7)
  - Pipelines longer than cache TTL failed with "cache expired" errors
  - Fixed via proactive renewal with configurable buffer

- **Cache reuse creation time bug** (#7)
  - Reused caches incorrectly used current time instead of actual creation time
  - Caused expiration detection to fail (thought cache was newer than it was)
  - Fixed to use `cache.create_time.timestamp()` from Google's servers

### Tests

- **17 new tests added** (95 tests total, all passing)
  - `test_shared_strategies.py` - 6 tests for shared strategy optimization
  - `test_gemini_api_versions.py` - 3 tests for API version detection
  - `test_token_tracking.py` - 8 tests for cached token tracking

- **Updated existing tests**
  - `test_gemini_cached_strategy_lifecycle` - Updated for new cleanup() behavior
  - `test_gemini_cached_strategy_auto_renewal` - Renamed and updated for auto-renewal

### Performance

- **No performance regression** - All optimizations are additions
- **Shared strategies reduce API calls** - One cache creation instead of multiple
- **Token tracking is O(n)** - Simple sum over results

---

## [0.1.0] - 2025-11-09 (untagged)

### ⚠️ Breaking Changes

This major release introduces the **LLM Call Strategy Pattern**,
providing a flexible, provider-agnostic architecture for batch LLM processing.

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
from async_batch_llm import PydanticAIStrategy

strategy = PydanticAIStrategy(agent=agent)
work_item = LLMWorkItem(
    item_id="item_1",
    strategy=strategy,
    prompt="Test prompt",
)
```

See **[Migration Guide](docs/archive/MIGRATION_V0_1.md)** for complete upgrade instructions.

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
- **Proactive rate limiting** to prevent hitting API rate limits
  - Configure via `ProcessorConfig.max_requests_per_minute`
  - Throttles requests before they hit the API (prevents 429 errors)
  - Uses `aiolimiter` for token bucket rate limiting
  - Coordinates across all workers (shared limiter instance)
  - Complements reactive rate limit handling (cooldown after 429s)
  - Optional: Set to `None` to disable (default behavior)
  - Validation warns if rate limit is lower than worker count

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
- **`docs/archive/MIGRATION_V0_1.md`** - Comprehensive v0.0.x → v0.1 migration guide
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

- **`async_batch_llm.testing.MockAgent`** - Mock agent for testing without API calls
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
- **Error classification logic bug handling** - Error classifiers now properly distinguish logic bugs from
  transient failures
  - `DefaultErrorClassifier` and `GeminiErrorClassifier` now explicitly check for logic bug exceptions
    (`ValueError`, `TypeError`, `AttributeError`, etc.)
  - Logic bugs are marked as non-retryable to avoid wasting retry attempts and tokens on deterministic failures
  - Pydantic `ValidationError` is explicitly marked as retryable (LLM might generate valid output on retry)
  - Generic `Exception` instances remain retryable (allows custom transient errors and test mocks)
  - Prevents wasting `max_attempts` retries on programming errors that won't be fixed by retrying
  - Added regression test `test_logic_bugs_fail_fast` to ensure logic bugs fail after 1 attempt
  - Fixed `test_token_usage_tracked_across_retries` exception chain construction

---

### Migration Path

**Upgrading from v0.0.x?** Follow these steps:

1. **Read the Migration Guide**: `docs/archive/MIGRATION_V0_1.md`
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
