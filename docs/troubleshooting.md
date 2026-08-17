# Troubleshooting and FAQ

Start with the terminal `WorkItemResult`, its `error_category`, and
`result.timing`. Enable normal Python logging for `async_batch_llm` before
raising concurrency or retry budgets; more work often amplifies the original
problem.

## Throughput stops below the worker count

**Symptom.** Increasing `concurrency` or `max_workers` no longer raises
throughput, and admission or HTTP pool waits grow.

**Likely cause.** The httpx/OpenAI client connection pool is smaller than the
worker pool, an explicit provider-admission limit is lower, the provider is
throttling, or the process is close to its file-descriptor limit. A configured
startup ramp also intentionally begins below steady-state concurrency.

**How to confirm.** Read capacity warnings at processor startup. Compare
`concurrency`, `max_workers`, `max_provider_concurrency`, the strategy's
`max_concurrency`, and client `max_connections`. Inspect
`result.admission_wait_seconds`, provider timing, HTTP pool metrics, and
`ulimit -n` on Unix.

**Fix.** Prefer unified `concurrency=N` with built-in clients so aligned limits
resize together. If using an existing client, size its pool explicitly and
advertise truthful `CallableStrategy(max_concurrency=N)`. Lower concurrency or
raise the OS soft descriptor limit deliberately; do not hide a small pool
behind more workers. Allow the startup ramp to reach steady state before
judging throughput.

**Related guide.** [Choosing Your Limits](choosing-your-limits.md) and the
[Production Checklist](production-checklist.md).

## Persistent 429 responses

**Symptom.** Many items repeatedly enter cooldown, total time grows, or adding
workers makes the run slower.

**Likely cause.** The provider account's request/token quota is below offered
load, a suggested wait is long, the rate-limit retry budget is too generous,
or a gateway and ABL are both retrying the same transport failure.

**How to confirm.** Check provider quota dashboards and response headers. Look
for coordinated-cooldown logs, repeated logical attempt numbers, suggested
waits, and `RateLimitConfig.max_rate_limit_retries` exhaustion. Inspect gateway
logs for hidden retries beneath each ABL attempt.

**Fix.** Reduce concurrency or set a conservative
`max_requests_per_minute`; request higher provider quota when appropriate.
Honor coordinated cooldowns rather than starting more workers. Choose one
layer to own transport retries at full strength and keep the other layer's
policy minimal. Bound rate-limit retries so a permanently throttled endpoint
eventually produces a terminal result.

**Related guide.** [Choosing Your Limits](choosing-your-limits.md)
and [Callable integration](callable-integration.md#retry-recovery).

## Items wait before provider capacity even though connections are free

**Cause.** A scoped RPM/TPM gate admits work before provider capacity. This is
intentional: no connection or provider slot is held while a quota refills.

**Confirm and fix.** Compare `result.quota_wait_seconds` with
`result.admission_wait_seconds` and inspect quota wait percentiles. Tune the
real RPM/TPM budget or estimator; adding connections cannot overcome quota.

## Token estimate exceeds the configured per-minute limit

**Cause.** One estimated input-plus-output total is larger than the bucket, so
it can never become admissible by waiting.

**Fix.** Correct an inflated estimator, reduce the expected output allowance,
raise the limit to the actual account budget, or route the call to a genuinely
independent quota scope. Do not split one provider request merely to satisfy a
local configuration unless the application semantics permit it.

## TPM limiting appears conservative after timeouts or 429s

**Cause.** Once provider work starts, unknown usage retains its token
reservation. A timeout, transport error, or 429 without reliable usage is not
the same as an explicit zero-token response.

**Confirm and fix.** Monitor `unknown_usage_attempts` and reconciliation
events. Extract provider-reported failed usage when available. Otherwise keep
the conservative accounting or increase headroom based on measured behavior;
do not treat missing usage as zero.

## Two providers unexpectedly share a quota

**Cause.** Their strategies return the same `quota_scope` identity, often
because quota scope defaults to a shared concurrency scope.

**Fix.** Give genuinely independent accounts distinct stable scope objects.
Keep them shared if the upstream provider or gateway actually enforces one
account budget.

## Two models do not share an account quota

**Cause.** Separate strategy/model instances default to separate scopes even
though the provider bills them against one account-level quota.

**Fix.** Pass the same `quota_scope` object to both strategies. Scope sharing
also coordinates cooldown, so verify the ownership matches the upstream
account rather than just the model name.

## Large items delay small items

**Cause.** Quota admission is FIFO. A large estimate at the head waits for
enough tokens and later small estimates do not bypass it indefinitely.

**Fix.** Improve estimates and expected-output bands. Separate workloads only
when they spend independent upstream quotas; otherwise head-of-line waiting is
the fairness trade-off of one shared budget.

## Gateway retrying makes reported usage differ from ABL attempts

**Cause.** A gateway may perform several upstream attempts inside one visible
`strategy.execute()` call. ABL can account only for the usage the gateway
returns.

**Fix.** Choose one strong retry owner, inspect gateway logs, and configure the
gateway to return aggregate usage. Do not claim per-upstream-attempt visibility
when the gateway hides it.

## Dry-run or replay has no quota events

**Cause.** This is the contract. Dry-run, compatible replay, and
middleware-filtered items perform no live admission and mutate no RPM/TPM
state.

**Fix.** None. Test quota behavior with a credential-free live fake strategy
instead of using dry-run or replay as a quota simulation.

## Why quota wait is separate from admission wait

Quota wait measures cooldown-following RPM/TPM admission. Admission wait
measures provider-capacity semaphore wait. Keeping them separate shows whether
the account budget or transport/provider capacity is the bottleneck and
prevents double-counting in summaries.

**Related guide.** [Token-Aware Admission](token-aware-admission.md).

## An item takes longer than `attempt_timeout`

**Symptom.** A logical item runs for several multiples of the configured
attempt timeout.

**Likely cause.** `attempt_timeout` limits one call to `strategy.execute()`.
Provider-capacity admission, HTTP-client waiting, coordinated cooldowns, retry
backoff, and later attempts are outside that single-attempt budget.

**How to confirm.** Compare each `result.timing.attempts` entry with total item
time and admission/cooldown/backoff fields. Distinguish the SDK/httpx timeout
from ABL's attempt timeout. On Python 3.10 ABL uses its compatibility timeout
path, but the semantics are the same.

**Fix.** Set the HTTP client timeout for network phases, `attempt_timeout` for
one provider attempt, `GuardrailConfig.total_timeout_per_item` for the whole
logical item, and `GuardrailConfig.batch_timeout` for the run. Size them in
that order; a batch deadline can stop admission while active work is drained
or cancelled according to `abort_mode`.

**Related guide.** [Deadlines and Fail-Fast Guardrails](guardrails.md).

## Empty, blocked, or tool-only responses

**Symptom.** A provider returns HTTP success but there is no usable text, or an
item fails with `EmptyResponseError`.

**Likely cause.** Content filtering or a Gemini safety block, length
termination, a response containing only tool calls, or an actually empty
choice. Text-oriented built-in strategies do not silently treat tool-only
responses as text.

**How to confirm.** Inspect `result.exception`, provider finish/stop reason,
typed provider metadata, safety ratings, tool-call views, and token usage. A
billed empty response can still contribute failed-attempt tokens.

**Fix.** Adjust the prompt or safety configuration where policy permits,
increase the output limit for length termination, or use a strategy/parser
that intentionally handles tool calls. Do not retry a policy block blindly.
Use `TokenTrackingError` in a custom callable when a billed response fails
application parsing so its usage is retained.

**Related guide.** Provider guides and
[Callable usage and metadata](callable-integration.md#usage-and-metadata).

## OpenRouter reports an error inside HTTP 200

**Symptom.** The HTTP request succeeds but the OpenRouter payload contains an
upstream provider error.

**Likely cause.** OpenRouter can encode a provider failure in an otherwise
successful HTTP response.

**How to confirm.** The built-in model raises `ProviderResponseError`; inspect
its embedded code and message rather than only the HTTP status. Embedded 429s
are classified as rate limits.

**Fix.** Use `OpenRouterStrategy` and its recommended classifier, or reproduce
the same payload check in a custom client and supply an appropriate
`error_classifier`. Keep retry ownership clear if OpenRouter or another
gateway also retries upstream failures.

**Related guide.** [OpenRouter Integration](OPENROUTER_INTEGRATION.md).

## A run appears to hang

**Symptom.** No new result is observed even though the process is alive.

**Likely cause.** The consumer may be slow, a bounded result queue may be
applying intended backpressure, a post-processor or callback may block, a
provider call may be long, or workers may be in cooldown/backoff. Artifact I/O
can also delay checkpoint-before-publication. At the low level, preloading a
bounded queue before workers start cannot make progress. A consumer that stops
draining `process_stream()` leaves publishers waiting until generator cleanup.
An external gateway may be retrying invisibly underneath ABL.

**How to confirm.** Check in this order:

1. Is the stream consumer still awaiting the next result?
2. Are `max_result_queue_size` slots full while the sink writes slowly?
3. Is a post-processor or custom callback doing blocking I/O?
4. Do attempt/provider timings show one long call?
5. Do logs show cooldown or retry backoff?
6. Is artifact storage slow or unavailable?
7. Was a low-level bounded queue filled before `start()`/`process_all()`?
8. Do gateway logs show nested retries?

At debug time, inspect unfinished `asyncio.all_tasks()` and their stacks; task
names/stacks distinguish workers waiting on result capacity from provider,
artifact, callback, and producer tasks. Observer events and per-result timing
provide a less invasive production view.

**Fix.** Keep the consumer draining, move blocking callbacks off the event
loop, size input/output buffers for expected sink latency, and use total/batch
deadlines. In low-level streaming call `start()` before feeding and always call
`finish()`. Close the async generator explicitly when manually stopping early.

**Related guide.** [Bounded Work and Backpressure](bounded-work.md).

## An artifact record is incompatible

**Symptom.** Resume executes the provider instead of replaying a record, or a
strict artifact operation reports incompatibility.

**Likely cause.** The compatibility fingerprint changed. It covers the item ID,
prompt, participating context, and artifact identity, which can include model,
provider, prompt version, parser version, and application version.

**How to confirm.** Compare the current prompt/context and `ArtifactIdentity`
with the artifact header and record fingerprint. Check whether code changed a
model route, parser schema, or application version.

**Fix.** Restore the compatible identity only if the behavior truly is
compatible; otherwise allow execution and append a new record. Strictness
prevents returning stale output after semantic changes.

**Related guide.** [Results, Artifacts, and Resume](results-and-artifacts.md#compatibility-matching).

## `CallableStrategy` cannot prepare an artifact identity

**Symptom.** Artifact preparation fails before the callable runs and asks for
`ArtifactIdentity`.

**Likely cause.** A Python callable does not reveal a stable provider, model,
gateway route, parser version, or application version. Lambdas, closure IDs,
memory addresses, and `repr()` are unsafe replay identities.

**How to confirm.** Check that neither `JsonlArtifactStore(identity=...)` nor
`CallableStrategy(identity=...)` supplies an identity.

**Fix.** Provide a stable explicit identity and bump its version fields when
prompting, parsing, routing, or application semantics change. A store identity
overrides the strategy identity when both are present.

**Related guide.** [Callable integration](callable-integration.md#artifact-identity-and-replay).

## Resume did not call the provider

**Symptom.** A resumed item completes without an SDK/gateway invocation.

**Likely cause.** This is successful replay: a compatible record matched the
selected resume policy.

**How to confirm.** Inspect `result.replayed_from_artifact` and the artifact
path/identity. Replayed usage remains available for audit but is excluded from
current-run consumption totals.

**Fix.** None when replay was intended. Use `ResumePolicy.NONE` for a live run,
or change a genuine compatibility component rather than fabricating a version
bump merely to bypass cache.

**Related guide.** [Results, Artifacts, and Resume](results-and-artifacts.md).

## Resume did call the provider

**Symptom.** An item expected to replay instead executes again.

**Likely cause.** Its fingerprint or identity changed; the previous result was
a failure under `REUSE_SUCCESSES`; output persistence was disabled; the JSONL
record is malformed/unsupported; or the callable/model identity changed.

**How to confirm.** Check the resume policy, `replayed_from_artifact`, artifact
warnings, stored record type/output, and current identity. Compare prompt and
context byte-for-byte after their canonical serialization.

**Fix.** Use `REUSE_ALL` only when replaying prior terminal failures is desired.
Persist outputs required for replay and repair/replace malformed artifacts.
Do not weaken identity compatibility to force reuse.

**Related guide.** [Results, Artifacts, and Resume](results-and-artifacts.md#resume-policies).

## Memory use grows

**Symptom.** Resident memory increases with a very large run.

**Likely cause.** `process_prompts()` retains the final `BatchResult`, an input
or output queue is unbounded, the application retains streamed results, or
concurrency permits many worker-local results. `JsonlArtifactStore` also loads
record history and builds replay indexes in memory when preparing an existing
artifact.

**How to confirm.** Check whether the code uses `process_stream()` with a lazy
source, positive `max_queue_size` and `max_result_queue_size`, and an
incremental sink. Measure application-held collections and artifact history,
not just queue sizes.

**Fix.** Stream and persist results instead of collecting them. Bound both
queues, keep concurrency proportional to memory, discard consumer-held data,
and rotate/compact long JSONL histories according to application retention
policy. The approximate live result bound is result-queue capacity plus active
workers plus the consumer's current item—not result-queue capacity alone.

**Related guide.** [Bounded Work and Backpressure](bounded-work.md).

## Progress is missing, noisy, or has a growing total

**Symptom.** `progress=True` logs instead of drawing a bar, totals increase, or
a custom callback fires much more often than the terminal display.

**Likely cause.** tqdm is not installed; a lazy producer continues accepting
items; or the custom callback's per-item contract is being confused with the
bundled reporter's throttled rendering.

**How to confirm.** Install/import tqdm and check the fallback notice. Inspect
`ProcessorConfig.progress_refresh_interval_seconds` separately from
`progress_interval`. Verify whether `progress` is `True` or a callable.

**Fix.** Install `async-batch-llm[progress]`. Tune
`progress_refresh_interval_seconds` for the bundled reporter; the logging
fallback uses at least a one-second cadence. Expect lazy-source totals to grow.
Aggregate inside a custom callback if its documented per-item delivery is too
frequent for an external metrics backend.

**Related guide.** [Getting Started](getting-started.md#5-add-progress).

## Is `LLMCallPool` an AI gateway?

**Symptom.** Code expects `LLMCallPool` to provide an HTTP endpoint, credential
store, provider routing, model catalog, or centralized policy plane.

**Likely cause.** Its compatibility name, `LLMGateway`, historically suggested
a network gateway. Both names refer to the same in-process shared executor.

**How to confirm.** `LLMCallPool is LLMGateway`; calls use the shared executor
directly and there is no background dispatcher or service.

**Fix.** Use LiteLLM or another real gateway for central routing/governance.
Place ABL above it when an application run also needs ABL's bounded execution,
recovery, deadlines, and artifacts.

**Related guide.** [Single Call and Shared Call Pool](api/single-gateway.md) and
[Choosing Between ABL and Alternatives](comparison.md#litellm-and-other-ai-gateways).

## Deprecation warnings after upgrading

**Symptom.** Code warns about `timeout_per_item` or calling
`cache_hit_rate()`.

**Likely cause.** v0.20 uses `attempt_timeout` to make per-attempt semantics
explicit, and exposes `cache_hit_rate` as a property. Compatibility shims remain
so v0.18 applications can migrate incrementally.

**How to confirm.** Run tests with deprecations visible and locate
`ProcessorConfig(timeout_per_item=...)`, `.timeout_per_item`, or
`.cache_hit_rate()`.

**Fix.** Replace them with `ProcessorConfig(attempt_timeout=...)` and
`.cache_hit_rate`. The old forms remain supported without behavior changes in
v0.20 and are planned for removal in the next major release.

**Related guide.** [Migrating from v0.18.x to v0.20.0](MIGRATION_V0_20.md).
