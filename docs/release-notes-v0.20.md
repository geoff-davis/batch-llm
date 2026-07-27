# v0.20.0 Release Notes

v0.20 makes async-batch-llm useful as an execution layer around clients and
application services you already have, while completing end-to-end streaming
backpressure and the onboarding path from v0.18.

## Highlights

1. **Bring your own async client.** `CallableStrategy` and `CallOutcome` put an
   existing SDK, gateway client, PydanticAI agent, or internal service through
   the same retry, cooldown, deadline, accounting, artifact, and observer path
   as built-in strategies.
2. **End-to-end streaming backpressure.** `max_result_queue_size` bounds
   completed results waiting for a slow consumer independently of the input
   queue, while control messages remain deliverable.
3. **Correct item-private recovery.** Each item owns its `RetryState`; shared
   strategies no longer risk leaking validation feedback or escalation state
   across concurrent work.
4. **Easier onboarding and progress.** The primary quick start uses `llm()`,
   `process_prompts()`, unified concurrency, coalesced progress, and
   `BatchResult.summary()`. A no-key Colab notebook and application demo are
   included.
5. **Checkpointing and result ergonomics.** Built-in strategies support
   zero-configuration checkpoint identity, summaries and output iteration are
   first-class, and lazy streams can bound both input and result handoffs.
6. **Clearer shared calls.** `LLMCallPool` is the preferred name for the
   queue-less in-process shared executor; `LLMGateway` remains an exact,
   warning-free alias.

The bundled `progress=True` reporter now observes exact per-item counts without
creating one thread-dispatch task per item. It renders the first update,
coalesces intermediate terminal refreshes, supports growing lazy-source totals,
and forces one exact final state on every shutdown path. Custom callbacks remain
per-item.

## Compatibility for v0.18 users

Upgrade directly from v0.18.x. `timeout_per_item` still aliases
`attempt_timeout`, and callable `cache_hit_rate()` still aliases the property;
both warn and are planned for removal in the next major release. Existing
tuple-returning strategies, `LLMGateway` imports, completion-order defaults,
and the v1 checkpoint schema remain supported.

See [Migrating from v0.18.x to v0.20.0](MIGRATION_V0_20.md) for before/after
examples and checkpoint guidance.

## Why there is no v0.19 release

v0.19.0 was not published. Its planned onboarding and ergonomics improvements
were completed alongside embedded application integration and are included in
v0.20.0. No v0.19 package, tag, synthetic changelog section, or intermediate
migration step is needed.

## Intentional boundaries

This release does not add generic job inputs, distributed execution, a network
gateway, provider routing, native provider batch submission, scheduling, DAGs,
or a cost database. Bespoke Curator, AI gateways, native batch APIs, and durable
workflow engines are often better fits for those jobs; see
[Choosing Between ABL and Alternatives](comparison.md).
