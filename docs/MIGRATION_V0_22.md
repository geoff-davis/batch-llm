# Migrating from v0.21.x to v0.22

v0.22 adds scoped proactive token admission without changing default execution.
Applications that do not configure RPM or TPM keep their existing behavior.

## Additive configuration

`ProcessorConfig` adds:

```python
ProcessorConfig(
    max_tokens_per_minute=200_000,
    token_estimator=my_estimator,
)
```

`max_requests_per_minute` remains optional and now participates in the same
atomic per-scope gate as TPM. Positive sub-one RPM values remain supported.
Boolean and non-finite RPM values are rejected.

TPM requires an explicit estimator. Supply `ProcessorConfig.token_estimator`,
`CallableStrategy(token_estimator=...)`, `ModelStrategy(token_estimator=...)`,
or override `LLMCallStrategy.estimate_tokens()`. Enabling TPM without one
produces the non-retryable `TokenEstimatorRequired` before provider work.

## Scope ownership

Strategies now expose `quota_scope`, which owns coordinated cooldown, RPM, and
TPM. Its default follows the existing `concurrency_scope`; most single-model
applications need no change. When multiple strategy instances spend one
provider/account quota, return or pass the same stable scope object. Keep
independent accounts in distinct scopes.

v0.22 also resolves the error classifier from the effective strategy after
middleware. Mixed-provider runs therefore use each provider's classifier and
cooldown/quota scope instead of accidentally inheriting one host-wide choice.

## Admission behavior changes

- Dry-run, compatible replay, and middleware-filtered items no longer consume
  proactive RPM or mutate live cooldown/quota state.
- Each physical retry gets a fresh atomic RPM+TPM reservation.
- Known usage refunds overestimates or records underestimation debt. Explicit
  zero is known; missing/unreliable usage retains the reservation as unknown.
- Cancellation before provider start refunds request and token reservations.
- Quota wait occurs before provider-capacity admission and is reported
  separately from `admission_wait_seconds`.

If an application depended on dry-run reducing a live RPM bucket, remove that
assumption. Dry-run is intentionally admission-neutral.

## Timing, events, and serialization

`AttemptTiming` appends optional/defaulted quota fields, and
`WorkItemResult.quota_wait_seconds` sums them. New observer events
`QUOTA_ADMITTED` and `QUOTA_RECONCILED` expose run-local ordinal scope IDs and
reconciliation evidence. Stats and metrics add bounded quota wait percentiles,
reservation, refund, debt, known-zero/unknown, estimator-failure, and scope
counts.

Result and artifact schema versions remain `1`. Decoders default absent v0.22
timing fields, so v0.21 records remain readable. `reported_tokens=None`
(unknown) and `reported_tokens=0` (known zero) remain distinct on new record
round trips. No checkpoint migration or invalidation is required.

## Dependency change

The runtime no longer depends on `aiolimiter`; the scoped FIFO gate is owned by
ABL so RPM and TPM can reserve atomically and reconcile exactly once. If your
application imported `aiolimiter` only because it arrived transitively, add it
to your own dependencies before upgrading. ABL does not expose `aiolimiter`
types in its public API.

## Before deployment

1. Identify quota ownership across every strategy/model route.
2. Validate estimator error on representative prompt and output sizes.
3. Confirm the largest estimate fits the configured bucket.
4. Exercise recoverable failed usage, unknown usage, cancellation, and retry.
5. Monitor refunds, debt, unknown attempts, and quota wait percentiles.

See [Token-Aware Admission](token-aware-admission.md) for the complete contract.
