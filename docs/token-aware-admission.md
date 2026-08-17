# Token-Aware Admission

Token-aware admission is an optional local quota smoother for provider request
and token budgets. It controls when an attempt may start; it does not change
provider concurrency, transport capacity, or retry policy.

## Four controls, four jobs

| Control | Bounds | Typical owner |
| --- | --- | --- |
| Worker concurrency | Framework tasks doing useful work | `concurrency` / `max_workers` |
| Provider concurrency | Calls holding a provider-capacity slot | `max_provider_concurrency` or the strategy/model |
| Requests per minute (RPM) | Physical attempts admitted per quota scope | `max_requests_per_minute` |
| Tokens per minute (TPM) | Estimated then reconciled token load per quota scope | `max_tokens_per_minute` plus an estimator |

Equal request counts are not equal load. Ten classification requests may fit
inside one long generation's token budget. Raising workers or connection-pool
size cannot overcome RPM or TPM; it only creates more tasks waiting at the
quota gate.

The live-attempt order is:

```text
coordinated cooldown → token estimation → atomic RPM+TPM reservation
→ provider-capacity wait → provider start
```

No provider-capacity slot is held during estimation or quota waiting.
`WorkItemResult.quota_wait_seconds` is therefore separate from
`admission_wait_seconds`, which measures provider-capacity wait.

## Configure RPM and TPM

```python
from async_batch_llm import CharacterTokenEstimator, ProcessorConfig

config = ProcessorConfig(
    concurrency=32,
    max_requests_per_minute=500,
    max_tokens_per_minute=200_000,
    token_estimator=CharacterTokenEstimator(
        characters_per_token=4.0,
        expected_output_tokens=400,
    ),
)
```

TPM is opt-in. When `max_tokens_per_minute` is configured, every live
provider attempt needs a `TokenEstimate`. Supply `token_estimator` on the
config, on `CallableStrategy`, or through a strategy's `estimate_tokens()`
hook. A missing estimator fails before provider work with
`TokenEstimatorRequired`. An individual estimate larger than the configured
bucket fails immediately with `TokenEstimateExceedsLimit`; it cannot become
admissible by waiting.

`CharacterTokenEstimator` uses a character-ratio heuristic and a fixed
expected output allowance. It is approximate and is never enabled
automatically. Prefer a provider tokenizer and a workload-specific output
estimate when accuracy matters.

## Reservation and reconciliation

Each physical attempt atomically reserves one RPM unit and its estimated input
plus output tokens from one FIFO gate. It never reserves RPM and then waits
separately for TPM.

After provider start:

- Known usage below the estimate refunds the difference.
- Known usage above the estimate records debt. Availability may go negative;
  later attempts wait until refill pays the debt.
- Explicit known zero refunds the full token reservation but still consumes
  the RPM unit because a provider attempt started.
- Unknown usage retains the estimate conservatively. This is distinct from
  known zero.
- Recoverable failed-attempt usage is reconciled before the retry. Every
  physical retry then receives a fresh estimate and reservation.

Cancellation before provider start refunds both the RPM unit and token
reservation. Cancellation after provider start follows the same known/unknown
usage rules as any other started attempt. An item or batch deadline may expire
while waiting; the waiting attempt makes no provider call and leaves no live
reservation.

Dry-run, compatible artifact replay, and middleware-filtered items bypass live
quota admission. They emit no quota-admission events and mutate no RPM/TPM
state.

## Quota scopes

`quota_scope` identifies the account or upstream budget shared by strategies.
Object identity defines sharing. By default it follows `concurrency_scope`, so
existing strategies preserve their established ownership. Override it when
one account quota spans multiple models or when one shared client serves
independent accounts.

Quota scope and concurrency scope answer different questions:

- `quota_scope`: which calls spend the same RPM/TPM budget and share cooldown?
- `concurrency_scope`: which calls contend for the same client/provider
  capacity?

Two models sharing one account quota:

```python
from async_batch_llm import ArtifactIdentity, CallableStrategy, TokenEstimate

shared_quota = object()


async def call_model_a(prompt, attempt, state):
    ...


async def call_model_b(prompt, attempt, state):
    ...


def estimate_a(prompt, *, strategy, attempt, state):
    return TokenEstimate(input_tokens=count_model_a(prompt), output_tokens=300)


def estimate_b(prompt, *, strategy, attempt, state):
    return TokenEstimate(input_tokens=count_model_b(prompt), output_tokens=600)


strategy_a = CallableStrategy(
    call_model_a,
    quota_scope=shared_quota,
    token_estimator=estimate_a,
    identity=ArtifactIdentity(
        provider="example",
        model="model-a",
        prompt_version="v1",
        parser_version="v1",
        application_version="v1",
    ),
)
strategy_b = CallableStrategy(
    call_model_b,
    quota_scope=shared_quota,
    token_estimator=estimate_b,
    identity=ArtifactIdentity(
        provider="example",
        model="model-b",
        prompt_version="v1",
        parser_version="v1",
        application_version="v1",
    ),
)
```

For independent account budgets, use distinct identities:

```python
account_a_quota = object()
account_b_quota = object()

strategy_a = CallableStrategy(
    call_model_a,
    quota_scope=account_a_quota,
    token_estimator=estimate_a,
    identity=identity_a,
)
strategy_b = CallableStrategy(
    call_model_b,
    quota_scope=account_b_quota,
    token_estimator=estimate_b,
    identity=identity_b,
)
```

Do not derive scope identities from credentials or place arbitrary scope
representations in logs or metrics. Events expose only run-local ordinal scope
IDs.

## Retry-aware estimators

The estimator receives the effective middleware-replaced strategy and prompt,
the logical attempt number, and the item's `RetryState`. This allows validation
recovery or model escalation to change the estimate:

```python
def estimate(prompt, *, strategy, attempt, state):
    escalated = state is not None and state.get("model") == "large"
    output_allowance = 1_200 if escalated else 300
    return TokenEstimate(
        input_tokens=provider_tokenizer(prompt),
        output_tokens=output_allowance,
    )
```

Synchronous estimators run off the event loop. Asynchronous estimators are
awaited directly. Estimator failures are redacted framework errors and are not
sent through strategy or middleware recovery hooks.

## Visibility boundaries

Token accounting covers attempts visible to ABL, including recoverable
failed-attempt usage. Retries hidden inside an upstream gateway require
gateway-reported usage to be visible. If a gateway returns only aggregate
usage, ABL cannot reconstruct its internal attempts or exact reservation
deltas.

Providers implement quotas differently: fixed windows, rolling windows,
weighted model budgets, cached-token rules, and account-level policies all
exist. ABL's gate is a local continuously refilled smoother. It reduces bursts
but does not claim to reproduce a provider's enforcement exactly.

TPM admission also does not schedule active GPU sequences, KV-cache occupancy,
or instantaneous decode-token capacity. Use a serving system or gateway that
owns those resources when that is the real constraint.

## FIFO trade-off

Admission is FIFO within each quota scope. This prevents later small requests
from repeatedly bypassing an earlier large request, but a large head item can
delay small items behind it. Split unrelated workloads into intentional quota
scopes only when they truly have independent upstream budgets; otherwise the
head-of-line behavior reflects a real shared limit.

## Observe and troubleshoot

Attempt timing reports estimate components, reserved and reported tokens,
reconciliation delta, quota wait, and run-local scope ordinal. Processor stats
and `MetricsObserver` aggregate bounded quota wait percentiles, reservations,
reported tokens, refunds, debt, known-zero/unknown attempts, estimator
failures, and scope count. `QUOTA_ADMITTED` and `QUOTA_RECONCILED` events expose
attempt-level evidence without using item or scope IDs as metric labels.

If an estimate exceeds the limit, either increase the configured bucket,
reduce expected output, or route that workload to its real independent quota
scope. Waiting cannot fix a request larger than the bucket. If timeouts or 429s
make admission look conservative, inspect `unknown_usage_attempts`: without a
reliable provider total, retaining the reservation is intentional.

See [Choosing Your Limits](choosing-your-limits.md) for sizing order and
[Troubleshooting](troubleshooting.md) for symptom-based guidance.
