# Results, Audit Artifacts, and Resume

Production runs often need two different outputs: a convenient batch summary and
an append-only checkpoint that survives interruption. `BatchResult` serialization
serves the first use case; `JsonlArtifactStore` serves the second.

## Completion order and input order

Batch processing and `process_stream()` publish results in **completion order** by
default. This keeps fast items visible while a slow or retrying item is still in
flight and preserves existing behavior.

Use `preserve_order=True` when collecting a batch in input order:

```python
result = await process_prompts(
    strategy,
    [("duplicate", "slow prompt"), ("duplicate", "fast prompt")],
    preserve_order=True,
)
```

The processor assigns an internal `submission_index` when each item is accepted.
It does not derive order from `item_id`, so duplicate IDs are safe. To reorder an
existing batch without mutating it, call `ordered = result.in_input_order()`.
This raises `ValueError` if any result predates submission indexes; it never
guesses.

Each `LLMWorkItem` object is a single submission. Duplicate IDs on distinct
objects are supported, but submitting the identical mutable object twice raises
`ValueError`; create a new work item for each queue entry.

`process_stream()` intentionally remains completion ordered. Ordered streaming
would block all later results behind one slow early item and require a reorder
buffer that can grow without a safe bound.

## JSON result serialization

Result mappings use the named `async-batch-llm-result` schema and an integer
schema version. They include terminal status, submission index, error category,
replay state, token usage, complete attempt/item timing, and batch termination
metadata.

```python
from pathlib import Path
from async_batch_llm import BatchResult

value = result.to_json()
Path("summary.json").write_text(value, encoding="utf-8")

restored = BatchResult.from_json(
    Path("summary.json").read_text(encoding="utf-8")
)

result.to_jsonl("summary.jsonl")
restored_lines = BatchResult.from_jsonl("summary.jsonl")
```

JSONL result files contain one complete versioned `WorkItemResult` record per
line. A zero-result batch uses one versioned batch-metadata record so termination
state still round-trips. These are summary exports, not resumable checkpoints;
use an artifact store for checkpoint/resume.

The safe encoder accepts normal JSON primitives plus dataclasses, Pydantic
models, enums, `datetime`/`date`/`time`, UUIDs, filesystem paths, tuples, and
sets. Tuples and sets normalize to JSON lists; dataclasses and Pydantic models
normalize to mappings; dates, UUIDs, and paths normalize to strings. Sets are
ordered deterministically. Without explicit decoders, those normalized values
remain JSON-native mappings, lists, and strings after deserialization.

Application-specific values need an explicit encoder:

```python
payload = result.to_json(
    encoder=lambda value: {"widget_id": value.id}
)
```

`WorkItemResult.from_dict()` and the `BatchResult.from_*()` methods accept
`output_decoder` and `context_decoder` hooks for trusted type reconstruction.
Deserialization never imports a class named by untrusted data. Exceptions are
stored only as module name, class name, and redacted message; the restored
runtime `exception` is `None`. Tracebacks and raw exception objects are never
persisted. Framework-controlled error text and values under structured
authorization/API-key/token keys are redacted. Other user-controlled strings
round-trip without text rewriting, so applications must keep credentials out
of prompts, outputs, context, and metadata.

Unsupported values, malformed input, and future schema versions raise
`ResultSerializationError`; values are never silently replaced with `repr()`.

## Resumable JSONL artifacts

An artifact begins with a version-1 manifest followed by versioned item records.
The manifest records UTC creation time, package version, canonical identity,
its SHA-256 fingerprint, and optional user metadata. Each terminal item record
contains input fingerprints, current submission index, strategy class,
identity/provenance, safe result data, timing, token use, error category, replay
eligibility, and optional caller-calculated cost.

The minimal form needs only a path (v0.20.0):

```python
from async_batch_llm import JsonlArtifactStore, ResumePolicy, process_prompts

result = await process_prompts(
    strategy,
    prompts,
    artifact_store=JsonlArtifactStore("runs/customer-tagging.jsonl"),
    resume=ResumePolicy.REUSE_SUCCESSES,
)
```

When no `ArtifactIdentity` is given, `provider` and `model` are inferred from
ordinary model-backed strategies at run start (built-in models map to their
provider names; custom models use their class name) and the remaining identity
fields default to `"unversioned"`. Prompt — and, by default, context — still
participate in the per-item compatibility fingerprint, so a changed prompt or
a changed model never silently replays a stale result.

Automatic identity is homogeneous within one live store instance/run. The first
prepared item pins its inferred identity; every later item is inferred and must
match before artifact lookup, append, or provider work begins. Use separate
stores for heterogeneous strategies, or pass one deliberate explicit
`ArtifactIdentity` that describes the mixed/routed run. A later store instance
may pin a different automatic identity and append to the same physical
artifact—older identities remain audit history and only compatible records
participate in replay.

`CallableStrategy` is intentionally stricter: an arbitrary function cannot
safely reveal its provider, model, route, parser, or application version. Pass
`identity=ArtifactIdentity(...)` to `CallableStrategy` or directly to
`JsonlArtifactStore`. Omitting both fails before the invocation callback runs;
ABL never derives identity from a lambda, closure, object ID, memory address, or
unstable `repr()`. An explicit store identity takes precedence over the
strategy's identity.

**When to use the full identity:** versioned production pipelines. An
explicit `ArtifactIdentity` lets a prompt-template change, parser change, or
application release invalidate replay even when the literal prompt text is
unchanged (e.g. context assembled outside the prompt, or a new parser reading
the same outputs):

```python
from async_batch_llm import (
    ArtifactIdentity,
    JsonlArtifactStore,
    ResumePolicy,
    process_prompts,
)

store = JsonlArtifactStore(
    "runs/customer-tagging.jsonl",
    identity=ArtifactIdentity(
        provider="openai",
        model="example-model",
        prompt_version="v3",
        parser_version="v2",
        application_version="2026.07",
    ),
)

result = await process_prompts(
    strategy,
    prompts,
    artifact_store=store,
    resume=ResumePolicy.REUSE_SUCCESSES,
)
```

An explicit identity is the caller-owned compatibility boundary. It may
intentionally describe a mixed-strategy run, but the caller must change it
whenever routing, provider, model, parser, prompt policy, or application
semantics should invalidate replay.

A terminal record is flushed before its result is returned or yielded. Set
`fsync=True` for an operating-system durability barrier after every record;
flush-only is the default. A crash-truncated final line is ignored on reopening,
while malformed complete or middle records fail clearly.

### Compatibility matching

Replay requires all of the following to match:

- item ID;
- SHA-256 prompt fingerprint;
- context fingerprint when `context_in_identity=True` (the default);
- combined input fingerprint; and
- complete `ArtifactIdentity` fingerprint and supported artifact schema.

Matching only an item ID is never sufficient. Changing provider, model,
prompt/parser/application version, identity `extra`, prompt, or participating
context invalidates the old record. When several records match, the newest
complete record wins.

Context is canonically JSON-encoded before a provider call. Supply `encoder=`
or `context_fingerprinter=` when an application context is not supported; the
store raises `ArtifactSerializationError` rather than silently excluding that
identity component.

Sensitive structured values are redacted from persisted identity, context,
output, and metadata mappings. Their original values still feed the one-way
context/identity fingerprint, so a credential change invalidates replay without
writing the credential itself to the artifact.

### Resume policies

- `ResumePolicy.NONE` never reuses old results but still checkpoints new ones.
- `ResumePolicy.REUSE_SUCCESSES` reuses the newest compatible success and reruns
  failures, missing items, and stale items.
- `ResumePolicy.REUSE_ALL` also reuses the newest compatible terminal failure.

Replayed items do not call the provider and are not appended a second time.
They retain historical output, timing, error, and token use, receive the current
run's `submission_index` and current work-item context, and set
`replayed_from_artifact=True`. Persisted historical context is never decoded or
substituted during replay. Historical
tokens remain visible on the item for audit but are excluded from newly consumed
provider-token statistics returned by `processor.get_stats()`. In contrast,
`BatchResult` aggregate token fields are computed from all returned results and
therefore include replayed historical usage. This makes live processor stats a
"spent this run" view and the collected batch an auditable result-history view.

For a callable, compatible replay likewise bypasses the invocation callback.
Changing its explicit identity invalidates replay and executes it again.

When a billed response fails parsing or validation, raise
`TokenTrackingError(token_usage=...)` from the callback. That attempt's usage is
retained and added to later successful or terminal usage. ABL cannot account
for transport retries hidden inside an upstream gateway unless the gateway
reports them.

Setting `include_output=False` makes a success record audit-only and therefore
ineligible for replay. A failure remains reusable under `REUSE_ALL` because it
does not need a successful output.

### Privacy controls

Prompt and context hashes are always stored for matching, but raw prompts and
raw context are excluded by default. Raw provider responses are never stored.
The independent options are:

- `include_output=True` (default);
- `include_metadata=True` (default);
- `include_prompt=False` (default); and
- `include_context=False` (default).

Outputs and metadata can themselves contain sensitive application data. Review
their shape before enabling artifacts, or disable either field. Raw prompt and
context persistence must be explicitly enabled.

### Generate, review, apply

Artifacts support a provider-free review/apply phase:

```python
from async_batch_llm import JsonlArtifactStore

review = JsonlArtifactStore.read_results(
    "runs/customer-tagging.jsonl",
    successes_only=True,
)
for item in review.results:
    await apply_approved_output(item.item_id, item.output)
```

Inspection restores raw context only when the writer explicitly used
`include_context=True`. Without a `context_decoder`, the restored value remains
JSON-native; a supplied decoder is applied only when stored context exists.
Context omitted by the default privacy policy, including an original `None`,
inspects as `None`.

Replay never substitutes that historical audit context for the live input and
does not invoke the stored-context decoder. It always attaches the current work
item's context and submission index to the replayed result.

For asynchronous inspection through an open store, use
`async for item in store.iter_results(successes_only=True)`.

Costs are optional and caller-supplied because this package does not maintain a
provider price database:

```python
store = JsonlArtifactStore(
    "run.jsonl",
    identity=identity,
    cost_calculator=lambda item: calculate_cost_from_current_prices(item),
)
```

The artifact stores `null` when there is no calculator.

## Indexed SQLite artifacts for large restartable runs

`SqliteArtifactStore` (v0.21) stores the same logical version-1 records in a
standard-library SQLite database with indexed replay lookup and bounded
iteration. It exists for 100k–1M-item restartable runs where scanning or
retaining a JSONL history on open becomes the bottleneck; JSONL remains the
recommended portable, human-inspectable audit format.

```python
from async_batch_llm import ResumePolicy, SqliteArtifactStore, process_stream

store = SqliteArtifactStore("runs/customer-tagging.sqlite")
async for item in process_stream(
    strategy,
    prompts,
    artifact_store=store,
    resume=ResumePolicy.REUSE_SUCCESSES,
):
    await sink(item)
```

Everything documented above for JSONL applies unchanged: identity resolution
and explicit `ArtifactIdentity`, the complete compatibility fingerprint, the
three resume policies, replay accounting, privacy controls
(`include_output` / `include_metadata` / `include_prompt` /
`include_context`), custom encoders/decoders and `context_fingerprinter`, and
optional `cost_calculator`. The two backends share one codec, so a record
that would replay from JSONL replays from SQLite under the same conditions.
Context-free items match through a NULL-safe indexed predicate. There is no
automatic JSONL→SQLite import; opt in by creating a new store, and existing
JSONL files remain fully supported.

| Backend | Best fit | Reopen behavior | Write behavior | Human inspection |
| --- | --- | --- | --- | --- |
| JSONL | Small/medium runs, portable audit log | Resume scans/indexes complete history; `iter_results()` streams line pages | One flushed line per record | Excellent |
| SQLite | 100k–1M restartable runs | Indexed replay and bounded keyset iteration — no history decode on open | Batched transactions with WAL checkpoints | Use SQLite tools/API |

“100k+” is a recommendation, not a cliff — the crossover depends on record
size and environment.

### Batched commits and durability

Appends are grouped into transactions (`commit_batch_size`, default 100,
within `commit_interval_seconds`, default 0.01). An `append()` — and therefore
the processor's publication of that item's result — completes only after its
transaction commits, preserving checkpoint-before-result-publication. A
transaction failure rolls back without exposing partial rows.

`SqliteDurability` selects the synchronization policy:

- `BALANCED` (default) — WAL with `synchronous=NORMAL`. A normal process
  crash retains a consistent database and committed history. A sudden OS or
  power failure may lose transactions committed since the last WAL sync;
  the loss window can span more than one transaction.
- `FULL` — WAL with `synchronous=FULL`; stronger power-loss durability at
  lower throughput. Exact guarantees always depend on SQLite, the OS,
  filesystem, and storage hardware.

WAL auto-checkpointing keeps the log at a bounded plateau during healthy
writes; `close()` drains accepted appends, attempts a truncating checkpoint,
and terminates the store's worker thread. An external long-lived reader can
make that final checkpoint report “busy” — recorded, not an error and not
data loss. SQLite may leave an `-shm` sidecar after a healthy close, and the
database directory will transiently contain `-wal`/`-shm` files during a run.
ABL does not encrypt SQLite artifacts; use filesystem permissions and volume
encryption as your deployment requires.

Cleanup finishes before `close()` reports an error. A previously unobserved
writer or cancelled-caller operation error is delivered first; a distinct
checkpoint or connection-close error is retained for the next `close()`.
Each store-level error is delivered once, and later closes return silently.
Observing an error never restores a failed store to usability.

### Inspection

Live `store.iter_results()` first settles detached errors, prepares the
writable store, and flushes its accepted writes. It then streams a finite
committed high-water snapshot in keyset-paged batches (`read_batch_size`),
decoding one row at a time. No cursor or read transaction is held across an
async yield, so a slow consumer never blocks the writer or WAL checkpointing.

`SqliteArtifactStore.read_results(path)` is a separate path-based,
materializing convenience. It opens an internal SQLite `mode=ro` connection,
never creates a schema or identity, never starts a writer, never changes WAL or
synchronization policy, and never checkpoints an active writer. It requires no
write permission on a clean database or its parent directory. The method
captures a committed high-water sequence and excludes rows committed later;
a subsequent read sees them. Its connection and owned reader thread close
after success, error, or cancellation. Unlike the synchronous JSONL
counterpart, it is an **async** classmethod because SQLite I/O stays off the
event loop:

```python
from async_batch_llm import SqliteArtifactStore

review = await SqliteArtifactStore.read_results(
    "runs/customer-tagging.sqlite",
    successes_only=True,
)
```

Opening a writable store validates schema and version markers, records the
current run's distinct identity when needed (a sequential identity history for
provenance), and never decodes stored history before a lookup needs it. The
read-only path validates the same application ID, versions, tables, columns,
required index names, and manifest without recording an identity. Each item row
consumed by either path is checked against the supported logical artifact
schema before its result or persisted context JSON is decoded; unsupported or
malformed row versions fail loudly instead of falling back to an older
compatible row.

## Process-safety boundary

Both stores are single-process, single-writer designs. `JsonlArtifactStore`
serializes writes with an `asyncio.Lock`, guaranteeing complete,
non-interleaved JSON records for concurrent workers sharing one store
instance in one process. `SqliteArtifactStore` owns one connection, one
worker thread, and one writer task per instance; sequential reopen is
supported and a concurrently open external reader obeys the configured busy
timeout. Neither implements distributed work claiming, leasing, or
cross-process exactly-once execution — use one writer per artifact.

On open, the JSONL store still builds an in-memory replay index over complete
item records for constant-time resume lookup (its `iter_results()` streams
bounded line pages instead of materializing history, as of v0.21). Very large
audit histories are exactly what `SqliteArtifactStore` is for.
