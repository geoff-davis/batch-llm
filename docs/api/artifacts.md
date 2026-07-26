# Artifact and Serialization API Reference

## ArtifactIdentity

::: async_batch_llm.ArtifactIdentity

## ResumePolicy

::: async_batch_llm.ResumePolicy

## ArtifactStore

::: async_batch_llm.ArtifactStore

## JsonlArtifactStore

::: async_batch_llm.JsonlArtifactStore

## SqliteArtifactStore

Indexed SQLite backend for large restartable runs (v0.21). Note that
`SqliteArtifactStore.read_results()` is an **async** classmethod, unlike the
synchronous JSONL convenience. It materializes a finite high-water snapshot
through an operationally read-only path connection; live
`store.iter_results()` instead flushes and inspects that writable store.

::: async_batch_llm.SqliteArtifactStore

## SqliteDurability

::: async_batch_llm.SqliteDurability

## ResultSerializationError

::: async_batch_llm.ResultSerializationError

## Artifact errors

::: async_batch_llm.ArtifactError

::: async_batch_llm.ArtifactSerializationError

::: async_batch_llm.ArtifactIOError

::: async_batch_llm.ArtifactFormatError
