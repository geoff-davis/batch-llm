"""Concrete LLM model implementations.

Model classes wrap a provider's client and handle API calls, token extraction,
and response normalization. Strategies receive an LLMModel and call generate()
without knowing about provider-specific details.

Added in v0.6.0.
"""

import json
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar

from .base import LLMResponse
from .strategies.errors import EmptyResponseError, ProviderResponseError

# A user-supplied hook that maps a raw provider response onto extra metadata
# keys. Returns None (or an empty dict) to contribute nothing. Extractors run
# best-effort and merge on top of the model's built-in metadata — see
# _run_extractors. Added in v0.15.0 (issue #52).
MetadataExtractor = Callable[[Any], dict[str, Any] | None]

# Sentinel prefix for encoding cache_tags into Gemini's CachedContent.display_name.
# google-genai's CreateCachedContentConfig does not expose a metadata field, so we
# round-trip tags through display_name, marked with this prefix so we can tell
# async-batch-llm-tagged caches apart from caches with user-chosen display names.
_TAG_DISPLAY_NAME_PREFIX = "abl-tags:"

# Conditional imports for optional dependencies
if TYPE_CHECKING:
    from google import genai
    from google.genai.types import Content
    from openai import AsyncOpenAI
else:
    try:
        from google import genai
        from google.genai.types import Content
    except ImportError:
        genai = None  # type: ignore[assignment]
        Content = Any  # type: ignore[misc,assignment]
    try:
        from openai import AsyncOpenAI
    except ImportError:
        AsyncOpenAI = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# Bound to OpenAICompatibleModel so from_api_key() returns the calling
# subclass type — subclass overrides don't need a cast(). See issue #10.
TM = TypeVar("TM", bound="OpenAICompatibleModel")


def _build_openai_http_client(max_connections: int) -> Any:
    """Build a sized client using the transport bundled with the OpenAI SDK.

    OpenAI SDK releases before 3.x use ``httpx`` while 3.x uses ``httpx2``.
    Both expose their matching limits instance and default async client at the
    package root, so deriving the limits class from the SDK supports both
    transport generations without taking a direct dependency on either one.
    """
    from openai import DEFAULT_CONNECTION_LIMITS, DefaultAsyncHttpxClient

    limits_type = type(DEFAULT_CONNECTION_LIMITS)
    return DefaultAsyncHttpxClient(
        limits=limits_type(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
        )
    )


def _encode_tags_to_display_name(tags: dict[str, str]) -> str:
    """Encode cache_tags as a deterministic string for the CachedContent display_name.

    Uses sorted, compact JSON so equal tag sets always produce the same display_name —
    critical for cache lookup to match. Prefixed with a sentinel so we can tell our
    tag encoding apart from a user-assigned display name.
    """
    encoded = json.dumps(tags, sort_keys=True, separators=(",", ":"))
    return f"{_TAG_DISPLAY_NAME_PREFIX}{encoded}"


def _decode_tags_from_display_name(display_name: str | None) -> dict[str, str] | None:
    """Decode cache_tags from a CachedContent display_name.

    Returns None when the display_name is absent or was not produced by
    _encode_tags_to_display_name. Callers should treat None as "this cache has no
    tag metadata we can match against".
    """
    if not display_name or not display_name.startswith(_TAG_DISPLAY_NAME_PREFIX):
        return None
    try:
        decoded = json.loads(display_name[len(_TAG_DISPLAY_NAME_PREFIX) :])
    except ValueError:
        return None
    if not isinstance(decoded, dict):
        return None
    return decoded


def _extract_metadata(response: Any) -> dict[str, Any] | None:
    """Extract safety ratings and finish reason from a Gemini response."""
    metadata: dict[str, Any] = {}

    try:
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]

            # Safety ratings
            if hasattr(candidate, "safety_ratings") and candidate.safety_ratings:
                ratings: dict[str, str] = {}
                for rating in candidate.safety_ratings:
                    category = str(rating.category) if hasattr(rating, "category") else "UNKNOWN"
                    probability = (
                        str(rating.probability) if hasattr(rating, "probability") else "UNKNOWN"
                    )
                    ratings[category] = probability
                metadata["safety_ratings"] = ratings

            # Finish reason
            if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                metadata["finish_reason"] = str(candidate.finish_reason)

        # Grounding (google_search tool) — only present when the caller
        # requested it, so default payloads are unchanged for everyone else.
        grounding = _extract_grounding(response)
        if grounding:
            metadata["grounding"] = grounding
    except Exception as e:
        # Metadata extraction is best-effort; missing/partial attributes on the
        # provider response shouldn't break the call. Keep `except Exception` so
        # KeyboardInterrupt still propagates.
        logger.warning(f"Failed to extract response metadata: {e}", exc_info=True)

    return metadata or None


def _run_extractors(
    response: Any,
    extractors: list[MetadataExtractor] | None,
    base: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge user-supplied metadata extractors on top of the built-in ``base``.

    Each extractor runs best-effort: a failing extractor is logged and skipped
    so it can't break the call. Later extractors win key collisions, and any
    user extractor overrides a built-in key of the same name. Returns ``None``
    when nothing was extracted, preserving the "empty → None" contract that
    ``LLMResponse.metadata`` expects.

    Added in v0.15.0 (issue #52).
    """
    merged: dict[str, Any] = dict(base) if base else {}
    for extractor in extractors or ():
        try:
            extra = extractor(response)
        except Exception as e:
            # Keep `except Exception` (not bare) so KeyboardInterrupt propagates.
            logger.warning(f"Metadata extractor {extractor!r} failed: {e}", exc_info=True)
            continue
        if extra:
            merged.update(extra)
    return merged or None


def _extract_grounding(response: Any) -> dict[str, Any] | None:
    """Extract Gemini grounding into the documented plain-dict shape.

    Returns the inner ``metadata['grounding']`` value — ``{"sources":
    [{"uri", "title"}], "queries": [...], "supports": [...]}`` — or ``None``
    when the response carries no grounding metadata. This dict shape is the
    public compatibility contract that the typed ``Grounding`` view
    (``provider_output.py``) reads.
    """
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return None
    gm = getattr(candidates[0], "grounding_metadata", None)
    if gm is None:
        return None

    grounding: dict[str, Any] = {}

    sources: list[dict[str, Any]] = []
    for chunk in getattr(gm, "grounding_chunks", None) or ():
        web = getattr(chunk, "web", None)
        if web is None:
            continue
        sources.append({"uri": getattr(web, "uri", None), "title": getattr(web, "title", None)})
    if sources:
        grounding["sources"] = sources

    queries = [q for q in (getattr(gm, "web_search_queries", None) or ()) if q]
    if queries:
        grounding["queries"] = queries

    supports: list[dict[str, Any]] = []
    for support in getattr(gm, "grounding_supports", None) or ():
        segment = getattr(support, "segment", None)
        supports.append(
            {
                "text": getattr(segment, "text", None),
                "start_index": getattr(segment, "start_index", None),
                "end_index": getattr(segment, "end_index", None),
                "chunk_indices": list(getattr(support, "grounding_chunk_indices", None) or ()),
            }
        )
    if supports:
        grounding["supports"] = supports

    return grounding or None


def grounding_metadata_extractor(response: Any) -> dict[str, Any] | None:
    """Metadata extractor for Gemini grounding (``google_search``).

    Since v0.16.0 the built-in Gemini models emit ``metadata['grounding']``
    **by default** whenever the response carries grounding metadata, so
    passing this extractor to ``GeminiModel``/``GeminiCachedModel`` is
    redundant (harmless — it re-emits the same key with the same value). It
    remains exported for custom models and for callers who prefer the
    explicit opt-in in their configuration.

    Returns ``{"grounding": {...}}`` with ``sources`` (``[{"uri",
    "title"}]``), ``queries`` (the ``web_search_queries`` the model issued),
    and ``supports`` (answer-span → source-index links), or ``None`` when the
    response carries no grounding metadata. Read the typed view via
    ``result.grounding`` (see ``provider_output.py``).

    Added in v0.15.0 (issue #52).
    """
    grounding = _extract_grounding(response)
    return {"grounding": grounding} if grounding else None


def _raise_empty_response(message: str, tokens: tuple[int, int, int, int]) -> NoReturn:
    """Raise EmptyResponseError carrying the tokens the provider already billed.

    The API call succeeded (and was billed) even though it produced no usable
    text — without attaching the counts, the framework's failed-attempt
    accounting would report zero spend for the attempt.
    """
    input_tokens, output_tokens, total_tokens, cached_tokens = tokens
    raise EmptyResponseError(
        message,
        token_usage={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_input_tokens": cached_tokens,
        },
    )


def _extract_tokens(response: Any) -> tuple[int, int, int, int]:
    """Extract token counts from a Gemini response.

    Returns:
        (input_tokens, output_tokens, total_tokens, cached_input_tokens)
    """
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata is None:
        logger.debug(
            "No usage_metadata on response (%s); token counts will be zero.",
            type(response).__name__,
        )
        return 0, 0, 0, 0

    input_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
    total_tokens = getattr(usage_metadata, "total_token_count", 0) or 0
    cached_tokens = 0
    if hasattr(usage_metadata, "cached_content_token_count"):
        cached_tokens = getattr(usage_metadata, "cached_content_token_count", 0) or 0

    return input_tokens, output_tokens, total_tokens, cached_tokens


class GeminiModel:
    """
    LLM model backed by the Google Gemini API.

    Wraps a genai.Client and model name, handling API calls, token extraction,
    and response normalization. Implements the LLMModel protocol.

    Example:
        >>> client = genai.Client(api_key="...")
        >>> model = GeminiModel("gemini-2.5-flash", client)
        >>> response = await model.generate("Hello!")
        >>> print(response.text, response.input_tokens)

    Added in v0.6.0.
    """

    def __init__(
        self,
        model: str,
        client: "genai.Client",
        *,
        safety_settings: list[dict[str, Any]] | None = None,
        system_instruction: str | None = None,
        metadata_extractors: list[MetadataExtractor] | None = None,
    ):
        """
        Args:
            model: Model name (e.g., "gemini-3.1-flash-lite-preview").
            client: Initialized genai.Client.
            safety_settings: Default safety settings for all calls.
            system_instruction: Default system instruction (overridable per-call).
            metadata_extractors: Optional hooks that contribute extra keys to
                ``LLMResponse.metadata`` (e.g. ``grounding_metadata_extractor``).
                Merged on top of the built-in ``safety_ratings``/``finish_reason``;
                user keys win. Added in v0.15.0.
        """
        if genai is None:
            raise ImportError(
                "google-genai is required for GeminiModel. "
                "Install with: pip install 'async-batch-llm[gemini]'"
            )

        self._model = model
        self._client = client
        self._safety_settings = safety_settings
        self._default_system_instruction = system_instruction
        self._metadata_extractors = metadata_extractors

    def _build_metadata(self, response: Any) -> dict[str, Any] | None:
        """Built-in Gemini metadata plus any user-supplied extractors."""
        return _run_extractors(response, self._metadata_extractors, _extract_metadata(response))

    async def generate(
        self,
        prompt: str | list[Any],
        *,
        temperature: float | None = 0.0,
        system_instruction: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate a response from Gemini.

        Args:
            prompt: Text prompt or list of content parts (multimodal).
            temperature: Sampling temperature. Pass ``None`` to omit it and
                use the model default.
            system_instruction: Override default system instruction.
            config: Additional provider-specific config entries.

        Returns:
            Normalized LLMResponse.
        """
        # Build config dict
        call_config: dict[str, Any] = {}
        if temperature is not None:
            call_config["temperature"] = temperature

        si = system_instruction or self._default_system_instruction
        if si is not None:
            call_config["system_instruction"] = si

        if self._safety_settings:
            call_config["safety_settings"] = self._safety_settings

        if config:
            call_config.update(config)

        # Make the API call (config is built as a dict; the SDK accepts this at runtime
        # even though the type stubs say GenerateContentConfig)
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=call_config,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        )

        # Extract tokens
        tokens = _extract_tokens(response)
        input_tokens, output_tokens, total_tokens, cached_tokens = tokens

        # Extract text (may be None if safety-blocked)
        text = response.text
        if text is None:
            metadata = _extract_metadata(response)
            safety_info = ""
            if metadata and "safety_ratings" in metadata:
                safety_info = f" Safety ratings: {metadata['safety_ratings']}"
            _raise_empty_response(
                f"Empty response from model (likely blocked by safety filter).{safety_info}",
                tokens,
            )

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached_tokens,
            metadata=self._build_metadata(response),
            raw=response,
        )


class GeminiCachedModel:
    """
    LLM model backed by Google Gemini with context caching.

    Wraps a genai.Client with cache lifecycle management. Implements the
    ManagedLLMModel protocol: call prepare() before first use, cleanup()
    when done.

    IMPORTANT — share one instance across work items.
        Create ONE GeminiCachedModel and reuse it across every LLMWorkItem that
        should share the cached context. Constructing a new instance per item
        defeats caching entirely and can cost 10× more. The framework calls
        prepare() exactly once per unique instance, so sharing is the intended
        lifecycle. See examples/example_llm_strategies.py for the pattern.

    This provides 70-90% cost savings when shared correctly.

    Example:
        >>> model = GeminiCachedModel(
        ...     "gemini-2.5-flash", client,
        ...     cached_content=[system_instruction, context_docs],
        ... )
        >>> await model.prepare()  # finds or creates cache
        >>> response = await model.generate("Process this")
        >>> await model.cleanup()  # preserves cache for reuse

    Added in v0.6.0.
    """

    def __init__(
        self,
        model: str,
        client: "genai.Client",
        cached_content: list["Content"],
        *,
        cache_ttl_seconds: int = 3600,
        cache_renewal_buffer_seconds: int = 300,
        auto_renew: bool = True,
        cache_tags: dict[str, str] | None = None,
        safety_settings: list[dict[str, Any]] | None = None,
        metadata_extractors: list[MetadataExtractor] | None = None,
    ):
        """
        Args:
            model: Model name (e.g., "gemini-2.5-flash").
            client: Initialized genai.Client.
            cached_content: Content to cache (system instructions, documents).
            cache_ttl_seconds: Cache TTL in seconds (default: 3600 = 1 hour).
            cache_renewal_buffer_seconds: Renew this many seconds before expiry
                (default: 300 = 5 minutes).
            auto_renew: Auto-renew expired caches in generate() (default: True).
            cache_tags: Tags for precise cache matching. Encoded into the cache's
                ``display_name`` at creation (google-genai ``CreateCachedContentConfig``
                has no ``metadata`` field) and decoded on lookup. Keep tag values
                short — Gemini's ``display_name`` has a 128-character limit.
            safety_settings: Default safety settings for all calls.
            metadata_extractors: Optional hooks that contribute extra keys to
                ``LLMResponse.metadata`` (e.g. ``grounding_metadata_extractor``).
                Merged on top of the built-in metadata; user keys win.
                Added in v0.15.0.
        """
        if genai is None:
            raise ImportError(
                "google-genai is required for GeminiCachedModel. "
                "Install with: pip install 'async-batch-llm[gemini]'"
            )

        if cache_renewal_buffer_seconds >= cache_ttl_seconds:
            raise ValueError(
                f"cache_renewal_buffer_seconds ({cache_renewal_buffer_seconds}) "
                f"must be less than cache_ttl_seconds ({cache_ttl_seconds})."
            )

        if 10 <= cache_ttl_seconds < 60:
            import warnings

            warnings.warn(
                f"cache_ttl_seconds ({cache_ttl_seconds}) is less than 60 seconds. "
                f"Very short TTLs defeat the purpose of caching. "
                f"Recommended minimum: 300 seconds (5 minutes).",
                UserWarning,
                stacklevel=2,
            )

        if cache_renewal_buffer_seconds < 60:
            import warnings

            warnings.warn(
                f"cache_renewal_buffer_seconds ({cache_renewal_buffer_seconds}) is less than "
                f"60 seconds. Small buffers risk renewing on every call if generation takes "
                f"longer than the buffer. Recommended minimum: 60 seconds.",
                UserWarning,
                stacklevel=2,
            )

        self._model = model
        self._client = client
        self._cached_content = cached_content
        self._cache_ttl_seconds = cache_ttl_seconds
        self._cache_renewal_buffer_seconds = cache_renewal_buffer_seconds
        self._auto_renew = auto_renew
        self._cache_tags = cache_tags or {}
        self._safety_settings = safety_settings
        self._metadata_extractors = metadata_extractors

        self._cache: Any = None
        self._cache_created_at: float | None = None
        self._cache_lock: Any = None
        self._prepared = False

    def _build_metadata(self, response: Any) -> dict[str, Any] | None:
        """Built-in Gemini metadata plus any user-supplied extractors."""
        return _run_extractors(response, self._metadata_extractors, _extract_metadata(response))

    @property
    def cache_name(self) -> str | None:
        """The name of the active cache, or None."""
        return self._cache.name if self._cache else None

    # ── Lifecycle ───────────────────────────────────────────────

    async def prepare(self) -> None:
        """Find or create the Gemini cache. Idempotent."""
        if self._prepared:
            return

        import asyncio

        if self._cache_lock is None:
            self._cache_lock = asyncio.Lock()

        async with self._cache_lock:
            if self._prepared:
                return
            await self._find_or_create_cache()
            self._prepared = True

    async def cleanup(self) -> None:
        """Preserve cache for reuse (does not delete). Idempotent."""
        if self._cache:
            logger.info(
                f"Leaving cache active for reuse: {self._cache.name} "
                f"(TTL: {self._cache_ttl_seconds}s, will expire naturally)"
            )

    async def delete_cache(self) -> None:
        """Explicitly delete the cache.

        Safe to call concurrently: the cache lock serializes delete attempts
        so the provider API fires at most once, and late callers that arrive
        after the cache is cleared return silently.
        """
        if self._cache is None:
            return

        import asyncio as _asyncio

        if getattr(self, "_cache_lock", None) is None:
            self._cache_lock = _asyncio.Lock()

        async with self._cache_lock:
            cache = self._cache
            if cache is None:
                # A concurrent caller already finished the delete.
                return

            # Capture the name up front so log messages don't depend on
            # self._cache still existing after concurrent callers clear it.
            cache_name = cache.name
            # Clear state BEFORE the API call so concurrent tasks that
            # re-enter see an empty cache and no-op.
            self._cache = None
            self._cache_created_at = None
            self._prepared = False

            try:
                await self._client.aio.caches.delete(name=cache_name)
                logger.info(f"Deleted Gemini cache: {cache_name}")
            except Exception as e:
                # Keep Exception (not BaseException) so KeyboardInterrupt still propagates;
                # cache-delete failures are best-effort — caches expire on their own.
                logger.warning(
                    f"Failed to delete Gemini cache '{cache_name}': {e}. "
                    "Cache may have already expired or been deleted.",
                    exc_info=True,
                )

    # ── Generate ────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str | list[Any],
        *,
        temperature: float | None = 0.0,
        system_instruction: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate a response using the cached context.

        Args:
            prompt: Text prompt or multimodal content parts.
            temperature: Sampling temperature. Pass ``None`` to omit it and
                use the model default.
            system_instruction: Not supported with caching — raises ValueError.
            config: Additional provider-specific config entries.

        Returns:
            Normalized LLMResponse.
        """
        if system_instruction is not None:
            raise ValueError(
                "system_instruction cannot be overridden per-call with cached models. "
                "The system instruction is baked into the cache at creation time."
            )

        # Auto-renew if expired
        if self._auto_renew and self._is_cache_expired():
            import asyncio

            if self._cache_lock is None:
                self._cache_lock = asyncio.Lock()

            async with self._cache_lock:
                if self._is_cache_expired():
                    age_str = (
                        f"{time.time() - self._cache_created_at:.0f}s"
                        if self._cache_created_at is not None
                        else "unknown (cache not yet initialized)"
                    )
                    logger.info(
                        "Cache expired or about to expire, renewing before API call "
                        f"(age: {age_str}, "
                        f"renewal buffer: {self._cache_renewal_buffer_seconds}s)"
                    )
                    await self._renew_cache()

        if self._cache is None:
            raise RuntimeError("Cache not initialized — call prepare() first")

        # Build config with cache reference
        call_config: dict[str, Any] = {
            "cached_content": self._cache.name,
        }
        if temperature is not None:
            call_config["temperature"] = temperature

        if self._safety_settings:
            call_config["safety_settings"] = self._safety_settings

        if config:
            call_config.update(config)

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=call_config,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        )

        tokens = _extract_tokens(response)
        input_tokens, output_tokens, total_tokens, cached_tokens = tokens

        text = response.text
        if text is None:
            metadata = _extract_metadata(response)
            safety_info = ""
            if metadata and "safety_ratings" in metadata:
                safety_info = f" Safety ratings: {metadata['safety_ratings']}"
            _raise_empty_response(
                f"Empty response from model (likely blocked by safety filter).{safety_info}",
                tokens,
            )

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached_tokens,
            metadata=self._build_metadata(response),
            raw=response,
        )

    # ── Cache internals ─────────────────────────────────────────

    def _is_cache_expired(self) -> bool:
        if self._cache is None or self._cache_created_at is None:
            return True
        cache_age = time.time() - self._cache_created_at
        expires_in = self._cache_ttl_seconds - cache_age
        return expires_in <= self._cache_renewal_buffer_seconds

    def _cache_within_renewal_buffer(self, cache: Any) -> bool:
        """True if ``cache`` expires within the renewal buffer.

        Reads the provider's ``expire_time`` so renewal lookups can skip a
        candidate that is itself about to expire. Returns False when the
        expiry is unknown (don't skip what we can't evaluate).
        """
        expire_time = getattr(cache, "expire_time", None)
        if expire_time is None:
            return False
        try:
            expires_in = expire_time.timestamp() - time.time()
        except (AttributeError, TypeError, ValueError):
            return False
        return bool(expires_in <= self._cache_renewal_buffer_seconds)

    async def _renew_cache(self) -> None:
        """Extend the current cache's TTL in place, or find/create a fresh one.

        Prefers ``caches.update()`` so the SAME cache's TTL is pushed out. This
        fixes the old renewal bug: ``_find_or_create_cache`` would re-list and
        re-adopt the very same near-expiry cache, extending nothing — so every
        ``generate()`` call inside the renewal-buffer window kept doing a
        ``caches.list()`` round-trip and the cache eventually lapsed anyway.

        Falls back to find-or-create (skipping near-expiry candidates) when
        there is no current cache or the update call fails.
        """
        cache = self._cache
        if cache is not None and getattr(cache, "name", None):
            try:
                from google.genai.types import UpdateCachedContentConfig

                updated = await self._client.aio.caches.update(
                    name=cache.name,
                    config=UpdateCachedContentConfig(ttl=f"{self._cache_ttl_seconds}s"),
                )
                # Some SDK versions return the refreshed cache object; keep our
                # reference current when they do.
                if updated is not None:
                    self._cache = updated
                self._cache_created_at = time.time()
                self._prepared = True
                logger.info(
                    f"Renewed Gemini cache TTL in place: {self._cache.name} "
                    f"(TTL: {self._cache_ttl_seconds}s)"
                )
                return
            except Exception as e:
                # Keep Exception (not BaseException) so KeyboardInterrupt still
                # propagates; an in-place renew failure is recoverable via the
                # find-or-create fallback below.
                logger.warning(
                    f"Failed to update TTL for cache '{getattr(cache, 'name', '?')}': {e}. "
                    "Falling back to find-or-create."
                )

        # Drop the stale reference and find/create — but skip caches that are
        # themselves inside the renewal buffer so we don't re-adopt a dud.
        self._cache = None
        self._cache_created_at = None
        self._prepared = False
        await self._find_or_create_cache(renewing=True)
        self._prepared = True

    async def _find_or_create_cache(self, renewing: bool = False) -> None:
        try:
            caches = await self._client.aio.caches.list()

            async for cache in caches:
                if not cache.model or not cache.model.endswith(self._model):
                    continue

                # During renewal, ignore any candidate that is itself about to
                # expire — adopting it would just put us straight back into the
                # renewal-buffer window.
                if renewing and self._cache_within_renewal_buffer(cache):
                    logger.debug(
                        f"Skipping cache {cache.name} during renewal: "
                        "within renewal buffer of expiry"
                    )
                    continue

                if self._cache_tags:
                    cache_tags = _decode_tags_from_display_name(
                        getattr(cache, "display_name", None)
                    )
                    if cache_tags is None:
                        logger.debug(
                            f"Skipping cache {cache.name}: no abl-tags display_name "
                            f"(want {self._cache_tags})"
                        )
                        continue
                    tags_match = all(cache_tags.get(k) == v for k, v in self._cache_tags.items())
                    if not tags_match:
                        logger.debug(
                            f"Skipping cache {cache.name}: tags don't match "
                            f"(want {self._cache_tags}, has {cache_tags})"
                        )
                        continue

                self._cache = cache
                if hasattr(cache, "create_time") and cache.create_time:
                    self._cache_created_at = cache.create_time.timestamp()
                else:
                    self._cache_created_at = time.time() - self._cache_ttl_seconds

                tag_info = f" with tags {self._cache_tags}" if self._cache_tags else ""
                age = time.time() - self._cache_created_at
                logger.info(
                    f"Reusing existing Gemini cache: {self._cache.name}{tag_info} (age: {age:.0f}s)"
                )
                return
        except Exception as e:
            logger.warning(f"Failed to list existing caches: {e}")

        await self._create_new_cache()

    async def _create_new_cache(self) -> None:
        from google.genai.types import CreateCachedContentConfig

        config_kwargs: dict[str, Any] = {
            "contents": self._cached_content,
            "ttl": f"{self._cache_ttl_seconds}s",
        }
        if self._cache_tags:
            # google-genai's CreateCachedContentConfig has no `metadata` field —
            # round-trip tags through `display_name` with a sentinel prefix.
            config_kwargs["display_name"] = _encode_tags_to_display_name(self._cache_tags)

        self._cache = await self._client.aio.caches.create(
            model=self._model,
            config=CreateCachedContentConfig(**config_kwargs),
        )

        self._cache_created_at = time.time()
        tag_info = f" with tags {self._cache_tags}" if self._cache_tags else ""
        logger.info(
            f"Created new Gemini cache: {self._cache.name}{tag_info} "
            f"(TTL: {self._cache_ttl_seconds}s)"
        )


# ── OpenAI-compatible models ────────────────────────────────────────────────


class OpenAICompatibleModel:
    """Base class for OpenAI chat-completions-compatible providers.

    Wraps an ``AsyncOpenAI`` client pointed at any chat-completions endpoint
    (OpenAI itself, OpenRouter, DeepSeek, HuggingFace Inference Providers,
    Together, Fireworks, local vLLM, etc.). Subclasses customize the default
    base URL, the install-extras hint, the env var read by
    :meth:`from_api_key`, and optionally the token/metadata extractors.

    Implements the ``ManagedLLMModel`` protocol — :meth:`cleanup` closes the
    underlying ``AsyncOpenAI`` client when this model owns it (i.e. it was
    constructed via :meth:`from_api_key`). User-provided clients are left
    alone.

    Models built with :meth:`from_api_key` and ``max_connections=N`` expose
    ``max_concurrency=N`` so execution surfaces can diagnose worker/pool
    mismatches. It remains ``None`` for caller-supplied clients because their
    effective transport capacity cannot be inspected reliably.

    Added in v0.9.0.
    """

    # Subclasses override.
    _default_base_url: str | None = None
    _install_extras: str = "openai"
    # Env var :meth:`from_api_key` reads when ``api_key`` is None.
    # ``None`` means: let the OpenAI SDK pick up its own default
    # (``OPENAI_API_KEY``); subclasses set it to read a different env var
    # ourselves and forward to the SDK explicitly.
    _api_key_env_var: str | None = None

    def __init__(
        self,
        model: str,
        client: "AsyncOpenAI",
        *,
        system_instruction: str | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        metadata_extractors: list[MetadataExtractor] | None = None,
    ):
        """
        Args:
            model: Provider model id (e.g. "gpt-4o-mini" or
                "anthropic/claude-haiku-4-5").
            client: Initialized AsyncOpenAI (point ``base_url`` at the desired
                endpoint). The model does NOT take ownership of the client —
                use :meth:`from_api_key` if you want the model to manage the
                client's lifecycle.
            system_instruction: Default system message prepended to each call.
                Per-call ``system_instruction`` argument takes precedence.
            extra_headers: Default headers forwarded on every call (e.g.
                OpenRouter's ``HTTP-Referer``/``X-Title``).
            extra_body: Default extra body fields forwarded on every call
                (e.g. OpenRouter ``provider`` routing config).
            metadata_extractors: Optional hooks that contribute extra keys to
                ``LLMResponse.metadata`` (e.g. ``reasoning_content`` or
                ``logprobs``). Merged on top of the built-in
                ``finish_reason``/``model`` metadata; user keys win.
                Added in v0.15.0.
        """
        if AsyncOpenAI is None:
            raise ImportError(
                f"openai is required for {type(self).__name__}. "
                f"Install with: pip install 'async-batch-llm[{self._install_extras}]'"
            )

        self._model = model
        self._client = client
        self._default_system_instruction = system_instruction
        self._default_extra_headers = extra_headers
        self._default_extra_body = extra_body
        self._metadata_extractors = metadata_extractors
        # Known only when this class constructs and sizes the underlying httpx
        # pool via from_api_key(max_connections=...). User-supplied clients are
        # intentionally not introspected because their transport may be wrapped.
        self.max_concurrency: int | None = None
        # Set to True only by from_api_key(); cleanup() uses this to decide
        # whether to close the underlying httpx connections.
        self._owns_client: bool = False
        # SDK constructor kwargs stashed by from_api_key() when it built the
        # client with the *default* httpx pool — lets request_concurrency()
        # rebuild the client with a right-sized pool before the first request.
        # None when the caller supplied the client or an explicit pool.
        self._rebuild_client_kwargs: dict[str, Any] | None = None

    async def generate(
        self,
        prompt: str | list[Any],
        *,
        temperature: float | None = 0.0,
        system_instruction: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Call ``client.chat.completions.create`` and normalize the response.

        Args:
            prompt: A string (becomes a single user message) or a list of
                OpenAI-shaped message dicts (passed through unchanged — used
                for multimodal content and Anthropic-via-OpenRouter
                ``cache_control`` markers).
            temperature: Sampling temperature. Pass ``None`` to omit the
                parameter so the provider uses its own default — required for
                OpenAI reasoning models (o1/o3/etc.) that reject an explicit
                ``temperature``.
            system_instruction: Per-call override for the system message.
            config: Per-call extra kwargs forwarded to the SDK call (merged
                over the instance's ``extra_body``). Use this to pass
                ``max_tokens``, ``response_format``, etc.

        Returns:
            Normalized LLMResponse.
        """
        messages = _coerce_to_messages(prompt)
        si = system_instruction or self._default_system_instruction
        if si is not None and not _has_system_message(messages):
            messages = [{"role": "system", "content": si}, *messages]

        # Merge extra_body: instance defaults + per-call config overrides.
        extra_body: dict[str, Any] = {}
        if self._default_extra_body:
            extra_body.update(self._default_extra_body)
        if config:
            extra_body.update(config)

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        # Omit temperature entirely when None so providers that reject an
        # explicit value (OpenAI reasoning models) use their own default.
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        if extra_body:
            call_kwargs["extra_body"] = extra_body
        if self._default_extra_headers:
            call_kwargs["extra_headers"] = self._default_extra_headers

        response = await self._client.chat.completions.create(**call_kwargs)

        # Some gateways report upstream failures inside an HTTP-200 body;
        # surface those before interpreting the (empty) choices list.
        self._raise_on_response_error(response)

        # Extract tokens up front: even a no-content response was billed for
        # the prompt (and, for finish_reason="length", the full output).
        tokens = self._extract_tokens(response)
        input_tokens, output_tokens, total_tokens, cached_tokens = tokens

        # Validate content present (None typically means a tool call or a
        # finish-reason like "length"/"content_filter").
        if not response.choices:
            _raise_empty_response(f"No choices returned from {type(self).__name__}.", tokens)
        message = response.choices[0].message
        text = getattr(message, "content", None)
        if text is None:
            finish_reason = getattr(response.choices[0], "finish_reason", "unknown")
            _raise_empty_response(
                f"Empty response content from model "
                f"(finish_reason={finish_reason!r}). "
                "This typically indicates a tool call, content filter, "
                "or token limit was reached.",
                tokens,
            )

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached_tokens,
            metadata=self._build_metadata(response),
            raw=response,
        )

    # ── Overridable extraction hooks ────────────────────────────────────

    def _raise_on_response_error(self, response: Any) -> None:
        """Hook: raise if the provider embedded an error in a 200 response.

        No-op here — OpenAI proper signals errors via HTTP status codes.
        Gateway subclasses that use error-in-body semantics (OpenRouter)
        override this.
        """
        return

    def _build_metadata(self, response: Any) -> dict[str, Any] | None:
        """Built-in ``_extract_metadata`` plus any user-supplied extractors.

        Subclasses customize the built-in payload by overriding
        ``_extract_metadata``; user extractors merge on top of that here.
        """
        return _run_extractors(
            response, self._metadata_extractors, self._extract_metadata(response)
        )

    def _extract_tokens(self, response: Any) -> tuple[int, int, int, int]:
        """Extract (input, output, total, cached) token counts.

        Reads OpenAI's ``usage.prompt_tokens`` / ``completion_tokens`` /
        ``total_tokens`` and the nested ``prompt_tokens_details.cached_tokens``
        when present. Subclasses (e.g. for DeepSeek's
        ``prompt_cache_hit_tokens``) override this.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0, 0, 0, 0

        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

        cached_tokens = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached_tokens = int(getattr(details, "cached_tokens", 0) or 0)

        return input_tokens, output_tokens, total_tokens, cached_tokens

    def _extract_metadata(self, response: Any) -> dict[str, Any] | None:
        """Extract finish_reason, routed model name, and the reserved
        provider-output keys (``reasoning``, ``tool_calls``, ``logprobs``).

        The ``isinstance`` guards below are deliberate: they keep the
        metadata dict JSON-serializable against SDK drift (and keep test
        doubles like ``MagicMock``, whose auto-attributes are truthy, from
        leaking into it). Emitted shapes are the documented contract read by
        the typed views in ``provider_output.py``.
        """
        metadata: dict[str, Any] = {}
        try:
            if response.choices:
                choice = response.choices[0]
                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason is not None:
                    metadata["finish_reason"] = str(finish_reason)

                message = getattr(choice, "message", None)
                if message is not None:
                    # Reasoning trace: DeepSeek reasoning_content, falling
                    # back to OpenRouter's reasoning.
                    reasoning = getattr(message, "reasoning_content", None)
                    if not isinstance(reasoning, str) or not reasoning:
                        reasoning = getattr(message, "reasoning", None)
                    if isinstance(reasoning, str) and reasoning:
                        metadata["reasoning"] = reasoning

                    # Tool calls -> plain dicts; arguments stays the raw JSON
                    # string. Visibility only — the framework never executes.
                    calls: list[dict[str, Any]] = []
                    raw_calls = getattr(message, "tool_calls", None)
                    if raw_calls:
                        try:
                            iterator = iter(raw_calls)
                        except TypeError:
                            iterator = iter(())
                        for tc in iterator:
                            fn = getattr(tc, "function", None)
                            name = getattr(fn, "name", None)
                            if not isinstance(name, str) or not name:
                                continue
                            args = getattr(fn, "arguments", None)
                            tc_id = getattr(tc, "id", None)
                            calls.append(
                                {
                                    "id": tc_id if isinstance(tc_id, str) else None,
                                    "name": name,
                                    "arguments": args if isinstance(args, str) else "",
                                }
                            )
                    if calls:
                        metadata["tool_calls"] = calls

                # Logprobs: model_dump() keeps it JSON-serializable.
                logprobs = getattr(choice, "logprobs", None)
                if logprobs is not None:
                    dumped = logprobs.model_dump() if hasattr(logprobs, "model_dump") else logprobs
                    if isinstance(dumped, (dict, list)):
                        metadata["logprobs"] = dumped

            routed_model = getattr(response, "model", None)
            if routed_model:
                metadata["model"] = str(routed_model)
        except Exception as e:
            logger.warning(f"Failed to extract response metadata: {e}", exc_info=True)
        return metadata or None

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def prepare(self) -> None:
        """No-op; OpenAI-compatible models have nothing to initialize."""
        return

    async def request_concurrency(self, concurrency: int) -> bool:
        """Ask the model to support ``concurrency`` parallel requests.

        Called by execution surfaces when ``ProcessorConfig.concurrency`` is
        set (v0.20.0, issue #97), before the first request. When this model
        owns its client (built via :meth:`from_api_key`) and no explicit
        ``max_connections`` was given, the ``AsyncOpenAI`` client is rebuilt
        with an httpx pool sized to ``concurrency`` (the SDK's default pool of
        ~100 connections would otherwise silently cap throughput) and
        ``max_concurrency`` starts advertising the new size.

        Returns True when the pool was resized. Returns False — leaving the
        model untouched — for caller-supplied clients and for models built
        with an explicit ``max_connections`` (an explicit value always wins;
        a genuine contradiction is surfaced by the existing capacity warning
        instead).
        """
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1; got {concurrency}.")
        if (
            not self._owns_client
            or self.max_concurrency is not None
            or self._rebuild_client_kwargs is None
        ):
            return False
        new_client = AsyncOpenAI(
            http_client=_build_openai_http_client(concurrency),
            **self._rebuild_client_kwargs,
        )
        old_client = self._client
        self._client = new_client
        self.max_concurrency = concurrency
        # The old client made no requests yet (this runs before batch start);
        # close it so its default httpx transport doesn't linger.
        close = getattr(old_client, "close", None)
        if close is not None:
            try:
                result = close()
                if hasattr(result, "__await__"):
                    await result
            except Exception as e:
                logger.warning(
                    f"Failed to close replaced {type(self).__name__} client: {e}",
                    exc_info=True,
                )
        return True

    async def cleanup(self) -> None:
        """Close the underlying AsyncOpenAI client if this model owns it.

        Models constructed directly with ``OpenAIModel(model, client=...)``
        do NOT own the client — the caller is expected to close it. Models
        constructed via :meth:`from_api_key` do own the client and close
        it here so repeated processor runs don't leak httpx connections.
        """
        if not self._owns_client or self._client is None:
            return
        close = getattr(self._client, "close", None)
        if close is None:
            return
        try:
            result = close()
            if hasattr(result, "__await__"):
                await result
        except Exception as e:
            logger.warning(
                f"Failed to close {type(self).__name__} client: {e}",
                exc_info=True,
            )

    # ── Convenience constructor ─────────────────────────────────────────

    @classmethod
    def from_api_key(
        cls: type[TM],
        model: str,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        system_instruction: str | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        json_mode: bool = False,
        max_connections: int | None = None,
        metadata_extractors: list[MetadataExtractor] | None = None,
        **client_kwargs: Any,
    ) -> TM:
        """Build the model with a freshly-constructed AsyncOpenAI client.

        The returned model owns the client — its connections are released
        when the framework calls :meth:`cleanup` (typically when the
        ``ParallelBatchProcessor`` exits).

        Uses ``base_url`` (if provided) or the class's ``_default_base_url``.
        Pass ``client_kwargs`` to forward additional kwargs (timeout,
        max_retries, http_client, etc.) to the SDK constructor.

        Args:
            model: Provider model id.
            api_key: API key. If ``None``:

                - For ``OpenAIModel`` the OpenAI SDK auto-reads
                  ``OPENAI_API_KEY``.
                - For ``OpenRouterModel`` (and other subclasses with
                  ``_api_key_env_var`` set) we read the env var ourselves
                  and forward it to the SDK explicitly.
                - If neither path resolves, raises ``ValueError``.
            json_mode: When ``True``, request JSON output by adding
                ``response_format={"type": "json_object"}`` to ``extra_body``
                (forwarded on every call). A convenience over hand-passing it
                yourself; an explicit ``response_format`` in ``extra_body``
                takes precedence. Most providers still require the word "JSON"
                somewhere in your prompt/system instruction for this to take
                effect. Pair with :func:`async_batch_llm.pydantic_json_parser`
                on the strategy, since some providers (DeepSeek) still wrap the
                JSON in markdown fences even in JSON mode (issue #26).
            max_connections: Size of the underlying httpx connection pool
                (both ``max_connections`` and ``max_keepalive_connections``).
                **Set this to at least ``ProcessorConfig.max_workers``** — the
                openai SDK otherwise uses httpx's default pool (~100), so
                raising ``max_workers`` above that gives no extra throughput;
                the excess workers just block waiting for a connection (see
                issue #25). High-concurrency providers like DeepSeek (which
                allow thousands of concurrent connections) hit this ceiling
                first. Mutually exclusive with passing your own
                ``http_client``; raises ``ValueError`` if you pass both.
                The value is also exposed as ``model.max_concurrency`` and
                forwarded by ``ModelStrategy`` for processor/gateway validation.
        """
        if AsyncOpenAI is None:
            raise ImportError(
                f"openai is required for {cls.__name__}. "
                f"Install with: pip install 'async-batch-llm[{cls._install_extras}]'"
            )
        if max_connections is not None:
            if "http_client" in client_kwargs:
                raise ValueError(
                    "Pass either max_connections or http_client, not both. "
                    "max_connections is a convenience for sizing the default "
                    "httpx pool; if you build your own http_client, set its "
                    "limits there instead."
                )
            if max_connections < 1:
                raise ValueError(f"max_connections must be >= 1; got {max_connections}.")
            client_kwargs["http_client"] = _build_openai_http_client(max_connections)
        effective_base_url = base_url or cls._default_base_url
        if effective_base_url is not None:
            client_kwargs.setdefault("base_url", effective_base_url)

        resolved_key = api_key
        if resolved_key is None and cls._api_key_env_var is not None:
            import os as _os

            resolved_key = _os.environ.get(cls._api_key_env_var)
            if not resolved_key:
                raise ValueError(
                    f"No API key for {cls.__name__}: pass api_key= or set the "
                    f"{cls._api_key_env_var} environment variable."
                )

        # When resolved_key is None and the SDK can self-resolve (OpenAIModel),
        # don't pass api_key at all — letting the SDK raise its own clear
        # error if neither path produces one.
        if resolved_key is not None:
            client_kwargs["api_key"] = resolved_key
        client = AsyncOpenAI(**client_kwargs)

        if json_mode:
            # Inject a JSON response_format, letting any explicit caller-supplied
            # response_format in extra_body win.
            effective_extra_body: dict[str, Any] = {"response_format": {"type": "json_object"}}
            if extra_body:
                effective_extra_body.update(extra_body)
            extra_body = effective_extra_body

        instance = cls(
            model,
            client,
            system_instruction=system_instruction,
            extra_headers=extra_headers,
            extra_body=extra_body,
            metadata_extractors=metadata_extractors,
        )
        instance.max_concurrency = max_connections
        instance._owns_client = True
        if max_connections is None and "http_client" not in client_kwargs:
            # Default-pool client: keep the constructor kwargs so
            # request_concurrency() can rebuild it with a right-sized pool.
            instance._rebuild_client_kwargs = dict(client_kwargs)
        return instance


class OpenAIModel(OpenAICompatibleModel):
    """LLM model backed by OpenAI's chat completions API.

    Uses the OpenAI SDK's default base URL (``https://api.openai.com/v1``).
    OpenAI's automatic prompt cache surfaces in ``cached_input_tokens`` for
    prompts longer than ~1024 tokens.

    Example:
        >>> model = OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-...")
        >>> response = await model.generate("Hello!")
        >>> print(response.text, response.cached_input_tokens)

    Added in v0.9.0.
    """

    _default_base_url: str | None = None
    _install_extras: str = "openai"


class OpenRouterModel(OpenAICompatibleModel):
    """LLM model backed by OpenRouter (https://openrouter.ai).

    OpenRouter exposes a unified OpenAI-compatible API for many upstream
    providers (Anthropic, OpenAI, Google, Mistral, DeepSeek, etc.). Model
    ids are prefixed with the provider, e.g. ``"anthropic/claude-haiku-4-5"``.

    Caching is provider-dependent:

    - **OpenAI / Gemini (implicit) / DeepSeek** — automatic; ``cached_input_tokens``
      is populated when the upstream cache hits.
    - **Anthropic** — opt-in. Pass ``prompt`` as a list of message dicts with
      ``cache_control: {"type": "ephemeral"}`` markers on the blocks you want
      cached.

    Example:
        >>> model = OpenRouterModel.from_api_key(
        ...     "anthropic/claude-haiku-4-5",
        ...     api_key="sk-or-...",
        ...     referer="https://my-app.example.com",
        ...     title="My App",
        ... )
        >>> response = await model.generate("Hello!")

    Added in v0.9.0.
    """

    _default_base_url: str | None = "https://openrouter.ai/api/v1"
    _install_extras: str = "openrouter"
    _api_key_env_var: str | None = "OPENROUTER_API_KEY"

    def _raise_on_response_error(self, response: Any) -> None:
        """Raise :class:`ProviderResponseError` for error-in-200 bodies.

        OpenRouter reports upstream failures (no provider available,
        upstream errors, upstream rate limits) as HTTP 200 with an
        ``error`` object and empty ``choices``, so the SDK never raises.
        Without this check the empty choices list would surface as a
        non-retryable ``EmptyResponseError`` even though these are
        precisely the transient failures the classifier is built to retry.
        """
        error = getattr(response, "error", None)
        if not error:
            return
        if isinstance(error, dict):
            message = error.get("message") or str(error)
            code = error.get("code")
        else:
            message = str(getattr(error, "message", None) or error)
            code = getattr(error, "code", None)
        raise ProviderResponseError(
            f"OpenRouter returned an error in the response body (code={code}): {message}",
            code=code if isinstance(code, int) else None,
            provider_error=error,
        )

    def _extract_metadata(self, response: Any) -> dict[str, Any] | None:
        """Add OpenRouter's ``provider`` field to the metadata."""
        metadata = super()._extract_metadata(response) or {}
        provider = getattr(response, "provider", None)
        if provider:
            metadata["provider"] = str(provider)
        return metadata or None

    @classmethod
    def from_api_key(  # type: ignore[override]
        cls,
        model: str,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        system_instruction: str | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        json_mode: bool = False,
        max_connections: int | None = None,
        referer: str | None = None,
        title: str | None = None,
        metadata_extractors: list[MetadataExtractor] | None = None,
        **client_kwargs: Any,
    ) -> "OpenRouterModel":
        """Build an OpenRouterModel.

        If ``api_key`` is None, reads ``OPENROUTER_API_KEY`` from the
        environment and raises ``ValueError`` if neither is set. (The
        OpenAI SDK doesn't know about ``OPENROUTER_API_KEY``, so we have
        to read it ourselves rather than relying on the SDK's default.)

        ``referer`` and ``title`` map to OpenRouter's optional
        ``HTTP-Referer`` and ``X-Title`` headers (used for app attribution
        on openrouter.ai's leaderboard).
        """
        merged_headers = dict(extra_headers) if extra_headers else {}
        if referer is not None:
            merged_headers.setdefault("HTTP-Referer", referer)
        if title is not None:
            merged_headers.setdefault("X-Title", title)

        # super().from_api_key is generic over cls (returns the calling
        # subclass type), so this returns OpenRouterModel directly — no cast
        # needed. See issue #10.
        return super().from_api_key(
            model,
            api_key,
            base_url=base_url,
            system_instruction=system_instruction,
            extra_headers=merged_headers or None,
            extra_body=extra_body,
            json_mode=json_mode,
            max_connections=max_connections,
            metadata_extractors=metadata_extractors,
            **client_kwargs,
        )


def _merge_thinking(
    extra_body: dict[str, Any] | None, thinking: bool | None
) -> dict[str, Any] | None:
    """Fold a DeepSeek ``thinking`` toggle into an ``extra_body`` dict.

    ``thinking=None`` leaves ``extra_body`` untouched (use the model's default).
    ``True``/``False`` map to DeepSeek's ``{"thinking": {"type": "enabled"}}`` /
    ``{"type": "disabled"}`` request field. An explicit ``thinking`` key already
    present in ``extra_body`` wins.
    """
    if thinking is None:
        return extra_body
    merged = dict(extra_body) if extra_body else {}
    merged.setdefault("thinking", {"type": "enabled" if thinking else "disabled"})
    return merged


class DeepSeekModel(OpenAICompatibleModel):
    """LLM model backed by DeepSeek's OpenAI-compatible API.

    Points at ``https://api.deepseek.com`` and reads
    ``DEEPSEEK_API_KEY`` in :meth:`from_api_key`. Model ids are bare DeepSeek
    names, e.g. ``"deepseek-chat"`` or ``"deepseek-reasoner"``.

    DeepSeek's automatic context cache reports hits at the **top level** of the
    ``usage`` object (``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens``)
    rather than under OpenAI's nested ``prompt_tokens_details.cached_tokens`` —
    so this subclass overrides :meth:`_extract_tokens` to surface them in
    ``cached_input_tokens``. Use :attr:`CachedTokenRates.DEEPSEEK` (~2%) when
    computing billable tokens.

    (Calling DeepSeek *through OpenRouter* uses :class:`OpenRouterModel`
    instead; the native cache fields aren't reliably forwarded there, which is
    why direct access via this class gives better cache telemetry.)

    **Thinking mode.** DeepSeek's V4 models (``deepseek-v4-flash`` /
    ``deepseek-v4-pro``) default to *thinking*, which for a batch
    classification job is a surprising, expensive default — thinking can emit
    several times the output tokens (and cost, and latency) of non-thinking.
    Pass ``thinking=False`` to force non-thinking mode explicitly rather than
    relying on the ``deepseek-chat`` (non-thinking) / ``deepseek-reasoner``
    (thinking) aliases, which DeepSeek is deprecating. Under the hood this sends
    ``extra_body={"thinking": {"type": "disabled"}}``.

    Example:
        >>> model = DeepSeekModel.from_api_key(
        ...     "deepseek-v4-flash", api_key="sk-...", thinking=False
        ... )
        >>> response = await model.generate("Hello!")
        >>> print(response.text, response.cached_input_tokens)

    Added in v0.10.0.
    """

    _default_base_url: str | None = "https://api.deepseek.com"
    _install_extras: str = "deepseek"
    _api_key_env_var: str | None = "DEEPSEEK_API_KEY"

    def __init__(
        self,
        model: str,
        client: "AsyncOpenAI",
        *,
        system_instruction: str | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        thinking: bool | None = None,
        metadata_extractors: list[MetadataExtractor] | None = None,
    ):
        """See :class:`OpenAICompatibleModel`; adds the DeepSeek ``thinking``
        toggle (``True``/``False`` to force thinking on/off, ``None`` for the
        model default)."""
        super().__init__(
            model,
            client,
            system_instruction=system_instruction,
            extra_headers=extra_headers,
            extra_body=_merge_thinking(extra_body, thinking),
            metadata_extractors=metadata_extractors,
        )

    @classmethod
    def from_api_key(  # type: ignore[override]
        cls,
        model: str,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        system_instruction: str | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        json_mode: bool = False,
        max_connections: int | None = None,
        thinking: bool | None = None,
        metadata_extractors: list[MetadataExtractor] | None = None,
        **client_kwargs: Any,
    ) -> "DeepSeekModel":
        """Build a DeepSeekModel; reads ``DEEPSEEK_API_KEY`` when ``api_key`` is
        None. Adds the ``thinking`` toggle (see the class docstring) on top of
        the shared :meth:`OpenAICompatibleModel.from_api_key` arguments."""
        return super().from_api_key(
            model,
            api_key,
            base_url=base_url,
            system_instruction=system_instruction,
            extra_headers=extra_headers,
            extra_body=_merge_thinking(extra_body, thinking),
            json_mode=json_mode,
            max_connections=max_connections,
            metadata_extractors=metadata_extractors,
            **client_kwargs,
        )

    def _extract_tokens(self, response: Any) -> tuple[int, int, int, int]:
        """Extract tokens, preferring DeepSeek's native cache-hit field.

        Falls back to the OpenAI-shaped extraction for the
        input/output/total counts, then overrides the cached count with
        ``usage.prompt_cache_hit_tokens`` when DeepSeek provides it.
        """
        input_tokens, output_tokens, total_tokens, cached_tokens = super()._extract_tokens(response)

        usage = getattr(response, "usage", None)
        if usage is not None:
            hit = getattr(usage, "prompt_cache_hit_tokens", None)
            if hit is not None:
                cached_tokens = int(hit or 0)

        return input_tokens, output_tokens, total_tokens, cached_tokens


def _coerce_to_messages(prompt: str | list[Any]) -> list[Any]:
    """Convert a string prompt to a single user message; pass lists through."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return list(prompt)


def _has_system_message(messages: list[Any]) -> bool:
    """Return True if ``messages`` already contains a system-role entry."""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            return True
    return False
