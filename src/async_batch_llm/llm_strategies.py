"""LLM call strategies for flexible model configuration and execution.

This module provides strategy classes that encapsulate how LLM calls are made,
including model selection, response parsing, and retry behavior.

v0.6.0: Strategies now accept an LLMModel instead of raw client + model name.
        GeminiCachedStrategy and GeminiResponse removed — use
        GeminiStrategy(model=GeminiCachedModel(...)) instead.
"""

import inspect
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Generic, NoReturn, TypeVar, cast, overload

from .base import LLMResponse, RetryState, TokenUsage
from .core.protocols import ManagedLLMModel
from .strategies.errors import TokenTrackingError
from .token_estimation import TokenEstimate, TokenEstimator

# Conditional imports for optional dependencies
if TYPE_CHECKING:
    from pydantic_ai import Agent

    from .core.protocols import LLMModel
    from .strategies.errors import ErrorClassifier
else:
    try:
        from pydantic_ai import Agent
    except ImportError:
        Agent = Any  # type: ignore[misc,assignment]

# Module-level logger
logger = logging.getLogger(__name__)

TOutput = TypeVar("TOutput")


def _run_result_usage(result: Any) -> Any:
    """Read usage off a pydantic-ai run result across API generations.

    pydantic-ai 1.x exposes ``AgentRunResult.usage`` as a property (calling
    the returned ``RunUsage`` only works through a deprecation shim slated
    for removal); older results — and test doubles like ``MockAgent``'s —
    expose a ``usage()`` method. Prefer the property form, calling only
    bound methods.
    """
    usage = getattr(result, "usage", None)
    if inspect.ismethod(usage) or inspect.isfunction(usage):
        return usage()
    return usage


def _usage_field(usage: Any, *names: str) -> int:
    """Read the first present, non-None attribute in ``names`` off ``usage``.

    Lets us read pydantic-ai usage without triggering DeprecationWarnings:
    we ask for the 1.x name (``input_tokens``) before the deprecated 0.x alias
    (``request_tokens``). Returns 0 when ``usage`` is falsy or none match.
    """
    if not usage:
        return 0
    for name in names:
        value = getattr(usage, name, None)
        if value is not None:
            return int(value)
    return 0


def _attach_token_usage(exception: Exception, tokens: "TokenUsage") -> NoReturn:
    """Attach token usage to ``exception`` and re-raise it.

    When a response parser raises, the underlying API has usually already
    billed the tokens — so we surface them to the framework's failure-path
    token accounting before propagating the error.

    Built-in exceptions without a writable ``__dict__`` are wrapped in a
    :class:`TokenTrackingError` (preserving the original via ``__cause__``);
    everything else gets ``_failed_token_usage`` stamped on it. Never returns.
    """
    if not hasattr(exception, "__dict__"):
        wrapped = TokenTrackingError(str(exception), token_usage=tokens)
        wrapped.__cause__ = exception
        raise wrapped from exception
    exception.__dict__["_failed_token_usage"] = tokens
    raise exception


class LLMCallStrategy(ABC, Generic[TOutput]):
    """
    Abstract base class for LLM call strategies.

    A strategy encapsulates how LLM calls are made, including:
    - Resource initialization (caches, clients)
    - Call execution with retries
    - Resource cleanup

    The framework calls:
    1. prepare() once per unique strategy instance before its first execution
    2. execute() for each attempt (including retries)
    3. cleanup() once per prepared strategy when the processor exits or shuts down
    """

    async def prepare(self) -> None:
        """
        Initialize resources before making any LLM calls.

        Called once per unique strategy instance before the first work item using
        that instance executes. Use this to set up shared caches, clients, etc.
        Per-item retry state belongs in execute()/on_error() via RetryState.

        Default: no-op
        """
        pass

    @abstractmethod
    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: "RetryState | None" = None
    ) -> tuple[TOutput, TokenUsage] | tuple[TOutput, TokenUsage, dict[str, Any] | None]:
        """
        Execute an LLM call for the given attempt.

        Args:
            prompt: The prompt to send to the LLM
            attempt: Which *logical* retry attempt this is (1, 2, 3, ...).
                **Guarantee:** rate-limit errors do NOT advance this number. If a
                call is throttled (429 / coordinated cooldown), the framework
                retries the *same* ``attempt`` after the cooldown rather than
                burning an attempt. So a strategy that escalates models on higher
                attempts (e.g. attempt >= 2 → smarter model) escalates because the
                *output* was bad over ``max_attempts`` tries, never merely because
                the endpoint was busy. Rate-limit retries are bounded separately
                by ``RetryConfig.max_rate_limit_retries``.
            timeout: Maximum time to wait for response (seconds)
            state: Optional retry state that persists across attempts (v0.3.0)

        Returns:
            Either a 2-tuple ``(output, token_usage)`` or a 3-tuple
            ``(output, token_usage, metadata)``. ``token_usage`` is a
            TokenUsage dict with optional keys ``input_tokens``,
            ``output_tokens``, ``total_tokens``, ``cached_input_tokens``.
            ``metadata`` (v0.10.0) is a provider-specific dict forwarded into
            ``WorkItemResult.metadata`` — typically ``finish_reason``,
            ``model``, ``provider`` (OpenRouter), ``safety_ratings`` (Gemini);
            pass ``None`` if you have nothing to surface. The 2-tuple shape is
            supported for backward compatibility but will be removed in a
            future release; built-in strategies all return the 3-tuple shape.

        Raises:
            Exception: Any exception propagated to trigger a retry (if
                retryable) or a permanent failure.

        Note (v0.3.0):
            The state parameter allows strategies to maintain state across retry
            attempts for multi-stage retry patterns. See RetryState documentation
            for examples.
        """
        pass

    async def cleanup(self) -> None:
        """
        Clean up resources when the processor exits or shuts down.

        Called once per prepared strategy instance, not once per work item.

        **Use this for:**
        - Closing connections/sessions
        - Releasing locks
        - Logging final metrics
        - Deleting temporary files

        **Do NOT use this for:**
        - Deleting caches intended for reuse across runs
        - Destructive cleanup that prevents resource reuse

        **Note on Caches (v0.2.0):**
        For reusable resources like Gemini caches with TTLs, consider letting
        them expire naturally to enable cost savings across multiple pipeline
        runs. See `GeminiCachedModel` for an example.

        Default: no-op
        """
        pass

    async def on_error(
        self, exception: Exception, attempt: int, state: "RetryState | None" = None
    ) -> None:
        """
        Handle errors that occur during execute().

        Called by the framework when execute() raises an exception, before
        deciding whether to retry. This allows strategies to:
        - Inspect the error type to adjust retry behavior
        - Store error information for use in next attempt
        - Modify prompts based on validation errors
        - Track error patterns across attempts

        Args:
            exception: The exception that was raised during execute()
            attempt: Which attempt number failed (1, 2, 3, ...)
            state: Optional retry state that persists across attempts (v0.3.0)

        Default: no-op

        Example:
            async def on_error(
                self, exception: Exception, attempt: int, state: RetryState | None = None
            ) -> None:
                if state is not None:
                    # Track validation errors separately from other errors
                    if isinstance(exception, ValidationError):
                        count = state.get('validation_failures', 0) + 1
                        state.set('validation_failures', count)
                        # Save partial results for recovery
                        if hasattr(exception, 'partial_data'):
                            state.set('partial_data', exception.partial_data)
        """
        pass

    async def dry_run(self, prompt: str) -> tuple[TOutput, TokenUsage]:
        """
        Return mock output for dry-run mode (testing without API calls).

        Override this method to provide realistic mock data for testing.
        Default implementation returns placeholder values that may not match
        your output type.

        Args:
            prompt: The prompt that would have been sent to the LLM

        Returns:
            Tuple of (mock_output, mock_token_usage)

        Default behavior:
        - Returns string "[DRY-RUN] Mock output" as output
        - Returns mock token usage: 100 input, 50 output, 150 total
        """
        mock_output: TOutput = f"[DRY-RUN] Mock output for prompt: {prompt[:50]}..."  # ty:ignore[invalid-assignment]
        mock_tokens: TokenUsage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }
        return mock_output, mock_tokens

    def recommended_error_classifier(self) -> "ErrorClassifier | None":
        """Return the error classifier best suited to this strategy's provider.

        The execution host calls this once per strategy identity when the caller
        didn't pass ``error_classifier`` explicitly.

        Returns ``None`` by default ("no preference"), which lets the framework
        fall back to :class:`DefaultErrorClassifier`. Provider strategies
        (``GeminiStrategy``, ``OpenAIStrategy``, …) override this to return their
        matching classifier. An explicit ``error_classifier=`` on the processor
        always wins over this recommendation.
        """
        return None

    def estimate_tokens(
        self,
        prompt: str,
        attempt: int,
        state: RetryState | None,
    ) -> TokenEstimate | Awaitable[TokenEstimate] | None:
        """Return a local token estimate for TPM admission, if supported.

        The default supplies no estimate. Configure ``ProcessorConfig`` with a
        run-level estimator or override this hook. No provider call should be
        made solely to estimate tokens.
        """
        del prompt, attempt, state
        return None

    @property
    def max_concurrency(self) -> int | None:
        """Maximum safe concurrent calls advertised by this strategy.

        ``None`` means the capacity is unknown. Model-backed strategies forward
        capacity metadata from their model; custom strategies can override this
        property when they own a bounded client or transport.
        """
        return None

    @property
    def concurrency_scope(self) -> object:
        """Identity whose capacity is shared by concurrent calls."""
        return self

    @property
    def quota_scope(self) -> object:
        """Identity whose RPM, TPM, and coordinated cooldown are shared.

        The default follows :attr:`concurrency_scope` for backward-compatible
        ownership. Override it when one provider/account quota spans multiple
        clients or when one shared client serves independent quota budgets.
        Object identity, not equality, defines sharing.
        """
        return self.concurrency_scope


class ModelStrategy(LLMCallStrategy[TOutput]):
    """
    Base strategy for any provider exposed as an :class:`LLMModel`.

    Holds the machinery shared by all model-backed strategies: the model
    reference, an optional response parser, lifecycle delegation to
    :class:`ManagedLLMModel`, and an ``execute()`` that calls
    ``model.generate()``, parses the response, and forwards
    ``LLMResponse.metadata`` as the third tuple element.

    The provider-named subclasses (:class:`GeminiStrategy`,
    :class:`OpenAIStrategy`, :class:`OpenRouterStrategy`) are thin shells over
    this base — they exist so users can pick the strategy named after the
    provider they're using. Use this base directly for a custom
    :class:`LLMModel` you don't want to name a dedicated subclass for.

    Added in v0.10.0 (extracted from the formerly-duplicated provider
    strategy classes).
    """

    @overload
    def __init__(
        self: "ModelStrategy[str]",
        model: "LLMModel",
        response_parser: None = None,
        *,
        temperature: float | None = 0.0,
        generation_config: dict[str, Any] | None = None,
        quota_scope: object | None = None,
        token_estimator: TokenEstimator | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        model: "LLMModel",
        response_parser: Callable[[LLMResponse], TOutput],
        *,
        temperature: float | None = 0.0,
        generation_config: dict[str, Any] | None = None,
        quota_scope: object | None = None,
        token_estimator: TokenEstimator | None = None,
    ) -> None: ...

    def __init__(
        self,
        model: "LLMModel",
        response_parser: Callable[[LLMResponse], TOutput] | None = None,
        *,
        temperature: float | None = 0.0,
        generation_config: dict[str, Any] | None = None,
        quota_scope: object | None = None,
        token_estimator: TokenEstimator | None = None,
    ) -> None:
        """
        Initialize strategy.

        Args:
            model: An LLMModel instance (e.g., GeminiModel, OpenAIModel).
            response_parser: Function to parse LLMResponse into TOutput. Defaults to
                returning ``response.text`` — only valid when TOutput is ``str``.
                When TOutput is any other type, pass a response_parser (enforced
                by the @overload signatures).
            temperature: Default sampling temperature. Pass ``None`` to omit the
                parameter and use the provider default (e.g. for OpenAI
                reasoning models that reject an explicit temperature).
            generation_config: Provider-specific config forwarded to
                ``model.generate(config=...)`` on every call — e.g. Gemini's
                ``response_schema`` / ``response_mime_type`` / ``tools``, or an
                OpenAI-compatible ``response_format`` / ``max_tokens`` (merged
                into the model's ``extra_body``). Lets a built-in strategy carry
                native structured output or grounding without subclassing
                ``execute()``. A subclass that overrides ``execute()`` and needs a
                *per-attempt* config can read ``self.generation_config`` and merge.
            quota_scope: Optional identity shared by strategies consuming the
                same provider/account quota. Defaults to the model, matching
                the existing concurrency scope.
            token_estimator: Optional strategy-owned local estimator used when
                TPM admission is enabled and the processor has no override.
        """
        if token_estimator is not None and not callable(token_estimator):
            raise TypeError("token_estimator must be callable or None")
        self.model = model
        # The overloads restrict the None-parser path to TOutput=str, so the cast
        # below is sound at static-analysis time.
        self.response_parser = response_parser or (lambda response: cast(TOutput, response.text))
        self.temperature = temperature
        self.generation_config = generation_config
        self._quota_scope = quota_scope
        self._token_estimator = token_estimator

    @property
    def max_concurrency(self) -> int | None:
        """Forward optional client/transport capacity advertised by the model."""
        capacity = getattr(self.model, "max_concurrency", None)
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            return None
        return capacity

    @property
    def concurrency_scope(self) -> object:
        """Strategies wrapping the same model share one admission limit."""
        return self.model

    @property
    def quota_scope(self) -> object:
        """Strategies wrapping the same model share quota by default."""
        return self.model if self._quota_scope is None else self._quota_scope

    def estimate_tokens(
        self,
        prompt: str,
        attempt: int,
        state: RetryState | None,
    ) -> TokenEstimate | Awaitable[TokenEstimate] | None:
        if self._token_estimator is None:
            return None
        return self._token_estimator(
            prompt,
            strategy=self,
            attempt=attempt,
            state=state,
        )

    async def request_concurrency(self, concurrency: int) -> bool:
        """Forward a concurrency request to the model, when supported.

        Execution surfaces call this when ``ProcessorConfig.concurrency`` is
        set (v0.20.0) so built-in models can right-size their connection
        pools. Models without the hook (custom ``LLMModel`` implementations,
        Gemini) simply return False and are left untouched.
        """
        hook = getattr(self.model, "request_concurrency", None)
        if hook is None:
            return False
        result = await hook(concurrency)
        return bool(result)

    async def prepare(self) -> None:
        """Delegate to model.prepare() if the model has a managed lifecycle."""
        if isinstance(self.model, ManagedLLMModel):
            await self.model.prepare()

    async def cleanup(self) -> None:
        """Delegate to model.cleanup() if the model has a managed lifecycle."""
        if isinstance(self.model, ManagedLLMModel):
            await self.model.cleanup()

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[TOutput, TokenUsage, dict[str, Any] | None]:
        """Execute the LLM call via the model and parse the response.

        Args:
            prompt: The prompt to send to the LLM.
            attempt: Which retry attempt this is (1, 2, 3, ...).
            timeout: Maximum time for response (enforced by the framework).
            state: Optional retry state for cross-attempt persistence.

        Returns:
            3-tuple ``(parsed_output, token_usage, metadata)`` where ``metadata``
            is forwarded from ``LLMResponse.metadata`` (provider, finish_reason,
            model, safety_ratings, etc.). Added the metadata slot in v0.10.0; the
            framework still accepts the legacy 2-tuple shape from custom
            strategies via a compat shim.
        """
        # Only pass `config` when set, so a custom LLMModel whose generate()
        # doesn't accept the kwarg keeps working — the default path is identical
        # to before generation_config existed.
        gen_kwargs: dict[str, Any] = {"temperature": self.temperature}
        if self.generation_config is not None:
            gen_kwargs["config"] = self.generation_config
        provider_started = time.perf_counter()
        try:
            llm_response = await self.model.generate(prompt, **gen_kwargs)
        finally:
            if state is not None:
                state.set(
                    "_abl_last_provider_seconds",
                    max(0.0, time.perf_counter() - provider_started),
                )

        try:
            output = self.response_parser(llm_response)
        except Exception as e:
            # The API already billed for this call even though parsing failed —
            # attach the token usage so the framework can account for it.
            _attach_token_usage(e, llm_response.token_usage)

        return output, llm_response.token_usage, llm_response.metadata


class GeminiStrategy(ModelStrategy[TOutput]):
    """
    Strategy for calling a Gemini model and parsing the response.

    Accepts an LLMModel (e.g., GeminiModel or GeminiCachedModel) and a
    response parser. The model handles the API call and token extraction;
    the strategy handles response parsing and lifecycle delegation.

    For caching, use ``GeminiStrategy(model=GeminiCachedModel(...))``.

    v0.6.0: Accepts LLMModel instead of raw client + model string.

    Example:
        >>> model = GeminiModel("gemini-2.5-flash", client)
        >>> strategy = GeminiStrategy(model, response_parser=lambda r: r.text)
        >>>
        >>> # With caching:
        >>> cached_model = GeminiCachedModel("gemini-2.5-flash", client, cached_content=[...])
        >>> strategy = GeminiStrategy(cached_model, response_parser=lambda r: r.text)
    """

    def recommended_error_classifier(self) -> "ErrorClassifier":
        from .classifiers.gemini import GeminiErrorClassifier

        return GeminiErrorClassifier()


class OpenAIStrategy(ModelStrategy[TOutput]):
    """
    Strategy for calling an OpenAI-compatible model and parsing the response.

    Accepts an LLMModel (typically OpenAIModel) and an optional response
    parser. The model handles the API call and token extraction; the
    strategy handles response parsing and lifecycle delegation.

    Added in v0.9.0.

    Example:
        >>> model = OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-...")
        >>> strategy = OpenAIStrategy(model)
        >>>
        >>> # Structured output via response_parser:
        >>> strategy = OpenAIStrategy(
        ...     model,
        ...     response_parser=lambda r: MyModel.model_validate_json(r.text),
        ... )
    """

    def recommended_error_classifier(self) -> "ErrorClassifier":
        from .classifiers.openai import OpenAIErrorClassifier

        return OpenAIErrorClassifier()


class OpenRouterStrategy(ModelStrategy[TOutput]):
    """
    Strategy for calling an OpenRouter-backed model and parsing the response.

    Functionally identical to :class:`OpenAIStrategy` (both delegate to an
    ``LLMModel`` via :class:`ModelStrategy`); the separate class exists for
    provider-named symmetry so users can pick the strategy named after the
    provider they're using. For OpenRouter, ``LLMResponse.metadata`` typically
    includes ``provider`` (the upstream that served the request), ``model``
    (the actually-routed model), and ``finish_reason``.

    Added in v0.9.0.

    Example:
        >>> model = OpenRouterModel.from_api_key(
        ...     "anthropic/claude-haiku-4-5", api_key="sk-or-...",
        ... )
        >>> strategy = OpenRouterStrategy(model)
    """

    def recommended_error_classifier(self) -> "ErrorClassifier":
        from .classifiers.openrouter import OpenRouterErrorClassifier

        return OpenRouterErrorClassifier()


class DeepSeekStrategy(ModelStrategy[TOutput]):
    """
    Strategy for calling a DeepSeek model and parsing the response.

    Functionally identical to :class:`OpenAIStrategy` (both delegate to an
    ``LLMModel`` via :class:`ModelStrategy`); the separate class exists for
    provider-named symmetry. Pair it with :class:`DeepSeekModel`, which
    surfaces DeepSeek's native cache-hit token counts.

    Added in v0.10.0.

    Example:
        >>> model = DeepSeekModel.from_api_key("deepseek-chat", api_key="sk-...")
        >>> strategy = DeepSeekStrategy(model)
    """

    def recommended_error_classifier(self) -> "ErrorClassifier":
        # DeepSeek is OpenAI-compatible; reuse the OpenAI classifier.
        from .classifiers.openai import OpenAIErrorClassifier

        return OpenAIErrorClassifier()


class PydanticAIStrategy(LLMCallStrategy[TOutput]):
    """
    Strategy for using PydanticAI agents.

    This strategy wraps a PydanticAI agent, providing a clean interface
    for batch processing. The agent handles all model interaction, validation,
    and parsing.

    Best for: Structured output with Pydantic models, using PydanticAI's features.
    """

    def __init__(self, agent: "Agent[None, TOutput]"):
        """
        Initialize PydanticAI strategy.

        Args:
            agent: Configured PydanticAI agent
        """
        if Agent is Any:
            raise ImportError(
                "pydantic-ai is required for PydanticAIStrategy. "
                "Install with: pip install 'async-batch-llm[pydantic-ai]'"
            )

        self.agent = agent

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[TOutput, TokenUsage, dict[str, Any] | None]:
        """Execute PydanticAI agent call.

        Note: timeout parameter is provided for information but timeout enforcement
        is handled by the framework wrapping this call in asyncio.wait_for().

        Args:
            prompt: The prompt to send to the LLM
            attempt: Which retry attempt this is (1, 2, 3, ...)
            timeout: Maximum time to wait for response (seconds)
            state: Optional retry state (v0.3.0, unused by this strategy)

        Returns:
            3-tuple ``(output, token_usage, metadata)``. PydanticAI's result
            object doesn't expose provider-side metadata uniformly, so
            ``metadata`` is currently always ``None`` here. (v0.10.0)
        """
        provider_started = time.perf_counter()
        try:
            result = await self.agent.run(prompt)
        finally:
            if state is not None:
                state.set(
                    "_abl_last_provider_seconds",
                    max(0.0, time.perf_counter() - provider_started),
                )

        # Extract token usage FIRST (before accessing result.output which may fail validation).
        # pydantic-ai 1.x renamed request_tokens/response_tokens -> input_tokens/output_tokens
        # (the old names still exist but emit DeprecationWarning). Prefer the new
        # names, falling back to the legacy ones so both 0.x and 1.x work cleanly.
        usage = _run_result_usage(result)
        tokens: TokenUsage = {
            "input_tokens": _usage_field(usage, "input_tokens", "request_tokens"),
            "output_tokens": _usage_field(usage, "output_tokens", "response_tokens"),
            "total_tokens": _usage_field(usage, "total_tokens"),
        }

        # Access result.output (may raise validation errors)
        try:
            output = result.output
        except Exception as e:
            # Attach token usage to exception so framework can track it
            _attach_token_usage(e, tokens)

        return output, tokens, None

    async def dry_run(self, prompt: str) -> tuple[TOutput, TokenUsage]:
        """Return mock output based on agent's result_type for dry-run mode."""
        # Try to create a mock instance of the expected output type
        try:
            from pydantic import BaseModel

            # pydantic-ai 1.x renamed Agent.result_type -> Agent.output_type
            # (the old attribute no longer exists). Prefer the new name, fall
            # back to the legacy one for 0.x agents.
            result_type = getattr(self.agent, "output_type", None)
            if result_type is None:
                result_type = getattr(self.agent, "result_type", None)

            # If output type is a Pydantic model, try to create an instance
            if isinstance(result_type, type) and issubclass(result_type, BaseModel):
                # Use model_construct to create instance without validation
                # This allows creating instances even with required fields
                mock_output: TOutput = result_type.model_construct()
            else:
                # For non-Pydantic types, use base class default
                return await super().dry_run(prompt)

        except Exception:
            # If anything fails, fall back to base class default
            return await super().dry_run(prompt)

        # Return mock output with realistic token usage
        mock_tokens: TokenUsage = {
            "input_tokens": len(prompt.split()),  # Rough estimate
            "output_tokens": 50,
            "total_tokens": len(prompt.split()) + 50,
        }

        return mock_output, mock_tokens
