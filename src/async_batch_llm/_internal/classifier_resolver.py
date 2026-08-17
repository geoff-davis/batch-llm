"""Identity-safe per-strategy error-classifier resolution."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..llm_strategies import LLMCallStrategy
from ..strategies import DefaultErrorClassifier, ErrorClassifier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ClassifierEntry:
    """Keep the strategy alive so a recycled object ID cannot hit stale state."""

    strategy: LLMCallStrategy
    classifier: ErrorClassifier


class StrategyClassifierResolver:
    """Resolve one automatic classifier per strategy object identity.

    An explicit caller-supplied classifier is returned for every strategy. In
    automatic mode, each strategy recommendation is evaluated at most once and
    cached together with a strong reference to the strategy.
    """

    def __init__(self, explicit: ErrorClassifier | None = None) -> None:
        self._explicit = explicit
        self._entries: dict[int, _ClassifierEntry] = {}

    @property
    def compatibility_classifier(self) -> ErrorClassifier:
        """Host-wide debug alias retained for callers inspecting old internals."""
        return self._explicit or DefaultErrorClassifier()

    def resolve(self, strategy: LLMCallStrategy) -> ErrorClassifier:
        if self._explicit is not None:
            return self._explicit

        strategy_id = id(strategy)
        entry = self._entries.get(strategy_id)
        if entry is not None and entry.strategy is strategy:
            return entry.classifier

        try:
            recommended = strategy.recommended_error_classifier()
        except Exception as exc:
            logger.warning(
                "[WARN] %s.recommended_error_classifier() failed: %s. "
                "Using DefaultErrorClassifier.",
                type(strategy).__name__,
                exc,
            )
            recommended = None

        classifier = recommended or DefaultErrorClassifier()
        self._entries[strategy_id] = _ClassifierEntry(strategy, classifier)
        return classifier

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()


__all__ = ["StrategyClassifierResolver"]
