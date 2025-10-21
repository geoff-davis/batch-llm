"""Helper functions for creating test strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from ..llm_strategies import PydanticAIStrategy
    from .mocks import MockAgent

from ..llm_strategies import PydanticAIStrategy
from .mocks import MockAgent

TOutput = TypeVar("TOutput")


def mock_strategy(mock_agent: MockAgent[TOutput]) -> PydanticAIStrategy[TOutput]:
    """
    Convert a MockAgent to a PydanticAIStrategy for testing.

    This helper makes it easy to use MockAgent in tests that require strategies.

    Args:
        mock_agent: The MockAgent to wrap

    Returns:
        A PydanticAIStrategy that uses the MockAgent

    Example:
        >>> mock_agent = MockAgent(response_factory=lambda p: TestOutput(text=p))
        >>> strategy = mock_strategy(mock_agent)
        >>> work_item = LLMWorkItem(item_id="test", strategy=strategy, prompt="Hello")
    """
    return PydanticAIStrategy(agent=mock_agent)
