"""Type protocols for batch LLM processing framework."""

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

# Constrain output to Pydantic models for validation
TOutput = TypeVar("TOutput", bound=BaseModel)


class AgentLike(Protocol[TOutput]):
    """Protocol that any agent must satisfy."""

    async def run(self, prompt: str, **kwargs) -> "ResultLike[TOutput]":
        """Run the agent with the given prompt."""
        ...


class ResultLike(Protocol[TOutput]):
    """Protocol for agent results."""

    @property
    def output(self) -> TOutput:
        """Get the agent output."""
        ...

    def usage(self) -> "UsageLike | None":
        """Get token usage information."""
        ...

    def all_messages(self) -> list[Any]:
        """Get all messages in the conversation."""
        ...


class UsageLike(Protocol):
    """Protocol for token usage."""

    @property
    def request_tokens(self) -> int:
        """Number of tokens in the request."""
        ...

    @property
    def response_tokens(self) -> int:
        """Number of tokens in the response."""
        ...

    @property
    def total_tokens(self) -> int:
        """Total number of tokens."""
        ...
