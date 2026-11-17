"""Errors shared across layers. Each one says whether retrying can help."""

from __future__ import annotations


class AgentError(Exception):
    """Base class for all errors raised by the agent."""


class HelpdeskError(AgentError):
    def __init__(self, message: str, status_code: int | None = None, code: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class HelpdeskAuthError(HelpdeskError):
    """401: wrong key or email. Fatal for the whole run."""


class TicketNotFoundError(HelpdeskError):
    """404: the ticket is not in this account's set."""


class AlreadyHandledError(HelpdeskError):
    """409: the customer already got an answer. Never retry."""


class RequestRejectedError(HelpdeskError):
    """400 or 422: our request is wrong. Retrying the same body cannot help."""

    def __init__(
        self, message: str, status_code: int, code: str, problems: list[str] | None = None
    ) -> None:
        super().__init__(message, status_code, code)
        self.problems = problems or []


class HelpdeskUnavailableError(HelpdeskError):
    """Transient failures (503, 429, network) persisted after all retries."""


class OutcomeUnknownError(HelpdeskError):
    """A write was sent but no answer came back. It may or may not have been saved."""


class ModelError(AgentError):
    """The model call failed, refused, or returned output that does not fit the schema."""

    def __init__(self, message: str, cost_usd: float = 0.0) -> None:
        super().__init__(message)
        # Tokens may have been billed even though the call failed.
        self.cost_usd = cost_usd
