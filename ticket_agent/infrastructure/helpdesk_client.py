"""HTTP client for the Roosterly helpdesk API.

Retry rules follow the API reference:
- 503: nothing was saved, so every request is safe to retry with backoff.
- 429: wait Retry-After, then retry.
- 409: never retry. 400/401/404/422: retrying the same request cannot help.
- Network failure after a write was sent: the outcome is unknown, so we do not retry here;
  the runner asks the helpdesk whether the write landed first.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import Any

import httpx

from ticket_agent.application.errors import (
    AlreadyHandledError,
    HelpdeskAuthError,
    HelpdeskError,
    HelpdeskUnavailableError,
    OutcomeUnknownError,
    RequestRejectedError,
    TicketNotFoundError,
)
from ticket_agent.application.ports import RemoteTicketState
from ticket_agent.domain.models import Decision, Ticket
from ticket_agent.infrastructure.resilience import RateLimiter, RetryPolicy

logger = logging.getLogger(__name__)

PAGE_SIZE = 10
_USER_AGENT = "support-ticket-agent/0.1"
# The request never reached the server: safe to retry even for writes.
_NOT_SENT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
# Server-side failures where a write may have been applied.
_AMBIGUOUS_STATUSES = {500, 502, 504}


class HelpdeskClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        candidate_email: str,
        rate_limiter: RateLimiter,
        retry_policy: RetryPolicy,
        candidate_name: str = "",
        timeout_s: float = 15.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "X-Candidate-Email": candidate_email,
            "User-Agent": _USER_AGENT,
        }
        if candidate_name:
            headers["X-Candidate-Name"] = candidate_name
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            transport=transport,
        )
        self._limiter = rate_limiter
        self._retry = retry_policy
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._retry_count = 0

    @property
    def retry_count(self) -> int:
        return self._retry_count

    def close(self) -> None:
        self._http.close()

    # --- public API -------------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        return _as_dict(self._request("GET", "/status").json())

    def list_open_tickets(self, max_tickets: int) -> list[Ticket]:
        """Reads every page of open tickets up front, then the caller works through the list."""
        tickets: list[Ticket] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while len(tickets) < max_tickets:
            params: dict[str, str | int] = {"limit": PAGE_SIZE, "status": "open"}
            if cursor:
                params["cursor"] = cursor
            page = _as_dict(self._request("GET", "/tickets", params=params).json())
            items = _first_list(page, "tickets", "items", "data")
            tickets.extend(Ticket.model_validate(item) for item in items)
            cursor = page.get("next_cursor")
            if not cursor or cursor in seen_cursors:
                break
            seen_cursors.add(cursor)
        return tickets[:max_tickets]

    def get_ticket_state(self, ticket_id: str) -> RemoteTicketState:
        payload = _as_dict(self._request("GET", f"/tickets/{ticket_id}").json())
        return RemoteTicketState(ticket_id=ticket_id, handled=parse_handled(payload))

    def submit_decision(self, decision: Decision) -> None:
        self._request(
            "POST",
            f"/tickets/{decision.ticket_id}/{decision.action.value}",
            json=decision.to_api_payload(),
            idempotent=False,
        )

    def get_examples(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/examples").json()
        if isinstance(payload, list):
            return payload
        return _first_list(_as_dict(payload), "examples", "items", "data")

    def reset_practice(self) -> dict[str, Any]:
        # Re-opening is idempotent: doing it twice leaves the same state.
        return _as_dict(self._request("POST", "/practice/reset", idempotent=True).json())

    # --- transport --------------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str | int] | None = None,
        json: dict[str, Any] | None = None,
        idempotent: bool = True,
    ) -> httpx.Response:
        last_problem = ""
        for attempt in range(1, self._retry.max_attempts + 1):
            if attempt > 1:
                self._retry_count += 1
            self._limiter.acquire()
            try:
                response = self._http.request(method, path, params=params, json=json)
            except _NOT_SENT_ERRORS as exc:
                last_problem = f"{type(exc).__name__}: {exc}"
                self._backoff(attempt, method, path, last_problem)
                continue
            except httpx.TransportError as exc:
                if not idempotent:
                    raise OutcomeUnknownError(f"{method} {path}: {exc}") from exc
                last_problem = f"{type(exc).__name__}: {exc}"
                self._backoff(attempt, method, path, last_problem)
                continue

            status = response.status_code
            if status < 300:
                return response
            if status == 429:
                wait = self._retry.retry_after(
                    response.headers.get("Retry-After"), attempt, self._rng
                )
                logger.warning("rate limited: path=%s wait_s=%.2f", path, wait)
                self._limiter.pause(wait)
                last_problem = "429 rate_limited"
                continue
            if status == 503 or (status in _AMBIGUOUS_STATUSES and idempotent):
                last_problem = f"{status} {_error_code(response)}"
                self._backoff(attempt, method, path, last_problem)
                continue
            if status in _AMBIGUOUS_STATUSES:
                raise OutcomeUnknownError(f"{method} {path}: HTTP {status}", status)
            raise _error_for(response, method, path)

        raise HelpdeskUnavailableError(
            f"{method} {path} failed after {self._retry.max_attempts} attempts: {last_problem}"
        )

    def _backoff(self, attempt: int, method: str, path: str, problem: str) -> None:
        if attempt >= self._retry.max_attempts:
            return
        delay = self._retry.backoff(attempt, self._rng)
        logger.warning(
            "transient helpdesk error, retrying: method=%s path=%s attempt=%s delay_s=%.2f "
            "problem=%s",
            method,
            path,
            attempt,
            delay,
            problem,
        )
        self._sleep(delay)


def parse_handled(payload: dict[str, Any]) -> bool:
    """Tells whether GET /tickets/{id} says the ticket was handled.

    The reference only says the response includes "how it was handled (if it was)", so we
    accept the likely shapes and refuse to guess on anything else: an unknown shape raises,
    and the runner leaves the ticket alone rather than risk a second answer.
    """
    for key in ("handled", "handling", "decision", "resolution"):
        if key in payload:
            value = payload[key]
            return value if isinstance(value, bool) else value not in (None, {}, [], "")
    status = payload.get("status")
    if isinstance(status, str) and status:
        return status.lower() != "open"
    nested = payload.get("ticket")
    if isinstance(nested, dict):
        return parse_handled(nested)
    raise HelpdeskError(f"cannot tell whether ticket is handled from keys {sorted(payload)}")


def _error_for(response: httpx.Response, method: str, path: str) -> HelpdeskError:
    status = response.status_code
    code = _error_code(response)
    message = f"{method} {path}: HTTP {status} {code}".strip()
    if status == 401:
        return HelpdeskAuthError(message, status, code)
    if status == 404:
        return TicketNotFoundError(message, status, code)
    if status == 409:
        return AlreadyHandledError(message, status, code)
    if status in (400, 422):
        return RequestRejectedError(message, status, code, _error_problems(response))
    return HelpdeskError(message, status, code)


def _error_body(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _error_code(response: httpx.Response) -> str:
    code = _error_body(response).get("error", "")
    return code if isinstance(code, str) else ""


def _error_problems(response: httpx.Response) -> list[str]:
    problems = _error_body(response).get("problems", [])
    return [str(p) for p in problems] if isinstance(problems, list) else [str(problems)]


def _as_dict(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HelpdeskError(f"expected a JSON object, got {type(payload).__name__}")
    return payload


def _first_list(payload: dict[str, Any], *keys: str) -> list[Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    raise HelpdeskError(f"no list under any of {keys} in response keys {sorted(payload)}")
