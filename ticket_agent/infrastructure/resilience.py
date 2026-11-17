"""Client-side rate limiting and retry backoff."""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


class RateLimiter:
    """Spaces calls evenly so we never exceed the helpdesk's 5 requests per second.

    Even spacing (rather than bursts) keeps us under a sliding-window limit too.
    """

    def __init__(
        self,
        requests_per_second: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self._interval = 1.0 / requests_per_second
        self._clock = clock
        self._sleep = sleep
        self._next_slot = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = self._clock()
            wait = self._next_slot - now
            if wait > 0:
                self._sleep(wait)
            self._next_slot = max(now, self._next_slot) + self._interval

    def pause(self, seconds: float) -> None:
        """Pushes the next slot back, e.g. after the server sent Retry-After."""
        with self._lock:
            self._next_slot = max(self._next_slot, self._clock() + seconds)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 6
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    max_retry_after_s: float = 30.0

    def backoff(self, attempt: int, rng: random.Random) -> float:
        """Exponential backoff with full jitter. attempt starts at 1."""
        ceiling = min(self.max_delay_s, self.base_delay_s * 2 ** (attempt - 1))
        return rng.uniform(0, ceiling)

    def retry_after(self, header_value: str | None, attempt: int, rng: random.Random) -> float:
        """Honours Retry-After (seconds) when present and sane, else falls back to backoff."""
        if header_value:
            try:
                seconds = float(header_value)
            except ValueError:
                seconds = -1.0
            if seconds >= 0:
                return min(seconds, self.max_retry_after_s)
        return self.backoff(attempt, rng)
