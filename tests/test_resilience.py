import random

from ticket_agent.infrastructure.resilience import RateLimiter, RetryPolicy


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limiter_spaces_calls_evenly():
    clock = FakeClock()
    limiter = RateLimiter(4, clock=clock, sleep=clock.sleep)
    for _ in range(9):
        limiter.acquire()
    # 9 calls at 4/s need at least 2 seconds; never more than 5 calls in any second.
    assert clock.now >= 2.0
    assert all(abs(s - 0.25) < 1e-9 for s in clock.sleeps)


def test_rate_limiter_does_not_sleep_when_calls_are_slow():
    clock = FakeClock()
    limiter = RateLimiter(4, clock=clock, sleep=clock.sleep)
    limiter.acquire()
    clock.now += 1.0
    limiter.acquire()
    assert clock.sleeps == []


def test_pause_delays_the_next_call():
    clock = FakeClock()
    limiter = RateLimiter(4, clock=clock, sleep=clock.sleep)
    limiter.pause(2.0)
    limiter.acquire()
    assert clock.now == 2.0


def test_backoff_is_bounded_and_jittered():
    policy = RetryPolicy(base_delay_s=0.5, max_delay_s=8.0)
    rng = random.Random(1)
    delays = [policy.backoff(attempt, rng) for attempt in range(1, 20)]
    assert all(0 <= d <= 8.0 for d in delays)
    assert len(set(delays)) > 1


def test_retry_after_header_is_honoured_and_capped():
    policy = RetryPolicy(max_retry_after_s=30)
    rng = random.Random(1)
    assert policy.retry_after("2", 1, rng) == 2.0
    assert policy.retry_after("600", 1, rng) == 30.0
    assert 0 <= policy.retry_after("soon", 1, rng) <= policy.base_delay_s
