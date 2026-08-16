"""Token counting via Anthropic's public count_tokens API
(client.messages.count_tokens) -- genuinely different from every other
systems/*/model.py: no local tokenizer, no vocab file, no byte-span
reconstruction. Every count is a real network round-trip, and the response
is just a total (SDK's MessageTokensCount has only `input_tokens: int` --
no token ids/strings/boundaries). So this can report compression rate /
fertility / token parity (derivable from a bare count) but NOT Rényi
efficiency or Gini, which need the per-token frequency distribution. Real
structural API limitation, reported honestly as narrower output (see
report_claude_eval), never faked with placeholder token identities to force
renyi/gini to compute.

RATE LIMITING: count_tokens has no batch endpoint -- one call per document.
Anthropic's published RPM limits by tier: Start=2000, Build=4000,
Scale=8000. RateLimiter enforces a strict rolling-window cap (thread-safe,
shared across the worker pool) so a large run never exceeds the configured
RPM regardless of thread count; concurrency is needed to actually approach
that cap since a sequential loop is latency-bound, not rate-limit-bound.

PREREQUISITE: needs a real ANTHROPIC_API_KEY (env var or --api-key) -- a
live account credential, unlike most of hf_frontier's public HF repos.
count_tokens itself is free to call but still requires auth and is subject
to the RPM limits above.
"""

import threading
import time
from collections import deque

import anthropic


class RateLimiter:
    """Thread-safe rolling-window rate limiter: at most `max_calls` calls in
    any trailing `period` seconds, across all threads sharing one instance.
    acquire() blocks (sleeps, doesn't raise) until a slot is free -- callers
    just call it before each request. ClaudeTokenCounter.count still retries
    actual 429s from the server too, since other concurrent usage on the
    same account (outside this process) isn't something this limiter alone
    can prevent."""

    def __init__(self, max_calls, period=60.0):
        self.max_calls = max_calls
        self.period = period
        self._calls = deque()
        self._lock = threading.Lock()

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] > self.period:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                sleep_for = self.period - (now - self._calls[0])
            time.sleep(max(sleep_for, 0.01))


class ClaudeTokenCounter:
    """One (model, shared rate limiter) pair -- .count(text) -> int.

    Retries transient failures with bounded exponential backoff:
    RateLimitError (429) and 5xx are subclasses of APIStatusError
    (`status_code` attribute); APIConnectionError (incl. APITimeoutError)
    covers connection drops/timeouts. Any other error (e.g. 401/bad
    request) is NOT retried and propagates immediately -- retrying a
    request that will never succeed just wastes rate-limit budget.
    """

    def __init__(self, model, rate_limiter, api_key=None, max_retries=5):
        self.model = model
        self.rate_limiter = rate_limiter
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.max_retries = max_retries

    def count(self, text):
        last_exc = None
        for attempt in range(self.max_retries):
            self.rate_limiter.acquire()
            try:
                resp = self.client.messages.count_tokens(
                    model=self.model,
                    messages=[{"role": "user", "content": text}],
                )
                return resp.input_tokens
            except anthropic.APIStatusError as e:
                last_exc = e
                if e.status_code == 429 or e.status_code >= 500:
                    time.sleep(min(2**attempt, 30))
                    continue
                raise
            except anthropic.APIConnectionError as e:
                last_exc = e
                time.sleep(min(2**attempt, 30))
                continue
        raise RuntimeError(
            f"count_tokens failed after {self.max_retries} attempts for model {self.model!r}"
        ) from last_exc
