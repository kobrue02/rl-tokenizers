"""Token counting via Anthropic's own public count_tokens API
(client.messages.count_tokens) -- genuinely different from every other
systems/*/model.py in this repo: there is no local tokenizer object at all,
no vocabulary file, no byte-span reconstruction. Every single count is a
real network round-trip to Anthropic's API, and the response is JUST a
total count (confirmed via the installed SDK's own MessageTokensCount type:
a single `input_tokens: int` field, nothing else -- no token ids, no token
strings, no per-token boundaries). That means this integration can report
compression rate / fertility / token parity (all derivable from a bare
count) but CANNOT report Rényi efficiency or Gini, which need the actual
per-token FREQUENCY distribution -- which specific tokens repeat, not just
how many there were -- see common.eval.cross_tokenizer.evaluate_on_groups's
own docstring for what those two need. This is a real, structural
limitation of the public API, not a shortcut taken here: reported honestly
as narrower output (see evaluate.py's own report_claude_eval), never faked
with placeholder token identities to force renyi/gini to compute (that
would produce a meaningless, misleadingly-perfect result -- an explicit
decision, not an oversight, made when this module was built).

RATE LIMITING: count_tokens has no batch endpoint -- one call per document.
Anthropic's own published limits (requests per minute, by usage tier):
Start=2000, Build=4000, Scale=8000. RateLimiter below enforces a strict
rolling-window cap (thread-safe, shared across a whole run's worker pool)
so a large run (e.g. BOUQuET test's ~272k rows) never exceeds whatever RPM
the caller configures, regardless of how many worker threads are issuing
requests concurrently -- concurrency is still needed to actually approach
that cap: a purely sequential loop is latency-bound (a few hundred ms per
call is typical for a small API request), not rate-limit-bound, so on its
own it would sit well under even the lowest 2000/min tier.

PREREQUISITE: needs a real ANTHROPIC_API_KEY (env var, or --api-key) -- a
live account credential, not a public/anonymous endpoint the way most of
systems/hf_frontier's own HF repos are. Confirmed live: the count_tokens
endpoint itself is free to call (no per-token billing), but still requires
authentication and is subject to the RPM limits above.
"""

import threading
import time
from collections import deque

import anthropic


class RateLimiter:
    """Thread-safe rolling-window rate limiter: at most `max_calls` calls
    permitted in any trailing `period` seconds, across every thread sharing
    one instance. acquire() blocks (sleeping, not busy-waiting, and not
    raising) until a slot is free -- callers just call acquire() before each
    request and it paces itself, no separate retry-on-429 logic needed for
    the "too many calls from US" case (ClaudeTokenCounter.count still
    retries actual 429 responses from the server, since a shared account
    limit across OTHER concurrent usage this process doesn't know about is
    a real, separate possibility this limiter alone can't prevent)."""

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

    Retries transient failures with bounded exponential backoff (confirmed
    live against the installed anthropic SDK's own exception hierarchy,
    not guessed): RateLimitError (429) and InternalServerError-class 5xx
    responses are both subclasses of APIStatusError, which carries a real
    `status_code` attribute; APIConnectionError (network-level, no
    status_code at all -- APITimeoutError is its own subclass) covers
    connection drops/timeouts. Anything else (4xx other than 429 -- e.g. a
    genuine 401/bad request) is NOT retried and propagates immediately,
    since retrying a request that will never succeed just burns rate-limit
    budget and time for no benefit.
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
