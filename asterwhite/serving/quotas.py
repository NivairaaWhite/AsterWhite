"""
AsterWhite :: serving.quotas

Concurrency control: a plain counter guarded by an asyncio.Lock, sized
from inference.max_concurrent_requests. A request that arrives at
capacity is REJECTED with 429 immediately (spec: "Enforce per-key or
per-client quotas and return a clear error when exceeded") rather than
queued -- an explicit, immediate rejection is more honest than an
invisible queue that lets tail latency degrade unbounded.

(Deliberately not asyncio.Semaphore: Semaphore has no public non-blocking
try-acquire, and reaching into its private _value to peek is the kind of
fragile-internals trick this codebase avoids elsewhere -- an explicit
counter is both simpler and fully correct.)
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import HTTPException, status

from ..observability.metrics import ACTIVE_REQUESTS, QUEUE_REJECTIONS


class ConcurrencyLimiter:
    def __init__(self, max_concurrent: int):
        self.max_concurrent = max_concurrent
        self._active = 0
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        async with self._lock:
            if self._active >= self.max_concurrent:
                QUEUE_REJECTIONS.inc()
                raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                     detail=f"server is at its concurrency limit "
                                            f"({self.max_concurrent} concurrent requests) -- retry shortly")
            self._active += 1
        ACTIVE_REQUESTS.inc()
        try:
            yield
        finally:
            async with self._lock:
                self._active -= 1
            ACTIVE_REQUESTS.dec()
