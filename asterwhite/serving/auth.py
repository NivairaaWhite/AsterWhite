"""
AsterWhite :: serving.auth

API-key check that accepts however the calling wrapper actually sends its
key -- `Authorization: Bearer <key>` (every OpenAI-client library),
`X-API-Key: <key>` (a very common alternate convention), or an `api_key`
query parameter (what a few simpler HTTP clients/browser tools default
to). All three are checked; any one matching is accepted.

Keys come ONLY from the ASTERWHITE_API_KEYS environment variable
(comma-separated) -- never from a config file or source control (spec
4.2). If that variable is unset or empty, auth is OPEN (no key required)
-- a deliberate, loudly-logged default for local/dev use, not a silent
security gap: see app.py's startup log line.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import Header, HTTPException, Query, status

from ..safety.pii_scrubber import scrub
from ..utils.error_tracing import get_logger

logger = get_logger("asterwhite.serving.auth")


def configured_keys() -> set:
    raw = os.environ.get("ASTERWHITE_API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


async def require_api_key(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    api_key: Optional[str] = Query(None),
) -> Optional[str]:
    valid = configured_keys()
    if not valid:
        return None  # open mode -- logged once at startup in app.py, not per-request

    presented = None
    if authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:].strip()
    elif x_api_key:
        presented = x_api_key
    elif api_key:
        presented = api_key

    if presented not in valid:
        logger.warning(f"rejected request with invalid/missing API key (presented={scrub(presented or '(none)')})")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                             detail="invalid or missing API key -- send Authorization: Bearer <key>, "
                                    "X-API-Key: <key>, or ?api_key=<key>")
    return presented
