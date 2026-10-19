"""
AsterWhite :: safety.safe_mode

A single, process-wide flag other modules can check before doing
something risky (executing a tool, writing to shared/external storage).
Deliberately simple -- a module-level bool guarded by a context manager,
not a distributed feature-flag service -- because the spec's ask here
("a safe mode that disables tool execution or external actions") is a
per-process kill switch, not a fleet-wide rollout mechanism.

ASTERWHITE_SAFE_MODE is read ONCE, at first import of this module --
changing the environment variable while the process is already running
has no effect (confirmed by this module's own tests). This is
intentional for a kill switch: incident response
(docs/INTENDED_USE_AND_INCIDENT_RESPONSE.md) sets the variable and
restarts the process, which is the correct/expected action for this kind
of control, not a limitation to work around. For a scoped, in-process
override (mainly useful in tests), use the `safe_mode()` context manager
below instead of the environment variable.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

_safe_mode_enabled = os.environ.get("ASTERWHITE_SAFE_MODE", "0") == "1"


def is_safe_mode() -> bool:
    return _safe_mode_enabled


def require_not_safe_mode(action: str) -> None:
    """Call this at the top of anything safe mode should block (tool
    execution, external writes). Raises rather than silently no-opping,
    so a blocked action is visible, not mysteriously skipped."""
    if _safe_mode_enabled:
        from ..utils.error_tracing import AsterWhiteError
        raise AsterWhiteError(f"blocked by safe mode: {action}. Set ASTERWHITE_SAFE_MODE=0 or unset it to allow.")


@contextmanager
def safe_mode(enabled: bool = True) -> Iterator[None]:
    """Scoped override, mainly for tests: `with safe_mode(): ...` forces
    safe mode on for the block, restoring the prior value on exit."""
    global _safe_mode_enabled
    previous = _safe_mode_enabled
    _safe_mode_enabled = enabled
    try:
        yield
    finally:
        _safe_mode_enabled = previous
