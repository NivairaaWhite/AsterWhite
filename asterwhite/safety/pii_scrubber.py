"""
AsterWhite :: safety.pii_scrubber

Regex-based redaction for logs/telemetry -- spec: "Logs and telemetry
MUST avoid storing raw personal data or secrets." This is a real,
deterministic, testable filter (see tests/test_safety.py); it is
pattern-matching, not a trained PII-detection model -- it will miss
context-dependent PII a human or ML classifier would catch, and that
tradeoff (recall vs. false positives on a fixed pattern set) is the right
default for automatic log redaction, where a missed edge case is bad but
a classifier confidently redacting the wrong thing (or being fooled
adversarially) is worse for a security-relevant default.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Pattern

_PATTERNS: Dict[str, Pattern] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "api_key": re.compile(r"\b(sk|pk|key|token|secret)[-_][A-Za-z0-9]{16,}\b", re.IGNORECASE),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "bearer_token": re.compile(r"\bBearer\s+[A-Za-z0-9._-]{10,}\b"),
}


def scrub(text: Optional[str]) -> Optional[str]:
    """Replaces every match with [REDACTED:<category>] -- category names
    are visible (useful for debugging what got redacted), the matched
    value itself never is. None passes through as None (a caller logging
    an optional field shouldn't need to guard every scrub() call)."""
    if text is None:
        return None
    out = text
    for category, pattern in _PATTERNS.items():
        out = pattern.sub(f"[REDACTED:{category}]", out)
    return out


def scrub_dict(d: dict, keys_to_scrub: tuple = ("prompt", "completion", "message", "content", "text")) -> dict:
    """Shallow helper for request/response logging: scrub only the
    free-text fields likely to contain user content, leaving structural
    fields (model name, request id, timestamps, ...) untouched and
    readable."""
    out = dict(d)
    for k in keys_to_scrub:
        if k in out and isinstance(out[k], str):
            out[k] = scrub(out[k])
    return out
