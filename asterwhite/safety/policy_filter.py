"""
AsterWhite :: safety.policy_filter

A basic, deterministic, keyword/regex content filter. This is NOT a
trained safety classifier and is not a substitute for red-teaming or a
real moderation model -- it exists as (a) a working default so
generated/served content passes through SOME check rather than none, and
(b) the heuristic label source for training.narrative_objectives.
safety_policy_loss's supervised probe. Treat a "clean" result from this
filter as "nothing obvious matched," never as "verified safe."

Extend it by adding terms to _CATEGORY_TERMS, or -- better, for a real
deployment -- replace PolicyFilter.check's body with a call to a trained
classifier while keeping the same PolicyCheckResult shape so callers
(asterwhite.serving) don't need to change.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Deliberately small and illustrative, not an exhaustive enumeration --
# this is a coarse first-pass filter, not a comprehensive taxonomy.
_CATEGORY_TERMS: Dict[str, List[str]] = {
    "violence": ["kill you", "slaughter", "massacre"],
    "self_harm": ["kill myself", "suicide method", "how to overdose"],
    "csam": ["child porn", "cp of", "underage sex"],
    "weapons_uplift": ["synthesize sarin", "build a bomb", "enrich uranium"],
    "hate_speech": ["subhuman race", "gas the "],
}

_COMPILED: List[Tuple[str, re.Pattern]] = [
    (category, re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE))
    for category, terms in _CATEGORY_TERMS.items()
]


@dataclass
class PolicyCheckResult:
    flagged: bool
    categories: List[str]


class PolicyFilter:
    def check(self, text: str) -> PolicyCheckResult:
        hits = [category for category, pattern in _COMPILED if pattern.search(text)]
        return PolicyCheckResult(flagged=bool(hits), categories=hits)
