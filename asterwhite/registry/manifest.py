"""
AsterWhite :: registry.manifest

Model registry records: what a training run was, and what a releasable
model IS, as plain structured data (JSON-serializable) rather than tribal
knowledge. `PromotionStage` is the release lifecycle
(Development -> Candidate -> Staging -> Canary -> Production -> Deprecated
-> Revoked) the spec asks for.

Signing: a real production deployment signs releases with an org-managed
key (HSM/KMS) verified by a separate trust root -- that key-management
infrastructure is outside what a repository can contain. What this module
DOES do for real: compute a stable SHA-256 "release hash" over every
artifact (weights, tokenizer, config, data manifest), and -- IF the
operator supplies a private key via `AsterWhite_SIGNING_KEY` -- sign that
hash with HMAC-SHA256, so the machinery is genuinely wired up and testable
even though no key is bundled or generated here.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import torch


class PromotionStage(str, Enum):
    DEVELOPMENT = "development"
    CANDIDATE = "candidate"
    STAGING = "staging"
    CANARY = "canary"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    REVOKED = "revoked"


_ALLOWED_TRANSITIONS = {
    PromotionStage.DEVELOPMENT: {PromotionStage.CANDIDATE, PromotionStage.REVOKED},
    PromotionStage.CANDIDATE: {PromotionStage.STAGING, PromotionStage.DEVELOPMENT, PromotionStage.REVOKED},
    PromotionStage.STAGING: {PromotionStage.CANARY, PromotionStage.CANDIDATE, PromotionStage.REVOKED},
    PromotionStage.CANARY: {PromotionStage.PRODUCTION, PromotionStage.STAGING, PromotionStage.REVOKED},
    PromotionStage.PRODUCTION: {PromotionStage.DEPRECATED, PromotionStage.CANARY, PromotionStage.REVOKED},
    PromotionStage.DEPRECATED: {PromotionStage.REVOKED},
    PromotionStage.REVOKED: set(),
}


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown (not a git checkout or git unavailable)"


@dataclass
class RunRecord:
    """What the spec's section 15 requires every training/alignment run to record."""
    run_id: str
    parent_run_id: Optional[str]
    code_git_commit: str
    config: Dict[str, Any]
    dataset_versions: List[str]
    tokenizer_version: Optional[str]
    metrics: Dict[str, float]
    artifacts: List[str]
    hardware_environment: Dict[str, Any]
    operator: str
    started_at_utc: str
    updated_at_utc: str

    @classmethod
    def start(cls, run_id: str, config: Dict[str, Any], operator: str = "unspecified",
              parent_run_id: Optional[str] = None, dataset_versions: Optional[List[str]] = None,
              tokenizer_version: Optional[str] = None, hardware_environment: Optional[Dict[str, Any]] = None) -> "RunRecord":
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return cls(run_id=run_id, parent_run_id=parent_run_id, code_git_commit=_git_commit(),
                   config=config, dataset_versions=dataset_versions or [], tokenizer_version=tokenizer_version,
                   metrics={}, artifacts=[], hardware_environment=hardware_environment or {},
                   operator=operator, started_at_utc=now, updated_at_utc=now)

    def record_metric(self, name: str, value: float) -> None:
        self.metrics[name] = value
        self.updated_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def record_artifact(self, path: str) -> None:
        self.artifacts.append(path)
        self.updated_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


@dataclass
class ModelRelease:
    """What the spec's section 15 requires every releasable model to include."""
    model_name: str
    version: str
    run_id: str
    weights_path: str
    tokenizer_path: Optional[str]
    config: Dict[str, Any]
    model_card_path: Optional[str]
    data_card_paths: List[str]
    evaluation_report: Dict[str, Any]
    safety_report: Dict[str, Any]
    license: str
    known_limitations: List[str]
    rollback_target: Optional[str]
    stage: PromotionStage = PromotionStage.DEVELOPMENT
    release_hash_sha256: str = ""
    signature_hmac_sha256: Optional[str] = None
    compatibility: Dict[str, str] = field(default_factory=lambda: {
        "torch_version": torch.__version__, "python_version": platform.python_version(),
    })

    def compute_release_hash(self) -> str:
        h = hashlib.sha256()
        payload = json.dumps({
            "model_name": self.model_name, "version": self.version, "run_id": self.run_id,
            "weights_path": self.weights_path, "tokenizer_path": self.tokenizer_path,
            "config": self.config, "data_card_paths": sorted(self.data_card_paths),
        }, sort_keys=True).encode("utf-8")
        h.update(payload)
        self.release_hash_sha256 = h.hexdigest()
        return self.release_hash_sha256

    def sign(self, signing_key: Optional[str] = None) -> Optional[str]:
        """Signs compute_release_hash()'s output with HMAC-SHA256 if a key
        is available (explicit arg, or AsterWhite_SIGNING_KEY env var).
        Returns None (and logs nothing scary) if no key is configured --
        signing is opt-in, not silently skipped-but-claimed."""
        key = signing_key or os.environ.get("ASTERWHITE_SIGNING_KEY")
        if not key:
            return None
        if not self.release_hash_sha256:
            self.compute_release_hash()
        sig = hmac.new(key.encode("utf-8"), self.release_hash_sha256.encode("utf-8"), hashlib.sha256).hexdigest()
        self.signature_hmac_sha256 = sig
        return sig

    @staticmethod
    def verify(release_hash: str, signature: str, signing_key: str) -> bool:
        expected = hmac.new(signing_key.encode("utf-8"), release_hash.encode("utf-8"), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)

    def promote(self, target: PromotionStage) -> None:
        allowed = _ALLOWED_TRANSITIONS[self.stage]
        if target not in allowed:
            raise ValueError(f"cannot promote release from {self.stage.value} to {target.value}; "
                              f"allowed next stages are {[s.value for s in allowed]}")
        self.stage = target

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        d = asdict(self)
        d["stage"] = self.stage.value
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ModelRelease":
        with open(path) as f:
            d = json.load(f)
        d["stage"] = PromotionStage(d["stage"])
        return cls(**d)
