"""
AsterWhite :: observability.logging_config

utils.error_tracing.get_logger's human-readable format is what the CLI
uses (and stays that way -- it's the right choice for a terminal). This
module adds an OPT-IN structured JSON formatter for production log
aggregation (spec: "Emit structured logs with a request id, model
version, latency and outcome"), used by asterwhite.serving when
ASTERWHITE_JSON_LOGS=1 is set.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        }
        for extra_key in ("request_id", "model_version", "latency_ms", "outcome", "route"):
            if hasattr(record, extra_key):
                payload[extra_key] = getattr(record, extra_key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_json_logging(logger_name: Optional[str] = None) -> logging.Logger:
    """Replaces the given logger's handlers with a single JSON-formatted
    StreamHandler. Pass None for the root logger (affects every
    not-yet-configured logger); pass a specific name to scope it (e.g. to
    "asterwhite.serving" only, leaving CLI subcommands human-readable even
    within the same process)."""
    logger = logging.getLogger(logger_name)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def log_request_outcome(logger: logging.Logger, request_id: str, route: str, model_version: str,
                         latency_ms: float, outcome: str, level: int = logging.INFO) -> None:
    """One structured line per request -- request id, route, model
    version, latency, outcome -- exactly the fields spec section 18
    lists. Works whether or not JSON formatting is active (the extra
    fields are attached either way; JsonFormatter surfaces them as JSON
    keys, the plain formatter just won't show them, which is why serving
    should call configure_json_logging first)."""
    logger.log(level, f"{route} {outcome} in {latency_ms:.1f}ms",
               extra={"request_id": request_id, "route": route, "model_version": model_version,
                      "latency_ms": latency_ms, "outcome": outcome})
