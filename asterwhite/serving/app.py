"""
AsterWhite :: serving.app

The FastAPI app `train.py serve` launches via uvicorn. Configuration
comes entirely from environment variables (set by train.py's cmd_serve,
or directly by anyone running `uvicorn asterwhite.serving.app:app`
themselves) -- never a hardcoded path, per spec 4.2's "never depend on
editing source" principle extended to the serving entry point:

  ASTERWHITE_SERVE_CHECKPOINT_DIR   trained weights directory. UNSET -> the bundled pretrained model
                                    (pretrained/asterwhite-small) if present; set to the empty string ->
                                    untrained random weights, loudly logged (tests / pipeline checks).
  ASTERWHITE_SERVE_TOKENIZER_DIR    tokenizer directory (empty -> <checkpoint>/tokenizer if the checkpoint is
                                    a bundle, else the byte-level fallback)
  ASTERWHITE_SERVE_PROFILE          configs/<name>.yaml for serving settings (sampling defaults, concurrency,
                                    timeouts). Empty -> the checkpoint's own stored settings. The
                                    ARCHITECTURE always comes from the checkpoint's stored config.
  ASTERWHITE_SERVE_DEVICE           auto/cpu/cuda/rocm/mps/xpu (default auto)
  ASTERWHITE_API_KEYS               comma-separated valid keys (empty -> open/no-auth, logged loudly)
  ASTERWHITE_JSON_LOGS              "1" -> structured JSON logs instead of human-readable
  ASTERWHITE_CORS_ORIGINS           comma-separated allowed origins (default "*")
"""
from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..config import AsterWhiteConfig
from ..inference.engine import InferenceEngine
from ..observability.logging_config import configure_json_logging
from ..observability.metrics import registry as metrics_registry
from ..profiles import load_profile
from ..registry.pretrained import adopt_checkpoint_config, default_bundle_dir
from ..safety.pii_scrubber import scrub
from ..utils.error_tracing import AsterWhiteError, get_logger
from .auth import configured_keys
from .quotas import ConcurrencyLimiter
from .routes import router

logger = get_logger("asterwhite.serving.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("ASTERWHITE_JSON_LOGS") == "1":
        configure_json_logging("asterwhite.serving")

    profile = os.environ.get("ASTERWHITE_SERVE_PROFILE") or None
    checkpoint_env = os.environ.get("ASTERWHITE_SERVE_CHECKPOINT_DIR")
    if checkpoint_env is None:
        checkpoint_dir = default_bundle_dir()
        if checkpoint_dir is None:
            raise AsterWhiteError(
                "no model to serve: ASTERWHITE_SERVE_CHECKPOINT_DIR is not set and no bundled pretrained model "
                "was found (looked for pretrained/asterwhite-small). Pass --checkpoint-dir <dir> to `train.py "
                "serve`, or set ASTERWHITE_SERVE_CHECKPOINT_DIR to the empty string to serve untrained "
                "weights for a pipeline check.")
        logger.info(f"no checkpoint configured -- serving the bundled pretrained model at {checkpoint_dir}")
    else:
        checkpoint_dir = checkpoint_env or None
    tokenizer_dir = os.environ.get("ASTERWHITE_SERVE_TOKENIZER_DIR") or None
    device = os.environ.get("ASTERWHITE_SERVE_DEVICE") or "auto"

    if profile:
        cfg: AsterWhiteConfig = load_profile(profile)
    elif checkpoint_dir:
        # No explicit serving profile: use the checkpoint's own stored
        # settings (for the bundled model, defaults tuned for it).
        cfg = adopt_checkpoint_config(None, checkpoint_dir, keep_profile_settings=False)
    else:
        cfg = load_profile("production_inference")

    engine = InferenceEngine(cfg, checkpoint_dir=checkpoint_dir, tokenizer_dir=tokenizer_dir,
                              device_override=None if device == "auto" else device)
    app.state.engine = engine
    app.state.limiter = ConcurrencyLimiter(cfg.inference.max_concurrent_requests)
    app.state.cfg = cfg

    if configured_keys():
        logger.info(f"serving started: auth ENABLED ({len(configured_keys())} key(s) configured)")
    else:
        logger.warning("serving started: auth DISABLED (ASTERWHITE_API_KEYS is unset/empty) -- "
                        "every request is accepted with no key. Fine for local dev, not for an "
                        "internet-facing deployment.")
    logger.info(f"model_version={engine.model_version} backend={engine.device_info.backend} "
                f"quantized={cfg.inference.quantize} max_concurrent={cfg.inference.max_concurrent_requests}")
    yield
    # No explicit teardown needed: the model and any open threads are
    # process-owned and exit with the process.


app = FastAPI(
    title="AsterWhite Inference API",
    description="OpenAI-compatible, Ollama-compatible, and generic REST endpoints over an AsterWhite checkpoint.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("ASTERWHITE_CORS_ORIGINS", "*").split(",")],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_and_timeout(request: Request, call_next):
    """Attaches a request id (used by routes.py's structured log lines)
    and enforces inference.request_timeout_seconds end-to-end -- a
    request that runs long gets a clean 504, not a hung connection."""
    request.state.request_id = uuid.uuid4().hex[:16]
    timeout = getattr(getattr(request.app.state, "cfg", None), "inference", None)
    timeout_seconds = timeout.request_timeout_seconds if timeout else 60.0
    t0 = time.time()
    try:
        import asyncio
        return await asyncio.wait_for(call_next(request), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(f"request {request.state.request_id} timed out after {time.time() - t0:.1f}s "
                        f"(limit {timeout_seconds}s)")
        return JSONResponse(status_code=504, content={"error": "request exceeded the server's timeout"})


@app.exception_handler(AsterWhiteError)
async def asterwhite_error_handler(request: Request, exc: AsterWhiteError):
    logger.error(f"AsterWhiteError on {request.url.path}: {scrub(str(exc))}")
    return JSONResponse(status_code=400, content={"error": scrub(str(exc))})


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    """Never let a raw traceback reach the client -- log the real
    exception server-side (scrubbed), return a generic message to the
    caller. spec 17: don't leak internals through error responses."""
    logger.error(f"unhandled exception on {request.url.path}: {type(exc).__name__}: {scrub(str(exc))}")
    return JSONResponse(status_code=500, content={"error": "internal server error"})


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(metrics_registry), media_type=CONTENT_TYPE_LATEST)


app.include_router(router)
