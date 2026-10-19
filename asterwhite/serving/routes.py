"""
AsterWhite :: serving.routes

Three wire conventions, each with the format its real ecosystem actually
uses -- not a single lookalike shape stretched to cover all of them:
  - OpenAI:  Server-Sent Events, "data: {json}\\n\\n" chunks, terminated by
             a literal "data: [DONE]\\n\\n" line.
  - Ollama:  newline-delimited JSON objects (NDJSON) -- no "data:" prefix,
             no [DONE] sentinel; the last object carries "done": true.
  - generic: a single JSON object, or (if stream=true) a plain
             text/event-stream of raw text pieces for anything that
             just wants bytes as they're produced.

Every route: (1) checks the API key, (2) acquires a concurrency slot,
(3) runs InferenceEngine (blocking/torch) in a worker thread via
asyncio.to_thread so the event loop is never blocked by a generation
call, (4) scrubs prompts/completions before they reach the logger,
(5) records Prometheus metrics and a structured log line.
"""
from __future__ import annotations

import asyncio
import queue as queue_module
import threading
import time
import uuid
from typing import AsyncIterator, Callable, Iterator, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import MODEL_NAME
from ..observability.logging_config import log_request_outcome
from ..observability.metrics import REQUEST_COUNT, REQUEST_LATENCY, TOKENS_GENERATED
from ..safety.pii_scrubber import scrub
from ..safety.policy_filter import PolicyFilter
from ..utils.error_tracing import get_logger
from .auth import require_api_key
from .schemas import (
    GenericGenerateRequest, GenericGenerateResponse, HealthResponse,
    OllamaChatRequest, OllamaGenerateRequest, OpenAIChatCompletionRequest, OpenAICompletionRequest,
    ollama_chat_chunk, ollama_generate_chunk, ollama_options_to_sampling,
    openai_chat_chunk, openai_chat_completion, openai_completion, openai_completion_chunk,
)

logger = get_logger("asterwhite.serving")
policy_filter = PolicyFilter()
router = APIRouter()


def _engine(request: Request):
    return request.app.state.engine


def _limiter(request: Request):
    return request.app.state.limiter


async def _bridge_sync_stream(sync_gen_factory: Callable[[], Iterator[str]]) -> AsyncIterator[str]:
    """Runs a synchronous generator (InferenceEngine.generate_stream, which
    calls blocking torch ops) in a background thread, bridging its output
    to an async generator via a thread-safe queue -- so the event loop
    stays responsive to OTHER requests while one request streams tokens."""
    q: "queue_module.Queue" = queue_module.Queue()
    _SENTINEL = object()
    loop = asyncio.get_event_loop()

    def _produce() -> None:
        try:
            for piece in sync_gen_factory():
                q.put(piece)
        except Exception as exc:  # noqa: BLE001 -- forwarded to the consumer, not swallowed
            q.put(exc)
        finally:
            q.put(_SENTINEL)

    threading.Thread(target=_produce, daemon=True).start()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is _SENTINEL:
            return
        if isinstance(item, Exception):
            raise item
        yield item


def _record(route: str, request_id: str, model_version: str, t0: float, outcome: str,
            completion_tokens: int = 0) -> None:
    dt = time.time() - t0
    REQUEST_COUNT.labels(route=route, outcome=outcome).inc()
    REQUEST_LATENCY.labels(route=route).observe(dt)
    TOKENS_GENERATED.inc(completion_tokens)
    log_request_outcome(logger, request_id=request_id, route=route, model_version=model_version,
                         latency_ms=dt * 1000, outcome=outcome)


def _flag_if_policy_violation(route: str, text: str) -> None:
    result = policy_filter.check(text)
    if result.flagged:
        logger.warning(f"{route}: policy_filter flagged categories={result.categories} "
                        f"(text={scrub(text)[:200]}...)")


# --------------------------------------------------------------------------- #
# generic
# --------------------------------------------------------------------------- #

@router.get("/health", response_model=HealthResponse)
@router.get("/healthz", response_model=HealthResponse)
@router.get("/ready", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    engine = _engine(request)
    return HealthResponse(status="ok", model_version=engine.model_version,
                           backend=engine.device_info.backend, device=str(engine.device_info.device))


@router.post("/generate")
async def generate_generic(body: GenericGenerateRequest, request: Request,
                            _api_key: Optional[str] = Depends(require_api_key)):
    engine, limiter = _engine(request), _limiter(request)
    request_id = f"gen-{uuid.uuid4().hex[:12]}"
    t0 = time.time()
    _flag_if_policy_violation("/generate", body.prompt)

    kwargs = dict(max_new_tokens=body.max_tokens, temperature=body.temperature, top_p=body.top_p,
                  top_k=body.top_k, repetition_penalty=body.repetition_penalty)
    async with limiter.slot():
        if body.stream:
            async def event_source():
                async for piece in _bridge_sync_stream(lambda: engine.generate_stream(body.prompt, **kwargs)):
                    yield piece
                _record("/generate", request_id, engine.model_version, t0, "success")
            return StreamingResponse(event_source(), media_type="text/plain")

        result = await asyncio.to_thread(engine.generate, body.prompt, **kwargs)
        _flag_if_policy_violation("/generate", result.text)
        _record("/generate", request_id, engine.model_version, t0, "success", result.completion_tokens)
        return GenericGenerateResponse(text=result.text, prompt_tokens=result.prompt_tokens,
                                        completion_tokens=result.completion_tokens,
                                        finish_reason=result.finish_reason, latency_ms=result.latency_ms,
                                        model_version=engine.model_version)


# --------------------------------------------------------------------------- #
# OpenAI-compatible
# --------------------------------------------------------------------------- #

@router.get("/v1/models")
async def list_models(request: Request):
    engine = _engine(request)
    return {"object": "list", "data": [{"id": MODEL_NAME.lower(), "object": "model", "owned_by": "you",
                                          "version": engine.model_version}]}


def _messages_to_prompt(messages) -> str:
    """No chat template is bundled (spec: don't fabricate one the model
    wasn't actually trained against) -- this is a plain, clearly-labeled
    role transcript. Train your own chat-formatting convention into the
    tokenizer's control tokens (asterwhite.tokenization's CHARACTER_TOKEN
    etc.) for anything more structured."""
    return "\n".join(f"{m.role}: {m.content}" for m in messages) + "\nassistant:"


@router.post("/v1/chat/completions")
async def chat_completions(body: OpenAIChatCompletionRequest, request: Request,
                            _api_key: Optional[str] = Depends(require_api_key)):
    engine, limiter = _engine(request), _limiter(request)
    request_id = f"chatcmpl-{uuid.uuid4().hex[:20]}"
    t0 = time.time()
    prompt = _messages_to_prompt(body.messages)
    _flag_if_policy_violation("/v1/chat/completions", prompt)
    kwargs = dict(max_new_tokens=body.max_tokens, temperature=body.temperature, top_p=body.top_p,
                  repetition_penalty=(1.0 + body.frequency_penalty) if body.frequency_penalty else None)

    async with limiter.slot():
        if body.stream:
            async def event_source():
                async for piece in _bridge_sync_stream(lambda: engine.generate_stream(prompt, **kwargs)):
                    yield f"data: {_json(openai_chat_chunk(request_id, body.model, piece))}\n\n"
                yield f"data: {_json(openai_chat_chunk(request_id, body.model, '', finish_reason='stop'))}\n\n"
                yield "data: [DONE]\n\n"
                _record("/v1/chat/completions", request_id, engine.model_version, t0, "success")
            return StreamingResponse(event_source(), media_type="text/event-stream")

        result = await asyncio.to_thread(engine.generate, prompt, **kwargs)
        _flag_if_policy_violation("/v1/chat/completions", result.text)
        _record("/v1/chat/completions", request_id, engine.model_version, t0, "success", result.completion_tokens)
        return openai_chat_completion(request_id, body.model, result.text, result.prompt_tokens,
                                       result.completion_tokens, result.finish_reason)


@router.post("/v1/completions")
async def completions(body: OpenAICompletionRequest, request: Request,
                       _api_key: Optional[str] = Depends(require_api_key)):
    engine, limiter = _engine(request), _limiter(request)
    request_id = f"cmpl-{uuid.uuid4().hex[:20]}"
    t0 = time.time()
    prompt = body.prompt if isinstance(body.prompt, str) else body.prompt[0]
    _flag_if_policy_violation("/v1/completions", prompt)
    kwargs = dict(max_new_tokens=body.max_tokens, temperature=body.temperature, top_p=body.top_p,
                  repetition_penalty=(1.0 + body.frequency_penalty) if body.frequency_penalty else None)

    async with limiter.slot():
        if body.stream:
            async def event_source():
                async for piece in _bridge_sync_stream(lambda: engine.generate_stream(prompt, **kwargs)):
                    yield f"data: {_json(openai_completion_chunk(request_id, body.model, piece))}\n\n"
                yield f"data: {_json(openai_completion_chunk(request_id, body.model, '', finish_reason='stop'))}\n\n"
                yield "data: [DONE]\n\n"
                _record("/v1/completions", request_id, engine.model_version, t0, "success")
            return StreamingResponse(event_source(), media_type="text/event-stream")

        result = await asyncio.to_thread(engine.generate, prompt, **kwargs)
        _flag_if_policy_violation("/v1/completions", result.text)
        _record("/v1/completions", request_id, engine.model_version, t0, "success", result.completion_tokens)
        return openai_completion(request_id, body.model, result.text, result.prompt_tokens,
                                  result.completion_tokens, result.finish_reason)


# --------------------------------------------------------------------------- #
# Ollama-compatible
# --------------------------------------------------------------------------- #

@router.get("/api/tags")
async def ollama_tags(request: Request):
    engine = _engine(request)
    return {"models": [{"name": MODEL_NAME.lower(), "model": MODEL_NAME.lower(),
                          "modified_at": "", "size": 0, "digest": engine.model_version}]}


@router.post("/api/generate")
async def ollama_generate(body: OllamaGenerateRequest, request: Request,
                           _api_key: Optional[str] = Depends(require_api_key)):
    engine, limiter = _engine(request), _limiter(request)
    request_id = f"ollama-{uuid.uuid4().hex[:12]}"
    t0 = time.time()
    _flag_if_policy_violation("/api/generate", body.prompt)
    kwargs = ollama_options_to_sampling(body.options)

    async with limiter.slot():
        if body.stream:
            async def ndjson_source():
                async for piece in _bridge_sync_stream(lambda: engine.generate_stream(body.prompt, **kwargs)):
                    yield _json(ollama_generate_chunk(body.model, piece, done=False)) + "\n"
                yield _json(ollama_generate_chunk(body.model, "", done=True,
                                                    total_duration=int((time.time() - t0) * 1e9))) + "\n"
                _record("/api/generate", request_id, engine.model_version, t0, "success")
            return StreamingResponse(ndjson_source(), media_type="application/x-ndjson")

        result = await asyncio.to_thread(engine.generate, body.prompt, **kwargs)
        _flag_if_policy_violation("/api/generate", result.text)
        _record("/api/generate", request_id, engine.model_version, t0, "success", result.completion_tokens)
        return JSONResponse(ollama_generate_chunk(body.model, result.text, done=True,
                                                    total_duration=int(result.latency_ms * 1e6)))


@router.post("/api/chat")
async def ollama_chat(body: OllamaChatRequest, request: Request,
                       _api_key: Optional[str] = Depends(require_api_key)):
    engine, limiter = _engine(request), _limiter(request)
    request_id = f"ollama-{uuid.uuid4().hex[:12]}"
    t0 = time.time()
    prompt = "\n".join(f"{m.role}: {m.content}" for m in body.messages) + "\nassistant:"
    _flag_if_policy_violation("/api/chat", prompt)
    kwargs = ollama_options_to_sampling(body.options)

    async with limiter.slot():
        if body.stream:
            async def ndjson_source():
                async for piece in _bridge_sync_stream(lambda: engine.generate_stream(prompt, **kwargs)):
                    yield _json(ollama_chat_chunk(body.model, piece, done=False)) + "\n"
                yield _json(ollama_chat_chunk(body.model, "", done=True,
                                                total_duration=int((time.time() - t0) * 1e9))) + "\n"
                _record("/api/chat", request_id, engine.model_version, t0, "success")
            return StreamingResponse(ndjson_source(), media_type="application/x-ndjson")

        result = await asyncio.to_thread(engine.generate, prompt, **kwargs)
        _flag_if_policy_violation("/api/chat", result.text)
        _record("/api/chat", request_id, engine.model_version, t0, "success", result.completion_tokens)
        return JSONResponse(ollama_chat_chunk(body.model, result.text, done=True,
                                                total_duration=int(result.latency_ms * 1e6)))


def _json(d: dict) -> str:
    import json
    return json.dumps(d)
