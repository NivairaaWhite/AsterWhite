"""
AsterWhite :: serving.schemas

Wire formats for every convention the serving layer speaks. "Compatible
with everything" in practice means the two conventions the actual wrapper
ecosystem has standardized on (OpenAI's REST shape -- LangChain,
LlamaIndex, most chat UIs, continue.dev, etc. -- and Ollama's REST shape,
which a large parallel set of local-model UIs speak natively) plus one
deliberately minimal generic shape for anything that speaks neither.
Pydantic validates every request before it reaches InferenceEngine, per
spec: "Validate every request against a schema before it reaches the
model."
"""
from __future__ import annotations

import time
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# OpenAI-compatible
# --------------------------------------------------------------------------- #

class OpenAIMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAIChatCompletionRequest(BaseModel):
    model: str = "asterwhite"
    messages: List[OpenAIMessage] = Field(..., min_length=1)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = Field(None, alias="max_tokens")
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    frequency_penalty: Optional[float] = None  # mapped to repetition_penalty (see routes.py's note)
    user: Optional[str] = None

    @field_validator("n")
    @classmethod
    def _single_completion_only(cls, v):
        if v != 1:
            raise ValueError("n > 1 is not supported yet -- request one completion at a time")
        return v


class OpenAICompletionRequest(BaseModel):
    model: str = "asterwhite"
    prompt: Union[str, List[str]]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    frequency_penalty: Optional[float] = None


def openai_chat_chunk(request_id: str, model: str, delta: str, finish_reason: Optional[str] = None) -> dict:
    return {
        "id": request_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": ({"content": delta} if delta else {}),
                     "finish_reason": finish_reason}],
    }


def openai_chat_completion(request_id: str, model: str, text: str, prompt_tokens: int,
                            completion_tokens: int, finish_reason: str) -> dict:
    return {
        "id": request_id, "object": "chat.completion", "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                   "total_tokens": prompt_tokens + completion_tokens},
    }


def openai_completion(request_id: str, model: str, text: str, prompt_tokens: int,
                       completion_tokens: int, finish_reason: str) -> dict:
    return {
        "id": request_id, "object": "text_completion", "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason, "logprobs": None}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                   "total_tokens": prompt_tokens + completion_tokens},
    }


def openai_completion_chunk(request_id: str, model: str, text: str, finish_reason: Optional[str] = None) -> dict:
    """The legacy /v1/completions streaming shape differs from chat's:
    choices[0].text (a plain string delta), not choices[0].delta.content."""
    return {
        "id": request_id, "object": "text_completion", "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason, "logprobs": None}],
    }


# --------------------------------------------------------------------------- #
# Ollama-compatible
# --------------------------------------------------------------------------- #

class OllamaGenerateRequest(BaseModel):
    model: str = "asterwhite"
    prompt: str
    stream: bool = True
    options: Optional[dict] = None  # {"temperature":.., "top_p":.., "top_k":.., "repeat_penalty":..}


class OllamaChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OllamaChatRequest(BaseModel):
    model: str = "asterwhite"
    messages: List[OllamaChatMessage] = Field(..., min_length=1)
    stream: bool = True
    options: Optional[dict] = None


def ollama_options_to_sampling(options: Optional[dict]) -> dict:
    options = options or {}
    out = {}
    if "temperature" in options:
        out["temperature"] = options["temperature"]
    if "top_p" in options:
        out["top_p"] = options["top_p"]
    if "top_k" in options:
        out["top_k"] = options["top_k"]
    if "repeat_penalty" in options:
        out["repetition_penalty"] = options["repeat_penalty"]
    if "num_predict" in options:
        out["max_new_tokens"] = options["num_predict"]
    return out


def ollama_generate_chunk(model: str, response: str, done: bool, **extra) -> dict:
    payload = {"model": model, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "response": response, "done": done}
    if done:
        payload.update(extra)
    return payload


def ollama_chat_chunk(model: str, content: str, done: bool, **extra) -> dict:
    payload = {"model": model, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
               "message": {"role": "assistant", "content": content}, "done": done}
    if done:
        payload.update(extra)
    return payload


# --------------------------------------------------------------------------- #
# Generic (anything that speaks neither convention)
# --------------------------------------------------------------------------- #

class GenericGenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    stream: bool = False


class GenericGenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    latency_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_version: str
    backend: str
    device: str
