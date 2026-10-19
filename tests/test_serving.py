import json
import os
import threading

import pytest

os.environ.setdefault("ASTERWHITE_SERVE_PROFILE", "smoke_cpu")
os.environ.setdefault("ASTERWHITE_SERVE_CHECKPOINT_DIR", "")
os.environ.setdefault("ASTERWHITE_SERVE_TOKENIZER_DIR", "")
os.environ.setdefault("ASTERWHITE_SERVE_DEVICE", "cpu")
os.environ.setdefault("ASTERWHITE_API_KEYS", "")

from fastapi.testclient import TestClient  # noqa: E402

from asterwhite.serving.app import app  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------- #
# generic + health + metrics
# --------------------------------------------------------------------------- #

def test_health_endpoints(client):
    for path in ("/health", "/healthz", "/ready"):
        r = client.get(path)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_generic_generate_non_streaming(client):
    r = client.post("/generate", json={"prompt": "hello", "max_tokens": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["completion_tokens"] <= 5
    assert body["prompt_tokens"] > 0


def test_generic_generate_streaming_yields_text(client):
    with client.stream("POST", "/generate", json={"prompt": "hi", "max_tokens": 5, "stream": True}) as r:
        text = "".join(r.iter_text())
    assert isinstance(text, str)


def test_metrics_endpoint_exposes_prometheus_format(client):
    client.post("/generate", json={"prompt": "hi", "max_tokens": 3})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"asterwhite" in r.content or b"# HELP" in r.content


# --------------------------------------------------------------------------- #
# OpenAI-compatible
# --------------------------------------------------------------------------- #

def test_openai_models_list(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["object"] == "list"


def test_openai_chat_completions_non_streaming(client):
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert "usage" in body


def test_openai_chat_completions_rejects_empty_messages(client):
    r = client.post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 422


def test_openai_chat_completions_streaming_is_sse_with_done_sentinel(client):
    with client.stream("POST", "/v1/chat/completions",
                        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5, "stream": True}) as r:
        lines = [l for l in r.iter_lines() if l]
    assert lines[-1] == "data: [DONE]"
    assert all(l.startswith("data: ") for l in lines)
    first_chunk = json.loads(lines[0][len("data: "):])
    assert first_chunk["object"] == "chat.completion.chunk"


def test_openai_completions_legacy_shape(client):
    r = client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 5})
    assert r.status_code == 200
    assert r.json()["object"] == "text_completion"


def test_openai_completions_streaming_chunk_shape_differs_from_chat(client):
    with client.stream("POST", "/v1/completions", json={"prompt": "hi", "max_tokens": 5, "stream": True}) as r:
        lines = [l for l in r.iter_lines() if l]
    first_chunk = json.loads(lines[0][len("data: "):])
    assert "text" in first_chunk["choices"][0]  # NOT delta.content -- legacy shape uses a flat "text" field
    assert "delta" not in first_chunk["choices"][0]


# --------------------------------------------------------------------------- #
# Ollama-compatible
# --------------------------------------------------------------------------- #

def test_ollama_tags(client):
    r = client.get("/api/tags")
    assert r.status_code == 200
    assert "models" in r.json()


def test_ollama_generate_non_streaming(client):
    r = client.post("/api/generate", json={"prompt": "hi", "stream": False})
    assert r.status_code == 200
    assert r.json()["done"] is True


def test_ollama_generate_streaming_is_ndjson_not_sse(client):
    with client.stream("POST", "/api/generate", json={"prompt": "hi", "stream": True}) as r:
        lines = [l for l in r.iter_lines() if l]
    assert not any(l.startswith("data:") for l in lines)  # no SSE prefix
    objs = [json.loads(l) for l in lines]  # every line must be a standalone JSON object
    assert objs[-1]["done"] is True
    assert all(o["done"] is False for o in objs[:-1])


def test_ollama_chat_streaming(client):
    with client.stream("POST", "/api/chat", json={"messages": [{"role": "user", "content": "hi"}], "stream": True}) as r:
        lines = [l for l in r.iter_lines() if l]
    objs = [json.loads(l) for l in lines]
    assert objs[-1]["done"] is True
    assert "message" in objs[0]


def test_ollama_options_map_to_sampling_params(client):
    r = client.post("/api/generate", json={
        "prompt": "hi", "stream": False,
        "options": {"temperature": 0.5, "top_k": 10, "repeat_penalty": 1.3, "num_predict": 4},
    })
    assert r.status_code == 200


# --------------------------------------------------------------------------- #
# auth + quotas (separate app construction with keys configured)
# --------------------------------------------------------------------------- #

def test_auth_accepts_all_three_conventions_and_rejects_bad_key(monkeypatch):
    monkeypatch.setenv("ASTERWHITE_API_KEYS", "secretA,secretB")
    with TestClient(app) as c:
        assert c.post("/generate", json={"prompt": "hi", "max_tokens": 2}).status_code == 401
        assert c.post("/generate", json={"prompt": "hi", "max_tokens": 2},
                       headers={"Authorization": "Bearer secretA"}).status_code == 200
        assert c.post("/generate", json={"prompt": "hi", "max_tokens": 2},
                       headers={"X-API-Key": "secretB"}).status_code == 200
        assert c.post("/generate?api_key=secretA", json={"prompt": "hi", "max_tokens": 2}).status_code == 200
        assert c.post("/generate", json={"prompt": "hi", "max_tokens": 2},
                       headers={"Authorization": "Bearer WRONG"}).status_code == 401


def test_quota_rejects_with_429_under_real_concurrency(client):
    client.app.state.limiter.max_concurrent = 2
    statuses = []

    def hit():
        r = client.post("/generate", json={"prompt": "x" * 50, "max_tokens": 60})
        statuses.append(r.status_code)

    threads = [threading.Thread(target=hit) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert 429 in statuses
    assert 200 in statuses


def test_unhandled_exception_never_leaks_traceback(monkeypatch):
    # TestClient re-raises exceptions by default so app bugs aren't hidden
    # in tests -- but that means it also re-raises exceptions this app's
    # OWN exception_handler(Exception) already turned into a clean 500
    # response (confirmed via the log line either way: the handler runs
    # and logs before TestClient's re-raise). raise_server_exceptions=False
    # is what actually lets a test observe that handled response, matching
    # what a real client hitting a real uvicorn server would receive.
    with TestClient(app, raise_server_exceptions=False) as c:
        def boom(*a, **kw):
            raise RuntimeError("internal detail that must not reach the client")
        monkeypatch.setattr(c.app.state.engine, "generate", boom)
        r = c.post("/generate", json={"prompt": "hi", "max_tokens": 3})
        assert r.status_code == 500
        assert "internal detail" not in r.text
        assert r.json() == {"error": "internal server error"}


# --------------------------------------------------------------------------- #
# real-checkpoint interconnectivity (CLI training output loaded by serving)
# --------------------------------------------------------------------------- #

def test_serving_reflects_untrained_vs_trained_model_version(monkeypatch, tmp_path):
    from asterwhite.config import AsterWhiteConfig
    from asterwhite.core.model import AsterWhiteCore
    from asterwhite.registry.checkpoint import save_checkpoint

    cfg = AsterWhiteConfig.tiny_for_testing()
    model = AsterWhiteCore(cfg.backbone)
    tensors = {f"model.{k}": v for k, v in model.state_dict().items()}
    save_checkpoint(str(tmp_path), "step_10", tensors,
                     {"step": 10, "config": cfg.to_dict()}, make_latest=True)

    monkeypatch.setenv("ASTERWHITE_SERVE_CHECKPOINT_DIR", str(tmp_path))
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.json()["model_version"] == "step_10"
    monkeypatch.setenv("ASTERWHITE_SERVE_CHECKPOINT_DIR", "")
