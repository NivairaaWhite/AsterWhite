"""AsterWhite :: tests.test_pretrained_bundle -- everything that makes a
pretrained model "just work": prompts without a trailing EOS, a checkpoint
directory that describes itself (config + tokenizer), `train.py export`,
defaulting to the bundled model, serving it with no flags, the --max-steps
resume semantics, and the conditioning prefix scale."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest
import torch

from asterwhite.config import AsterWhiteConfig
from asterwhite.core.model import AsterWhiteCore
from asterwhite.profiles import load_profile
from asterwhite.registry.pretrained import (
    DEFAULT_BUNDLE_NAME, PRETRAINED_DIR_ENV, adopt_checkpoint_config, bundled_tokenizer_dir, default_bundle_dir,
    effective_context_length, stored_config,
)
from asterwhite.tokenization import AsterWhiteTokenizer
from asterwhite.training import Trainer
from asterwhite.utils.error_tracing import AsterWhiteError

TRAIN_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train.py")


def _run(*args: str, env_extra=None, cwd=None) -> subprocess.CompletedProcess:
    env = {**os.environ, **(env_extra or {})}
    return subprocess.run([sys.executable, TRAIN_PY, *args], capture_output=True, text=True, timeout=240,
                          env=env, cwd=cwd)


def _tiny_tokenizer(tmp_path, vocab_size=300):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("The quick brown fox jumps over the lazy dog. " * 60 +
                      "A curious fox explored the forest and found glowing flowers. " * 60)
    tok = AsterWhiteTokenizer.train([str(corpus)], vocab_size=vocab_size)
    tok_dir = tmp_path / "tok"
    tok.save(str(tok_dir))
    return tok, str(tok_dir)


def _training_checkpoint(tmp_path, tok, d_model=48):
    """A real Trainer-written checkpoint (optimizer + RNG + EMA + stored config) whose
    architecture deliberately differs from every shipped profile."""
    cfg = AsterWhiteConfig.tiny_for_testing()
    cfg.backbone.vocab_size = tok.vocab_size
    cfg.backbone.d_model = d_model
    cfg.backbone.max_seq_len = 96
    cfg.training.ema_decay = 0.9
    cfg.inference.max_total_tokens = 4096           # deliberately beyond the model's context
    cfg.inference.default_max_new_tokens = 16
    model = AsterWhiteCore(cfg.backbone)
    trainer = Trainer(model, cfg.training, torch.device("cpu"), tokenizer_hash=tok.compute_hash())
    trainer.train_step([torch.randint(0, tok.vocab_size, (2, 24))])
    ckpt_dir = tmp_path / "ckpt"
    trainer.save_checkpoint(str(ckpt_dir), config_dict=cfg.to_dict())
    return cfg, str(ckpt_dir)


@pytest.fixture()
def exported_bundle(tmp_path):
    tok, tok_dir = _tiny_tokenizer(tmp_path)
    cfg, ckpt_dir = _training_checkpoint(tmp_path, tok)
    out = tmp_path / "bundle"
    r = _run("export", "--checkpoint-dir", ckpt_dir, "--tokenizer-dir", tok_dir, "--out", str(out))
    assert r.returncode == 0, r.stderr
    return str(out), cfg, tok


# --------------------------------------------------------------------------- #
# tokenizer: prompts must not end with EOS
# --------------------------------------------------------------------------- #

def test_encode_prompt_has_bos_but_no_eos(tmp_path):
    tok, _ = _tiny_tokenizer(tmp_path)
    prompt = tok.encode_prompt("The quick fox")
    assert prompt[0] == tok.bos_id and tok.eos_id not in prompt
    assert prompt[1:] == tok.encode("The quick fox", add_special_tokens=False)


def test_encode_default_still_wraps_a_whole_document(tmp_path):
    tok, _ = _tiny_tokenizer(tmp_path)
    doc = tok.encode("The quick fox")
    assert doc[0] == tok.bos_id and doc[-1] == tok.eos_id
    plain = tok.encode("The quick fox", add_special_tokens=False)
    assert tok.bos_id not in plain and tok.eos_id not in plain


def test_empty_prompt_is_just_bos(tmp_path):
    tok, _ = _tiny_tokenizer(tmp_path)
    assert tok.encode_prompt("") == [tok.bos_id]


# --------------------------------------------------------------------------- #
# bundle discovery
# --------------------------------------------------------------------------- #

def test_default_bundle_dir_respects_override_env(tmp_path, monkeypatch):
    monkeypatch.setenv(PRETRAINED_DIR_ENV, str(tmp_path / "nothing_here"))
    assert default_bundle_dir() is None
    bundle = tmp_path / "root" / DEFAULT_BUNDLE_NAME
    bundle.mkdir(parents=True)
    (bundle / "latest.json").write_text('{"name": "model"}')
    monkeypatch.setenv(PRETRAINED_DIR_ENV, str(tmp_path / "root"))
    assert default_bundle_dir() == str(bundle)


def test_bundled_tokenizer_dir(tmp_path):
    assert bundled_tokenizer_dir(None) is None
    assert bundled_tokenizer_dir(str(tmp_path)) is None
    tok, tok_dir = _tiny_tokenizer(tmp_path)
    holder = tmp_path / "holder"
    (holder / "tokenizer").mkdir(parents=True)
    tok.save(str(holder / "tokenizer"))
    assert bundled_tokenizer_dir(str(holder)) == str(holder / "tokenizer")


# --------------------------------------------------------------------------- #
# config comes from the checkpoint
# --------------------------------------------------------------------------- #

def test_stored_config_round_trips_the_training_config(tmp_path):
    tok, _ = _tiny_tokenizer(tmp_path)
    cfg, ckpt_dir = _training_checkpoint(tmp_path, tok)
    assert stored_config(ckpt_dir).backbone == cfg.backbone


def test_adopt_takes_architecture_from_checkpoint_and_keeps_profile_serving_settings(tmp_path):
    tok, _ = _tiny_tokenizer(tmp_path)
    cfg, ckpt_dir = _training_checkpoint(tmp_path, tok)
    profile = load_profile("smoke_cpu")                      # a different architecture on purpose
    assert profile.backbone != cfg.backbone
    merged = adopt_checkpoint_config(profile, ckpt_dir, keep_profile_settings=True)
    assert merged.backbone == cfg.backbone                   # architecture: the checkpoint's
    assert merged.inference.default_max_new_tokens == profile.inference.default_max_new_tokens  # serving: the profile's
    assert profile.backbone.d_model == 32                    # the caller's object was not mutated


def test_adopt_without_a_profile_uses_the_whole_stored_config_and_caps_context(tmp_path):
    tok, _ = _tiny_tokenizer(tmp_path)
    cfg, ckpt_dir = _training_checkpoint(tmp_path, tok)
    merged = adopt_checkpoint_config(None, ckpt_dir, keep_profile_settings=False)
    assert merged.inference.default_max_new_tokens == 16     # the checkpoint's own setting
    # stored max_total_tokens=4096 > the model's 96-token context -> capped
    assert merged.inference.max_total_tokens == effective_context_length(merged.backbone) == 96


def test_adopt_with_no_stored_config_and_no_fallback_is_a_clear_error(tmp_path):
    from asterwhite.registry.checkpoint import save_checkpoint
    save_checkpoint(str(tmp_path / "bare"), "model", {"model.w": torch.zeros(2)}, {"step": 1})
    with pytest.raises(AsterWhiteError, match="no stored config"):
        adopt_checkpoint_config(None, str(tmp_path / "bare"), keep_profile_settings=False)


def test_effective_context_length_accounts_for_rope_scaling():
    b = AsterWhiteConfig.tiny_for_testing().backbone
    assert effective_context_length(b) == b.max_seq_len
    b.rope_scaling.strategy, b.rope_scaling.factor = "yarn", 4.0
    assert effective_context_length(b) == b.max_seq_len * 4


# --------------------------------------------------------------------------- #
# export + generate/serve straight from a bundle
# --------------------------------------------------------------------------- #

def test_export_writes_a_self_contained_inference_only_bundle(exported_bundle):
    out, cfg, tok = exported_bundle
    names = set(os.listdir(out))
    assert {"latest.json", "model.json", "model.safetensors", "tokenizer"} <= names
    assert AsterWhiteTokenizer.load(os.path.join(out, "tokenizer")).compute_hash() == tok.compute_hash()
    # inference-only: no optimizer/RNG/EMA tensors rode along
    from safetensors import safe_open
    with safe_open(os.path.join(out, "model.safetensors"), "pt") as f:
        assert all(k.startswith("model.") for k in f.keys())
    assert stored_config(out).backbone == cfg.backbone


def test_export_refuses_a_checkpoint_from_a_different_tokenizer(tmp_path):
    tok, _ = _tiny_tokenizer(tmp_path)
    _, ckpt_dir = _training_checkpoint(tmp_path, tok)
    other_corpus = tmp_path / "other.txt"
    other_corpus.write_text("Completely different words appear here instead of those. " * 80)
    other = AsterWhiteTokenizer.train([str(other_corpus)], vocab_size=300)
    other.save(str(tmp_path / "other_tok"))
    r = _run("export", "--checkpoint-dir", ckpt_dir, "--tokenizer-dir", str(tmp_path / "other_tok"),
             "--out", str(tmp_path / "bad"))
    assert r.returncode != 0 and "tokenizer_hash" in r.stderr


def test_export_ema_requires_ema_weights(tmp_path):
    tok, tok_dir = _tiny_tokenizer(tmp_path)
    cfg = AsterWhiteConfig.tiny_for_testing()
    cfg.backbone.vocab_size = tok.vocab_size
    trainer = Trainer(AsterWhiteCore(cfg.backbone), cfg.training, torch.device("cpu"), tokenizer_hash=tok.compute_hash())
    trainer.save_checkpoint(str(tmp_path / "noema"), config_dict=cfg.to_dict())
    r = _run("export", "--checkpoint-dir", str(tmp_path / "noema"), "--tokenizer-dir", tok_dir,
             "--out", str(tmp_path / "o"), "--ema")
    assert r.returncode != 0 and "no EMA weights" in r.stderr


def test_ema_export_has_different_weights_and_does_not_claim_the_raw_val_loss(tmp_path):
    """The training checkpoint's val_loss was measured on the RAW weights; an EMA export is different
    weights, so the bundle must not present that number as its own."""
    from safetensors.torch import load_file
    tok, tok_dir = _tiny_tokenizer(tmp_path)
    cfg, ckpt_dir = _training_checkpoint(tmp_path, tok)
    # re-save with a recorded val_loss, as `pretrain --val-data` does
    model = AsterWhiteCore(cfg.backbone)
    trainer = Trainer(model, cfg.training, torch.device("cpu"), tokenizer_hash=tok.compute_hash())
    trainer.train_step([torch.randint(0, tok.vocab_size, (2, 24))])
    trainer.best_val_loss = 3.5
    ckpt2 = str(tmp_path / "ckpt_val")
    trainer.save_checkpoint(ckpt2, config_dict=cfg.to_dict(), extra_metadata={"val_loss": 3.5})
    raw_out, ema_out = str(tmp_path / "raw"), str(tmp_path / "ema")
    for out, extra in ((raw_out, []), (ema_out, ["--ema"])):
        r = _run("export", "--checkpoint-dir", ckpt2, "--tokenizer-dir", tok_dir, "--out", out, *extra)
        assert r.returncode == 0, r.stderr
    raw_meta = json.load(open(os.path.join(raw_out, "model.json")))
    ema_meta = json.load(open(os.path.join(ema_out, "model.json")))
    assert raw_meta["weights_source"] == "raw" and raw_meta["val_loss"] == 3.5
    assert ema_meta["weights_source"] == "ema"
    assert ema_meta["val_loss"] is None and ema_meta["best_val_loss"] is None
    assert ema_meta["raw_weights_val_loss"] == 3.5
    raw_w, ema_w = load_file(os.path.join(raw_out, "model.safetensors")), load_file(os.path.join(ema_out, "model.safetensors"))
    assert raw_w.keys() == ema_w.keys()
    assert any(not torch.equal(raw_w[k], ema_w[k]) for k in raw_w), "EMA export is identical to the raw weights"


def test_generate_from_a_bundle_needs_no_profile_and_no_tokenizer_flag(exported_bundle):
    out, _cfg, _tok = exported_bundle
    r = _run("generate", "--load", out, "--prompt", "The quick", "--max-new-tokens", "8", "--seed", "1")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() != ""


def test_generate_is_reproducible_with_a_seed(exported_bundle):
    out, _cfg, _tok = exported_bundle
    args = ("generate", "--load", out, "--prompt", "The quick", "--max-new-tokens", "12", "--seed", "7")
    assert _run(*args).stdout == _run(*args).stdout


def test_generate_without_load_uses_the_bundled_default(exported_bundle, tmp_path):
    out, _cfg, _tok = exported_bundle
    root = tmp_path / "root"
    root.mkdir()
    shutil.copytree(out, root / DEFAULT_BUNDLE_NAME)           # (a copy, not a symlink: Windows-safe)
    r = _run("generate", "--prompt", "The quick", "--max-new-tokens", "6", "--seed", "1",
             env_extra={PRETRAINED_DIR_ENV: str(root)})
    assert r.returncode == 0, r.stderr
    assert "bundled pretrained model" in r.stderr


def test_generate_without_load_and_without_a_bundle_still_refuses(tmp_path):
    r = _run("generate", "--prompt", "hi", env_extra={PRETRAINED_DIR_ENV: str(tmp_path / "none")})
    assert r.returncode != 0
    assert "refuses to construct a fresh, untrained model" in r.stderr


def test_conditioning_scale_zero_runs_a_plain_continuation(exported_bundle):
    out, _cfg, _tok = exported_bundle
    r = _run("generate", "--load", out, "--prompt", "The quick", "--max-new-tokens", "6", "--seed", "1",
             "--conditioning-scale", "0")
    assert r.returncode == 0, r.stderr


# --------------------------------------------------------------------------- #
# serving
# --------------------------------------------------------------------------- #

def test_inference_engine_takes_architecture_from_the_checkpoint_and_finds_its_tokenizer(exported_bundle):
    from asterwhite.inference.engine import InferenceEngine
    out, cfg, tok = exported_bundle
    engine = InferenceEngine(load_profile("smoke_cpu"), checkpoint_dir=out, device_override="cpu")
    assert engine.cfg.backbone == cfg.backbone
    assert engine.tokenizer is not None and engine.tokenizer.compute_hash() == tok.compute_hash()
    assert engine.cfg.inference.max_total_tokens <= 96        # capped at the model's context
    ids = engine._encode("The quick fox")
    assert ids[0] == tok.bos_id and ids[-1] != tok.eos_id     # a prompt, not a finished document
    assert engine.generate("The quick", max_new_tokens=6).completion_tokens <= 6


def test_serve_app_with_no_env_uses_the_bundle_and_its_own_settings(exported_bundle, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from asterwhite.serving.app import app
    out, cfg, _tok = exported_bundle
    root = tmp_path / "root"
    root.mkdir()
    shutil.copytree(out, root / DEFAULT_BUNDLE_NAME)
    monkeypatch.setenv(PRETRAINED_DIR_ENV, str(root))
    monkeypatch.delenv("ASTERWHITE_SERVE_CHECKPOINT_DIR", raising=False)
    monkeypatch.setenv("ASTERWHITE_SERVE_PROFILE", "")
    monkeypatch.setenv("ASTERWHITE_SERVE_TOKENIZER_DIR", "")
    monkeypatch.setenv("ASTERWHITE_SERVE_DEVICE", "cpu")
    with TestClient(app) as client:
        assert client.get("/health").json()["model_version"].startswith("step_")   # trained weights, not 'untrained'
        r = client.post("/generate", json={"prompt": "The quick", "max_tokens": 5})
        assert r.status_code == 200 and r.json()["completion_tokens"] <= 5


def test_serve_app_with_no_env_and_no_bundle_fails_with_a_clear_message(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from asterwhite.serving.app import app
    monkeypatch.setenv(PRETRAINED_DIR_ENV, str(tmp_path / "none"))
    monkeypatch.delenv("ASTERWHITE_SERVE_CHECKPOINT_DIR", raising=False)
    with pytest.raises(AsterWhiteError, match="no model to serve"):
        with TestClient(app):
            pass


# --------------------------------------------------------------------------- #
# pretrain --resume: --max-steps is the TOTAL target step
# --------------------------------------------------------------------------- #

def test_resume_counts_max_steps_as_the_total_not_additional(tmp_path):
    common = ("pretrain", "--profile", "smoke_cpu", "--dry-run")
    r1 = _run(*common, "--max-steps", "4", cwd=str(tmp_path))
    assert r1.returncode == 0, r1.stderr
    assert "(step 4," in r1.stdout
    ckpt = os.path.join(str(tmp_path), "checkpoints", "smoke_cpu")

    r2 = _run(*common, "--max-steps", "4", "--resume", ckpt, cwd=str(tmp_path))     # already there
    assert r2.returncode == 0, r2.stderr
    assert "nothing to do" in r2.stdout and "(step 4," in r2.stdout

    r3 = _run(*common, "--max-steps", "6", "--resume", ckpt, cwd=str(tmp_path))     # two more, not six more
    assert r3.returncode == 0, r3.stderr
    assert "(step 6," in r3.stdout


def test_finetune_from_a_bundle_takes_architecture_and_tokenizer_from_it(exported_bundle, tmp_path):
    """`pretrain --load-base <bundle>` with no --profile/--tokenizer-dir used to build the DEFAULT (huge)
    config and be refused; the bundle's weights only fit its own architecture."""
    out, cfg, tok = exported_bundle
    work = tmp_path / "ft"
    work.mkdir()
    corpus = tmp_path / "corpus.txt"                      # written by _tiny_tokenizer
    r = _run("pretrain", "--load-base", out, "--data", str(corpus), "--max-steps", "2", cwd=str(work))
    assert r.returncode == 0, r.stderr
    assert "(step 2," in r.stdout
    new_ckpt = str(work / "checkpoints")
    assert stored_config(new_ckpt).backbone == cfg.backbone
    assert os.path.isfile(os.path.join(new_ckpt, "tokenizer", "tokenizer.json"))   # tokenizer travelled along


# --------------------------------------------------------------------------- #
# conditioning prefix scale
# --------------------------------------------------------------------------- #

def _orchestrator(conditioning_scale=1.0):
    from generation_orchestrator import GenerationOrchestrator, PlotCalculator
    from universe_tracker import DynamicStateTracker, WorldState
    cfg = AsterWhiteConfig.tiny_for_testing()
    model = AsterWhiteCore(cfg.backbone).eval()
    plot = PlotCalculator(plan_dim=cfg.narrative.plan_dim, max_children=cfg.narrative.max_plan_beats).eval()
    tracker = DynamicStateTracker(world_state_dim=cfg.narrative.world_state_dim).eval()
    orch = GenerationOrchestrator(model, plot, WorldState("u0"), tracker, max_window=64,
                                  conditioning_scale=conditioning_scale)
    orch.begin(torch.randn(cfg.narrative.plan_dim), torch.randint(0, cfg.backbone.vocab_size, (1, 6)), branching=[3])
    return orch


def test_conditioning_prefix_is_rescaled_to_the_embedding_scale():
    orch = _orchestrator(conditioning_scale=1.0)
    emb_rms = orch.model.tok_emb.weight.detach().pow(2).mean().sqrt()
    prefix = orch._conditioning_prefix(None)
    assert prefix.shape == (1, 2, orch.model.cfg.d_model)
    beat_rms = prefix[0, 0].pow(2).mean().sqrt()
    assert torch.isclose(beat_rms, emb_rms, rtol=1e-3)         # NOT the planner's native ~1.0
    assert torch.count_nonzero(prefix[0, 1]) == 0              # empty world state stays exactly zero
    assert beat_rms < 0.5                                       # i.e. far below an unscaled LayerNorm'd vector


def test_conditioning_scale_multiplies_the_prefix_and_zero_disables_it():
    assert _orchestrator(0.0)._conditioning_prefix(None) is None
    half = _orchestrator(0.5)
    emb_rms = half.model.tok_emb.weight.detach().pow(2).mean().sqrt()
    assert torch.isclose(half._conditioning_prefix(None)[0, 0].pow(2).mean().sqrt(), 0.5 * emb_rms, rtol=1e-3)


def test_negative_conditioning_scale_is_rejected():
    with pytest.raises(ValueError, match="conditioning_scale"):
        _orchestrator(-1.0)


@pytest.mark.parametrize("scale", [0.0, 1.0])
def test_generate_next_runs_with_and_without_a_prefix(scale):
    orch = _orchestrator(scale)
    out = orch.generate_next(max_new_tokens=5)
    assert out.shape == (1, 5)
