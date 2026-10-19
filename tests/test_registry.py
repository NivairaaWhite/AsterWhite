import json
import os

import pytest
import torch

from asterwhite.registry.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION, capture_rng_state, flatten_tensors, load_checkpoint,
    restore_rng_state, save_checkpoint, unflatten_tensors,
)
from asterwhite.registry.manifest import ModelRelease, PromotionStage, RunRecord
from asterwhite.utils.error_tracing import AsterWhiteError


def test_interrupted_save_never_leaves_latest_pointer_broken(tmp_path, monkeypatch):
    """Simulates a crash mid-save (e.g. Ctrl+C, node failure) by making
    tensor serialization raise partway through, and confirms the 'latest'
    pointer is left completely untouched -- never pointing at a partial or
    corrupted checkpoint. save_checkpoint writes the tensor file, then the
    metadata file, and only writes latest.json/best.json AFTER both are
    already safely committed (see registry/checkpoint.py's save_checkpoint) --
    this is what that ordering guarantees, verified directly rather than
    just by reading the code."""
    import asterwhite.registry.checkpoint as ckpt_module

    directory = str(tmp_path / "ckpt")
    good_tensors = {"model.weight": torch.randn(4, 4)}

    # A first, GOOD checkpoint -- this is the one 'latest' must still point
    # to after the interrupted second save below.
    save_checkpoint(directory, "step_1", good_tensors, {"step": 1}, make_latest=True)
    with open(os.path.join(directory, "latest.json")) as f:
        pointer_before = json.load(f)

    def exploding_serialize(*args, **kwargs):
        raise RuntimeError("simulated crash mid-save (e.g. Ctrl+C, node failure)")

    monkeypatch.setattr(ckpt_module, "_serialize_tensors", exploding_serialize)
    with pytest.raises(RuntimeError):
        save_checkpoint(directory, "step_2", {"model.weight": torch.randn(4, 4)}, {"step": 2}, make_latest=True)

    with open(os.path.join(directory, "latest.json")) as f:
        pointer_after = json.load(f)
    assert pointer_after == pointer_before  # untouched -- still points at step_1, never step_2
    assert not os.path.exists(os.path.join(directory, "step_2.safetensors"))  # no partial file left behind either
    assert not any(f.startswith(".tmp-") for f in os.listdir(directory))  # nor an orphaned temp file

    # And loading 'latest' afterward still works, unaffected by the failed attempt.
    tensors, metadata = load_checkpoint(directory, which="latest")
    assert metadata["step"] == 1


def test_save_load_round_trip(tmp_path):
    tensors = {"w1": torch.randn(4, 4), "w2": torch.randn(2)}
    save_checkpoint(str(tmp_path), "ckpt", tensors, {"note": "hello"})
    loaded, meta = load_checkpoint(str(tmp_path), name="ckpt")
    assert torch.allclose(loaded["w1"], tensors["w1"])
    assert meta["note"] == "hello"
    assert meta["schema_version"] == CHECKPOINT_SCHEMA_VERSION


def test_latest_pointer_resolves_without_explicit_name(tmp_path):
    save_checkpoint(str(tmp_path), "step_1", {"w": torch.zeros(2)}, {}, make_latest=True)
    save_checkpoint(str(tmp_path), "step_2", {"w": torch.ones(2)}, {}, make_latest=True)
    loaded, meta = load_checkpoint(str(tmp_path))  # which="latest" by default
    assert torch.allclose(loaded["w"], torch.ones(2))


def test_checksum_mismatch_is_refused(tmp_path):
    save_checkpoint(str(tmp_path), "ckpt", {"w": torch.randn(3)}, {})
    with open(os.path.join(tmp_path, "ckpt.safetensors"), "r+b") as f:
        f.seek(f.tell(), os.SEEK_END)
        f.write(b"\x00corruption")
    with pytest.raises(AsterWhiteError, match="checksum"):
        load_checkpoint(str(tmp_path), name="ckpt")


def test_schema_version_mismatch_is_refused(tmp_path):
    save_checkpoint(str(tmp_path), "ckpt", {"w": torch.randn(2)}, {})
    meta_path = os.path.join(tmp_path, "ckpt.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["schema_version"] = CHECKPOINT_SCHEMA_VERSION + 99
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    with pytest.raises(AsterWhiteError, match="schema_version"):
        load_checkpoint(str(tmp_path), name="ckpt")


def test_missing_files_refused_not_partial_load(tmp_path):
    with pytest.raises(AsterWhiteError, match="missing"):
        load_checkpoint(str(tmp_path), name="nonexistent")


def test_tokenizer_hash_mismatch_refused(tmp_path):
    save_checkpoint(str(tmp_path), "ckpt", {"w": torch.randn(2)}, {"tokenizer_hash": "abc"})
    with pytest.raises(AsterWhiteError, match="tokenizer"):
        load_checkpoint(str(tmp_path), name="ckpt", expected_tokenizer_hash="different")
    load_checkpoint(str(tmp_path), name="ckpt", expected_tokenizer_hash="abc")  # must not raise


def test_tied_weights_deduplicated_and_restored(tmp_path):
    shared = torch.randn(5, 5)
    tensors = {"tok_emb.weight": shared, "lm_head.weight": shared}  # same storage -- would break safetensors raw
    handle = save_checkpoint(str(tmp_path), "ckpt", tensors, {})
    assert len(handle.metadata["tensor_aliases"]) == 1
    loaded, _ = load_checkpoint(str(tmp_path), name="ckpt")
    assert set(loaded.keys()) == {"tok_emb.weight", "lm_head.weight"}
    assert torch.allclose(loaded["tok_emb.weight"], loaded["lm_head.weight"])


def test_flatten_unflatten_round_trip_nested_structure():
    obj = {"a": torch.randn(3), "b": [torch.randn(2), 5, "text"], "c": {"nested": torch.randn(1)}, "d": None}
    tensors, skeleton = flatten_tensors(obj)
    restored = unflatten_tensors(skeleton, tensors)
    assert torch.allclose(restored["a"], obj["a"])
    assert torch.allclose(restored["b"][0], obj["b"][0])
    assert restored["b"][1:] == [5, "text"]
    assert torch.allclose(restored["c"]["nested"], obj["c"]["nested"])
    assert restored["d"] is None


def test_rng_state_capture_restore_reproduces_next_draw():
    state = capture_rng_state()
    a = torch.randn(5)
    restore_rng_state(state)
    b = torch.randn(5)
    assert torch.allclose(a, b)


def test_run_record_and_model_release_lifecycle(tmp_path):
    run = RunRecord.start(run_id="run-1", config={"x": 1}, operator="test")
    run.record_metric("loss", 0.5)
    run.save(str(tmp_path / "run.json"))
    assert json.loads((tmp_path / "run.json").read_text())["metrics"]["loss"] == 0.5

    release = ModelRelease(model_name="AsterWhite", version="0.0.1", run_id="run-1",
                            weights_path="w.safetensors", tokenizer_path=None, config={},
                            model_card_path=None, data_card_paths=[], evaluation_report={},
                            safety_report={}, license="Apache-2.0", known_limitations=["untrained"],
                            rollback_target=None)
    h = release.compute_release_hash()
    assert len(h) == 64
    assert release.sign(signing_key="secret") is not None
    assert ModelRelease.verify(h, release.signature_hmac_sha256, "secret") is True
    assert ModelRelease.verify(h, release.signature_hmac_sha256, "wrong") is False

    release.promote(PromotionStage.CANDIDATE)
    with pytest.raises(ValueError):
        release.promote(PromotionStage.PRODUCTION)  # candidate -> production is not an allowed direct jump
    release.save(str(tmp_path / "release.json"))
    reloaded = ModelRelease.load(str(tmp_path / "release.json"))
    assert reloaded.stage == PromotionStage.CANDIDATE
