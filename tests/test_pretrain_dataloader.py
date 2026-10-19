"""AsterWhite :: tests.test_pretrain_dataloader -- train.py's cmd_pretrain /
_make_loader, run for real (not --dry-run) against every dataset type
data_pipeline.build_dataset can return, since PyTorch's DataLoader has
genuinely different (and easy to get wrong) constructor rules for a
map-style Dataset vs. an IterableDataset.

Regression coverage for a real bug: build_dataset automatically returns a
StreamingRecordDataset (an IterableDataset) for any corpus over its
mb_threshold (256MB default) -- the COMMON case for real training data,
not an edge case -- and _make_loader unconditionally passed shuffle=... to
DataLoader regardless of dataset type. PyTorch's DataLoader raises
ValueError if shuffle is passed AT ALL (even shuffle=False) for an
IterableDataset. This was never caught before because every existing test
corpus was well under the threshold, so only the map-style path (which
tolerates shuffle=) had ever actually run through cmd_pretrain.
"""
from __future__ import annotations

import argparse
import os

import pytest

import train as cli
from data_pipeline import build_dataset as _real_build_dataset


def _pretrain_args(data_dir: str, checkpoint_dir: str, **overrides) -> argparse.Namespace:
    base = dict(
        profile="smoke_cpu", device=None, allow_oversized_config=False,
        data=[data_dir], val_data=None, tokenizer_dir=None, dry_run=False,
        max_steps=1, resume=None, load_base=None, lora=False, lora_r=8, lora_alpha=16,
        lora_targets=["q_proj", "v_proj"], operator="test",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _write_corpus(tmp_path, n_repeats: int = 400) -> str:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "book.txt").write_text(
        "The quick brown fox jumps over the lazy dog and finds a hidden path through the forest. " * n_repeats
    )
    return str(corpus_dir)


def _patched_profile_checkpoint_dir(monkeypatch, checkpoint_dir: str):
    """cmd_pretrain resolves --profile smoke_cpu -> load_profile('smoke_cpu'),
    which points checkpoint_dir at the repo's own ./checkpoints/smoke_cpu --
    redirect it into tmp_path so tests never touch/pollute the real repo
    directory, matching the pattern used elsewhere for --profile-based CLI tests."""
    from asterwhite.profiles import load_profile as real_load_profile

    def patched(name):
        cfg = real_load_profile(name)
        cfg.training.checkpoint_dir = checkpoint_dir
        return cfg

    monkeypatch.setattr(cli, "load_profile", patched)


def test_pretrain_with_a_small_in_memory_dataset(tmp_path, monkeypatch):
    """Map-style path (InMemoryTextDataset) -- the one that was already
    exercised before; kept here as the baseline the streaming test below
    is contrasted against."""
    _patched_profile_checkpoint_dir(monkeypatch, str(tmp_path / "ckpt"))
    data_dir = _write_corpus(tmp_path, n_repeats=5)  # small -> well under the streaming threshold
    args = _pretrain_args(data_dir, str(tmp_path / "ckpt"))
    cli.cmd_pretrain(args)  # must not raise
    assert os.path.exists(os.path.join(str(tmp_path / "ckpt"), "latest.json"))


def test_pretrain_with_a_streaming_iterable_dataset_forced(tmp_path, monkeypatch):
    """The actual regression test: force build_dataset's streaming
    (IterableDataset) path -- exactly what any real corpus over 256MB
    gets automatically -- using a small, fast-to-write corpus (forcing the
    scale rather than writing an actual 256MB+ file), and confirm
    cmd_pretrain's real _make_loader handles it without the
    'expected unspecified shuffle option' crash."""
    _patched_profile_checkpoint_dir(monkeypatch, str(tmp_path / "ckpt"))

    def forced_streaming_build_dataset(paths, **kwargs):
        kwargs.pop("scale", None)
        return _real_build_dataset(paths, scale="streaming", **kwargs)

    monkeypatch.setattr(cli, "build_dataset", forced_streaming_build_dataset)

    data_dir = _write_corpus(tmp_path, n_repeats=400)
    args = _pretrain_args(data_dir, str(tmp_path / "ckpt"))
    cli.cmd_pretrain(args)  # must not raise ValueError from DataLoader
    assert os.path.exists(os.path.join(str(tmp_path / "ckpt"), "latest.json"))


def test_pretrain_streaming_path_with_val_data_too(tmp_path, monkeypatch):
    """Both loaders _make_loader builds (train AND val) go through the
    same function -- confirm the val_data path is equally covered, not
    just the train one."""
    _patched_profile_checkpoint_dir(monkeypatch, str(tmp_path / "ckpt"))

    def forced_streaming_build_dataset(paths, **kwargs):
        kwargs.pop("scale", None)
        return _real_build_dataset(paths, scale="streaming", **kwargs)

    monkeypatch.setattr(cli, "build_dataset", forced_streaming_build_dataset)

    train_dir = _write_corpus(tmp_path, n_repeats=400)
    val_dir = str(tmp_path / "val_corpus")
    os.makedirs(val_dir)
    with open(os.path.join(val_dir, "val.txt"), "w") as f:
        f.write("A held-out validation passage about rivers and mountains. " * 300)

    args = _pretrain_args(train_dir, str(tmp_path / "ckpt"), val_data=[val_dir])
    cli.cmd_pretrain(args)  # must not raise for either the train or the val loader
    assert os.path.exists(os.path.join(str(tmp_path / "ckpt"), "latest.json"))


def test_make_loader_logic_directly_for_every_dataset_type(tmp_path):
    """Lower-level, distributed-agnostic confirmation of the actual fixed
    rule (isolated from cmd_pretrain's full setup cost): a DataLoader
    built for a dataset with no __len__ (IterableDataset) must never be
    constructed with an explicit shuffle= kwarg or a sampler, while a
    map-style dataset still gets shuffle=True for training data exactly
    as before -- this is what the fix in train.py's _make_loader encodes,
    reproduced here directly against torch.utils.data.DataLoader to
    confirm the underlying PyTorch behavior these tests all depend on."""
    from torch.utils.data import DataLoader
    from data_pipeline import InMemoryTextDataset, StreamingRecordDataset

    data_dir = _write_corpus(tmp_path, n_repeats=50)
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    map_ds = InMemoryTextDataset(files, block_size=16)
    assert hasattr(map_ds, "__len__")
    DataLoader(map_ds, batch_size=2, shuffle=True)  # must not raise

    iterable_ds = StreamingRecordDataset(files, block_size=16)
    assert not hasattr(iterable_ds, "__len__")
    with pytest.raises(ValueError, match="shuffle"):
        DataLoader(iterable_ds, batch_size=2, shuffle=True)  # confirms WHY the fix is needed
    DataLoader(iterable_ds, batch_size=2)  # the fixed call shape -- must not raise
