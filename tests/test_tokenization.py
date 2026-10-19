import os

import pytest
import torch

from asterwhite.tokenization import AsterWhiteTokenizer, CONTROL_TOKENS, remap_embeddings_for_new_tokenizer
from asterwhite.utils.error_tracing import AsterWhiteError


def _train_tiny_tokenizer(tmp_path, vocab_size=300):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "The quick brown fox jumps over the lazy dog. " * 50 +
        "A curious fox explored the forest and found glowing flowers. " * 50
    )
    return AsterWhiteTokenizer.train([str(corpus)], vocab_size=vocab_size)


def test_train_and_round_trip(tmp_path):
    tok = _train_tiny_tokenizer(tmp_path)
    samples = ["The fox ran.", "Hello, world! 123", "A curious fox explored the forest."]
    assert tok.round_trip_ok(samples)


def test_train_from_a_relative_path_after_changing_directory(tmp_path, monkeypatch):
    """Regression test for a real bug reported on Windows: the Rust
    tokenizers library's own file-opening did not reliably resolve a
    relative path against the current process's working directory the
    same way Python's os.path.exists just had, raising 'os error 3' for a
    file that genuinely existed. AsterWhiteTokenizer.train now resolves
    every corpus path to absolute before handing it to the library --
    this exercises that exact scenario (a relative path, from a specific
    cwd) rather than only ever using tmp_path's already-absolute paths
    like every other test here does."""
    corpus_dir = tmp_path / "corpus_dir"
    corpus_dir.mkdir()
    (corpus_dir / "book.txt").write_text("The quick brown fox jumps over the lazy dog. " * 80)
    monkeypatch.chdir(tmp_path)
    tok = AsterWhiteTokenizer.train(["corpus_dir/book.txt"], vocab_size=300)  # relative to the new cwd
    assert tok.round_trip_ok(["The fox ran."])


def test_train_uses_train_from_iterator_not_the_files_based_rust_path(tmp_path, monkeypatch):
    """The redesigned train() reads every corpus file in PYTHON (via
    data_pipeline.iter_raw_text_for_tokenizer_training) and hands the
    Rust tokenizers library plain strings through train_from_iterator --
    never a file path via the files=-based Tokenizer.train. This is a
    stronger fix than just absolute-pathing files= (see save/load's
    equivalent test below, which still need it): it removes the Rust
    library's own file-opening from the training call path entirely, so
    whatever OS-specific relative-path quirks that call had (the original
    reported bug) cannot recur here regardless of the corpus paths given."""
    from tokenizers import Tokenizer

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("The quick brown fox jumps over the lazy dog. " * 80)
    calls = {"train_from_iterator": 0, "train": 0}

    original_train_from_iterator = Tokenizer.train_from_iterator
    original_train = Tokenizer.train

    def capturing_train_from_iterator(self, iterator, trainer=None, length=None):
        calls["train_from_iterator"] += 1
        return original_train_from_iterator(self, iterator, trainer=trainer, length=length)

    def capturing_train(self, files, trainer=None):
        calls["train"] += 1
        return original_train(self, files=files, trainer=trainer)

    monkeypatch.setattr(Tokenizer, "train_from_iterator", capturing_train_from_iterator)
    monkeypatch.setattr(Tokenizer, "train", capturing_train)
    AsterWhiteTokenizer.train([str(corpus)], vocab_size=300)

    assert calls["train_from_iterator"] == 1
    assert calls["train"] == 0


def test_save_and_load_pass_only_absolute_paths_to_the_underlying_library(tmp_path, monkeypatch):
    from tokenizers import Tokenizer

    tok = _train_tiny_tokenizer(tmp_path)
    seen = {}

    original_save = Tokenizer.save
    original_from_file = Tokenizer.from_file

    def capturing_save(self, path, pretty=True):
        seen["save_path"] = path
        return original_save(self, path, pretty=pretty)

    def capturing_from_file(path):
        seen["load_path"] = path
        return original_from_file(path)

    monkeypatch.setattr(Tokenizer, "save", capturing_save)
    monkeypatch.setattr(Tokenizer, "from_file", capturing_from_file)

    out_dir = str(tmp_path / "saved_tokenizer")
    tok.save(out_dir)
    assert "save_path" in seen and os.path.isabs(seen["save_path"])

    AsterWhiteTokenizer.load(out_dir)
    assert "load_path" in seen and os.path.isabs(seen["load_path"])


def test_train_from_a_directory_directly(tmp_path):
    """Regression test for a real bug reported on Windows: pointing --data
    at a directory reached AsterWhiteTokenizer.train with the DIRECTORY
    path itself (never expanded into the files inside it), which the
    underlying library's file-opening rejected with 'Access is denied'
    (Windows) trying to open a directory as if it were a regular file."""
    (tmp_path / "book0.txt").write_text("The quick brown fox jumps over the lazy dog. " * 80)
    (tmp_path / "book1.txt").write_text("A curious fox explored the forest at dawn. " * 80)
    tok = AsterWhiteTokenizer.train([str(tmp_path)], vocab_size=300)
    assert tok.round_trip_ok(["The fox ran."])


def test_train_from_a_nested_directory_and_mixed_formats(tmp_path):
    """The other half of the same fix: directories, nested subdirectories,
    AND multiple formats (.txt + .jsonl here) mixed together in one call."""
    (tmp_path / "book0.txt").write_text("The quick brown fox jumps over the lazy dog. " * 60)
    sub = tmp_path / "more_books"
    sub.mkdir()
    (sub / "book1.txt").write_text("A nested story about mountains and rivers. " * 60)
    import json
    with open(tmp_path / "records.jsonl", "w", encoding="utf-8") as f:
        for i in range(15):
            f.write(json.dumps({"text": f"A jsonl record number {i} about traveling merchants."}) + "\n")

    tok = AsterWhiteTokenizer.train([str(tmp_path)], vocab_size=300)
    assert tok.round_trip_ok(["The fox ran.", "A jsonl record."])


def test_train_respects_text_column_for_jsonl(tmp_path):
    """Handing the tokenizer trainer raw JSON syntax (braces, quotes, key
    names) instead of just the prose field would pollute its learned
    vocabulary with that structural noise -- verify only the selected
    field's content is what gets trained on, not the record's other
    field(s) or its JSON punctuation."""
    import json
    with open(tmp_path / "records.jsonl", "w", encoding="utf-8") as f:
        for i in range(30):
            f.write(json.dumps({
                "label": "should-not-appear-in-vocab-verbatim",
                "prose": f"A short prose passage number {i} about wandering through quiet forests. " * 3,
            }) + "\n")
    tok = AsterWhiteTokenizer.train([str(tmp_path)], vocab_size=300, text_column="prose")
    assert tok.round_trip_ok(["wandering through quiet forests"])


def test_train_on_empty_corpus_raises_clear_error(tmp_path):
    (tmp_path / "empty.txt").write_text("")
    with pytest.raises(AsterWhiteError, match="usable text"):
        AsterWhiteTokenizer.train([str(tmp_path / "empty.txt")], vocab_size=300)


def test_cmd_tokenizer_train_cli_handles_a_directory_end_to_end(tmp_path):
    """CLI-level regression test for the SECOND bug found alongside the
    first: the post-training round-trip sanity check opened args.data[0]
    (the raw, possibly-a-directory CLI argument) directly with open(),
    which would have crashed immediately after a successful training run
    fixed the first bug. Runs the real cmd_tokenizer_train, not just the
    library call, so this covers the actual CLI path a user hits."""
    import argparse
    import train as cli

    (tmp_path / "book0.txt").write_text("The quick brown fox jumps over the lazy dog. " * 80)
    (tmp_path / "book1.txt").write_text("A curious fox explored the forest at dawn. " * 80)
    out_dir = str(tmp_path / "tokenizer_out")

    args = argparse.Namespace(data=[str(tmp_path)], vocab_size=300, min_frequency=2,
                               text_column=None, out=out_dir)
    cli.cmd_tokenizer_train(args)
    assert os.path.exists(os.path.join(out_dir, "tokenizer.json"))



    tok = _train_tiny_tokenizer(tmp_path)
    for special in CONTROL_TOKENS:
        assert tok.token_id(special) is not None
    ids = tok.encode("<|character|> Ash <|/character|> said hello.")
    assert tok.token_id("<|character|>") in ids
    assert tok.token_id("<|/character|>") in ids


def test_save_load_hash_verified(tmp_path):
    tok = _train_tiny_tokenizer(tmp_path)
    out_dir = tmp_path / "tok_out"
    tok.save(str(out_dir))
    reloaded = AsterWhiteTokenizer.load(str(out_dir))
    assert reloaded.compute_hash() == tok.compute_hash()
    assert reloaded.encode("The fox ran.") == tok.encode("The fox ran.")


def test_hash_differs_for_differently_trained_tokenizers(tmp_path):
    tok_a = _train_tiny_tokenizer(tmp_path, vocab_size=300)
    tok_b = _train_tiny_tokenizer(tmp_path, vocab_size=280)
    assert tok_a.compute_hash() != tok_b.compute_hash()


def test_fertility_returns_positive_ratio(tmp_path):
    tok = _train_tiny_tokenizer(tmp_path)
    report = tok.fertility({"english": ["The quick brown fox jumps."]})
    assert report.as_dict()["english"] > 0


def test_embedding_remap_copies_shared_tokens_by_string(tmp_path):
    tok_a = _train_tiny_tokenizer(tmp_path, vocab_size=300)
    tok_b = _train_tiny_tokenizer(tmp_path, vocab_size=300)  # same corpus -> same vocab strings, in practice same ids too
    old_weight = torch.randn(tok_a.vocab_size, 16)
    new_weight = remap_embeddings_for_new_tokenizer(tok_a, tok_b, old_weight)
    assert new_weight.shape == (tok_b.vocab_size, 16)
    # every control token must have been copied (same string exists in both vocabs)
    bos_a, bos_b = tok_a.bos_id, tok_b.bos_id
    assert torch.allclose(new_weight[bos_b], old_weight[bos_a])
