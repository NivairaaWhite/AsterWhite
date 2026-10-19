"""AsterWhite :: tests.test_data_paths -- data_pipeline.resolve_data_paths,
which is what makes --data accept individual files, directories (searched
recursively), and glob patterns interchangeably, with a specific error
naming the exact bad entry rather than an obscure failure deep inside a
Dataset/DataLoader."""
from __future__ import annotations

import os

import pytest

from asterwhite.utils.error_tracing import AsterWhiteError
from data_pipeline import SUPPORTED_DATA_EXTENSIONS, build_dataset, resolve_data_paths


def _write(path: str, text: str = "hello world. " * 50) -> None:
    with open(path, "w") as f:
        f.write(text)


def test_resolves_directory_recursively_by_extension(tmp_path):
    (tmp_path / "sub").mkdir()
    _write(str(tmp_path / "book1.txt"))
    _write(str(tmp_path / "sub" / "book2.md"))
    (tmp_path / "ignore.png").write_bytes(b"\x89PNG")

    resolved = resolve_data_paths(str(tmp_path))
    names = sorted(os.path.basename(p) for p in resolved)
    assert names == ["book1.txt", "book2.md"]


def test_resolves_glob_pattern():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        _write(os.path.join(d, "a.txt"))
        _write(os.path.join(d, "b.txt"))
        _write(os.path.join(d, "c.jsonl"))
        resolved = resolve_data_paths(os.path.join(d, "*.txt"))
        assert len(resolved) == 2


def test_accepts_a_plain_individual_file(tmp_path):
    p = tmp_path / "corpus.txt"
    _write(str(p))
    resolved = resolve_data_paths(str(p))
    assert resolved == [os.path.normpath(str(p))]


def test_mixed_files_and_directories_and_globs_in_one_call(tmp_path):
    (tmp_path / "dir1").mkdir()
    _write(str(tmp_path / "dir1" / "x.txt"))
    _write(str(tmp_path / "solo.txt"))
    (tmp_path / "glob_dir").mkdir()
    _write(str(tmp_path / "glob_dir" / "g1.txt"))
    _write(str(tmp_path / "glob_dir" / "g2.txt"))

    resolved = resolve_data_paths([str(tmp_path / "dir1"), str(tmp_path / "solo.txt"),
                                    str(tmp_path / "glob_dir" / "*.txt")])
    assert len(resolved) == 4


def test_deduplicates_overlapping_entries(tmp_path):
    p = tmp_path / "corpus.txt"
    _write(str(p))
    resolved = resolve_data_paths([str(p), str(p)])
    assert len(resolved) == 1


def test_missing_path_raises_specific_actionable_error(tmp_path):
    missing = str(tmp_path / "does_not_exist.txt")
    with pytest.raises(AsterWhiteError) as exc_info:
        resolve_data_paths(missing)
    assert missing in str(exc_info.value)


def test_error_messages_never_escape_backslashes_in_paths():
    """Regression test for a real bug caught on Windows: an early version
    used repr() (f"{entry!r}") to quote paths in error messages, which
    escapes backslashes (C:\\Users\\... -> 'C:\\\\Users\\\\...' in the
    displayed text) -- so the exact path a Windows user typed or sees in
    Explorer never appeared verbatim in the error. Constructs a
    backslash-containing path string directly (rather than relying on
    tmp_path, which uses forward slashes on this test runner's own OS) so
    this is caught on any platform, not only by actually running on
    Windows."""
    fake_windows_path = r"C:\Users\example\data\does_not_exist.txt"
    with pytest.raises(AsterWhiteError) as exc_info:
        resolve_data_paths(fake_windows_path)
    message = str(exc_info.value)
    assert fake_windows_path in message
    assert "\\\\" not in message  # no doubled backslashes anywhere in the message


def test_empty_directory_raises_clear_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(AsterWhiteError):
        resolve_data_paths(str(empty))


def test_glob_matching_nothing_raises_clear_error(tmp_path):
    with pytest.raises(AsterWhiteError):
        resolve_data_paths(str(tmp_path / "*.doesnotexist"))


def test_empty_list_raises():
    with pytest.raises(AsterWhiteError):
        resolve_data_paths([])


def test_directory_with_only_unsupported_extensions_raises(tmp_path):
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "archive.zip").write_bytes(b"PK")
    with pytest.raises(AsterWhiteError):
        resolve_data_paths(str(tmp_path))


def test_in_memory_dataset_too_small_for_one_block_raises_clear_error(tmp_path):
    """Regression test: a corpus with fewer tokens than block_size+1 used
    to silently construct a zero-length dataset, which only failed later,
    deep inside PyTorch's DataLoader/sampler internals, with an opaque
    'num_samples should be a positive integer value, but got
    num_samples=0'. Now raises immediately with a specific, actionable
    message instead."""
    from data_pipeline import InMemoryTextDataset
    (tmp_path / "tiny.txt").write_text("short")
    with pytest.raises(AsterWhiteError, match="block_size"):
        InMemoryTextDataset([str(tmp_path / "tiny.txt")], block_size=512)


def test_in_memory_dataset_reproduces_the_original_downstream_crash_if_bypassed():
    """Confirms WHY the fix matters: constructing a zero-length map-style
    dataset directly (bypassing the new guard, as if it still existed) and
    handing it to a real DataLoader really does crash with exactly the
    cryptic error the fix now prevents users from ever seeing."""
    from torch.utils.data import DataLoader, Dataset

    class ZeroLengthDataset(Dataset):
        def __len__(self):
            return 0

        def __getitem__(self, idx):
            raise IndexError

    with pytest.raises(ValueError, match="num_samples"):
        DataLoader(ZeroLengthDataset(), batch_size=1, shuffle=True)


def test_build_dataset_too_small_for_one_block_raises_clear_error(tmp_path):
    from data_pipeline import build_dataset
    (tmp_path / "tiny.txt").write_text("short")
    with pytest.raises(AsterWhiteError, match="block_size"):
        build_dataset(str(tmp_path / "tiny.txt"), block_size=512)


def test_memmap_dataset_too_small_for_one_block_raises_clear_error(tmp_path):
    from data_pipeline import MemmapTokenDataset, build_tokenized_binary
    (tmp_path / "corpus.txt").write_text("short")
    bin_path = str(tmp_path / "corpus.bin")
    build_tokenized_binary(str(tmp_path / "corpus.txt"), bin_path)
    with pytest.raises(AsterWhiteError, match="block_size"):
        MemmapTokenDataset(bin_path, block_size=512)


def test_adequately_sized_corpus_still_works_normally(tmp_path):
    """The guard must not be overzealous -- a corpus that genuinely has
    enough tokens still works exactly as before."""
    from data_pipeline import InMemoryTextDataset
    (tmp_path / "book.txt").write_text("word " * 2000)
    ds = InMemoryTextDataset([str(tmp_path / "book.txt")], block_size=64)
    assert len(ds) > 0
    assert ds[0].shape == (65,)



    from data_pipeline import iter_raw_text_for_tokenizer_training
    _write(str(tmp_path / "a.txt"), "hello world")
    _write(str(tmp_path / "b.md"), "goodbye world")
    texts = list(iter_raw_text_for_tokenizer_training(str(tmp_path)))
    combined = "".join(texts)
    assert "hello world" in combined and "goodbye world" in combined


def test_iter_raw_text_extracts_jsonl_text_column_only(tmp_path):
    from data_pipeline import iter_raw_text_for_tokenizer_training
    import json
    with open(tmp_path / "records.jsonl", "w") as f:
        for i in range(5):
            f.write(json.dumps({"label": f"skip-me-{i}", "prose": f"keep me {i}"}) + "\n")
    texts = list(iter_raw_text_for_tokenizer_training(str(tmp_path / "records.jsonl"), text_column="prose"))
    assert len(texts) == 5
    assert all(t.startswith("keep me") for t in texts)
    assert not any("skip-me" in t for t in texts)


def test_iter_raw_text_jsonl_default_column_is_first_field(tmp_path):
    from data_pipeline import iter_raw_text_for_tokenizer_training
    import json
    with open(tmp_path / "records.jsonl", "w") as f:
        f.write(json.dumps({"first_field": "this one", "second_field": "not this one"}) + "\n")
    texts = list(iter_raw_text_for_tokenizer_training(str(tmp_path / "records.jsonl")))
    assert texts == ["this one"]


def test_iter_raw_text_skips_malformed_jsonl_lines_without_raising(tmp_path):
    from data_pipeline import iter_raw_text_for_tokenizer_training
    with open(tmp_path / "records.jsonl", "w") as f:
        f.write('{"text": "a good line"}\n')
        f.write("not valid json at all\n")
        f.write('{"text": "another good line"}\n')
    texts = list(iter_raw_text_for_tokenizer_training(str(tmp_path / "records.jsonl")))
    assert texts == ["a good line", "another good line"]


def test_iter_raw_text_mixed_directory_all_formats_together(tmp_path):
    from data_pipeline import iter_raw_text_for_tokenizer_training
    import json
    _write(str(tmp_path / "a.txt"), "plain text content")
    sub = tmp_path / "nested"
    sub.mkdir()
    _write(str(sub / "b.txt"), "nested plain text")
    with open(tmp_path / "c.jsonl", "w") as f:
        f.write(json.dumps({"text": "jsonl content"}) + "\n")
    texts = list(iter_raw_text_for_tokenizer_training(str(tmp_path)))
    combined = "".join(texts)
    assert "plain text content" in combined
    assert "nested plain text" in combined
    assert "jsonl content" in combined



    _write(str(tmp_path / "book1.txt"), "Chapter one. " * 200)
    _write(str(tmp_path / "book2.txt"), "Chapter two. " * 200)
    ds = build_dataset(str(tmp_path), block_size=32)
    assert len(ds) > 0


def test_supported_extensions_constant_matches_what_resolve_actually_accepts(tmp_path):
    for ext in SUPPORTED_DATA_EXTENSIONS:
        if ext in (".csv", ".parquet", ".pq"):
            continue  # need real tabular content, covered by data_pipeline's own format tests
        _write(str(tmp_path / f"file{ext}"))
    resolved = resolve_data_paths(str(tmp_path))
    found_exts = {os.path.splitext(p)[1] for p in resolved}
    assert found_exts == {e for e in SUPPORTED_DATA_EXTENSIONS if e not in (".csv", ".parquet", ".pq")}
