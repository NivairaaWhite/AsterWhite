"""
AsterWhite :: data_pipeline

Dataset-scale-agnostic loading, from a few megabytes up to terabytes,
without ever requiring the whole corpus to fit in RAM:

  - MB:      InMemoryTextDataset      -- standard in-memory Dataset.
  - GB:      StreamingRecordDataset   -- IterableDataset streaming raw
             CSV/Parquet/JSONL/txt records chunk-by-chunk straight from disk.
  - GB/TB, pre-tokenized: MemmapTokenDataset -- numpy.memmap over a single
             flat binary token array; the OS pages in only the bytes
             actually touched, so a multi-terabyte token file costs the
             same RAM as a megabyte one. This is the standard technique
             behind most large-scale from-scratch LLM pretraining loaders.

Every record-parsing boundary is wrapped so a corrupted segment raises a
DataCorruptionError carrying exact file/offset context (see
asterwhite.utils.error_tracing) and is logged + skipped rather than
crashing an hours-long streaming run.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from asterwhite.utils.error_tracing import AsterWhiteError, DataCorruptionError, get_logger, trace_errors

logger = get_logger("asterwhite.data_pipeline")

EncodeFn = Callable[[str], Sequence[int]]

# Extensions recognized when a --data argument is a directory (searched
# recursively) rather than an individual file -- kept as one constant so
# resolve_data_paths, docs, and any future format-sniffing stay in sync.
SUPPORTED_DATA_EXTENSIONS = (".txt", ".md", ".jsonl", ".ndjson", ".csv", ".parquet", ".pq")


def resolve_data_paths(paths: Union[str, Sequence[str]]) -> List[str]:
    """Turns whatever a person reasonably passes as --data into a flat,
    validated, deterministic list of real file paths -- called internally
    by build_dataset (so this applies no matter what constructs a Dataset),
    and safe to call again yourself before that for an early, specific
    error rather than an obscure one from deep inside a Dataset/DataLoader.

    Accepts, in any combination:
      - an individual file path ("book1.txt")
      - a directory ("./my_corpus/") -- walked RECURSIVELY, keeping every
        file whose extension is in SUPPORTED_DATA_EXTENSIONS; this is what
        makes "point it at a folder of books" work with zero flags.
      - a glob pattern ("books/*.txt", "data/**/*.jsonl" with recursive
        globbing) -- expanded here rather than relying on the shell, since
        Windows cmd/PowerShell (unlike bash/zsh) do NOT expand wildcards
        before handing them to the program, so a pattern that "just works"
        on Linux/macOS would otherwise silently become one literal
        (nonexistent) filename on Windows.

    Raises AsterWhiteError immediately, naming the exact offending entry,
    if something doesn't exist, resolves to zero files, or (for an
    explicit non-glob, non-directory path) isn't a regular file --
    deliberately BEFORE any device/model/tokenizer setup runs, so a typo'd
    path fails in milliseconds with a clear message instead of after
    minutes of otherwise-successful startup."""
    raw = [paths] if isinstance(paths, str) else list(paths)
    if not raw:
        raise AsterWhiteError("no data paths given (empty --data list).")

    resolved: List[str] = []
    for entry in raw:
        is_pattern = any(ch in entry for ch in "*?[")
        if is_pattern:
            matches = sorted(glob.glob(entry, recursive=True))
            matches = [m for m in matches if os.path.isfile(m)]
            if not matches:
                raise AsterWhiteError(
                    f"--data pattern '{entry}' matched zero files. Check the pattern is correct for your "
                    f"shell (on Windows, quote it so this program expands it: --data \"books\\*.txt\")."
                )
            resolved.extend(matches)
            continue

        if not os.path.exists(entry):
            raise AsterWhiteError(
                f"--data path '{entry}' does not exist (checked relative to cwd='{os.getcwd()}'). "
                f"Pass an existing file, an existing directory, or a glob pattern."
            )
        if os.path.isdir(entry):
            found = []
            for root, _dirs, files in os.walk(entry):
                for fn in files:
                    if os.path.splitext(fn)[1].lower() in SUPPORTED_DATA_EXTENSIONS:
                        found.append(os.path.join(root, fn))
            if not found:
                raise AsterWhiteError(
                    f"--data directory '{entry}' contains no files with a supported extension "
                    f"({', '.join(SUPPORTED_DATA_EXTENSIONS)}), searched recursively."
                )
            resolved.extend(sorted(found))
            continue

        if not os.path.isfile(entry):
            raise AsterWhiteError(f"--data path '{entry}' exists but is neither a regular file nor a directory "
                                   f"(a socket, device file, etc.) -- not something this pipeline can read.")
        resolved.append(entry)

    deduped = sorted(dict.fromkeys(os.path.normpath(p) for p in resolved))
    logger.info(f"resolved {len(raw)} --data argument(s) -> {len(deduped)} file(s)")
    return deduped


def iter_raw_text_for_tokenizer_training(paths: Union[str, Sequence[str]],
                                          text_column: Optional[str] = None) -> Iterator[str]:
    """Yields plain text strings from ANY combination of files, directories
    (searched recursively), and glob patterns (via resolve_data_paths),
    across every SUPPORTED_DATA_EXTENSIONS format -- what
    tokenization.AsterWhiteTokenizer.train feeds to the tokenizers
    library's train_from_iterator so a fresh BPE tokenizer can be trained
    directly on --data ./my_corpus/ the same way pretraining can, without
    needing every file pre-converted to plain .txt first.

    This matters beyond just "directories work": for .jsonl/.ndjson/.csv/
    .parquet, only the actual prose (text_column, or the first field/
    column if unset -- the SAME convention StreamingRecordDataset uses,
    kept consistent on purpose) is yielded. Handing the tokenizer trainer
    raw JSON/CSV syntax instead (braces, quotes, column headers, commas)
    would train real vocabulary entries for that structural noise instead
    of the prose inside it -- a real, silent quality bug, not just a
    directory-handling one.

    Corrupt/unreadable individual records are logged and skipped (never
    raised) -- appropriate for tokenizer training, a best-effort pass over
    a potentially huge, messy corpus where one bad JSON line should not
    abort an hours-long run the way it should for pretraining's exact
    token-accounting (see StreamingRecordDataset's on_corruption='raise'
    default, which this intentionally does NOT share)."""
    for path in resolve_data_paths(paths):
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".csv":
                yield from _iter_raw_text_csv(path, text_column)
            elif ext in (".parquet", ".pq"):
                yield from _iter_raw_text_parquet(path, text_column)
            elif ext in (".jsonl", ".ndjson"):
                yield from _iter_raw_text_jsonl(path, text_column)
            else:
                yield from _iter_raw_text_plain(path)
        except AsterWhiteError:
            raise  # a missing optional dependency (pandas/pyarrow) is not something to silently skip past
        except Exception as exc:
            logger.warning(f"skipping unreadable file {path} while extracting tokenizer-training text: "
                            f"{type(exc).__name__}: {exc}")


def _iter_raw_text_plain(path: str) -> Iterator[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(1 << 16)  # 64KB at a time, matching StreamingRecordDataset._iter_txt
            if not chunk:
                break
            yield chunk


def _iter_raw_text_jsonl(path: str, text_column: Optional[str]) -> Iterator[str]:
    import json
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for offset, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                text = record.get(text_column) if text_column else next(iter(record.values()))
            except (json.JSONDecodeError, StopIteration, AttributeError) as exc:
                logger.warning(f"skipping malformed record in {path}:{offset}: {type(exc).__name__}: {exc}")
                continue
            if isinstance(text, str) and text:
                yield text


def _iter_raw_text_csv(path: str, text_column: Optional[str]) -> Iterator[str]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise AsterWhiteError(
            f"reading {path} as CSV requires pandas, which is not installed in this environment. "
            f"Run: pip install pandas>=2.0    (or convert this file to .txt/.jsonl, which need no extra "
            f"dependency)."
        ) from exc
    for chunk in pd.read_csv(path, chunksize=8192):
        col = text_column or chunk.columns[0]
        for text in chunk[col].tolist():
            if isinstance(text, str) and text:
                yield text


def _iter_raw_text_parquet(path: str, text_column: Optional[str]) -> Iterator[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise AsterWhiteError(
            f"reading {path} as Parquet requires pyarrow, which is not installed in this environment. "
            f"Run: pip install pyarrow>=14.0    (or convert this file to .txt/.jsonl, which need no extra "
            f"dependency)."
        ) from exc
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=8192):
        table = batch.to_pydict()
        col = text_column or list(table.keys())[0]
        for text in table[col]:
            if isinstance(text, str) and text:
                yield text


def byte_encode(text: str) -> List[int]:
    """Byte-level fallback 'tokenizer' -- every input is representable (any
    byte 0-255), so this module is runnable standalone with no external
    vocab file. Swap in a real subword tokenizer's .encode by passing your
    own EncodeFn to any function/class below."""
    return list(text.encode("utf-8", errors="replace"))


# --------------------------------------------------------------------------- #
# MB scale: in-memory
# --------------------------------------------------------------------------- #

class InMemoryTextDataset(Dataset):
    """Loads one or more small text files fully into memory and slices them
    into fixed-length token blocks. Appropriate up to roughly a few hundred
    MB -- beyond that, prefer StreamingRecordDataset or MemmapTokenDataset."""

    def __init__(self, paths: Union[str, Sequence[str]], block_size: int = 512,
                 encode_fn: EncodeFn = byte_encode):
        paths = resolve_data_paths(paths)
        self.block_size = block_size
        buf: List[int] = []
        for p in paths:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                buf.extend(encode_fn(f.read()))
        if len(buf) <= block_size:
            raise AsterWhiteError(
                f"corpus has only {len(buf)} tokens after encoding, but block_size={block_size} needs at "
                f"least {block_size + 1} to form even one training example -- add more data, point --data "
                f"at your full corpus rather than a small sample, or lower backbone.max_seq_len. (This would "
                f"otherwise fail later with an opaque PyTorch error from an empty DataLoader.)"
            )
        self.tokens = torch.tensor(buf, dtype=torch.long)

    def __len__(self) -> int:
        return max(0, (len(self.tokens) - 1) // self.block_size)

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.block_size
        return self.tokens[start: start + self.block_size + 1]


# --------------------------------------------------------------------------- #
# GB scale: streaming, chunked reads straight from disk
# --------------------------------------------------------------------------- #

@dataclass
class _FileShard:
    path: str
    kind: str  # "csv" | "parquet" | "jsonl" | "txt"


class StreamingRecordDataset(IterableDataset):
    """IterableDataset reading CSV / Parquet / JSONL / raw-text records
    chunk-by-chunk from disk, never materializing more than one chunk at a
    time. Correctly shards files across DataLoader workers (via
    get_worker_info) so multi-worker loading doesn't duplicate data.
    Corrupted records are logged and skipped by default, not fatal."""

    def __init__(self, paths: Sequence[str], text_column: Optional[str] = None,
                 encode_fn: EncodeFn = byte_encode, block_size: int = 512,
                 csv_chunksize: int = 10_000, on_corruption: str = "skip",
                 rank: Optional[int] = None, world_size: Optional[int] = None):
        assert on_corruption in ("skip", "raise")
        self.shards = [_FileShard(p, self._infer_kind(p)) for p in resolve_data_paths(paths)]
        self.text_column = text_column
        self.encode_fn = encode_fn
        self.block_size = block_size
        self.csv_chunksize = csv_chunksize
        self.on_corruption = on_corruption
        # Auto-detect from an already-initialized process group if the
        # caller didn't say -- but an explicit rank/world_size always wins,
        # so a caller under a non-torch.distributed launcher can still
        # shard correctly.
        if world_size is None:
            world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.rank = rank
        self.world_size = world_size

    @staticmethod
    def _infer_kind(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            return "csv"
        if ext in (".parquet", ".pq"):
            return "parquet"
        if ext in (".jsonl", ".ndjson"):
            return "jsonl"
        return "txt"

    def _shards_for_this_worker(self) -> List[_FileShard]:
        """Shards by (rank, DataLoader worker) JOINTLY: previously this
        only used get_worker_info(), so under torch.distributed every rank
        would independently iterate the FULL shard set -- N ranks each
        re-reading 100% of the data instead of a disjoint 1/N slice.
        global_id/global_num_shards fold both dimensions into one index."""
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1
        global_id = self.rank * num_workers + worker_id
        global_num_shards = self.world_size * num_workers
        return self.shards[global_id::global_num_shards]

    @trace_errors
    def _validate_and_encode(self, text: object, source: str, offset: int) -> torch.Tensor:
        # Missing/NaN/None cells (common in real CSV/Parquet/JSONL data) are
        # exactly the "corrupted segment" case this pipeline must survive --
        # checked on the raw value rather than trusting an upstream str()
        # conversion, since pandas NaN-handling has changed across versions.
        if not isinstance(text, str):
            raise DataCorruptionError(
                f"non-string/missing record in {source} at offset {offset} (got {type(text).__name__}: {text!r})"
            )
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise DataCorruptionError(f"invalid UTF-8 in {source} at offset {offset}: {exc}") from exc
        ids = self.encode_fn(text)
        if len(ids) == 0:
            raise DataCorruptionError(f"empty record after encoding in {source} at offset {offset}")
        return torch.tensor(ids, dtype=torch.long)

    def _handle_corruption(self, exc: DataCorruptionError) -> None:
        if self.on_corruption == "raise":
            raise exc
        logger.warning(exc.report())

    def _iter_csv(self, shard: _FileShard) -> Iterator[torch.Tensor]:
        try:
            import pandas as pd
        except ImportError as exc:
            raise AsterWhiteError(
                f"reading {shard.path} as CSV requires pandas, which is not installed in this "
                f"environment. Run: pip install pandas>=2.0    (or convert this file to .txt/.jsonl, "
                f"which need no extra dependency)."
            ) from exc
        offset = 0
        for chunk in pd.read_csv(shard.path, chunksize=self.csv_chunksize):
            col = self.text_column or chunk.columns[0]
            for text in chunk[col].tolist():
                try:
                    yield self._validate_and_encode(text, shard.path, offset)
                except DataCorruptionError as exc:
                    self._handle_corruption(exc)
                offset += 1

    def _iter_parquet(self, shard: _FileShard) -> Iterator[torch.Tensor]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise AsterWhiteError(
                f"reading {shard.path} as Parquet requires pyarrow, which is not installed in this "
                f"environment. Run: pip install pyarrow>=14.0    (or convert this file to .txt/.jsonl, "
                f"which need no extra dependency)."
            ) from exc
        pf = pq.ParquetFile(shard.path)
        offset = 0
        for batch in pf.iter_batches(batch_size=self.csv_chunksize):
            table = batch.to_pydict()
            col = self.text_column or list(table.keys())[0]
            for text in table[col]:
                try:
                    yield self._validate_and_encode(text, shard.path, offset)
                except DataCorruptionError as exc:
                    self._handle_corruption(exc)
                offset += 1

    def _iter_jsonl(self, shard: _FileShard) -> Iterator[torch.Tensor]:
        import json
        with open(shard.path, "r", encoding="utf-8", errors="replace") as f:
            for offset, line in enumerate(f):
                try:
                    record = json.loads(line)
                    text = record.get(self.text_column) if self.text_column else next(iter(record.values()))
                    yield self._validate_and_encode(text, shard.path, offset)
                except json.JSONDecodeError as exc:
                    self._handle_corruption(DataCorruptionError(f"malformed JSON in {shard.path}:{offset}: {exc}",
                                                                 cause=exc))
                except DataCorruptionError as exc:
                    self._handle_corruption(exc)

    def _iter_txt(self, shard: _FileShard) -> Iterator[torch.Tensor]:
        with open(shard.path, "r", encoding="utf-8", errors="replace") as f:
            offset = 0
            while True:
                chunk = f.read(1 << 16)  # 64KB at a time -- never the whole file
                if not chunk:
                    break
                try:
                    yield self._validate_and_encode(chunk, shard.path, offset)
                except DataCorruptionError as exc:
                    self._handle_corruption(exc)
                offset += len(chunk)

    def _iter_shard(self, shard: _FileShard) -> Iterator[torch.Tensor]:
        return {"csv": self._iter_csv, "parquet": self._iter_parquet,
                "jsonl": self._iter_jsonl, "txt": self._iter_txt}[shard.kind](shard)

    def __iter__(self) -> Iterator[torch.Tensor]:
        buf: List[int] = []
        for shard in self._shards_for_this_worker():
            for ids in self._iter_shard(shard):
                buf.extend(ids.tolist())
                while len(buf) >= self.block_size + 1:
                    yield torch.tensor(buf[: self.block_size + 1], dtype=torch.long)
                    buf = buf[self.block_size:]


# --------------------------------------------------------------------------- #
# GB/TB scale, pre-tokenized: memory-mapped flat binary
# --------------------------------------------------------------------------- #

class MemmapTokenDataset(Dataset):
    """Random-access dataset over a single flat binary file of token ids
    (uint16 or uint32), opened with numpy.memmap. The OS pages in only the
    bytes actually read per __getitem__ call, so this scales to a
    multi-terabyte token file on a machine with ordinary RAM."""

    def __init__(self, bin_path: str, block_size: int = 1024, dtype: str = "uint16"):
        self.bin_path = bin_path
        self.block_size = block_size
        self.np_dtype = np.dtype(dtype)
        meta_path = bin_path + ".meta.json"
        if os.path.exists(meta_path):
            # Written by build_tokenized_binary -- if present, a dtype
            # mismatch here is almost certainly a real mistake (built with
            # one integer width, opened with another), and reading it
            # anyway would silently reinterpret every byte as the wrong
            # token ids -- not an error, just completely wrong training
            # data with no indication anything is off. No sidecar (e.g. a
            # binary built by hand/another tool) means this check is
            # simply skipped -- dtype is trusted as given, exactly as before.
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("dtype") and meta["dtype"] != dtype:
                raise AsterWhiteError(
                    f"{bin_path} was built with dtype={meta['dtype']!r} (per {meta_path}) but opened here with "
                    f"dtype={dtype!r} -- these produce completely different, silently wrong token ids from the "
                    f"same bytes. Pass dtype={meta['dtype']!r} to match how it was built."
                )
        self._mmap: Optional[np.memmap] = None  # opened lazily; see _ensure_open
        self._length = os.path.getsize(bin_path) // self.np_dtype.itemsize
        if self._length <= block_size:
            raise AsterWhiteError(
                f"{bin_path} has only {self._length} tokens, but block_size={block_size} needs at "
                f"least {block_size + 1} to form even one training example -- this binary is too small (or "
                f"block_size is too large) to train on. (This would otherwise fail later with an opaque "
                f"PyTorch error from an empty DataLoader.)"
            )

    def _ensure_open(self) -> np.memmap:
        # Re-open per worker process -- memmaps don't reliably survive fork
        # across DataLoader worker processes on every platform.
        if self._mmap is None:
            self._mmap = np.memmap(self.bin_path, dtype=self.np_dtype, mode="r")
        return self._mmap

    def __len__(self) -> int:
        return max(0, (self._length - 1) // self.block_size)

    def __getitem__(self, idx: int) -> torch.Tensor:
        mm = self._ensure_open()
        start = idx * self.block_size
        chunk = np.asarray(mm[start: start + self.block_size + 1], dtype=np.int64)
        return torch.from_numpy(chunk)


@trace_errors
def build_tokenized_binary(paths: Union[str, Sequence[str]], output_path: str, encode_fn: EncodeFn = byte_encode,
                            dtype: str = "uint16", text_column: Optional[str] = None) -> str:
    """Streams arbitrarily large raw text (files, directories -- searched
    recursively, glob patterns, and every SUPPORTED_DATA_EXTENSIONS format
    mixed together -- same acceptance rules as everywhere else via
    resolve_data_paths/iter_raw_text_for_tokenizer_training) into a single
    flat binary token file, writing incrementally -- never holding the
    full input OR the full output in memory. This is what
    MemmapTokenDataset then reads; it scales the *build* step to TB inputs
    the same way reading does.

    dtype='uint16' (the default) supports vocab_size up to 65536; pass
    'uint32' for a larger tokenizer vocabulary -- MemmapTokenDataset must
    be opened with the SAME dtype used here or every token id after the
    first will silently misalign (uint16 vs uint32 read the same bytes as
    a different number of tokens of different values, not an error)."""
    np_dtype = np.dtype(dtype)
    max_representable = int(np.iinfo(np_dtype).max)
    resolved = resolve_data_paths(paths)
    n_tokens = 0
    with open(output_path, "wb") as out:
        for text in iter_raw_text_for_tokenizer_training(resolved, text_column=text_column):
            id_list = encode_fn(text)
            if not id_list:
                continue
            max_id = max(id_list)
            if max_id > max_representable:
                out.close()
                os.remove(output_path)
                raise AsterWhiteError(
                    f"encode_fn produced token id {max_id}, which exceeds dtype={dtype!r}'s max representable "
                    f"value ({max_representable}). Writing it anyway would silently wrap/corrupt that token id "
                    f"(and every read of it) rather than error -- use dtype='uint32' for a tokenizer with a "
                    f"vocabulary this large."
                )
            ids = np.asarray(id_list, dtype=np_dtype)
            out.write(ids.tobytes())
            n_tokens += len(ids)
    if n_tokens == 0:
        os.remove(output_path)
        raise AsterWhiteError(f"no usable text extracted from {resolved[:5]}"
                               f"{'...' if len(resolved) > 5 else ''} -- refusing to write an empty binary.")
    with open(output_path + ".meta.json", "w") as meta_f:
        json.dump({"dtype": dtype, "n_tokens": n_tokens}, meta_f)
    logger.info(f"built tokenized binary -> {output_path} ({n_tokens:,} tokens, "
                f"{os.path.getsize(output_path):,} bytes, dtype={dtype})")
    return output_path


# --------------------------------------------------------------------------- #
# Scale-auto factory
# --------------------------------------------------------------------------- #

def _total_size(paths: Sequence[str]) -> int:
    return sum(os.path.getsize(p) for p in paths if os.path.exists(p))


def build_dataset(paths: Union[str, Sequence[str]], block_size: int = 512, scale: str = "auto",
                   mb_threshold: int = 256 << 20, text_column: Optional[str] = None,
                   encode_fn: EncodeFn = byte_encode, rank: Optional[int] = None,
                   world_size: Optional[int] = None):
    """scale='auto' picks InMemoryTextDataset below `mb_threshold` total
    bytes, else StreamingRecordDataset. Pass scale='memmap' explicitly with
    a single pre-tokenized .bin path to get MemmapTokenDataset for GB/TB
    corpora (build one with build_tokenized_binary).

    rank/world_size: only meaningful for the streaming path (map-style
    InMemoryTextDataset/MemmapTokenDataset are correctly rank-sharded
    externally via torch.utils.data.DistributedSampler in train.py's
    DataLoader construction instead -- they don't need internal sharding
    logic the way an IterableDataset does)."""
    paths = resolve_data_paths(paths)
    auto_memmap = scale == "auto" and len(paths) == 1 and paths[0].lower().endswith(".bin")
    if scale == "memmap" or auto_memmap:
        assert len(paths) == 1, "MemmapTokenDataset takes exactly one pre-tokenized .bin file"
        meta_path = paths[0] + ".meta.json"
        dtype = "uint16"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                dtype = json.load(f).get("dtype", dtype)
        if auto_memmap:
            logger.info(f"{paths[0]} ends in .bin -- auto-detected as a pre-tokenized MemmapTokenDataset "
                        f"(dtype={dtype}, {'from ' + meta_path if os.path.exists(meta_path) else 'no .meta.json '
                        'sidecar found, defaulting'})")
        return MemmapTokenDataset(paths[0], block_size=block_size, dtype=dtype)

    total = _total_size(paths)
    use_streaming = scale == "streaming" or (scale == "auto" and total > mb_threshold)
    if use_streaming:
        logger.info(f"dataset size {total / (1 << 20):.1f} MB -> streaming pipeline")
        return StreamingRecordDataset(paths, text_column=text_column, encode_fn=encode_fn, block_size=block_size,
                                       rank=rank, world_size=world_size)
    logger.info(f"dataset size {total / (1 << 20):.1f} MB -> in-memory pipeline")
    return InMemoryTextDataset(paths, block_size=block_size, encode_fn=encode_fn)
