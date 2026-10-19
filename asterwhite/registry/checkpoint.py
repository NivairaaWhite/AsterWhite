"""
AsterWhite :: registry.checkpoint

Safe, atomic, checksummed, versioned checkpoint I/O. Replaces every bare
`torch.save`/`torch.load` in the repo (training checkpoints, generation
checkpoints): tensors are written through `safetensors` (a format that
cannot execute arbitrary code on load -- unlike pickle, which `torch.load`
uses by default), and everything else (optimizer step counts, RNG state,
world-state JSON, config) travels as a plain-JSON sidecar.

What "atomic, checksummed, versioned, safe" means concretely here:
  - atomic:      write to a temp path in the same directory, fsync, then
                 os.replace() into the final name (POSIX rename is atomic;
                 a crash mid-write leaves the old file or the temp file,
                 never a half-written final file).
  - checksummed: the tensor file's SHA-256 is computed after the fsync and
                 stored in the metadata sidecar; load() recomputes and
                 compares before touching the tensors, and refuses on a
                 mismatch instead of loading a possibly-corrupted file.
  - versioned:   every sidecar carries CHECKPOINT_SCHEMA_VERSION; load()
                 refuses a mismatched version rather than guessing.
  - safe:        no pickle for the tensor payload. Arbitrary nested
                 metadata (optimizer state, RNG state) is flattened so its
                 tensors also go through safetensors -- see
                 `flatten_tensors`/`unflatten_tensors` below.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ..utils.error_tracing import AsterWhiteError, get_logger

logger = get_logger("asterwhite.registry.checkpoint")

CHECKPOINT_SCHEMA_VERSION = 1

_TUPLE_MARK = "__tuple__"
_TENSOR_MARK = "__tensor_ref__"


# --------------------------------------------------------------------------- #
# Generic tensor/JSON flattening -- lets arbitrary nested structures
# (optimizer.state_dict(), RNG state, ...) travel through safetensors +
# JSON instead of pickle.
# --------------------------------------------------------------------------- #

def flatten_tensors(obj: Any, prefix: str = "t") -> Tuple[Dict[str, torch.Tensor], Any]:
    """Walks obj (nested dict/list/tuple of tensors + JSON-safe primitives),
    pulls every torch.Tensor out into a flat {key: tensor} dict, and returns
    a JSON-safe 'skeleton' with each tensor replaced by a small marker
    pointing at its flat key. Non-tensor leaves are left in place."""
    tensors: Dict[str, torch.Tensor] = {}
    counter = [0]

    def _walk(o: Any) -> Any:
        if isinstance(o, torch.Tensor):
            key = f"{prefix}.{counter[0]}"
            counter[0] += 1
            tensors[key] = o.detach().cpu().contiguous()
            return {_TENSOR_MARK: key}
        if isinstance(o, dict):
            return {"__dict__": {str(k): _walk(v) for k, v in o.items()}}
        if isinstance(o, tuple):
            return {_TUPLE_MARK: [_walk(v) for v in o]}
        if isinstance(o, list):
            return [_walk(v) for v in o]
        if isinstance(o, np.ndarray):
            key = f"{prefix}.{counter[0]}"
            counter[0] += 1
            tensors[key] = torch.from_numpy(o.copy())
            return {"__ndarray__": key, "dtype": str(o.dtype)}
        if isinstance(o, (int, float, str, bool)) or o is None:
            return o
        raise AsterWhiteError(f"flatten_tensors: unsupported leaf type {type(o)} -- extend the walker "
                               f"before storing this in a checkpoint (never pickle it as a fallback).")

    skeleton = _walk(obj)
    return tensors, skeleton


def unflatten_tensors(skeleton: Any, tensors: Dict[str, torch.Tensor]) -> Any:
    """Inverse of flatten_tensors."""
    def _walk(o: Any) -> Any:
        if isinstance(o, dict):
            if _TENSOR_MARK in o:
                return tensors[o[_TENSOR_MARK]]
            if "__ndarray__" in o:
                return tensors[o["__ndarray__"]].numpy().astype(o["dtype"])
            if _TUPLE_MARK in o:
                return tuple(_walk(v) for v in o[_TUPLE_MARK])
            if "__dict__" in o:
                return {k: _walk(v) for k, v in o["__dict__"].items()}
            raise AsterWhiteError(f"unflatten_tensors: unrecognized marker dict {list(o.keys())}")
        if isinstance(o, list):
            return [_walk(v) for v in o]
        return o

    return _walk(skeleton)


# --------------------------------------------------------------------------- #
# RNG state capture/restore -- one of the "MUST contain or reference" items
# for a production checkpoint (Python / NumPy / CPU / accelerator RNGs).
# --------------------------------------------------------------------------- #

def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


# --------------------------------------------------------------------------- #
# Low-level atomic + checksummed file helpers
# --------------------------------------------------------------------------- #

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_bytes(final_path: str, write_fn) -> None:
    """write_fn(file_object) writes the content. Writes to a temp file in
    the SAME directory (so os.replace stays on one filesystem -> atomic),
    fsyncs before renaming, and only then replaces the final path."""
    directory = os.path.dirname(final_path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=os.path.basename(final_path))
    try:
        with os.fdopen(fd, "wb") as f:
            write_fn(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, final_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _storage_ptr(t: torch.Tensor) -> int:
    try:
        return t.untyped_storage().data_ptr()
    except AttributeError:  # pragma: no cover -- older torch
        return t.storage().data_ptr()


def _deduplicate_shared_tensors(tensors: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """safetensors refuses to serialize two keys that back the same
    storage (e.g. tok_emb.weight / lm_head.weight under
    tie_word_embeddings=True -- discovered by this module's own tests,
    not assumed). Keeps the first key seen per storage identity, and
    returns {dropped_key: canonical_key} so the caller can put the dropped
    key(s) back after loading (see restore_aliases below)."""
    seen: Dict[int, str] = {}
    deduped: Dict[str, torch.Tensor] = {}
    aliases: Dict[str, str] = {}
    for key, tensor in tensors.items():
        ptr = _storage_ptr(tensor)
        if ptr in seen:
            aliases[key] = seen[ptr]
        else:
            seen[ptr] = key
            deduped[key] = tensor
    return deduped, aliases


def restore_aliases(tensors: Dict[str, torch.Tensor], aliases: Dict[str, str]) -> Dict[str, torch.Tensor]:
    """Inverse of _deduplicate_shared_tensors: re-adds each dropped key
    with a copy of its canonical tensor's data, so a state_dict built from
    the result has every original key again."""
    out = dict(tensors)
    for dropped_key, canonical_key in aliases.items():
        out[dropped_key] = tensors[canonical_key].clone()
    return out


@dataclass
class CheckpointHandle:
    name: str
    directory: str
    metadata_path: str
    tensor_path: str
    metadata: Dict[str, Any]


def save_checkpoint(directory: str, name: str, tensors: Dict[str, torch.Tensor],
                     metadata: Dict[str, Any], make_latest: bool = True,
                     make_best: bool = False) -> CheckpointHandle:
    """Atomically writes `tensors` (safetensors) + `metadata` (JSON, with
    schema version + tensor checksum injected) under `directory/{name}.*`,
    then atomically updates `latest.json` / `best.json` pointers."""
    os.makedirs(directory, exist_ok=True)
    tensor_path = os.path.join(directory, f"{name}.safetensors")
    meta_path = os.path.join(directory, f"{name}.json")

    deduped_tensors, tensor_aliases = _deduplicate_shared_tensors(tensors)
    _atomic_write_bytes(tensor_path, lambda f: f.write(_serialize_tensors(deduped_tensors)))
    checksum = _sha256_file(tensor_path)

    full_metadata = dict(metadata)
    full_metadata.update({
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "tensor_file": os.path.basename(tensor_path),
        "tensor_checksum_sha256": checksum,
        "tensor_aliases": tensor_aliases,
        "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    _atomic_write_bytes(meta_path, lambda f: f.write(json.dumps(full_metadata, indent=2).encode("utf-8")))

    if make_latest:
        _write_pointer(directory, "latest", name)
    if make_best:
        _write_pointer(directory, "best", name)

    logger.info(f"checkpoint saved -> {meta_path} ({len(deduped_tensors)} tensors "
                f"({len(tensor_aliases)} aliased/deduped), sha256={checksum[:12]}...)")
    return CheckpointHandle(name=name, directory=directory, metadata_path=meta_path,
                             tensor_path=tensor_path, metadata=full_metadata)


def _serialize_tensors(tensors: Dict[str, torch.Tensor]) -> bytes:
    # safetensors.torch.save_file writes to a path; route through a temp
    # file of its own so we can hand _atomic_write_bytes raw bytes and keep
    # ONE atomic-rename code path for every artifact this module writes.
    fd, tmp = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    try:
        save_file(tensors, tmp)
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        os.remove(tmp)


def _write_pointer(directory: str, pointer_name: str, checkpoint_name: str) -> None:
    path = os.path.join(directory, f"{pointer_name}.json")
    _atomic_write_bytes(path, lambda f: f.write(json.dumps({"name": checkpoint_name}).encode("utf-8")))


def resolve_checkpoint_name(directory: str, name: Optional[str] = None, which: str = "latest") -> str:
    if name is not None:
        return name
    pointer_path = os.path.join(directory, f"{which}.json")
    if not os.path.exists(pointer_path):
        raise AsterWhiteError(f"no {which!r} checkpoint pointer in {directory} -- pass an explicit "
                               f"name, or save at least one checkpoint with make_{which}=True first.")
    with open(pointer_path) as f:
        return json.load(f)["name"]


def load_checkpoint(directory: str, name: Optional[str] = None, which: str = "latest",
                     map_location: str = "cpu", expected_tokenizer_hash: Optional[str] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Verifies checksum + schema version BEFORE returning tensors. Raises
    AsterWhiteError (not a bare exception) on any integrity problem --
    corrupted checkpoints must fail loudly, never load partially."""
    resolved_name = resolve_checkpoint_name(directory, name, which)
    tensor_path = os.path.join(directory, f"{resolved_name}.safetensors")
    meta_path = os.path.join(directory, f"{resolved_name}.json")

    if not (os.path.exists(tensor_path) and os.path.exists(meta_path)):
        raise AsterWhiteError(f"checkpoint {resolved_name!r} is incomplete or missing in {directory} "
                               f"(need both {os.path.basename(tensor_path)} and {os.path.basename(meta_path)}) "
                               f"-- refusing to load a partial checkpoint.")

    with open(meta_path) as f:
        metadata = json.load(f)

    if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise AsterWhiteError(f"checkpoint {resolved_name!r} has schema_version="
                               f"{metadata.get('schema_version')!r}, this build expects "
                               f"{CHECKPOINT_SCHEMA_VERSION} -- refusing to load rather than guess "
                               f"at a compatibility mapping.")

    actual_checksum = _sha256_file(tensor_path)
    expected_checksum = metadata.get("tensor_checksum_sha256")
    if actual_checksum != expected_checksum:
        raise AsterWhiteError(f"checkpoint {resolved_name!r} FAILED checksum verification "
                               f"(expected {expected_checksum}, got {actual_checksum}) -- the tensor "
                               f"file is corrupted or was tampered with. Refusing to load.")

    if expected_tokenizer_hash is not None and metadata.get("tokenizer_hash") not in (None, expected_tokenizer_hash):
        raise AsterWhiteError(f"checkpoint {resolved_name!r} was trained with tokenizer_hash="
                               f"{metadata.get('tokenizer_hash')!r} but the currently loaded tokenizer "
                               f"hashes to {expected_tokenizer_hash!r} -- these are incompatible "
                               f"vocabularies. See asterwhite.tokenization for migration.")

    tensors = load_file(tensor_path, device=map_location)
    tensors = restore_aliases(tensors, metadata.get("tensor_aliases", {}))
    logger.info(f"checkpoint loaded <- {meta_path} (sha256 verified, schema_version="
                f"{metadata['schema_version']}, {len(metadata.get('tensor_aliases', {}))} alias(es) restored)")
    return tensors, metadata
