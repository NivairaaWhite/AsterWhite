"""
AsterWhite :: tokenization.tokenizer

A real, trainable subword tokenizer pipeline -- the production path the
spec (section 6) requires. The byte-level fallback (`data_pipeline.
byte_encode`) remains available and is what every test / --dry-run path
still uses by default for zero-setup operation; this module is what a real
run should train and load instead.

Design:
  - Byte-level BPE (same family as GPT-2/GPT-NeoX tokenizers): the
    pre-tokenizer operates on raw UTF-8 bytes, so ANY Unicode text encodes
    with zero "unknown character" failures -- the standard way to satisfy
    "support multilingual Unicode text" without a curated character set.
  - Reserved control tokens for BOS/EOS/PAD/UNK plus narrative-specific
    slots (plan beats, characters, scenes, revisions, tools, safety labels)
    per spec 6 -- added as atomic special tokens so they can never be
    split apart by BPE merges.
  - A stable SHA-256 hash over the serialized tokenizer is what
    registry.checkpoint.load_checkpoint's `expected_tokenizer_hash` check
    compares against, so a checkpoint trained with tokenizer A can't be
    silently loaded and fine-tuned with tokenizer B.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from ..utils.error_tracing import AsterWhiteError, get_logger

logger = get_logger("asterwhite.tokenization")

TOKENIZER_SCHEMA_VERSION = 1

# Fixed control-token vocabulary. Order matters -- these occupy the lowest
# token ids, stable across retrains as long as this list doesn't change
# (changing it is exactly the "vocabulary change" section 6 says must go
# through the explicit migration path below, never silently).
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
PAD_TOKEN = "<|pad|>"
UNK_TOKEN = "<|unk|>"
# Narrative control tokens (spec 6: "plan beats, characters, scenes,
# revisions, tools and safety labels"). Kept as a small fixed set rather
# than one-per-instance -- callers pair them with an id/argument in the
# surrounding text (e.g. "<|character|> Kestrel <|/character|>") rather
# than needing a token per character name, which would make the vocabulary
# unbounded.
BEAT_TOKEN, BEAT_END_TOKEN = "<|beat|>", "<|/beat|>"
CHARACTER_TOKEN, CHARACTER_END_TOKEN = "<|character|>", "<|/character|>"
SCENE_TOKEN, SCENE_END_TOKEN = "<|scene|>", "<|/scene|>"
REVISION_TOKEN, REVISION_END_TOKEN = "<|revision|>", "<|/revision|>"
TOOL_TOKEN, TOOL_END_TOKEN = "<|tool|>", "<|/tool|>"
SAFETY_TOKEN, SAFETY_END_TOKEN = "<|safety|>", "<|/safety|>"

CONTROL_TOKENS: List[str] = [
    BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN,
    BEAT_TOKEN, BEAT_END_TOKEN, CHARACTER_TOKEN, CHARACTER_END_TOKEN,
    SCENE_TOKEN, SCENE_END_TOKEN, REVISION_TOKEN, REVISION_END_TOKEN,
    TOOL_TOKEN, TOOL_END_TOKEN, SAFETY_TOKEN, SAFETY_END_TOKEN,
]


@dataclass
class FertilityReport:
    """Tokens-per-word by sample, the spec's 'measure fertility by language
    and script' -- a coarse proxy (word-splitting is script-dependent) but
    a real, computed number rather than an assumption."""
    samples: Dict[str, float]  # label -> mean tokens-per-word

    def as_dict(self) -> Dict[str, float]:
        return dict(self.samples)


class AsterWhiteTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self._tok = tokenizer

    # -- construction -------------------------------------------------- #

    @classmethod
    def train(cls, corpus_paths: Sequence[str], vocab_size: int = 32000,
              min_frequency: int = 2, text_column: Optional[str] = None) -> "AsterWhiteTokenizer":
        """corpus_paths accepts individual files, directories (searched
        recursively for every data_pipeline.SUPPORTED_DATA_EXTENSIONS
        format), and glob patterns, in any combination -- the exact same
        acceptance rules `pretrain --data` uses (data_pipeline.
        resolve_data_paths). For structured formats (.jsonl/.ndjson/.csv/
        .parquet), only the actual prose is used for training -- see
        data_pipeline.iter_raw_text_for_tokenizer_training's docstring for
        why raw JSON/CSV syntax would otherwise pollute the vocabulary.
        text_column selects which field/column holds that prose; None (the
        default) uses the first field/column, matching the pretraining
        data pipeline's own default."""
        # Imported here (not at module top) to avoid a hard import-time
        # dependency from tokenization on the top-level data_pipeline
        # module for code paths that never train a tokenizer at all.
        from data_pipeline import iter_raw_text_for_tokenizer_training, resolve_data_paths

        if vocab_size <= len(CONTROL_TOKENS):
            raise AsterWhiteError(f"vocab_size={vocab_size} must exceed the {len(CONTROL_TOKENS)} "
                                   f"reserved control tokens.")
        resolved = resolve_data_paths(corpus_paths)  # raises with a specific, actionable message if none exist
        tok = Tokenizer(BPE(unk_token=UNK_TOKEN))
        tok.normalizer = normalizers.NFC()
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency,
                              special_tokens=CONTROL_TOKENS,
                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        text_iter = iter_raw_text_for_tokenizer_training(resolved, text_column=text_column)
        try:
            first = next(text_iter)
        except StopIteration:
            preview = resolved[:5] + (["..."] if len(resolved) > 5 else [])
            raise AsterWhiteError(
                f"{len(resolved)} corpus file(s) resolved ({preview}) but none contained usable text -- "
                f"empty files, or (for .jsonl/.csv/.parquet) no matching text_column in any record. Pass "
                f"text_column explicitly if your text field isn't the first one."
            )
        tok.train_from_iterator(itertools.chain([first], text_iter), trainer=trainer)
        tok.post_processor = processors.TemplateProcessing(
            single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
            pair=f"{BOS_TOKEN} $A {EOS_TOKEN} {BOS_TOKEN} $B {EOS_TOKEN}",
            special_tokens=[(BOS_TOKEN, tok.token_to_id(BOS_TOKEN)), (EOS_TOKEN, tok.token_to_id(EOS_TOKEN))],
        )
        logger.info(f"trained tokenizer: vocab_size={tok.get_vocab_size()} over {len(resolved)} file(s)")
        return cls(tok)

    @classmethod
    def load(cls, directory: str) -> "AsterWhiteTokenizer":
        path = os.path.join(directory, "tokenizer.json")
        if not os.path.exists(path):
            raise AsterWhiteError(f"no tokenizer.json in {directory}")
        with open(os.path.join(directory, "tokenizer_meta.json")) as f:
            meta = json.load(f)
        if meta.get("schema_version") != TOKENIZER_SCHEMA_VERSION:
            raise AsterWhiteError(f"tokenizer schema_version={meta.get('schema_version')} incompatible "
                                   f"with expected {TOKENIZER_SCHEMA_VERSION}")
        # Same reasoning as train()'s existing_abs above: Tokenizer.from_file
        # opens the file in the tokenizers library's Rust core, which has
        # been observed not to reliably resolve a relative path against the
        # current process's working directory on Windows.
        tok = Tokenizer.from_file(str(Path(path).resolve()))
        instance = cls(tok)
        actual_hash = instance.compute_hash()
        if actual_hash != meta.get("hash_sha256"):
            raise AsterWhiteError(f"tokenizer.json in {directory} does not match its own tokenizer_meta.json "
                                   f"hash record (expected {meta.get('hash_sha256')}, got {actual_hash}) -- "
                                   f"the files may be mismatched or corrupted.")
        return instance

    def save(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "tokenizer.json")
        self._tok.save(str(Path(path).resolve()))
        meta = {
            "schema_version": TOKENIZER_SCHEMA_VERSION,
            "vocab_size": self.vocab_size,
            "hash_sha256": self.compute_hash(),
            "control_tokens": CONTROL_TOKENS,
        }
        with open(os.path.join(directory, "tokenizer_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"tokenizer saved -> {directory} (vocab_size={self.vocab_size}, hash={meta['hash_sha256'][:12]}...)")
        return path

    # -- core encode/decode (matches data_pipeline.EncodeFn signature) --- #

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """add_special_tokens=True (default) wraps the text as a whole
        DOCUMENT: [BOS] text [EOS] -- exactly what pretraining feeds the
        model per file/record. Do NOT use that to encode a PROMPT to be
        continued: the trailing EOS tells the model the document has already
        ended, so it starts a new one instead of continuing yours. Use
        encode_prompt() for that."""
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids

    def encode_prompt(self, text: str) -> List[int]:
        """[BOS] text -- with NO trailing EOS -- the right way to encode text
        the model is about to CONTINUE (generation, serving). Matches the
        start of every training document, without claiming the document is
        over."""
        return [self.bos_id] + self.encode(text, add_special_tokens=False)

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def token_id(self, token: str) -> Optional[int]:
        return self._tok.token_to_id(token)

    @property
    def bos_id(self) -> int:
        return self._tok.token_to_id(BOS_TOKEN)

    @property
    def eos_id(self) -> int:
        return self._tok.token_to_id(EOS_TOKEN)

    @property
    def pad_id(self) -> int:
        return self._tok.token_to_id(PAD_TOKEN)

    # -- hash / integrity ------------------------------------------------ #

    def compute_hash(self) -> str:
        """Stable hash over the serialized tokenizer -- what checkpoints
        tie their vocabulary identity to (registry.checkpoint's
        expected_tokenizer_hash)."""
        payload = self._tok.to_str(pretty=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    # -- required tests / measurements from spec 6 ----------------------- #

    def round_trip_ok(self, samples: Sequence[str]) -> bool:
        """encode -> decode -> compare. Byte-level BPE is lossless by
        construction, so this should always pass; it's still a real,
        runnable check rather than an assumption, and will catch a
        misconfigured normalizer (e.g. one that lowercases) immediately."""
        for s in samples:
            if self.decode(self.encode(s)).strip() != s.strip():
                return False
        return True

    def fertility(self, samples_by_label: Dict[str, Sequence[str]]) -> FertilityReport:
        out = {}
        for label, texts in samples_by_label.items():
            ratios = []
            for t in texts:
                n_words = max(1, len(t.split()))
                n_tokens = len(self.encode(t))
                ratios.append(n_tokens / n_words)
            out[label] = sum(ratios) / max(1, len(ratios))
        return FertilityReport(samples=out)


def remap_embeddings_for_new_tokenizer(old_tokenizer: AsterWhiteTokenizer, new_tokenizer: AsterWhiteTokenizer,
                                        old_embedding_weight) -> "object":
    """Spec 6: 'Support tokenizer migration only through an explicit
    embedding-remapping process.' For every string token that exists in
    BOTH vocabularies, copy its old embedding row into the new embedding
    matrix at the new id; tokens that only exist in the new vocabulary
    keep whatever init the caller already applied (typically random --
    pass a freshly-initialized nn.Embedding.weight in). This is a real,
    deterministic remap -- it does not fabricate a mapping for genuinely
    new subwords, which have no prior embedding to inherit.

    old_embedding_weight: a (old_vocab_size, d_model) tensor.
    Returns a new (new_vocab_size, d_model) tensor: `new_embedding_weight`,
    intended to be assigned into a freshly-constructed nn.Embedding before
    resuming training under the new tokenizer.
    """
    old_vocab = old_tokenizer._tok.get_vocab()   # token string -> old id
    new_vocab_size = new_tokenizer.vocab_size
    d_model = old_embedding_weight.shape[1]
    new_weight = old_embedding_weight.new_empty(new_vocab_size, d_model)
    new_weight.normal_(mean=0.0, std=0.02)  # same init scheme as AsterWhiteCore._init_weights

    copied = 0
    for token, old_id in old_vocab.items():
        new_id = new_tokenizer.token_id(token)
        if new_id is not None and old_id < old_embedding_weight.shape[0]:
            new_weight[new_id] = old_embedding_weight[old_id]
            copied += 1
    logger.info(f"tokenizer migration: copied {copied}/{new_vocab_size} embedding rows by matching "
                f"token string; {new_vocab_size - copied} rows are freshly initialized.")
    return new_weight
