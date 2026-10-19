#!/usr/bin/env python3
"""
AsterWhite :: scripts/prepare_corpus.py

Builds the training corpus the bundled pretrained model (pretrained/
asterwhite-small) was trained on -- reproducibly, from public-domain text:

  * source: the NLTK `gutenberg` corpus (Project Gutenberg texts, all
    public domain), fetched from the nltk_data GitHub repo, or read from a
    local copy via --source-zip if you are offline.
  * selection: FICTION only (Austen, Melville, Chesterton, Carroll,
    Edgeworth, Bryant, Burgess). Poetry, plays and scripture are left out on
    purpose -- AsterWhite is a prose/novel model and mixing verse or
    scripture cadence into a small model visibly leaks into its prose.
  * cleaning: the Gutenberg files are hard-wrapped at ~70 columns. A model
    trained on that learns to emit a line break every 70 characters, so
    paragraphs are un-wrapped (single newlines inside a paragraph become
    spaces; blank lines between paragraphs are kept) and runs of spaces are
    collapsed. Nothing else is altered.
  * split: the LAST `--val-fraction` of every book (cut on a paragraph
    boundary) becomes validation text; everything before it is training
    text. Validation text is therefore never seen in training.

Output layout (each book is one "document" -- the tokenizer wraps every
file in BOS/EOS, so documents stay separated during training):

    <out>/train/<book>.txt
    <out>/val/<book>.txt
    <out>/manifest.json            (what went in, sizes, source URL)

Usage:
    python scripts/prepare_corpus.py --out ./data
    python scripts/prepare_corpus.py --out ./data --source-zip gutenberg.zip
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import urllib.request
import zipfile

SOURCE_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip"

# Fiction only -- see module docstring for why verse/plays/scripture are excluded.
FICTION_BOOKS = [
    "austen-emma", "austen-persuasion", "austen-sense",
    "bryant-stories", "burgess-busterbrown", "carroll-alice",
    "chesterton-ball", "chesterton-brown", "chesterton-thursday",
    "edgeworth-parents", "melville-moby_dick",
]


def unwrap_paragraphs(raw: str) -> str:
    """Hard-wrapped Gutenberg text -> one line per paragraph."""
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n\s*\n", raw)
    cleaned = []
    for p in paragraphs:
        p = re.sub(r"\s*\n\s*", " ", p.strip())
        p = re.sub(r"[ \t]+", " ", p)
        if p:
            cleaned.append(p)
    return "\n\n".join(cleaned) + "\n"


def split_train_val(text: str, val_fraction: float):
    """Cut on a paragraph boundary so neither side starts/ends mid-paragraph."""
    paragraphs = text.strip().split("\n\n")
    total = sum(len(p) for p in paragraphs)
    target = total * (1.0 - val_fraction)
    running, cut = 0, len(paragraphs)
    for i, p in enumerate(paragraphs):
        running += len(p)
        if running >= target:
            cut = i + 1
            break
    train = "\n\n".join(paragraphs[:cut]) + "\n"
    val = "\n\n".join(paragraphs[cut:]) + "\n"
    return train, val


def load_zip(source_zip: str | None) -> zipfile.ZipFile:
    if source_zip:
        return zipfile.ZipFile(source_zip)
    print(f"downloading {SOURCE_URL}", file=sys.stderr)
    with urllib.request.urlopen(SOURCE_URL, timeout=120) as resp:
        return zipfile.ZipFile(io.BytesIO(resp.read()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory (created if missing)")
    ap.add_argument("--source-zip", default=None, help="local copy of nltk_data's gutenberg.zip (skips the download)")
    ap.add_argument("--val-fraction", type=float, default=0.04, help="tail of each book held out for validation")
    args = ap.parse_args()
    if not (0.0 < args.val_fraction < 0.5):
        ap.error("--val-fraction must be in (0, 0.5)")

    zf = load_zip(args.source_zip)
    names = {os.path.basename(n): n for n in zf.namelist() if n.endswith(".txt")}
    os.makedirs(os.path.join(args.out, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "val"), exist_ok=True)

    manifest = {"source": args.source_zip or SOURCE_URL, "val_fraction": args.val_fraction, "books": {}}
    train_chars = val_chars = 0
    for book in FICTION_BOOKS:
        member = names.get(f"{book}.txt")
        if member is None:
            raise SystemExit(f"{book}.txt not found in the source zip -- is this the NLTK gutenberg corpus?")
        text = unwrap_paragraphs(zf.read(member).decode("utf-8", errors="replace"))
        train, val = split_train_val(text, args.val_fraction)
        with open(os.path.join(args.out, "train", f"{book}.txt"), "w", encoding="utf-8") as f:
            f.write(train)
        with open(os.path.join(args.out, "val", f"{book}.txt"), "w", encoding="utf-8") as f:
            f.write(val)
        manifest["books"][book] = {"train_chars": len(train), "val_chars": len(val)}
        train_chars += len(train)
        val_chars += len(val)
        print(f"  {book:<24} train {len(train):>9,} chars   val {len(val):>8,} chars")

    manifest["train_chars_total"] = train_chars
    manifest["val_chars_total"] = val_chars
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"done -> {args.out}  (train {train_chars:,} chars, val {val_chars:,} chars)")


if __name__ == "__main__":
    main()
