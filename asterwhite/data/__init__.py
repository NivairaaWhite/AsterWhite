"""
AsterWhite :: data

Spec 4.1 places ingestion/cleaning/sharding/manifests/loaders under
asterwhite/data/. The actual implementation lives in the top-level
data_pipeline.py (imported throughout the repo -- train.py,
generation_orchestrator.py, tests/) and stayed there rather than being
physically moved: this package re-exports it so `from asterwhite.data
import build_dataset` also works, without a repo-wide import-path
rewrite of code that is already tested and working. Both import paths
resolve to the exact same module (see sys.modules -- there is only one
copy).
"""
import sys

import data_pipeline as _data_pipeline
from data_pipeline import (  # noqa: F401
    EncodeFn, InMemoryTextDataset, MemmapTokenDataset, StreamingRecordDataset,
    build_dataset, build_tokenized_binary, byte_encode,
)

sys.modules[__name__ + ".data_pipeline"] = _data_pipeline  # so asterwhite.data.data_pipeline also resolves

__all__ = [
    "build_dataset", "build_tokenized_binary", "byte_encode", "EncodeFn",
    "InMemoryTextDataset", "StreamingRecordDataset", "MemmapTokenDataset",
]
