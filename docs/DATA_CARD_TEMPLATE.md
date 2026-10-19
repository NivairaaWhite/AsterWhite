# Data Card -- <DATASET NAME>

## Provenance
- Source(s): `<where this came from -- own writing, licensed corpus, public domain, ...>`
- Collection method: `<...>`
- License: `<...>`
- Consent / rights basis (if it includes anyone's personal writing, likeness, etc.): `<...>`

## Composition
- Format: `<.txt/.csv/.parquet/.jsonl>`, size on disk: `<...>`, record count: `<...>`
- Languages/scripts present: `<...>` (informs
  `asterwhite.tokenization.AsterWhiteTokenizer.fertility` measurement)
- Known biases or skew: `<genre/topic/author skew, time period, ...>`

## Processing
- Cleaning steps applied: `<dedup, PII scrubbing (asterwhite.safety.pii_scrubber),
  language filtering, ...>`
- Corrupted/skipped records: `<count -- data_pipeline.py logs these with
  folder/file/line via asterwhite.utils.error_tracing>`
- Train/validation split method: `<...>`

## Restrictions
- Excluded content categories: `<...>`
- Redistribution restrictions: `<...>`
- Retention/deletion policy: `<...>`

## Versioning
- `dataset_version` (referenced by `registry.manifest.RunRecord.dataset_versions`): `<...>`
- Changes from previous version: `<...>`
