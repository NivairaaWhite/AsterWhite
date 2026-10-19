# Model Card -- AsterWhite <VERSION>

Fill in every section before setting a `registry.manifest.ModelRelease`
to `PromotionStage.CANDIDATE` or beyond. An unfilled placeholder left in
a real release is a documentation gap, not a formality -- it's exactly
the "known limitations documented honestly" requirement this project
takes as a hard constraint (see README.md's scope notes).

## Summary
- **Architecture**: hybrid grouped-query attention / selective state-space
  backbone (see `asterwhite/core/`), `<N>` layers, `<D>` hidden dim.
- **Parameters**: `<count>` (from `asterwhite.core.autosizer.count_parameters`)
- **Tokenizer**: `<byte-level fallback | trained AsterWhiteTokenizer, hash <...>>`
- **Training data**: see the linked Data Card(s): `<paths>`
- **Intended use**: `<narrative/fiction generation, character simulation, ...>`
- **Out-of-scope use**: `<factual QA, code generation, anything safety-critical, ...>`

## Training
- Base config / profile: `<configs/*.yaml name, or custom>`
- Hardware: `<DeviceInfo.as_manifest_dict() output>`
- Steps / tokens seen: `<...>`
- Alignment: `<HeuristicRewardComposite | trained RewardModel, version ...>`
- LoRA/adapters applied: `<yes/no, target modules, rank>`

## Evaluation
- Validation loss / perplexity: `<Trainer.evaluate() output over training>`
- Qualitative spot checks: `<link or paste a few samples, both good and bad>`
- Known failure modes: `<repetition, contradiction, drift, ...>`

## Safety
- `asterwhite.safety.PolicyFilter` categories checked: violence, self_harm,
  csam, weapons_uplift, hate_speech (heuristic keyword/regex -- NOT a
  trained classifier or red-teamed result; see safety/policy_filter.py's
  module docstring).
- Red-teaming performed: `<none bundled with this repository -- describe
  what you actually did before shipping>`
- Known limitations: `<list every one you know about, plainly>`

## Provenance
- `run_id`: `<RunRecord.run_id>`
- `code_git_commit`: `<RunRecord.code_git_commit>`
- `release_hash_sha256`: `<ModelRelease.compute_release_hash()>`
- Signed: `<yes (HMAC key id `<...>`) | no>`
