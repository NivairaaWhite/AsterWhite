# AsterWhite

A from-scratch, trainable narrative-generation architecture: a hybrid
attention/state-space transformer core (with optional sparse
mixture-of-experts, long-context RoPE scaling, QK-norm, and multi-token
prediction), hierarchical world/character simulation, GRPO alignment, a
multi-pass revision suite, and a full production-infrastructure layer
around it.

**It ships with a pretrained model**, `pretrained/asterwhite-small`
(~6.5M parameters, trained with this repository's own `pretrain` command
on public-domain English fiction), so `generate` and `serve` work the
moment you unzip it -- no profile, no tokenizer, no checkpoint flags. It
is a small, demonstration-scale model: it writes plausible
nineteenth-century-style English prose, not a coherent novel. See
[`pretrained/asterwhite-small/MODEL_CARD.md`](pretrained/asterwhite-small/MODEL_CARD.md)
for measured numbers and honest limits, and use it as a base to fine-tune
or as proof that the whole pipeline works end to end. For anything
bigger, train your own with the same commands and a larger profile.

Every class is real, working PyTorch/Python code, exercised by `tests/`
on tiny dimensions so the whole system runs in seconds on CPU.

## Quickstart

**Use the bundled pretrained model (nothing to train):**

```bash
pip install -r requirements.txt
python train.py doctor                                        # environment preflight -- run this first

python train.py generate --prompt "It was a dark and stormy night," --max-new-tokens 150
python train.py generate --prompt "Chapter One." --branching 4 --chapters 4 --max-new-tokens 200 --seed 7
python train.py serve                                         # OpenAI/Ollama/generic API on :8000, serving the bundled model
python train.py evaluate --load pretrained/asterwhite-small --data my_held_out_text.txt   # perplexity
```

With no `--load`, `generate`/`align`/`serve` use `pretrained/asterwhite-small`
(or `$ASTERWHITE_PRETRAINED_DIR/asterwhite-small`). A bundle is a checkpoint
directory with a `tokenizer/` folder inside it, so the architecture and the
vocabulary are read from the bundle itself -- no `--profile`/`--tokenizer-dir`
needed, and a profile can never disagree with the weights. `--seed N` makes
sampling reproducible. Point `--load` at any other bundle to use that one.

Fine-tune from the bundled weights (architecture and tokenizer come from the bundle):

```bash
python train.py pretrain --load-base pretrained/asterwhite-small --data my_text/ --max-steps 500
python train.py pretrain --load-base pretrained/asterwhite-small --lora --data my_text/ --max-steps 500
```

**Train your own, or continue from scratch:**

```bash
python train.py selftest                                      # runs the full tests/ suite

python train.py pretrain --dry-run                            # tiny config, no data, seconds on CPU
python train.py pretrain --profile smoke_cpu --data mycorpus.txt --max-steps 100
python train.py pretrain --profile smoke_cpu --data ./my_book_folder/ --max-steps 100  # a directory works too
python train.py align --profile smoke_cpu --load ./checkpoints/smoke_cpu --steps 50
python train.py generate --profile smoke_cpu --load ./checkpoints/smoke_cpu --prompt "Chapter One."

python train.py tokenizer train --data mycorpus.txt --vocab-size 32000 --out ./tokenizer
python train.py serve --checkpoint-dir ./checkpoints/smoke_cpu --tokenizer-dir ./tokenizer

# turn a training run into a self-contained, shareable bundle (weights + config + tokenizer)
python train.py export --checkpoint-dir ./checkpoints/smoke_cpu --tokenizer-dir ./tokenizer --out ./pretrained/my-model
python train.py generate --load ./pretrained/my-model --prompt "Chapter One."
```

`pretrain --resume DIR --max-steps N` treats N as the TOTAL target step
(resuming a run that stopped at step 750 with `--max-steps 3000` trains
2,250 more), so the cosine learning-rate schedule continues instead of
restarting at zero.

**Reproduce the bundled model** (about an hour on 2 CPU cores):

```bash
python scripts/prepare_corpus.py --out ./data                 # public-domain fiction -> data/train, data/val
python train.py tokenizer train --data ./data/train --vocab-size 4096 --out ./tokenizer
python train.py pretrain --profile pretrained_small --tokenizer-dir ./tokenizer \
    --data ./data/train --val-data ./data/val --seed 1234
```

`--data` accepts individual files, directories (searched recursively for
`.txt`/`.md`/`.csv`/`.parquet`/`.jsonl`/`.ndjson`), and glob patterns
(quote them on Windows), in any combination -- see
`data_pipeline.resolve_data_paths`. A bad/missing/empty path is reported
immediately, before any device or model setup, naming the exact entry.

`--profile <name>` loads `configs/<name>.yaml` -- 14 profiles from
`smoke_cpu` through `multi_node`, plus `long_context_extension` and
`pretrained_small` (the recipe behind the bundled model), plus
`base` (see [Configuration profiles](#configuration-profiles)). Omit it
for `--dry-run` or built-in defaults. `AsterWhiteConfig` in
`asterwhite/config.py` still holds every hyperparameter if you'd rather
edit Python directly.

Any uncaught error prints a short, specific diagnostic (not a wall of
library traceback) and exits non-zero; set `ASTERWHITE_DEBUG=1` for the
full Python traceback.

## Frontier architecture additions

All default OFF/unused -- zero behavior change unless you opt in via
`--profile` or `AsterWhiteConfig`:

| Feature | Config | What it does |
|---|---|---|
| Hierarchical story planner | `GenerationOrchestrator.begin(..., branching=[...])` | Lazily-materialized saga → volume → chapter tree (`HierarchicalPlan`, `PlotCalculator.expand`) replacing a flat, hard-capped-at-64-beats planner -- only the current root-to-chapter path is ever held in memory, which is what makes planning scale from a short story to tens of thousands of chapters. See "Writing a long-form story" below. |
| Sparse mixture-of-experts | `backbone.moe` | Top-k routed experts + optional always-on shared experts (DeepSeek-style) in place of a layer's dense MLP, with a load-balancing auxiliary loss. `asterwhite/core/moe.py`. Enabled on `multi_gpu_fsdp`/`multi_node` (FSDP shards the extra expert parameters); deliberately not on single-GPU tiers -- see those profiles' notes. |
| Long-context RoPE scaling | `backbone.rope_scaling` | `"linear"` / `"ntk"` / `"yarn"` context extension for continued pretraining past a checkpoint's original `max_seq_len`. See `configs/long_context_extension.yaml`. |
| QK-norm | `backbone.qk_norm` | Per-head RMSNorm on query/key before rotation (Gemma 2 / Qwen 3 style) -- a cheap real training-stability improvement at scale. Enabled from `nvidia_single_gpu` up. |
| Multi-token prediction | `backbone.mtp_n_future_tokens` | Extra parallel output heads (Gloeckle et al. 2024) predicting further future offsets from the same hidden state -- better training signal, and the drafter for self-speculative decoding below. |
| Self-speculative decoding | `asterwhite/inference/speculative.py` | Uses the MTP heads as a zero-extra-forward-pass drafter, verified via standard rejection sampling against the full model -- output is provably identical to sequential decoding (see `tests/test_speculative.py`), acceptance rate (and therefore speedup) depends on how well-trained the MTP heads are. |
| Activation checkpointing | `training.activation_checkpointing` | Now actually wired (`AsterWhiteCore.forward_hidden`) -- previously a documented-but-unused config flag. |
| `torch.compile` | `training.compile_model` | Only attempted on CUDA (compilation is lazy and has been observed to hang without a full build toolchain on CPU-only sandboxes); a no-op elsewhere, logged not silent. |
| EMA weights | `training.ema_decay` | Polyak-averaged shadow weights, persisted in/restored from checkpoints (`Trainer.ema_state_dict()`). |
| Fused AdamW | `training.optimizer_fused` | Tries `fused=True`, transparently falls back if unsupported on this device/torch build. |
| Attention sinks | `GenerationOrchestrator(..., attention_sink_size=...)` | StreamingLLM-style: keeps the first few positions' KV permanently across the sliding-window trim, preventing the perplexity spike plain trimming causes on long generations. |

## Writing a long-form story: short stories through saga-scale sagas

**Why this needs a hierarchical plan, not just a bigger number.** A flat
outline (one list of N plan beats) hits two walls fast: the list has to
fit in memory and in a fixed-size learned parameter tensor (this repo's
planner used to cap out at 64 beats total -- fine for one chapter's worth
of scenes, nowhere near enough for a real novel, let alone thousands of
chapters), and nothing about "beat #4,812 of 20,000" tells the model
anything about STRUCTURE -- is it early or late in its volume? Its arc?
`generation_orchestrator.HierarchicalPlan` fixes both: the same small
planner module (`PlotCalculator`) is applied recursively -- saga → volume
→ chapter, or as many levels as you configure -- and only the path from
the root to whichever chapter is CURRENTLY being written is ever
materialized. A story configured for 25,000 chapters costs the same
memory to plan as one configured for 25, until you actually get there.

**How -- a normal short story** (flat, single level):
```bash
python train.py generate --load ./checkpoints/tiny_cpu --tokenizer-dir ./tokenizer \
    --prompt "Chapter One." --branching 12 --chapters 12 --max-new-tokens 800
```
`--branching 12` plans 12 chapters total; `--chapters 12` writes all of
them in this one run.

**How -- a real saga**, spanning many separate runs over real time (this
is the actual target scale: `--branching 20 50` plans 20 volumes × 50
chapters/volume = 1,000 chapters; add a third number for more):
```bash
# Session 1: plan the saga, write the first 20 chapters, save the session
python train.py generate --load ./checkpoints/nvidia_single_gpu --tokenizer-dir ./tokenizer \
    --prompt "Chapter One." --branching 20 50 --chapters 20 --max-new-tokens 1500 \
    --save-checkpoint ./sessions/mysaga.ckpt

# Session 2 (later -- same day, next week, doesn't matter): pick up exactly
# where it left off, no re-planning, no re-reading the whole saga so far
python train.py generate --load ./checkpoints/nvidia_single_gpu \
    --resume-checkpoint ./sessions/mysaga.ckpt --chapters 20 --max-new-tokens 1500 \
    --save-checkpoint ./sessions/mysaga.ckpt
```
Repeat the resume step as many times as needed to reach the planned
chapter count -- each run only holds the current chapter's plan context
and a bounded attention window in memory, not the growing manuscript.

`--branching`/`--prompt` are ignored on `--resume-checkpoint` (the saved
session's own plan and history continue instead); `--chapters` always
means "how many chapters THIS run writes," never the total story length.

## What's here

| Domain | Where it lives | What it actually is |
|---|---|---|
| GB/TB-scale data engine, memory-mapped, rank-aware sharding | `data_pipeline.py` (facade: `asterwhite/data/`) | `MemmapTokenDataset` + `StreamingRecordDataset` (now shards by rank AND DataLoader worker jointly under `torch.distributed`, not worker-only) + `InMemoryTextDataset`. |
| Hierarchical universe state engine | `universe_tracker.py` | `WorldState` with a `validate()` pass run on every load (checks every character/location/faction cross-reference, catching a hand-edited or corrupted save before it drives generation) + `DynamicStateTracker` (GRU-updated per-entity embedding). |
| Agentic character simulator, dual-tier memory | `agent_simulation.py` | `Agent` + `DualTierMemory` + `AgentSimulationLoop`, unchanged from the original architecture. |
| Long-context generator, dynamic-length RoPE, editable checkpoints | `generation_orchestrator.py` | `PlotCalculator` + prose synthesis. RoPE position caches now grow on demand instead of hard-capping generation at the configured `max_seq_len`. `GenerationCheckpoint` now saves through the safe checkpoint format below instead of raw pickle. |
| International-company-grade backbone | `asterwhite/core/` | `AsterWhiteCore`: RoPE + grouped-query attention interleaved with a from-scratch selective state-space block. Sampling now supports top-k and repetition penalty alongside temperature/top-p, plus a context-budget guard. |
| Hardware management | `asterwhite/core/device.py`, `asterwhite/core/autosizer.py` | `DeviceManager`: CPU/CUDA/ROCm/MPS/XPU detection, explicit override, startup capability self-test, precision fallback (logged, never silent). `autosizer`: analytic memory estimate across params/gradients/optimizer state/activations/KV-cache; refuses an oversized config before allocating unless `--allow-oversized-config`. |
| Safe checkpoints | `asterwhite/registry/checkpoint.py` | Atomic (temp file + fsync + rename), SHA-256 checksummed, schema-versioned, `safetensors` instead of pickle for every tensor payload (training checkpoints, generation checkpoints, LoRA adapters, the reward model) -- nothing in this repo calls `torch.load(weights_only=False)` anymore. Tied weights (e.g. `tie_word_embeddings`) are deduplicated before serialization and transparently restored on load. |
| Model registry | `asterwhite/registry/manifest.py` | `RunRecord` (per-run tracking) and `ModelRelease` (release metadata, `PromotionStage` lifecycle, optional HMAC-signed release hash via `ASTERWHITE_SIGNING_KEY`). |
| Real tokenizer | `asterwhite/tokenization/` | Byte-level BPE (`tokenizers` library) with reserved narrative control tokens (`<\|beat\|>`, `<\|character\|>`, `<\|scene\|>`, `<\|revision\|>`, `<\|tool\|>`, `<\|safety\|>`), hash-versioned and tied to checkpoints, plus an explicit embedding-remap path for vocabulary migration. The zero-setup byte-level fallback (`data_pipeline.byte_encode`) still works with no tokenizer trained. |
| Parameter-efficient fine-tuning | `asterwhite/training/lora.py` | LoRA: apply/freeze/train/save/load/merge, targeting any named sub-module. |
| Training objectives for the narrative modules | `asterwhite/training/narrative_objectives.py` | Real, unit-tested loss functions for the planner, tracker, character encoder, revision suite, style checker, and a safety probe -- previously these modules had forward passes and no loss function anywhere. Needs your own paired narrative training examples to actually train against; see the module docstring. |
| Alignment / reward | `asterwhite/alignment/` | `GRPOTrainer` (now clips rewards before advantage normalization) + `HeuristicRewardComposite` (an auditable, inspectable default: style consistency + originality + length + repetition, replacing the old one-line placeholder) + `RewardModel` (trainable, versioned, Bradley-Terry preference training). |
| Distributed, memory-efficient training | `asterwhite/training/trainer.py` | Gradient accumulation + bf16/fp16/fp32 + DDP or FSDP (`training.distributed_strategy`) + a real validation loop + NaN/Inf detection that skips the bad step and raises after too many in a row + full resume (weights, optimizer, RNG state, step count). |
| Inference engine | `asterwhite/inference/engine.py`, `quantization.py` | `InferenceEngine`: loads a registered checkpoint + tokenizer onto the resolved device, thread-safe `generate()`/`generate_stream()`. Optional dynamic INT8 quantization for CPU inference. |
| **Serving API** | `asterwhite/serving/` | FastAPI app speaking three conventions on one port: **OpenAI**-compatible (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, SSE streaming), **Ollama**-compatible (`/api/generate`, `/api/chat`, `/api/tags`, NDJSON streaming), and a **generic** `/generate`. API-key auth (Bearer header, `X-API-Key` header, or `?api_key=`), per-deployment concurrency quotas (429 on overload), CORS, `/metrics` (Prometheus), `/health`/`/healthz`/`/ready`. |
| Safety | `asterwhite/safety/` | Regex/keyword PII scrubber for logs, a heuristic policy-violation filter, a process-wide safe-mode kill switch. Explicitly heuristic, not a trained/red-teamed classifier -- see each module's docstring. |
| Observability | `asterwhite/observability/` | Structured JSON logging, Prometheus metrics (request count/latency, active requests, queue rejections, tokens generated). |
| Character models reusable in RPGs/games | `asterwhite/narrative/character.py` | Unchanged: `CharacterProfile.export_to_game_card()`. |
| Accurate error location | `asterwhite/utils/error_tracing.py` | Unchanged: `AsterWhiteError` + `install()` + `@trace_errors`. |
| Case-sensitive "AsterWhite" name | `asterwhite/config.py` | `MODEL_NAME = "AsterWhite"`. |

## Configuration profiles

`configs/*.yaml`, loaded via `asterwhite/profiles.py`, one per hardware
tier (plus one training-recipe profile), each `extends:` a parent
(defaulting to `base`) and overrides only what differs:

`base`, `smoke_cpu`, `tiny_cpu`, `laptop_low_memory`, `apple_mps`,
`intel_xpu`, `nvidia_single_gpu`, `amd_rocm_single_gpu`, `multi_gpu_ddp`,
`multi_gpu_fsdp`, `multi_node`, `edge_inference`, `production_inference`,
`long_context_extension` (continued-pretraining recipe for extending an
already-trained checkpoint to a longer context via RoPE scaling -- not a
from-scratch tier, see that file's header comment), `pretrained_small`
(the exact recipe that produced `pretrained/asterwhite-small`).

Every profile is validated (`asterwhite.config.validate_config`) before
it can be used -- a bad profile or `--override` fails immediately, before
any model is allocated. CLI flags always win over the profile file.

## Deployment

`deploy/Dockerfile` (CPU by default, GPU via `--build-arg BASE_IMAGE=...`),
`deploy/docker-compose.yml` (single host; serves the bundled pretrained
model out of the box, with three commented lines to serve your own),
`deploy/k8s-deployment.yaml` (Deployment + Service +
HPA). The Docker/compose files are built and run in this repository's own
CPU sandbox as part of development; the Kubernetes manifest is authored
as a starting point and is **not** verified against a live cluster --
review resource limits and the image reference before applying it.

## Honest notes on scope

This ships with one small pretrained model (`pretrained/asterwhite-small`),
trained here, on CPU, with the repository's own tooling -- not downloaded
from anywhere, and not frontier-class. The infrastructure *around* the
architecture is real and tested end-to-end (verified by actually running
`pretrain` → a real checkpoint → `export` → loading that same bundle in
both `generate` and `serve` in this repository's own test suite). A few
things stay explicitly out of scope, because no amount of code changes
them:

- **Training quality**: the bundled model saw ~1.5M tokens of public-domain
  fiction for a few thousand steps. It learned spelling, grammar,
  punctuation, dialogue and period diction, and it does not plan, keep
  characters straight, or hold a plot. A model that writes good prose
  needs far more data, compute and training time than a CPU sandbox has --
  train a bigger profile on your own hardware for that.
- **The story planner / universe tracker are untrained**: the bundled
  model was pretrained as a plain language model. The hierarchical planner,
  state tracker and character encoder are randomly initialised modules
  whose vectors are fed to the backbone as a prefix; measured on held-out
  text, that prefix has no effect on loss (`GenerationOrchestrator`'s
  docstring has the numbers). `--conditioning-scale 0` turns it off.
  Training these needs paired narrative data -- see
  `asterwhite/training/narrative_objectives.py`.
- **The hierarchical planner solves tractability, not craft**: it's what
  makes planning a 20,000-chapter saga computationally possible without
  holding the whole outline in memory. Whether the PROSE stays coherent
  and well-crafted across that many chapters is a training/data problem,
  same as the point above -- no known model, trained by anyone, has
  actually demonstrated tight literary craft sustained at that length; the
  closest real precedent is long-running serialized web fiction, which is
  valued for different things than a tightly-plotted single novel.
- **Reward model / narrative objective weights**: `RewardModel` and the
  narrative loss heads are real, trainable, checkpointable machinery with
  zero preference/paired-narrative data bundled -- train them on your own
  before relying on them over the heuristic defaults.
- **Multi-node elastic restart**: `setup_distributed()` assumes a fixed
  world size for the life of a run; a node failure means stop, fix or
  replace it, relaunch every rank, and `--resume`.
- **MoE and speculative decoding need training to pay off**: enabling
  `backbone.moe` or `backbone.mtp_n_future_tokens` on an untrained model
  changes nothing about output quality or speed by itself -- MoE's routed
  experts specialize (and speculative decoding's acceptance rate climbs)
  only after real training. Both are correct and safe to enable from step
  zero; neither is a shortcut around needing to actually train.
- **ISO/NIST/EU AI Act-style certification, red-teaming, SBOM signing
  against real org PKI**: these are organizational/legal processes, not
  code -- `asterwhite.safety` and `docs/` give you real, working starting
  points (a heuristic filter, a signing hook, card templates), not a
  substitute for actually doing them.
- **ONNX export to mobile runtimes (CoreML/NNAPI/QNN/XNNPACK)**: not
  attempted -- verifying it needs the target hardware/OS, which this
  CPU-Linux sandbox doesn't have. A `torch.onnx.export` path is a
  reasonable next step if you need it.

Two things worth knowing before a long training run, carried over from
the original architecture notes:

- **Tokenization** now has a real, trainable path
  (`python train.py tokenizer train`) in addition to the zero-setup
  byte-level fallback -- swap in the tokenizer's `.encode`/`.decode` via
  `--tokenizer-dir` on any subcommand.
- **A few conditioning mechanisms remain learned-but-untrained
  stand-ins** by design, not oversight: `PassTagEmbedding` and the
  narrative objective heads are wired to be trainable, not pre-loaded
  with instructions or preference data that doesn't exist yet.

## Structure

```
AsterWhite/
├── train.py                       # CLI: pretrain / align / generate / serve / export / evaluate / tokenizer / selftest
├── data_pipeline.py                # MB in-memory, GB streaming (rank-aware), GB/TB memmap
├── universe_tracker.py             # cosmic/local/character graph + neural state + validate()
├── agent_simulation.py             # per-character agents, dual-tier memory
├── generation_orchestrator.py      # plan/prose split, long-context, safe checkpoints
├── asterwhite/
│   ├── config.py                   # AsterWhiteConfig, validate_config, schema versioning
│   ├── profiles.py                 # configs/*.yaml loader with inheritance
│   ├── core/                       # attention.py, ssm.py, moe.py, norm.py, ffn.py, layers.py, model.py,
│   │                                # device.py, autosizer.py
│   ├── data/                       # facade over data_pipeline.py
│   ├── tokenization/                # real BPE tokenizer, control tokens, migration
│   ├── narrative/                  # character.py, style_consistency.py, revision.py
│   ├── training/                   # trainer.py, lora.py, narrative_objectives.py
│   ├── alignment/                  # grpo.py, reward_model.py
│   ├── inference/                  # engine.py, quantization.py, speculative.py
│   ├── serving/                    # app.py, routes.py, schemas.py, auth.py, quotas.py
│   ├── safety/                     # pii_scrubber.py, policy_filter.py, safe_mode.py
│   ├── observability/               # logging_config.py, metrics.py
│   ├── registry/                   # checkpoint.py (safe I/O), manifest.py (run/release tracking), pretrained.py (bundle discovery)
│   └── utils/error_tracing.py
├── pretrained/asterwhite-small/    # the bundled model: weights + config + tokenizer + MODEL_CARD.md
├── scripts/prepare_corpus.py       # builds the public-domain fiction corpus the bundled model was trained on
├── configs/                        # 14 profiles (12 hardware tiers + long_context_extension + pretrained_small) + base.yaml
├── deploy/                         # Dockerfile, docker-compose.yml, k8s-deployment.yaml
├── docs/                           # model/data card templates, ADR template, incident response
└── tests/                          # run via `python train.py selftest`
```

## Requirements

See `requirements.txt` / `pyproject.toml`. CPU-only works (used for all
testing above); a CUDA GPU is picked up automatically via `DeviceManager`
and enables Flash-Attention-2/3 dispatch plus real mixed-precision
speedups. ROCm/MPS/XPU are detected and wired the same way but not
hardware-validated in this repository's CPU-only environment.
