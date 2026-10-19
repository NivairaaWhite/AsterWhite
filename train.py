#!/usr/bin/env python3
"""
AsterWhite :: train.py -- CLI entry point.

    python train.py doctor
    python train.py selftest
    python train.py tokenizer train --data corpus.txt --vocab-size 32000 --out ./tokenizer
    python train.py pretrain --profile tiny_cpu --data mycorpus.txt
    python train.py pretrain --profile tiny_cpu --data ./my_book_folder/
    python train.py pretrain --profile nvidia_single_gpu --data mycorpus.txt --resume ./checkpoints/nvidia_single_gpu
    python train.py align --profile tiny_cpu --load ./checkpoints/tiny_cpu
    python train.py generate --prompt "Chapter One."            # bundled pretrained model, no other flags needed
    python train.py generate --load ./checkpoints/tiny_cpu --tokenizer-dir ./tokenizer --prompt "Chapter One."
    python train.py serve                                       # serves the bundled pretrained model
    python train.py serve --checkpoint-dir ./checkpoints/tiny_cpu --tokenizer-dir ./tokenizer --profile production_inference
    python train.py export --checkpoint-dir ./checkpoints/tiny_cpu --tokenizer-dir ./tokenizer --out ./pretrained/mymodel
    python train.py evaluate --data ./data/val                  # validation loss / perplexity of the bundled model

Run `python train.py doctor` FIRST on a new machine -- it checks Python/
torch/dependency versions, every hardware backend, disk space, and runs
one real tiny forward+backward pass, printing a clear PASS/FAIL/WARN per
check instead of failing whichever real command you try first, deeper in
the stack, less specifically.

Every subcommand accepts --profile (see configs/*.yaml, asterwhite.profiles.list_profiles())
and --device (auto/cpu/cuda/rocm/mps/xpu -- default auto-detects).
--dry-run (pretrain/align/generate) runs on a freshly-initialized, UNTRAINED
model for pipeline verification only -- see each command's help text.
Omitting --load on generate/align/serve uses the bundled pretrained model
(pretrained/asterwhite-small) when it is present; a checkpoint directory is
self-describing (its stored config gives the architecture, and a tokenizer/
subfolder next to it is picked up automatically), so --profile and
--tokenizer-dir are optional there.

--data (pretrain/tokenizer train) accepts, in any combination: individual
files (.txt/.md/.csv/.parquet/.jsonl/.ndjson), a DIRECTORY (searched
recursively for those extensions -- "point it at a folder of books" needs
no other flag), or a glob pattern ("books/*.txt", quote it on Windows) --
see data_pipeline.resolve_data_paths. A bad/missing/empty path is reported
immediately, before any device or model setup, naming the exact entry.

Any uncaught error prints a short, specific diagnostic (what failed, and
the exact AsterWhite file/line, not a wall of library traceback) and exits
non-zero rather than dumping a raw stack trace; set ASTERWHITE_DEBUG=1 to
get the full Python traceback back (e.g. for filing a bug report).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile

import numpy as np

import torch
from torch.utils.data import DataLoader, DistributedSampler

from asterwhite.alignment import GRPOTrainer, HeuristicRewardComposite, HeuristicRewardWeights, RewardModel
from asterwhite.config import AsterWhiteConfig, MODEL_NAME, validate_config
from asterwhite.core.autosizer import estimate_memory, guard_or_raise
from asterwhite.core.device import DeviceManager
from asterwhite.core.model import AsterWhiteCore
from asterwhite.narrative import OriginalityGuard, StyleConsistencyChecker
from asterwhite.profiles import list_profiles, load_profile
from asterwhite.registry import RunRecord
from asterwhite.registry.checkpoint import load_checkpoint as registry_load_checkpoint
from asterwhite.registry.checkpoint import resolve_checkpoint_name, save_checkpoint as registry_save_checkpoint
from asterwhite.registry.pretrained import (
    TOKENIZER_SUBDIR, adopt_checkpoint_config, bundled_tokenizer_dir, default_bundle_dir, effective_context_length, is_checkpoint_dir,
)
from asterwhite.tokenization import AsterWhiteTokenizer
from asterwhite.training import (
    Trainer, apply_lora, load_lora_adapter, mark_only_lora_as_trainable,
    save_lora_adapter, setup_distributed,
)
from asterwhite.utils.error_tracing import AsterWhiteError, get_logger, install
from data_pipeline import (
    build_dataset, build_tokenized_binary, byte_encode, iter_raw_text_for_tokenizer_training, resolve_data_paths,
)
from generation_orchestrator import GenerationOrchestrator, PlotCalculator
from universe_tracker import DynamicStateTracker, WorldState

install()
logger = get_logger("asterwhite.cli")

UNTRAINED_WARNING = (
    "############################################################\n"
    "# WARNING: no --load checkpoint was given.                 #\n"
    "# This model has RANDOM, UNTRAINED weights. Output will be  #\n"
    "# incoherent noise -- this run is for pipeline verification #\n"
    "# only, not real generation. Train first, then pass --load. #\n"
    "############################################################"
)


def _add_common_args(p: argparse.ArgumentParser, need_data: bool = False) -> None:
    p.add_argument("--profile", default=None, choices=list_profiles() or None,
                    help="load configs/<name>.yaml (see configs/); omit to use --dry-run or built-in defaults")
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "rocm", "mps", "xpu"],
                    help="override auto-detected device")
    p.add_argument("--allow-oversized-config", action="store_true",
                    help="proceed even if asterwhite.core.autosizer predicts this won't fit in memory")
    if need_data:
        p.add_argument("--data", nargs="+",
                        help="corpus file(s) (.txt/.md/.csv/.parquet/.jsonl/.ndjson), directories "
                             "(searched recursively), and/or glob patterns, in any combination")
        p.add_argument("--val-data", nargs="+", default=None,
                        help="held-out validation corpus (optional; same accepted forms as --data)")
    p.add_argument("--tokenizer-dir", default=None,
                    help="directory from `tokenizer train` output; omit to use the zero-setup byte-level fallback")
    p.add_argument("--dry-run", action="store_true", help="tiny config, no real data required, seconds on CPU")


def _resolve_device_and_config(args: argparse.Namespace, need_training: bool = True) -> tuple:
    device_info = DeviceManager.resolve(None if args.device in (None, "auto") else args.device)
    if args.profile:
        cfg = load_profile(args.profile)
    else:
        cfg = AsterWhiteConfig.tiny_for_testing() if args.dry_run else AsterWhiteConfig()
    return cfg, device_info


def _resolve_tokenizer(args: argparse.Namespace, cfg: AsterWhiteConfig, with_prompt_encoder: bool = False):
    """(encode_fn, decode_fn, tokenizer_hash) -- or, with_prompt_encoder=True,
    (encode_fn, decode_fn, tokenizer_hash, prompt_encode_fn).

    encode_fn wraps text as a whole DOCUMENT ([BOS] text [EOS]) -- right for
    building training data. prompt_encode_fn is for text the model will
    CONTINUE: [BOS] text with no trailing EOS (an EOS there tells the model
    the document is already over)."""
    if args.tokenizer_dir:
        tok = AsterWhiteTokenizer.load(args.tokenizer_dir)
        if tok.vocab_size != cfg.backbone.vocab_size:
            logger.warning(f"tokenizer vocab_size={tok.vocab_size} overrides config's "
                            f"backbone.vocab_size={cfg.backbone.vocab_size}")
            cfg.backbone.vocab_size = tok.vocab_size
        result = (tok.encode, tok.decode, tok.compute_hash())
        return result + (tok.encode_prompt,) if with_prompt_encoder else result
    decode = lambda ids: bytes(i % 256 for i in ids).decode("utf-8", errors="replace")  # noqa: E731
    result = (byte_encode, decode, None)
    return result + (byte_encode,) if with_prompt_encoder else result


def _seed_everything(seed) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def _attach_checkpoint_environment(args: argparse.Namespace, cfg: AsterWhiteConfig, load_dir, *,
                                   keep_profile_settings=None):
    """Makes a checkpoint directory self-describing for generate/align: its
    stored config supplies the architecture (an explicit --profile still
    overrides serving/training settings, never the architecture), and a
    tokenizer/ subfolder next to it is used when --tokenizer-dir is omitted.
    keep_profile_settings=True keeps EVERYTHING but the architecture from `cfg`
    (used by `pretrain --load-base`, where the base model's old training recipe
    should not silently become the fine-tune's). Returns the config to use."""
    if not load_dir:
        return cfg
    if keep_profile_settings is None:
        keep_profile_settings = args.profile is not None
    cfg = adopt_checkpoint_config(cfg, load_dir, keep_profile_settings=keep_profile_settings)
    if args.tokenizer_dir is None:
        bundled = bundled_tokenizer_dir(load_dir)
        if bundled:
            logger.info(f"using the tokenizer bundled with the checkpoint: {bundled}")
            args.tokenizer_dir = bundled
    return cfg


def _save_tokenizer_next_to(checkpoint_dir: str, tokenizer_dir) -> None:
    """Copies the tokenizer into <checkpoint_dir>/tokenizer so the checkpoint
    directory is self-contained (see registry.pretrained)."""
    if not tokenizer_dir:
        return
    dest = os.path.join(checkpoint_dir, TOKENIZER_SUBDIR)
    if os.path.abspath(dest) != os.path.abspath(tokenizer_dir):
        shutil.copytree(tokenizer_dir, dest, dirs_exist_ok=True)


def _guard_memory(cfg: AsterWhiteConfig, device_info, override: bool) -> None:
    estimate = estimate_memory(cfg, device_info)
    logger.info(f"memory estimate: {estimate.report()}")
    guard_or_raise(estimate, override=override)


# --------------------------------------------------------------------------- #
# tokenizer
# --------------------------------------------------------------------------- #

def cmd_tokenizer_train(args: argparse.Namespace) -> None:
    tok = AsterWhiteTokenizer.train(args.data, vocab_size=args.vocab_size, min_frequency=args.min_frequency,
                                     text_column=args.text_column)
    # Sampled from the RESOLVED files (not args.data directly -- a raw
    # --data argument can be a directory or glob pattern, which open()
    # cannot read) via the same format-aware text extraction training
    # itself used, so this is a real, representative sample either way.
    resolved = resolve_data_paths(args.data)
    sample_text = next(iter_raw_text_for_tokenizer_training(resolved[:1], text_column=args.text_column), "")
    ok = tok.round_trip_ok([sample_text]) if sample_text.strip() else True
    if not ok:
        raise AsterWhiteError("tokenizer failed its own round-trip check (encode->decode != original) -- "
                               "refusing to save a broken tokenizer.")
    tok.save(args.out)
    print(f"tokenizer saved -> {args.out} (vocab_size={tok.vocab_size}, hash={tok.compute_hash()[:12]}...)")


def cmd_tokenizer_inspect(args: argparse.Namespace) -> None:
    tok = AsterWhiteTokenizer.load(args.tokenizer_dir)
    print(json.dumps({"vocab_size": tok.vocab_size, "hash": tok.compute_hash(),
                       "bos_id": tok.bos_id, "eos_id": tok.eos_id, "pad_id": tok.pad_id}, indent=2))


def cmd_tokenizer_prepare_binary(args: argparse.Namespace) -> None:
    if args.tokenizer_dir:
        tok = AsterWhiteTokenizer.load(args.tokenizer_dir)
        encode_fn, vocab_size = tok.encode, tok.vocab_size
    else:
        encode_fn, vocab_size = byte_encode, 256

    dtype = args.dtype
    if dtype is None:
        # uint16 tops out at 65535 -- must exceed the tokenizer's actual
        # vocab_size, not just be "probably big enough": every id up to
        # vocab_size - 1 has to fit, or build_tokenized_binary will refuse
        # (rather than silently overflow) the first time a high-id token
        # actually appears in the corpus.
        dtype = "uint16" if vocab_size <= 65536 else "uint32"
        logger.info(f"--dtype not given -- auto-selected {dtype!r} for tokenizer vocab_size={vocab_size}")

    out_path = build_tokenized_binary(args.data, args.out, encode_fn=encode_fn, dtype=dtype,
                                       text_column=args.text_column)
    print(f"pre-tokenized binary saved -> {out_path} ({os.path.getsize(out_path):,} bytes, dtype={dtype})")
    print(f"use it with: python train.py pretrain --data {out_path} ...  "
          f"(auto-detected as pre-tokenized from the .bin extension)")


# --------------------------------------------------------------------------- #
# pretrain
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _init_narrative_style_seed(cfg: AsterWhiteConfig, device: torch.device) -> StyleConsistencyChecker:
    checker = StyleConsistencyChecker(style_dim=cfg.backbone.d_model, vocab_size=cfg.backbone.vocab_size).to(device)
    checker.update_centroid(torch.randint(0, cfg.backbone.vocab_size, (2, 32)).to(device))
    return checker


def cmd_pretrain(args: argparse.Namespace) -> None:
    # Resolved/validated FIRST, before device detection, tokenizer setup, or
    # model construction: a bad --data path is the single most common first
    # mistake, and should fail in milliseconds with a specific message
    # instead of after everything else has already spun up.
    if args.data:
        args.data = resolve_data_paths(args.data)
    if args.val_data:
        args.val_data = resolve_data_paths(args.val_data)

    cfg, device_info = _resolve_device_and_config(args)
    if args.load_base and is_checkpoint_dir(args.load_base):
        # Fine-tuning from a model: its weights only fit ITS architecture, so that comes from the
        # checkpoint (and a bundled tokenizer/ is used if --tokenizer-dir is omitted). Training
        # settings stay whatever --profile / the defaults say -- a fine-tune should not silently
        # inherit the base model's pretraining recipe.
        cfg = _attach_checkpoint_environment(args, cfg, args.load_base, keep_profile_settings=True)
    if args.max_steps is not None:
        cfg.training.max_steps = args.max_steps
    encode_fn, _decode_fn, tokenizer_hash = _resolve_tokenizer(args, cfg)
    validate_config(cfg)
    _guard_memory(cfg, device_info, args.allow_oversized_config)
    _seed_everything(getattr(args, "seed", None))

    rank, world_size, local_rank = setup_distributed()
    distributed = world_size > 1
    model = AsterWhiteCore(cfg.backbone)

    if args.load_base:
        # Base-weights-only load, on the PLAIN model, before any LoRA
        # wrapping and before Trainer/the optimizer exist: this is "start a
        # NEW run from an existing checkpoint's weights" (step resets to
        # 0), not "resume THIS run" -- no optimizer/RNG/step state carries
        # over. Loading after LoRA-wrapping would fail: apply_lora renames
        # <layer>.weight to <layer>.base.weight, which would no longer
        # match a plain checkpoint's keys.
        tensors, _meta = registry_load_checkpoint(args.load_base, expected_tokenizer_hash=tokenizer_hash)
        model_sd = {k[len("model."):]: v for k, v in tensors.items() if k.startswith("model.")}
        model.load_state_dict(model_sd, strict=True)
        logger.info(f"loaded base weights from {args.load_base} for a fresh fine-tune run (step reset to 0)")

    if args.lora:
        # Must happen before Trainer() below constructs its optimizer, in
        # every case, or the optimizer would capture the wrong parameter
        # set (and, for --resume, the checkpoint's LoRA-shaped keys
        # wouldn't match an unwrapped model at load time):
        #   --load-base --lora : wrap the just-loaded base weights (new run)
        #   --resume     --lora : wrap to match a checkpoint that already
        #                         has LoRA-shaped keys from a prior run
        #   --lora alone        : LoRA from random init, step 0
        apply_lora(model, target_substrings=tuple(args.lora_targets), r=args.lora_r, alpha=args.lora_alpha)
        mark_only_lora_as_trainable(model)

    trainer = Trainer(model, cfg.training, device_info.device, distributed=distributed, local_rank=local_rank,
                       world_size=world_size, tokenizer_hash=tokenizer_hash)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    run_record = RunRecord.start(run_id=f"pretrain-{os.getpid()}", config=cfg.to_dict(), operator=args.operator,
                                  tokenizer_version=tokenizer_hash, hardware_environment=device_info.as_manifest_dict())

    if args.dry_run and not args.data:
        data_paths = None
    else:
        if not args.data:
            raise AsterWhiteError("--data is required unless --dry-run is also given with no --data")
        data_paths = args.data

    def _make_loader(paths, is_train: bool) -> DataLoader:
        ds = build_dataset(paths, block_size=cfg.backbone.max_seq_len, encode_fn=encode_fn,
                            rank=rank, world_size=world_size)
        is_iterable = not hasattr(ds, "__len__")
        if is_iterable:
            # StreamingRecordDataset (torch.utils.data.IterableDataset) --
            # automatic for any corpus over data_pipeline.build_dataset's
            # mb_threshold (256MB default), which real training data is
            # essentially always well past, so this is the COMMON path,
            # not an edge case. PyTorch's DataLoader raises a ValueError if
            # `shuffle` is passed AT ALL here (even shuffle=False) or if a
            # `sampler`/`batch_sampler` is given -- ordering and rank/
            # worker-aware sharding for this dataset type is entirely the
            # dataset's own __iter__'s job, already handled by the rank/
            # world_size passed into build_dataset above (see
            # StreamingRecordDataset's own docstring for its sharding
            # logic), not something a DataLoader-level sampler can express
            # against an object that has no __len__/random access at all.
            return DataLoader(ds, batch_size=cfg.training.micro_batch_size)
        sampler = None
        if distributed:
            sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=is_train)
        return DataLoader(ds, batch_size=cfg.training.micro_batch_size, sampler=sampler,
                           shuffle=(is_train and sampler is None))

    if data_paths:
        train_loader = _make_loader(data_paths, is_train=True)
        val_loader = _make_loader(args.val_data, is_train=False) if args.val_data else None
    else:
        # --dry-run, no --data: synthetic batches, exercises the exact same
        # Trainer.train_step code path with zero setup.
        train_loader = None
        val_loader = None

    accum_steps = max(1, cfg.training.batch_size // max(1, cfg.training.micro_batch_size))
    # --max-steps / training.max_steps is the TOTAL step the run ends at -- the
    # same absolute horizon Trainer._lr_for_step's cosine schedule decays to
    # zero over -- not "how many more steps to take". Counting from zero on a
    # --resume used to run max_steps ADDITIONAL steps, all of the ones past
    # the horizon at learning rate 0 (wasted compute, no learning).
    if args.max_steps is not None:
        target_step = args.max_steps
    elif args.dry_run:
        target_step = trainer.step_count + 3   # a dry run is always "a few more steps"
    else:
        target_step = cfg.training.max_steps
    steps_to_run = max(0, target_step - trainer.step_count)
    if steps_to_run == 0 and rank == 0:
        print(f"nothing to do: already at step {trainer.step_count} >= target step {target_step} "
              f"(raise --max-steps to train further)")
    elif trainer.step_count > 0 and rank == 0:
        logger.info(f"resumed at step {trainer.step_count}; training {steps_to_run} more step(s) to reach "
                    f"target step {target_step}")
    train_iter = iter(train_loader) if train_loader is not None else None
    epochs_completed = 0

    def _next_micro_batches():
        """Cycles train_loader when its iterator is exhausted -- pretraining
        is step-based, not epoch-based; a dataset smaller than
        steps_to_run * accum_steps * micro_batch_size tokens should train
        for multiple passes, not silently stop early with no indication
        why. Only genuinely empty data (0 batches on the very first pull)
        is treated as an error."""
        nonlocal train_iter, epochs_completed
        micros = []
        for _ in range(accum_steps):
            if train_iter is None:
                micros.append(torch.randint(0, cfg.backbone.vocab_size,
                                             (cfg.training.micro_batch_size, cfg.backbone.max_seq_len)))
                continue
            try:
                micros.append(next(train_iter))
            except StopIteration:
                epochs_completed += 1
                train_iter = iter(train_loader)
                try:
                    micros.append(next(train_iter))
                except StopIteration:
                    raise AsterWhiteError(f"train_loader yielded ZERO batches even from a fresh iterator "
                                           f"-- --data points at no usable examples for block_size="
                                           f"{cfg.backbone.max_seq_len} (check the corpus isn't empty/"
                                           f"shorter than one block).")
        return micros

    for _ in range(steps_to_run):
        micros = _next_micro_batches()
        stats = trainer.train_step(micros)
        run_record.record_metric("last_train_loss", stats.loss)

        if val_loader is not None and trainer.step_count % cfg.training.eval_every == 0:
            eval_stats = trainer.evaluate(val_loader)
            run_record.record_metric("last_val_loss", eval_stats.val_loss)
            if rank == 0:
                trainer.save_checkpoint(cfg.training.checkpoint_dir, config_dict=cfg.to_dict(),
                                         extra_metadata={"val_loss": eval_stats.val_loss})

        if rank == 0 and trainer.step_count % cfg.training.checkpoint_every == 0:
            trainer.save_checkpoint(cfg.training.checkpoint_dir, config_dict=cfg.to_dict())
            run_record.record_artifact(cfg.training.checkpoint_dir)

    if rank == 0:
        trainer.save_checkpoint(cfg.training.checkpoint_dir, config_dict=cfg.to_dict())
        if args.lora:
            save_lora_adapter(model, os.path.join(cfg.training.checkpoint_dir, "lora_adapter"))
        run_record.save(os.path.join(cfg.training.checkpoint_dir, "run_record.json"))
        _save_tokenizer_next_to(cfg.training.checkpoint_dir, args.tokenizer_dir)
        print(f"pretrain complete -> {cfg.training.checkpoint_dir} (step {trainer.step_count}, "
              f"{epochs_completed} full data pass(es) completed)")


# --------------------------------------------------------------------------- #
# align (GRPO)
# --------------------------------------------------------------------------- #

def cmd_align(args: argparse.Namespace) -> None:
    cfg, device_info = _resolve_device_and_config(args)
    load_dir = args.load
    if load_dir is None and not args.dry_run:
        load_dir = default_bundle_dir()
        if load_dir:
            logger.info(f"no --load given -- aligning the bundled pretrained model at {load_dir}")
    if load_dir is None and not args.dry_run:
        raise AsterWhiteError("align refuses to run on a freshly-initialized policy by default -- GRPO's "
                               "KL-to-reference anchor presupposes a reasonably-capable starting point. "
                               "Pass --load <checkpoint_dir> (from a prior `pretrain` run), or pass --dry-run "
                               "to explicitly acknowledge this is a pipeline-verification run on random weights. "
                               "(No bundled pretrained model was found to default to.)")
    cfg = _attach_checkpoint_environment(args, cfg, load_dir)
    encode_fn, _, tokenizer_hash, prompt_encode_fn = _resolve_tokenizer(args, cfg, with_prompt_encoder=True)
    validate_config(cfg)
    _guard_memory(cfg, device_info, args.allow_oversized_config)

    cfg.training.ema_decay = None            # alignment starts from the weights only; see cmd_evaluate
    model = AsterWhiteCore(cfg.backbone).to(device_info.device)
    if load_dir:
        trainer_for_load = Trainer(model, cfg.training, device_info.device, tokenizer_hash=tokenizer_hash)
        trainer_for_load.load_checkpoint(load_dir, restore_rng=False)
    else:
        print(UNTRAINED_WARNING, file=sys.stderr)

    style_checker = _init_narrative_style_seed(cfg, device_info.device)
    originality_guard = OriginalityGuard(n=4)

    if args.reward_model_dir:
        reward_model = RewardModel.load(args.reward_model_dir, expected_tokenizer_hash=tokenizer_hash).to(device_info.device)
        from asterwhite.training.narrative_objectives import pooled_hidden_states

        def reward_fn(prompt_ids, completion_ids):
            pooled = pooled_hidden_states(model, completion_ids.unsqueeze(0))
            return reward_model(pooled).squeeze(0)
        logger.info(f"using trained RewardModel from {args.reward_model_dir}")
    else:
        reward_fn = HeuristicRewardComposite(style_checker, originality_guard, HeuristicRewardWeights())
        logger.info("using HeuristicRewardComposite (no --reward-model-dir given) -- audit its "
                     "score_components() output before trusting it for anything beyond experimentation")

    grpo = GRPOTrainer(model, reward_fn, cfg.grpo, device=device_info.device)
    prompts = args.prompts or ["Write the opening of a story."]
    steps_to_run = 2 if args.dry_run else args.steps
    for step in range(steps_to_run):
        for prompt in prompts:
            prompt_ids = torch.tensor([prompt_encode_fn(prompt)], device=device_info.device)
            stats = grpo.step(prompt_ids)
            logger.info(f"align step {step}: reward={stats.mean_reward:.3f} kl={stats.kl:.4f} "
                        f"clip_fraction={stats.clip_fraction:.2f}")

    out_dir = args.out or os.path.join(cfg.training.checkpoint_dir, "aligned")
    os.makedirs(out_dir, exist_ok=True)
    trainer_for_save = Trainer(model, cfg.training, device_info.device, tokenizer_hash=tokenizer_hash)
    trainer_for_save.save_checkpoint(out_dir, config_dict=cfg.to_dict())
    _save_tokenizer_next_to(out_dir, args.tokenizer_dir)
    print(f"align complete -> {out_dir}")


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #

def cmd_generate(args: argparse.Namespace) -> None:
    cfg, device_info = _resolve_device_and_config(args)
    load_dir = args.load
    if load_dir is None and not args.dry_run:
        load_dir = default_bundle_dir()
        if load_dir:
            logger.info(f"no --load given -- using the bundled pretrained model at {load_dir}")

    if load_dir is None and not args.dry_run:
        raise AsterWhiteError("generate refuses to construct a fresh, untrained model by default (spec: "
                               "generation MUST load trained weights). Pass --load <checkpoint_dir>, or "
                               "pass --dry-run to explicitly acknowledge random weights for pipeline testing. "
                               "(No bundled pretrained model was found to default to -- it is expected at "
                               "pretrained/asterwhite-small.)")

    cfg = _attach_checkpoint_environment(args, cfg, load_dir)
    encode_fn, decode_fn, tokenizer_hash, prompt_encode_fn = _resolve_tokenizer(args, cfg, with_prompt_encoder=True)
    validate_config(cfg)
    _seed_everything(args.seed)

    model = AsterWhiteCore(cfg.backbone).to(device_info.device)
    if load_dir:
        tensors, _ = registry_load_checkpoint(load_dir, expected_tokenizer_hash=tokenizer_hash)
        model_sd = {k[len("model."):]: v for k, v in tensors.items() if k.startswith("model.")}
        model.load_state_dict(model_sd, strict=True)
        logger.info(f"loaded trained weights from {load_dir}")
    else:
        print(UNTRAINED_WARNING, file=sys.stderr)

    if args.lora_adapter:
        # Must happen AFTER the base --load above (so LoRALinear wraps the
        # already-loaded base nn.Linear layers, matching how apply_lora was
        # used at training time) and BEFORE .eval()/generation.
        try:
            apply_lora(model, target_substrings=tuple(args.lora_targets), r=args.lora_r, alpha=args.lora_alpha)
            load_lora_adapter(model, args.lora_adapter)
        except AsterWhiteError:
            raise
        except Exception as exc:
            raise AsterWhiteError(
                f"failed to apply --lora-adapter '{args.lora_adapter}' on top of --load '{load_dir}': "
                f"{type(exc).__name__}: {exc}\nThe most common cause: --load is ALREADY a checkpoint from a "
                f"`pretrain --lora` run (its own state dict already has the LoRA weights merged in) -- in that "
                f"case load it alone, with no --lora-adapter. --lora-adapter is only for combining a PLAIN base "
                f"checkpoint with a separately-saved adapter trained via `--load-base <that checkpoint> --lora`.",
                cause=exc,
            ) from exc
        logger.info(f"applied LoRA adapter from {args.lora_adapter}")
    model.eval()

    world_state = WorldState(universe_id="u0")
    # Same device as the backbone, and eval() like it: PlotCalculator wraps a
    # TransformerEncoder with dropout, which would otherwise stay ACTIVE at
    # generation time and make the story plan non-deterministic (and a
    # resumed run re-expand different sibling beats than the original run).
    tracker = DynamicStateTracker(world_state_dim=cfg.narrative.world_state_dim).to(device_info.device).eval()
    plot = PlotCalculator(plan_dim=cfg.narrative.plan_dim,
                          max_children=cfg.narrative.max_plan_beats).to(device_info.device).eval()

    if args.resume_checkpoint:
        orch = GenerationOrchestrator.resume_from_disk(args.resume_checkpoint, model, plot, tracker,
                                                         max_window=cfg.backbone.max_seq_len,
                                                         conditioning_scale=args.conditioning_scale)
        print(f"resumed at {orch.plan.current_label()} ({orch.token_ids.shape[1]} tokens so far)", file=sys.stderr)
    else:
        orch = GenerationOrchestrator(model, plot, world_state, tracker, max_window=cfg.backbone.max_seq_len,
                                      conditioning_scale=args.conditioning_scale)
        prompt_ids = torch.tensor([prompt_encode_fn(args.prompt)], device=device_info.device)
        premise_embedding = torch.randn(cfg.narrative.plan_dim,
                                         device=device_info.device)  # unbatched: begin() unsqueezes
        orch.begin(premise_embedding, prompt_ids, branching=args.branching)
        if len(args.branching) > 1:
            logger.info(f"hierarchical plan: branching={args.branching} -> "
                         f"{orch.plan.total_chapters():,} chapters total")

    max_new = args.max_new_tokens
    if max_new + orch.token_ids.shape[1] > cfg.inference.max_total_tokens:
        max_new = max(1, cfg.inference.max_total_tokens - orch.token_ids.shape[1])
        logger.warning(f"max_new_tokens reduced to {max_new} to respect inference.max_total_tokens="
                        f"{cfg.inference.max_total_tokens}")

    chapters_written = 0
    for _ in range(max(1, args.chapters)):
        label = orch.plan.current_label()
        out_ids = orch.generate_next(max_new_tokens=max_new, temperature=args.temperature, top_p=args.top_p,
                                      top_k=(args.top_k or None), repetition_penalty=args.repetition_penalty)
        text = decode_fn(out_ids[0].tolist())
        if args.chapters > 1:
            print(f"\n=== {label} ===")
        print(text)
        chapters_written += 1
        if args.chapters > 1 and not orch.advance_chapter():
            print(f"\n(plan complete -- all {orch.plan.total_chapters():,} planned chapters written)",
                  file=sys.stderr)
            break

    if args.save_checkpoint:
        orch.pause_to_disk(args.save_checkpoint)
        print(f"generation session saved -> {args.save_checkpoint} "
              f"({chapters_written} chapter(s) written this run) -- continue later with: "
              f"generate --resume-checkpoint {args.save_checkpoint} --load {load_dir or '<same --load>'} "
              f"--chapters <N>", file=sys.stderr)


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #

def cmd_evaluate(args: argparse.Namespace) -> None:
    """Held-out next-token loss / perplexity of a checkpoint on text files --
    the same number `pretrain --val-data` reports during training, so a
    model's quality claim can be re-checked by anyone with the data."""
    data = resolve_data_paths(args.data)
    cfg, device_info = _resolve_device_and_config(args)
    load_dir = args.load or default_bundle_dir()
    if load_dir is None:
        raise AsterWhiteError("evaluate needs a model: pass --load <checkpoint_dir> (no bundled pretrained model "
                               "was found to default to).")
    cfg = _attach_checkpoint_environment(args, cfg, load_dir)
    encode_fn, _decode_fn, tokenizer_hash = _resolve_tokenizer(args, cfg)
    validate_config(cfg)

    cfg.training.ema_decay = None            # loading weights only -- no EMA shadow copy to restore or warn about
    model = AsterWhiteCore(cfg.backbone)
    trainer = Trainer(model, cfg.training, device_info.device, tokenizer_hash=tokenizer_hash)
    trainer.load_checkpoint(load_dir, restore_rng=False)
    dataset = build_dataset(data, block_size=cfg.backbone.max_seq_len, encode_fn=encode_fn)
    stats = trainer.evaluate(DataLoader(dataset, batch_size=args.batch_size))
    print(json.dumps({"checkpoint": load_dir, "step": trainer.step_count, "block_size": cfg.backbone.max_seq_len,
                      "val_loss": round(stats.val_loss, 4), "perplexity": round(stats.perplexity, 2)}, indent=2))


# --------------------------------------------------------------------------- #
# export -- training checkpoint -> self-contained inference bundle
# --------------------------------------------------------------------------- #

def cmd_export(args: argparse.Namespace) -> None:
    """Packages a TRAINING checkpoint into a self-contained, inference-only
    model bundle (see asterwhite.registry.pretrained): just the weights
    (optimizer state, RNG state and EMA shadow copies stay behind -- they
    make a training checkpoint ~4x bigger and are useless for inference),
    the exact config they were trained with, and the tokenizer. The bundle
    is test-loaded and run once before anything is written, so a broken
    export fails here rather than at the user's first `generate`."""
    tok_dir = args.tokenizer_dir or bundled_tokenizer_dir(args.checkpoint_dir)
    if not tok_dir:
        raise AsterWhiteError(f"no tokenizer to bundle: pass --tokenizer-dir, or put one at "
                               f"{os.path.join(args.checkpoint_dir, TOKENIZER_SUBDIR)}")
    tok = AsterWhiteTokenizer.load(tok_dir)
    resolved_name = resolve_checkpoint_name(args.checkpoint_dir, args.name, args.which)
    tensors, meta = registry_load_checkpoint(args.checkpoint_dir, name=resolved_name,
                                              expected_tokenizer_hash=tok.compute_hash())
    stored = meta.get("config")
    if not stored:
        raise AsterWhiteError(f"checkpoint {resolved_name!r} has no stored config, so the bundle could not say "
                               f"what architecture its weights belong to -- re-save it from a Trainer that was "
                               f"given config_dict (cmd_pretrain always does).")
    cfg = AsterWhiteConfig.from_dict(stored)
    validate_config(cfg)
    if tok.vocab_size != cfg.backbone.vocab_size:
        raise AsterWhiteError(f"tokenizer vocab_size={tok.vocab_size} != checkpoint backbone.vocab_size="
                               f"{cfg.backbone.vocab_size}")
    # The bundle's stored serving defaults must never ask for more than the model's context.
    context = effective_context_length(cfg.backbone)
    cfg.inference.max_total_tokens = min(cfg.inference.max_total_tokens, context)
    cfg.inference.default_max_new_tokens = min(cfg.inference.default_max_new_tokens, cfg.inference.max_total_tokens)
    cfg.training.ema_decay = None            # a bundle carries no EMA shadow copy
    stored = cfg.to_dict()

    prefix = "ema." if args.ema else "model."
    if args.ema and not meta.get("has_ema"):
        raise AsterWhiteError(f"--ema was requested but checkpoint {resolved_name!r} has no EMA weights "
                               f"(train with training.ema_decay set).")
    weights = {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}
    if not weights:
        raise AsterWhiteError(f"checkpoint {resolved_name!r} contains no {prefix!r} tensors")

    model = AsterWhiteCore(cfg.backbone)
    model.load_state_dict(weights, strict=True)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor([tok.encode_prompt("Chapter I.")]))
    if not torch.isfinite(logits).all():
        raise AsterWhiteError("refusing to export: the weights produce non-finite logits on a test prompt")

    n_params = sum(p.numel() for p in model.parameters())
    export_meta = {
        "step": meta.get("step"), "strategy": "export", "config": stored,
        "tokenizer_hash": tok.compute_hash(), "has_ema": False,
        # The training checkpoint's val_loss was measured on its RAW weights. An EMA export is
        # different weights, so recording that number as the bundle's own would be wrong -- keep
        # it under an explicit name and leave val_loss for `evaluate` to establish.
        "best_val_loss": None if args.ema else meta.get("best_val_loss"),
        "val_loss": None if args.ema else meta.get("val_loss"),
        "raw_weights_val_loss": meta.get("val_loss") if args.ema else None,
        "weights_source": "ema" if args.ema else "raw", "n_parameters": n_params,
        "exported_from": os.path.abspath(args.checkpoint_dir), "exported_checkpoint": resolved_name,
        "description": args.description, "model_name": MODEL_NAME,
    }
    out_tensors = {f"model.{k}": v for k, v in model.state_dict().items()}
    registry_save_checkpoint(args.out, "model", out_tensors, export_meta, make_latest=True)
    _save_tokenizer_next_to(args.out, tok_dir)
    size_mb = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out)
                  if os.path.isfile(os.path.join(args.out, f))) / 1e6
    print(f"exported {resolved_name} ({'EMA' if args.ema else 'raw'} weights, {n_params:,} parameters, "
          f"{size_mb:.1f} MB) -> {args.out}\n"
          f"run it with: python train.py generate --load {args.out} --prompt \"Chapter One.\"")


# --------------------------------------------------------------------------- #
# serve
# --------------------------------------------------------------------------- #

def cmd_serve(args: argparse.Namespace) -> None:
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = default_bundle_dir()
        if checkpoint_dir is None:
            raise AsterWhiteError("nothing to serve: no --checkpoint-dir was given and no bundled pretrained model "
                                   "was found (expected at pretrained/asterwhite-small). Train one with "
                                   "`pretrain` and pass --checkpoint-dir, or restore the pretrained/ folder.")
        logger.info(f"no --checkpoint-dir given -- serving the bundled pretrained model at {checkpoint_dir}")
    os.environ["ASTERWHITE_SERVE_CHECKPOINT_DIR"] = checkpoint_dir
    os.environ["ASTERWHITE_SERVE_TOKENIZER_DIR"] = args.tokenizer_dir or ""
    os.environ["ASTERWHITE_SERVE_PROFILE"] = args.profile or ""
    os.environ["ASTERWHITE_SERVE_DEVICE"] = args.device or "auto"
    import uvicorn
    uvicorn.run("asterwhite.serving.app:app", host=args.host, port=args.port, workers=1, log_level="info")


# --------------------------------------------------------------------------- #
# doctor -- environment preflight, run this first on a new machine
# --------------------------------------------------------------------------- #

class _DoctorReport:
    def __init__(self) -> None:
        self.rows: list = []  # (status, name, detail)
        self.failed = False

    def ok(self, name: str, detail: str = "") -> None:
        self.rows.append(("PASS", name, detail))

    def warn(self, name: str, detail: str) -> None:
        self.rows.append(("WARN", name, detail))

    def fail(self, name: str, detail: str) -> None:
        self.rows.append(("FAIL", name, detail))
        self.failed = True

    def check(self, name: str, fn) -> None:
        """Runs fn() and records PASS(detail)/FAIL(exception) -- a single
        broken check (e.g. one missing optional package) can never abort
        the rest of the report, which is the entire point of `doctor`:
        one full picture of the environment in one run, not one error at
        a time across five separate attempts."""
        try:
            detail = fn()
            self.ok(name, detail or "")
        except Exception as exc:  # boundary: doctor must never itself crash
            self.fail(name, f"{type(exc).__name__}: {exc}")

    def print(self) -> None:
        width = max((len(r[1]) for r in self.rows), default=0)
        for status, name, detail in self.rows:
            marker = {"PASS": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[status]
            print(f"{marker} {name.ljust(width)}  {detail}")
        print()
        if self.failed:
            print("One or more checks FAILED -- fix those before `pretrain`/`align`/`generate`/`serve`; "
                  "everything else can be worked around (WARN) or ignored if you don't need that format/backend.")
        else:
            print("All checks passed (or warned-only). This machine should be able to run AsterWhite.")


def cmd_doctor(args: argparse.Namespace) -> None:
    import platform as _platform
    r = _DoctorReport()

    r.check("python version", lambda: (
        _platform.python_version() if sys.version_info >= (3, 9)
        else (_ for _ in ()).throw(RuntimeError(f"{_platform.python_version()} < 3.9 (minimum supported)"))
    ))
    r.check("torch import", lambda: f"torch {torch.__version__}")
    r.check("torch basic op (CPU)", lambda: (torch.randn(4, 4) @ torch.randn(4, 4)).sum().item() and "matmul OK")

    for backend_name, requested in (("cuda", "cuda"), ("mps", "mps"), ("xpu", "xpu")):
        try:
            info = DeviceManager.resolve(requested)
            r.ok(f"device: {backend_name}", f"available -> {info.device_name}")
        except AsterWhiteError:
            r.warn(f"device: {backend_name}", "not available on this machine (fine if you don't have one)")
        except Exception as exc:
            r.warn(f"device: {backend_name}", f"probe raised {type(exc).__name__}: {exc}")
    r.check("device: auto-resolve", lambda: (
        lambda info: f"backend={info.backend} device={info.device} name={info.device_name}"
    )(DeviceManager.resolve(None)))

    for pkg, min_hint in (("numpy", "1.24"), ("pandas", "2.0"), ("pyarrow", "14.0"), ("tokenizers", "0.20"),
                           ("safetensors", "0.4"), ("pydantic", "2.6"), ("yaml", "6.0"), ("psutil", "5.9"),
                           ("fastapi", "0.110"), ("uvicorn", "0.29")):
        def _import_check(pkg=pkg, min_hint=min_hint):
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            return f"{ver} (repo tested >= {min_hint})"
        r.check(f"dependency: {pkg}", _import_check)

    def _disk_check():
        target = args.path or os.getcwd()
        usage = shutil.disk_usage(target if os.path.isdir(target) else os.path.dirname(os.path.abspath(target)) or ".")
        free_gb = usage.free / (1 << 30)
        if free_gb < 1:
            raise RuntimeError(f"only {free_gb:.2f} GB free at '{target}' -- checkpoints will not fit")
        return f"{free_gb:.1f} GB free at '{target}'"
    r.check("disk space", _disk_check)

    def _write_check():
        target = args.path or os.getcwd()
        os.makedirs(target, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=target)
        os.close(fd)
        os.remove(tmp_path)
        return f"writable: '{target}'"
    r.check("write permission", _write_check)

    def _e2e_check():
        cfg = AsterWhiteConfig.tiny_for_testing()
        validate_config(cfg)
        model = AsterWhiteCore(cfg.backbone)
        ids = torch.randint(0, cfg.backbone.vocab_size, (1, 8))
        logits = model(ids)
        logits.float().pow(2).mean().backward()
        n_params = sum(p.numel() for p in model.parameters())
        return f"tiny model built, forward+backward OK ({n_params:,} params)"
    r.check("end-to-end tiny model (forward+backward)", _e2e_check)

    r.print()
    sys.exit(1 if r.failed else 0)


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #

def cmd_selftest(args: argparse.Namespace) -> None:
    import pytest
    device_info = DeviceManager.resolve(None)
    print(f"device: backend={device_info.backend} name={device_info.device_name}")
    sys.exit(pytest.main(["-q", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")]))


# --------------------------------------------------------------------------- #
# argparse wiring
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="train.py", description=f"{MODEL_NAME} CLI")
    sub = p.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("pretrain", help="next-token pretraining/fine-tuning")
    _add_common_args(pt, need_data=True)
    pt.add_argument("--max-steps", type=int, default=None)
    pt.add_argument("--resume", default=None, help="checkpoint dir to resume THIS run from (optimizer+RNG+step restored)")
    pt.add_argument("--load-base", default=None, help="checkpoint dir to load weights ONLY from, starting a new run at step 0")
    pt.add_argument("--lora", action="store_true", help="apply LoRA and freeze the base model")
    pt.add_argument("--lora-r", type=int, default=8)
    pt.add_argument("--lora-alpha", type=int, default=16)
    pt.add_argument("--lora-targets", nargs="+", default=["q_proj", "v_proj"])
    pt.add_argument("--operator", default=os.environ.get("USER", "unspecified"))
    pt.add_argument("--seed", type=int, default=None, help="seed Python/NumPy/torch RNGs for a reproducible run")
    pt.set_defaults(func=cmd_pretrain)

    al = sub.add_parser("align", help="GRPO alignment against a reward signal")
    _add_common_args(al, need_data=False)
    al.add_argument("--load", default=None, help="checkpoint dir of the base policy to align (default: the "
                    "bundled pretrained model; --dry-run for a pipeline check on random weights)")
    al.add_argument("--reward-model-dir", default=None, help="load a trained RewardModel instead of the heuristic default")
    al.add_argument("--prompts", nargs="+", default=None)
    al.add_argument("--steps", type=int, default=50)
    al.add_argument("--out", default=None)
    al.set_defaults(func=cmd_align)

    ge = sub.add_parser("generate", help="generate narrative text from a checkpoint")
    _add_common_args(ge, need_data=False)
    ge.add_argument("--load", default=None, help="checkpoint dir with trained weights (default: the bundled "
                    "pretrained model, pretrained/asterwhite-small; use --dry-run for random weights). The "
                    "directory's stored config sets the architecture and a tokenizer/ subfolder is used "
                    "automatically. "
                    "If this checkpoint was itself produced by a `pretrain --lora` run, load it alone (it already "
                    "contains the LoRA weights merged in) -- do NOT also pass --lora-adapter in that case.")
    ge.add_argument("--lora-adapter", default=None,
                    help="apply a SEPARATELY-trained LoRA adapter on top of --load's checkpoint. Only correct when "
                         "--load is a PLAIN base checkpoint (from a `pretrain` run WITHOUT --lora) and this "
                         "directory came from a later `pretrain --load-base <that checkpoint> --lora` run's saved "
                         "checkpoint_dir/lora_adapter -- passing an adapter from the SAME run as --load will fail "
                         "to load (that checkpoint already has the LoRA weights merged into its own state dict).")
    ge.add_argument("--lora-r", type=int, default=8, help="must match the r used when this adapter was trained")
    ge.add_argument("--lora-alpha", type=int, default=16, help="must match the alpha used when this adapter was trained")
    ge.add_argument("--lora-targets", nargs="+", default=["q_proj", "v_proj"],
                    help="must match the target_substrings used when this adapter was trained")
    ge.add_argument("--prompt", default="Chapter One.", help="opening prompt -- ignored (the loaded session's "
                     "prompt is used instead) when --resume-checkpoint is given")
    ge.add_argument("--branching", type=int, nargs="+", default=[6],
                     help="story plan tree branching factor, root to chapter. '--branching 20' plans a flat "
                          "20-chapter story (a normal-length short work). '--branching 20 50' plans 20 volumes x "
                          "50 chapters/volume = 1,000 chapters total; add more numbers for more levels/scale -- "
                          "see generation_orchestrator.HierarchicalPlan. Ignored when --resume-checkpoint is given "
                          "(the saved session's own plan continues instead).")
    ge.add_argument("--chapters", type=int, default=1,
                     help="how many chapters to write THIS run (each calls generate_next() once, then "
                          "advance_chapter()) -- not the total story length, which --branching sets. Stops early "
                          "if the planned story completes first. Use --save-checkpoint + a later "
                          "--resume-checkpoint to continue writing more chapters in a separate run.")
    ge.add_argument("--resume-checkpoint", default=None,
                     help="continue a previously --save-checkpoint'd session from the exact next token and "
                          "chapter position, instead of starting a new story")
    ge.add_argument("--max-new-tokens", type=int, default=128, help="token budget PER CHAPTER")
    ge.add_argument("--temperature", type=float, default=0.9)
    ge.add_argument("--top-p", type=float, default=0.9)
    ge.add_argument("--top-k", type=int, default=0)
    ge.add_argument("--repetition-penalty", type=float, default=1.0)
    ge.add_argument("--save-checkpoint", default=None, help="path to save the generation session for later resume")
    ge.add_argument("--seed", type=int, default=None, help="seed for reproducible output (story plan + sampling)")
    ge.add_argument("--conditioning-scale", type=float, default=1.0,
                    help="strength of the story-plan/world-state conditioning vectors, as a multiple of the "
                         "model's own token-embedding magnitude; 0 disables the conditioning prefix entirely "
                         "(plain continuation of --prompt). Note the planner modules are untrained random "
                         "projections unless you trained them, so the prefix carries no real plan information "
                         "either way.")
    ge.set_defaults(func=cmd_generate)

    se = sub.add_parser("serve", help="launch the OpenAI/Ollama-compatible inference API")
    se.add_argument("--checkpoint-dir", default=None,
                    help="model to serve (default: the bundled pretrained model, pretrained/asterwhite-small)")
    se.add_argument("--tokenizer-dir", default=None,
                    help="default: the tokenizer/ folder inside --checkpoint-dir, if there is one")
    se.add_argument("--profile", default=None, choices=list_profiles() or None,
                    help="serving settings (sampling defaults, concurrency, timeouts); default: the "
                         "checkpoint's own stored settings. The architecture always comes from the checkpoint.")
    se.add_argument("--device", default="auto")
    se.add_argument("--host", default="0.0.0.0")
    se.add_argument("--port", type=int, default=8000)
    se.set_defaults(func=cmd_serve)

    tok = sub.add_parser("tokenizer", help="train/inspect a real subword tokenizer")
    tok_sub = tok.add_subparsers(dest="tokenizer_command", required=True)
    tt = tok_sub.add_parser("train")
    tt.add_argument("--data", nargs="+", required=True,
                     help="file(s), directories (searched recursively), and/or glob patterns -- same rules as "
                          "pretrain's --data (data_pipeline.resolve_data_paths)")
    tt.add_argument("--vocab-size", type=int, default=32000)
    tt.add_argument("--min-frequency", type=int, default=2)
    tt.add_argument("--text-column", default=None,
                     help="for .jsonl/.ndjson/.csv/.parquet inputs, which field/column holds the prose "
                          "to train on; omit to use the first field/column (same default as pretrain's "
                          "data pipeline)")
    tt.add_argument("--out", required=True)
    tt.set_defaults(func=cmd_tokenizer_train)
    ti = tok_sub.add_parser("inspect")
    ti.add_argument("--tokenizer-dir", required=True)
    ti.set_defaults(func=cmd_tokenizer_inspect)
    tp = tok_sub.add_parser("prepare-binary", help="pre-tokenize a corpus into one flat binary token file for "
                                                     "MemmapTokenDataset -- the TB-scale pretraining path (constant "
                                                     "RAM regardless of corpus size, no re-tokenizing every epoch)")
    tp.add_argument("--data", nargs="+", required=True,
                     help="file(s), directories (searched recursively), and/or glob patterns -- same rules as "
                          "pretrain's --data (data_pipeline.resolve_data_paths)")
    tp.add_argument("--tokenizer-dir", default=None,
                     help="directory from `tokenizer train` output; omit to use the zero-setup byte-level "
                          "fallback (dtype=uint8-safe, vocab_size=256)")
    tp.add_argument("--text-column", default=None,
                     help="for .jsonl/.ndjson/.csv/.parquet inputs, which field/column holds the prose")
    tp.add_argument("--dtype", default=None, choices=["uint16", "uint32"],
                     help="integer width per token id; must be able to represent every id up to the "
                          "tokenizer's vocab_size - 1, or building refuses rather than silently overflowing. "
                          "Omit to auto-pick the smallest width that fits your tokenizer's actual vocab_size.")
    tp.add_argument("--out", required=True, help="output .bin path -- pass this directly as `pretrain --data "
                                                   "<this>.bin` afterward (auto-detected as pre-tokenized)")
    tp.set_defaults(func=cmd_tokenizer_prepare_binary)

    ex = sub.add_parser("export", help="package a training checkpoint + tokenizer into a self-contained, "
                                        "inference-only model bundle (weights + config + tokenizer)")
    ex.add_argument("--checkpoint-dir", required=True, help="training checkpoint directory (from `pretrain`)")
    ex.add_argument("--tokenizer-dir", default=None, help="default: <checkpoint-dir>/tokenizer")
    ex.add_argument("--out", required=True, help="bundle directory to create, e.g. ./pretrained/mymodel")
    ex.add_argument("--which", default="latest", choices=["latest", "best"],
                    help="which checkpoint pointer to export ('best' = lowest validation loss so far)")
    ex.add_argument("--name", default=None, help="export a specific checkpoint by name instead, e.g. step_2500")
    ex.add_argument("--ema", action="store_true", help="export the EMA (Polyak-averaged) weights instead of the raw ones")
    ex.add_argument("--description", default="", help="free-text note stored in the bundle's metadata")
    ex.set_defaults(func=cmd_export)

    ev = sub.add_parser("evaluate", help="validation loss / perplexity of a checkpoint on held-out text")
    ev.add_argument("--data", nargs="+", required=True, help="held-out text: files, directories and/or globs")
    ev.add_argument("--load", default=None, help="checkpoint/bundle dir (default: the bundled pretrained model)")
    ev.add_argument("--tokenizer-dir", default=None, help="default: the tokenizer/ folder inside --load")
    ev.add_argument("--profile", default=None, choices=list_profiles() or None)
    ev.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "rocm", "mps", "xpu"])
    ev.add_argument("--batch-size", type=int, default=8)
    ev.set_defaults(func=cmd_evaluate, dry_run=False)

    st = sub.add_parser("selftest", help="run the bundled test suite")
    st.set_defaults(func=cmd_selftest)

    doc = sub.add_parser("doctor", help="environment preflight: python/torch/deps/every device backend/disk, "
                                         "plus one real tiny forward+backward pass -- run this first")
    doc.add_argument("--path", default=None, help="directory to disk-space/writability-check (default: cwd)")
    doc.set_defaults(func=cmd_doctor)

    return p


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.stderr.write("\ninterrupted.\n")
        sys.exit(130)
    except SystemExit:
        raise
    except AsterWhiteError as exc:
        sys.stderr.write("\n" + exc.report() + "\n")
        if os.environ.get("ASTERWHITE_DEBUG"):
            raise
        sys.stderr.write("\n(Set ASTERWHITE_DEBUG=1 to see the full Python traceback, e.g. to file a bug report.)\n")
        sys.exit(1)
    except Exception as exc:
        # Any exception NOT already an AsterWhiteError (a third-party
        # library's own exception type, an OS-level error, etc.) still gets
        # the same clean, located report -- AsterWhiteError(cause=exc) walks
        # exc's own traceback to find the exact AsterWhite file/line that
        # called into whatever failed (see utils.error_tracing.locate).
        wrapped = AsterWhiteError(f"{type(exc).__name__}: {exc}", cause=exc)
        sys.stderr.write("\n" + wrapped.report() + "\n")
        if os.environ.get("ASTERWHITE_DEBUG"):
            raise
        sys.stderr.write("\n(Set ASTERWHITE_DEBUG=1 to see the full Python traceback, e.g. to file a bug report.)\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
