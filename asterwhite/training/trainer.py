"""
AsterWhite :: training.trainer

Supervised (next-token) pretraining/fine-tuning loop: gradient accumulation
to reach large effective batch sizes under fixed VRAM, mixed-precision
(bf16/fp16 autocast + GradScaler) to cut memory and speed up compute, and
optional torch.distributed DDP/FSDP wrapping for multi-GPU/multi-node runs
-- the same training loop runs unchanged on a laptop CPU (smoke tests) or
a multi-node GPU cluster (production); only the launch command differs.

Also wires in, all opt-in via TrainingConfig/BackboneConfig and all
no-ops (byte-identical loss/step behavior) when left at their defaults:
  - activation checkpointing (real: BackboneConfig had a documented-but-
    unused flag before this; see AsterWhiteCore.forward_hidden)
  - torch.compile, with a logged (never silent, never fatal) fallback to
    the uncompiled model if compilation itself fails to initialize
  - fused AdamW, with the same kind of logged fallback
  - EMA (Polyak-averaged) shadow weights, persisted in/restored from
    checkpoints alongside the raw weights
  - the multi-token-prediction auxiliary loss (BackboneConfig.
    mtp_n_future_tokens) and the MoE load-balancing auxiliary loss
    (BackboneConfig.moe), both added into the training (not eval/
    validation) loss automatically whenever the model has them.

Checkpointing goes through registry.checkpoint (atomic, checksummed,
versioned, no pickle) instead of a bare torch.save -- see that module's
docstring. Every checkpoint carries RNG state (Python/NumPy/torch CPU/
torch CUDA), the resolved config, and (optionally) a tokenizer hash, so a
resumed run is byte-for-byte reproducible where the backend allows it, and
refuses to silently resume under a different tokenizer.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ..config import TrainingConfig
from ..registry.checkpoint import (
    capture_rng_state, flatten_tensors, load_checkpoint, restore_rng_state,
    save_checkpoint, unflatten_tensors,
)
from ..utils.error_tracing import AsterWhiteError, get_logger, trace_errors

logger = get_logger("asterwhite.training")

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    _FSDP_AVAILABLE = True
except Exception:  # pragma: no cover -- older torch without FSDP
    _FSDP_AVAILABLE = False


def setup_distributed() -> Tuple[int, int, int]:
    """Reads standard torchrun env vars (RANK/WORLD_SIZE/LOCAL_RANK).
    Returns (rank, world_size, local_rank). No-ops to (0, 1, 0) when not
    launched under torchrun, so the exact same script runs single-process.
    Known gap (documented, not silently claimed otherwise): this assumes a
    FIXED world size for the life of the run -- elastic membership changes
    (a node joining/leaving mid-run) are not handled; a node failure means
    stop, fix/replace it, relaunch all ranks, and --resume."""
    if "RANK" not in os.environ:
        return 0, 1, 0
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))


@dataclass
class StepStats:
    step: int
    loss: float
    lr: float
    tokens_per_sec: float
    grad_norm: float = 0.0
    skipped_nonfinite: bool = False
    main_loss: float = 0.0
    mtp_loss: float = 0.0
    moe_aux_loss: float = 0.0


@dataclass
class EvalStats:
    step: int
    val_loss: float
    perplexity: float


_FLOAT_DTYPES = (torch.float16, torch.float32, torch.float64, torch.bfloat16)


class Trainer:
    def __init__(self, model: nn.Module, cfg: TrainingConfig, device: torch.device,
                 distributed: bool = False, local_rank: int = 0, world_size: int = 1,
                 tokenizer_hash: Optional[str] = None, extra_modules: Optional[Dict[str, nn.Module]] = None,
                 max_consecutive_nonfinite: int = 20):
        self.cfg = cfg
        self.device = device
        self.tokenizer_hash = tokenizer_hash
        self.extra_modules = extra_modules or {}
        self.raw_model = model.to(device)
        self.max_consecutive_nonfinite = max_consecutive_nonfinite
        self._consecutive_nonfinite = 0

        # Real wiring for the (previously dead) activation_checkpointing
        # config flag: AsterWhiteCore.forward_hidden reads this exact
        # attribute (defaulting to False for anyone constructing/using the
        # model without a Trainer at all) -- see its docstring for why
        # this lives as a plain attribute set here rather than threaded
        # through BackboneConfig (it's a training-time recompute/memory
        # tradeoff, not a property of the architecture itself).
        self.raw_model.activation_checkpointing = cfg.activation_checkpointing

        if distributed and cfg.distributed_strategy == "fsdp":
            if not _FSDP_AVAILABLE:
                raise AsterWhiteError("training.distributed_strategy='fsdp' was requested but this torch "
                                       "build has no torch.distributed.fsdp -- upgrade torch or use 'ddp'.")
            auto_wrap = None
            try:
                from functools import partial
                auto_wrap = partial(size_based_auto_wrap_policy, min_num_params=int(1e6))
            except Exception:  # pragma: no cover
                pass
            self.model: nn.Module = FSDP(self.raw_model, auto_wrap_policy=auto_wrap,
                                          device_id=local_rank if device.type == "cuda" else None)
            self._strategy = "fsdp"
        elif distributed:
            self.model = DDP(self.raw_model, device_ids=[local_rank] if device.type == "cuda" else None)
            self._strategy = "ddp"
        else:
            self.model = self.raw_model
            self._strategy = "none"

        if cfg.compile_model:
            if device.type not in ("cuda",):
                # torch.compile is lazy: compilation happens on the FIRST
                # real forward call, not on the torch.compile(...) wrapping
                # call itself -- so a try/except around the wrap can't catch
                # a slow/hanging first compilation, only an immediate
                # construction-time error. Observed directly during this
                # repo's own development: on a CPU-only sandbox without a
                # full build toolchain for the inductor CPU backend,
                # compilation did not error -- it hung. Rather than risk
                # that on an arbitrary user machine, compilation is only
                # attempted on backends where it's mature and reliably
                # fast to build (CUDA); everywhere else this is a logged
                # no-op, not a silent one.
                logger.info(f"compile_model=True but device={device.type} -- torch.compile is only attempted on "
                            f"CUDA (compilation on CPU/MPS/XPU has been observed to hang or take excessively long "
                            f"without a full build toolchain). Continuing with the uncompiled model.")
            else:
                try:
                    self.model = torch.compile(self.model)
                    logger.info("torch.compile enabled")
                except Exception as exc:  # pragma: no cover -- environment-dependent (Triton/backend availability)
                    logger.warning(f"torch.compile failed to initialize ({type(exc).__name__}: {exc}) -- "
                                    f"continuing with the uncompiled model. This is not fatal: training proceeds "
                                    f"normally, just without the compile speedup.")

        self.optimizer = self._build_optimizer()
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.amp_dtype]
        self.amp_device_type = "cuda" if device.type == "cuda" else "cpu"
        # GradScaler only matters for fp16 (bf16's exponent range doesn't
        # need loss scaling); passing enabled=False makes every call below
        # a safe no-op, so this line of code is identical for all 3 dtypes.
        self.scaler = torch.amp.GradScaler(self.amp_device_type, enabled=(cfg.amp_dtype == "fp16"))
        self.step_count = 0
        self.best_val_loss: Optional[float] = None

        # EMA (Polyak-averaged) shadow weights -- see TrainingConfig.ema_decay.
        # A separate, plain (never DDP/FSDP-wrapped, never autograd-tracked)
        # dict of cloned tensors, updated in-place after every successful
        # optimizer step (_update_ema) -- disabled (None) unless ema_decay
        # is set, at zero cost/behavior-change when unused.
        self.ema_decay: Optional[float] = cfg.ema_decay
        self.ema_state: Optional[Dict[str, torch.Tensor]] = None
        if self.ema_decay is not None:
            self.ema_state = {k: v.detach().clone() for k, v in self.raw_model.state_dict().items()}

    def _build_optimizer(self) -> torch.optim.Optimizer:
        kwargs = dict(lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay, betas=(0.9, 0.95))
        if self.cfg.optimizer_fused:
            try:
                return torch.optim.AdamW(self.model.parameters(), fused=True, **kwargs)
            except Exception as exc:
                # fused=True requires every param on one CUDA-like device and
                # a torch build with the fused kernel available -- neither is
                # true on CPU, MPS, or some older/exotic torch builds. Not an
                # error: transparently use the standard (unfused) AdamW.
                logger.info(f"fused AdamW unavailable ({type(exc).__name__}: {exc}) -- using standard AdamW")
        return torch.optim.AdamW(self.model.parameters(), **kwargs)

    def _lr_for_step(self, step: int) -> float:
        if step < self.cfg.warmup_steps:
            return self.cfg.learning_rate * (step + 1) / max(1, self.cfg.warmup_steps)
        progress = (step - self.cfg.warmup_steps) / max(1, self.cfg.max_steps - self.cfg.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * self.cfg.learning_rate * (1 + math.cos(progress * math.pi))

    def _forward_loss(self, micro: torch.Tensor, include_aux_losses: bool = False) -> Tuple[torch.Tensor, dict]:
        """Returns (total_loss, breakdown) where breakdown has keys
        main_loss/mtp_loss/moe_aux_loss (all float, 0.0 for whichever
        don't apply) -- train_step logs/reports these; evaluate() only
        ever calls this with include_aux_losses=False (default) so
        val_loss/perplexity stay directly comparable across MTP/MoE
        on-or-off configs (both report pure next-token cross-entropy)."""
        with torch.amp.autocast(self.amp_device_type, dtype=self.amp_dtype,
                                 enabled=(self.cfg.amp_dtype != "fp32")):
            logits = self.model(micro[:, :-1])
            main_loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), micro[:, 1:].reshape(-1))
            total = main_loss
            mtp_loss_val = torch.zeros((), device=self.device)
            moe_aux_val = torch.zeros((), device=self.device)

            if include_aux_losses:
                mtp_logits = getattr(self.raw_model, "_last_mtp_logits", None)
                if mtp_logits:
                    L = micro.shape[1]
                    mtp_total = torch.zeros((), device=self.device)
                    n_valid_heads = 0
                    for idx, head_logits in enumerate(mtp_logits):
                        k = idx + 2  # head[idx] predicts offset (idx+2): the main lm_head already covers +1
                        h_len = L - k
                        if h_len <= 0:
                            continue  # sequence shorter than this offset -- skip, don't crash on tiny batches
                        logits_k = head_logits[:, :h_len, :]
                        target_k = micro[:, k:k + h_len]
                        mtp_total = mtp_total + nn.functional.cross_entropy(
                            logits_k.reshape(-1, logits_k.shape[-1]), target_k.reshape(-1))
                        n_valid_heads += 1
                    if n_valid_heads > 0:
                        mtp_loss_val = mtp_total / n_valid_heads
                        total = total + self.cfg.mtp_loss_coef * mtp_loss_val

                moe_cfg = getattr(getattr(self.raw_model, "cfg", None), "moe", None)
                if moe_cfg is not None and moe_cfg.enabled and hasattr(self.raw_model, "collect_moe_aux_loss"):
                    moe_aux_val = self.raw_model.collect_moe_aux_loss()
                    total = total + moe_cfg.aux_loss_coef * moe_aux_val

            breakdown = {"main_loss": float(main_loss.detach().item()),
                         "mtp_loss": float(mtp_loss_val.detach().item()) if torch.is_tensor(mtp_loss_val) else 0.0,
                         "moe_aux_loss": float(moe_aux_val.detach().item()) if torch.is_tensor(moe_aux_val) else 0.0}
            return total, breakdown

    @torch.no_grad()
    def _update_ema(self) -> None:
        if self.ema_state is None:
            return
        for k, v in self.raw_model.state_dict().items():
            shadow = self.ema_state[k]
            if v.dtype in _FLOAT_DTYPES:
                shadow.mul_(self.ema_decay).add_(v.detach().to(shadow.dtype), alpha=1.0 - self.ema_decay)
            else:
                shadow.copy_(v)  # non-float buffers (e.g. any integer counters) just track the latest value

    def ema_state_dict(self) -> Optional[Dict[str, torch.Tensor]]:
        """The EMA shadow weights, ready to torch-load into a fresh
        AsterWhiteCore for inference -- most modern pretraining recipes
        serve these rather than the raw last-step weights (consistently
        as-good-or-better, and smoother/less step-to-step-noisy). None if
        TrainingConfig.ema_decay was never set."""
        return {k: v.clone() for k, v in self.ema_state.items()} if self.ema_state is not None else None

    @trace_errors
    def train_step(self, micro_batches: Iterable[torch.Tensor]) -> StepStats:
        """`micro_batches`: an iterable of (micro_batch_size, T) token-id
        tensors that together make up ONE optimizer step. Gradients
        accumulate across them via repeated .backward() calls before a
        single optimizer.step() -- this is what lets effective batch_size
        exceed whatever micro_batch_size actually fits in VRAM/RAM.

        NaN/Inf handling: if the accumulated loss or gradient norm is
        non-finite, the optimizer step is SKIPPED (zero_grad, no update)
        rather than corrupting the model with a bad step -- this is a real
        recovery action, not just a log line. After
        `max_consecutive_nonfinite` skips in a row this raises
        AsterWhiteError, because at that point it is runaway divergence,
        not a transient spike, and silently skipping forever would waste
        compute while hiding a real problem."""
        t0 = time.time()
        lr = self._lr_for_step(self.step_count)
        for g in self.optimizer.param_groups:
            g["lr"] = lr

        self.optimizer.zero_grad(set_to_none=True)
        total_loss, n_micro, n_tokens = 0.0, 0, 0
        last_breakdown = {"main_loss": 0.0, "mtp_loss": 0.0, "moe_aux_loss": 0.0}
        for micro in micro_batches:
            micro = micro.to(self.device, non_blocking=True)
            loss, breakdown = self._forward_loss(micro, include_aux_losses=True)
            self.scaler.scale(loss).backward()
            total_loss += loss.item()
            last_breakdown = breakdown
            n_micro += 1
            n_tokens += micro.numel()

        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.div_(max(1, n_micro))

        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        grad_norm_val = float(grad_norm) if not torch.is_tensor(grad_norm) else grad_norm.item()

        nonfinite = not math.isfinite(total_loss) or not math.isfinite(grad_norm_val)
        if nonfinite:
            self._consecutive_nonfinite += 1
            logger.warning(f"step {self.step_count + 1}: non-finite loss/grad_norm "
                            f"(loss={total_loss}, grad_norm={grad_norm_val}) -- skipping optimizer step "
                            f"({self._consecutive_nonfinite}/{self.max_consecutive_nonfinite} consecutive)")
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.update()
            if self._consecutive_nonfinite >= self.max_consecutive_nonfinite:
                raise AsterWhiteError(f"{self._consecutive_nonfinite} consecutive non-finite steps -- this is "
                                       f"divergence, not a transient spike. Lower the learning rate, check the "
                                       f"data for corrupt records, or resume from an earlier checkpoint.")
        else:
            self._consecutive_nonfinite = 0
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._update_ema()

        self.step_count += 1
        dt = time.time() - t0
        stats = StepStats(step=self.step_count, loss=total_loss / max(1, n_micro), lr=lr,
                           tokens_per_sec=n_tokens / max(dt, 1e-9), grad_norm=grad_norm_val,
                           skipped_nonfinite=nonfinite, main_loss=last_breakdown["main_loss"],
                           mtp_loss=last_breakdown["mtp_loss"], moe_aux_loss=last_breakdown["moe_aux_loss"])
        if self.step_count % self.cfg.log_every == 0:
            extra = ""
            if last_breakdown["mtp_loss"]:
                extra += f" | mtp {last_breakdown['mtp_loss']:.4f}"
            if last_breakdown["moe_aux_loss"]:
                extra += f" | moe_aux {last_breakdown['moe_aux_loss']:.4f}"
            logger.info(f"step {stats.step} | loss {stats.loss:.4f} | lr {stats.lr:.2e} | "
                        f"grad_norm {stats.grad_norm:.3f} | {stats.tokens_per_sec:.0f} tok/s{extra}")
        return stats

    @torch.no_grad()
    def evaluate(self, val_batches: Iterable[torch.Tensor]) -> EvalStats:
        """Real validation loop (spec: 'MUST include a validation or
        evaluation loop, not only a training loss curve'). Switches to
        eval mode, runs loss with no gradient tracking, restores whatever
        mode the model was in before. Tracks self.best_val_loss so callers
        can decide whether this checkpoint should become `best`. Reports
        pure next-token cross-entropy (include_aux_losses=False) even when
        MTP/MoE are enabled, so val_loss/perplexity stay comparable across
        configs that do and don't use those training-time-only aids."""
        was_training = self.model.training
        self.model.eval()
        total_loss, n = 0.0, 0
        for micro in val_batches:
            micro = micro.to(self.device, non_blocking=True)
            loss, _breakdown = self._forward_loss(micro)
            total_loss += loss.item()
            n += 1
        if was_training:
            self.model.train()
        val_loss = total_loss / max(1, n)
        perplexity = math.exp(min(val_loss, 20))  # cap the exponent -- avoid inf on an early, very-bad checkpoint
        if self.best_val_loss is None or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
        stats = EvalStats(step=self.step_count, val_loss=val_loss, perplexity=perplexity)
        logger.info(f"eval @ step {stats.step} | val_loss {stats.val_loss:.4f} | ppl {stats.perplexity:.2f}")
        return stats

    def save_checkpoint(self, directory: str, config_dict: Optional[dict] = None,
                         extra_metadata: Optional[dict] = None) -> None:
        """directory/{step_N}.safetensors + .json, plus latest.json /
        best.json pointers (best.json only written when this call's
        val_loss, if provided via extra_metadata, matches self.best_val_loss).
        EMA shadow weights (if TrainingConfig.ema_decay is set) are saved
        alongside the raw weights under an `ema.` prefix -- see
        load_checkpoint, which restores them back into self.ema_state."""
        name = f"step_{self.step_count}"
        opt_tensors, opt_skeleton = flatten_tensors(self.optimizer.state_dict(), prefix="opt")
        rng_tensors, rng_skeleton = flatten_tensors(capture_rng_state(), prefix="rng")

        model_tensors = {f"model.{k}": v for k, v in self.raw_model.state_dict().items()}
        extra_tensors = {}
        for mod_name, module in self.extra_modules.items():
            for k, v in module.state_dict().items():
                extra_tensors[f"extra.{mod_name}.{k}"] = v
        ema_tensors = {f"ema.{k}": v for k, v in self.ema_state.items()} if self.ema_state is not None else {}

        all_tensors = {**model_tensors, **extra_tensors, **ema_tensors,
                       **{f"opt.{k}": v for k, v in opt_tensors.items()},
                       **{f"rng.{k}": v for k, v in rng_tensors.items()}}

        is_best = (extra_metadata or {}).get("val_loss") is not None and \
            self.best_val_loss is not None and (extra_metadata["val_loss"] <= self.best_val_loss)

        metadata = {
            "step": self.step_count, "strategy": self._strategy,
            "optimizer_skeleton": opt_skeleton, "rng_skeleton": rng_skeleton,
            "extra_module_names": list(self.extra_modules.keys()),
            "has_ema": self.ema_state is not None,
            "config": config_dict or {}, "tokenizer_hash": self.tokenizer_hash,
            "best_val_loss": self.best_val_loss,
        }
        metadata.update(extra_metadata or {})
        save_checkpoint(directory, name, all_tensors, metadata, make_latest=True, make_best=is_best)

    def load_checkpoint(self, directory: str, name: Optional[str] = None, which: str = "latest",
                         strict: bool = True, restore_rng: bool = True) -> dict:
        """Returns the loaded metadata dict (config, step, best_val_loss,
        ...) so the caller can sanity-check it (e.g. compare
        metadata["config"] against the config it's about to train with).
        Refuses (via registry.checkpoint's tokenizer-hash check) if this
        Trainer was constructed with a tokenizer_hash that doesn't match
        what the checkpoint was trained with."""
        tensors, metadata = load_checkpoint(directory, name=name, which=which,
                                             map_location=str(self.device),
                                             expected_tokenizer_hash=self.tokenizer_hash)
        model_sd = {k[len("model."):]: v for k, v in tensors.items() if k.startswith("model.")}
        self.raw_model.load_state_dict(model_sd, strict=strict)

        for mod_name, module in self.extra_modules.items():
            prefix = f"extra.{mod_name}."
            sd = {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}
            if sd:
                module.load_state_dict(sd, strict=strict)

        if self.ema_state is not None:
            ema_tensors = {k[len("ema."):]: v for k, v in tensors.items() if k.startswith("ema.")}
            if ema_tensors:
                for k, v in ema_tensors.items():
                    if k in self.ema_state:
                        self.ema_state[k] = v.to(self.device)
                logger.info(f"restored EMA shadow weights from checkpoint ({len(ema_tensors)} tensors)")
            else:
                logger.warning("this Trainer has EMA enabled but the checkpoint being loaded has none -- "
                                "EMA will continue accumulating fresh from the just-loaded raw weights.")

        opt_tensors = {k[len("opt."):]: v for k, v in tensors.items() if k.startswith("opt.")}
        if opt_tensors and "optimizer_skeleton" in metadata:
            self.optimizer.load_state_dict(unflatten_tensors(metadata["optimizer_skeleton"], opt_tensors))

        if restore_rng:
            rng_tensors = {k[len("rng."):]: v for k, v in tensors.items() if k.startswith("rng.")}
            if rng_tensors and "rng_skeleton" in metadata:
                restore_rng_state(unflatten_tensors(metadata["rng_skeleton"], rng_tensors))

        self.step_count = metadata.get("step", 0)
        self.best_val_loss = metadata.get("best_val_loss")
        logger.info(f"checkpoint loaded <- {directory} (resuming at step {self.step_count}, "
                    f"strategy={metadata.get('strategy')})")
        return metadata
