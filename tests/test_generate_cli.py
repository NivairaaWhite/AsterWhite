"""AsterWhite :: tests.test_generate_cli -- train.py's `generate` subcommand
exercised as a real subprocess, covering the multi-chapter (--branching/
--chapters) and session resume (--save-checkpoint/--resume-checkpoint)
flows end to end through actual argparse, not just the underlying library
calls tested elsewhere."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

TRAIN_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train.py")


def _run(*args: str) -> subprocess.CompletedProcess:
    # ASTERWHITE_PRETRAINED_DIR -> a directory that has no bundles: these tests are about the explicit
    # --load / --dry-run contract, and must behave the same whether or not the shipped pretrained
    # model (which `generate` now defaults to when --load is omitted) is present in this checkout.
    env = {**os.environ, "ASTERWHITE_PRETRAINED_DIR": os.path.join(tempfile.gettempdir(), "asterwhite-no-bundles")}
    return subprocess.run([sys.executable, TRAIN_PY, *args], capture_output=True, text=True, timeout=120, env=env)


def test_dry_run_generate_single_chapter():
    r = _run("generate", "--dry-run", "--max-new-tokens", "10")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() != ""


def test_multi_chapter_generation_writes_each_labeled_chapter_in_order():
    r = _run("generate", "--dry-run", "--branching", "3", "4", "--chapters", "3", "--max-new-tokens", "10")
    assert r.returncode == 0, r.stderr
    assert r.stdout.index("=== Saga.1.1 ===") < r.stdout.index("=== Saga.1.2 ===") < r.stdout.index("=== Saga.1.3 ===")


def test_chapters_stops_early_when_plan_completes():
    # branching=[2] -> only 2 chapters total; requesting 5 must stop at 2, not error
    r = _run("generate", "--dry-run", "--branching", "2", "--chapters", "5", "--max-new-tokens", "10")
    assert r.returncode == 0, r.stderr
    assert "=== Saga.1 ===" in r.stdout and "=== Saga.2 ===" in r.stdout
    assert "=== Saga.3 ===" not in r.stdout
    assert "plan complete" in r.stderr


def test_save_and_resume_checkpoint_continues_without_gap_or_duplication():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "session.ckpt")

        r1 = _run("generate", "--dry-run", "--branching", "3", "4", "--chapters", "2",
                   "--max-new-tokens", "10", "--save-checkpoint", ckpt)
        assert r1.returncode == 0, r1.stderr
        assert "=== Saga.1.1 ===" in r1.stdout and "=== Saga.1.2 ===" in r1.stdout

        r2 = _run("generate", "--dry-run", "--resume-checkpoint", ckpt, "--chapters", "2", "--max-new-tokens", "10")
        assert r2.returncode == 0, r2.stderr
        assert "=== Saga.1.3 ===" in r2.stdout and "=== Saga.1.4 ===" in r2.stdout
        # no gap (1.3 immediately follows what was already written) and no
        # re-generation of already-written chapters
        assert "Saga.1.1" not in r2.stdout
        assert "Saga.1.2" not in r2.stdout


def test_resume_still_requires_load_or_dry_run():
    """--resume-checkpoint restores the WRITING SESSION (token history,
    plan position); it is not a substitute for --load/--dry-run, which
    select which trained MODEL WEIGHTS to generate with -- these are
    orthogonal, and omitting both must still be refused."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "session.ckpt")
        _run("generate", "--dry-run", "--chapters", "1", "--max-new-tokens", "5", "--save-checkpoint", ckpt)
        r = _run("generate", "--resume-checkpoint", ckpt, "--chapters", "1")
        assert r.returncode != 0
        assert "refuses to construct a fresh, untrained model" in r.stderr
