"""
AsterWhite :: utils.error_tracing

Every uncaught exception anywhere in AsterWhite is reported with an exact
folder / file / line / function location -- not just a bare Python
traceback. `install()` is called once at process start (train.py does this
automatically) and augments sys.excepthook. `@trace_errors` additionally
wraps any function so its own internal failure point is captured even if
the exception is caught and re-raised higher up the call stack.
"""
from __future__ import annotations

import functools
import logging
import os
import sys
import traceback
from typing import Any, Callable, Optional, TypeVar

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
F = TypeVar("F", bound=Callable[..., Any])


def _relocate(path: str) -> str:
    try:
        return os.path.relpath(path, _REPO_ROOT)
    except ValueError:
        return path


class ErrorLocation:
    __slots__ = ("folder", "file", "line", "function", "code_line")

    def __init__(self, folder: str, file: str, line: int, function: str, code_line: str):
        self.folder = folder
        self.file = file
        self.line = line
        self.function = function
        self.code_line = code_line

    def __str__(self) -> str:
        return f"{self.folder}/{self.file}:{self.line} in {self.function}()"


def locate(exc: BaseException) -> Optional[ErrorLocation]:
    """Walk an exception's traceback and return the deepest AsterWhite-owned
    frame -- the actual line of AsterWhite code that failed, not a frame
    inside a third-party library it happened to call into."""
    tb = exc.__traceback__
    frames = traceback.extract_tb(tb)
    if not frames:
        return None
    candidates = [fr for fr in frames if _REPO_ROOT in os.path.abspath(fr.filename)]
    frame = candidates[-1] if candidates else frames[-1]
    abs_path = os.path.abspath(frame.filename)
    rel_path = _relocate(abs_path)
    folder, file = os.path.split(rel_path)
    return ErrorLocation(
        folder=folder or ".",
        file=file,
        line=frame.lineno or -1,
        function=frame.name,
        code_line=(frame.line or "").strip(),
    )


class AsterWhiteError(Exception):
    """Base exception for AsterWhite. Auto-captures the precise source
    location of whatever frame raised it, so failures during long
    streaming/training runs point straight at folder + file + line instead
    of a wall of library traceback."""

    def __init__(self, message: str, *, cause: Optional[BaseException] = None):
        super().__init__(message)
        self.message = message
        self.cause = cause
        self.location: Optional[ErrorLocation] = locate(cause) if cause else None

    def report(self) -> str:
        loc = self.location
        header = "[AsterWhite Error]"
        if loc is None:
            return f"{header} {self.message}"
        return (
            f"{header} {self.message}\n"
            f"  Folder:   {loc.folder}\n"
            f"  File:     {loc.file}\n"
            f"  Line:     {loc.line}\n"
            f"  Function: {loc.function}()\n"
            f"  Source:   {loc.code_line}"
        )


class DataCorruptionError(AsterWhiteError):
    """Raised by data_pipeline.py when a record fails byte-level validation."""


class StateIntegrityError(AsterWhiteError):
    """Raised by universe_tracker.py when a graph mutation would leave
    universe/character state inconsistent."""


def trace_errors(func: F) -> F:
    """Decorator: on failure, re-raise as AsterWhiteError carrying the exact
    folder/file/line/function of the failure, with the original exception
    chained as __cause__."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AsterWhiteError:
            raise
        except Exception as exc:  # boundary: intentionally broad
            wrapped = AsterWhiteError(f"{type(exc).__name__}: {exc}", cause=exc)
            raise wrapped from exc

    return wrapper  # type: ignore[return-value]


def get_logger(name: str) -> logging.Logger:
    """Logger whose formatter includes source path + line for every
    message, so anything logged -- not just exceptions -- is traceable to
    its origin."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(pathname)s:%(lineno)d (%(funcName)s) | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def install() -> None:
    """Install a process-wide excepthook so ANY uncaught exception -- not
    just ones wrapped in @trace_errors -- prints an AsterWhite-style
    folder/file/line report before the normal traceback."""
    default_hook = sys.excepthook

    def _hook(exc_type, exc_value, exc_tb):
        exc_value.__traceback__ = exc_tb
        loc = locate(exc_value)
        if loc is not None:
            sys.stderr.write(
                "\n[AsterWhite Error Trace]\n"
                f"  Folder:   {loc.folder}\n"
                f"  File:     {loc.file}\n"
                f"  Line:     {loc.line}\n"
                f"  Function: {loc.function}()\n"
                f"  Source:   {loc.code_line}\n"
                f"  {exc_type.__name__}: {exc_value}\n\n"
            )
        default_hook(exc_type, exc_value, exc_tb)

    sys.excepthook = _hook
