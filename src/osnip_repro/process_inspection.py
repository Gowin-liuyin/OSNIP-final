"""Portable process command-line inspection for trainer identity monitors.

The identity monitors (scripts/verify_summary_scorer.py and
scripts/continue_after_legacy_run.py) parse a live process command line.
``ps`` does not exist on Windows and posix shlex splitting eats backslash
path separators, so both behaviours live here behind one helper pair.
"""
from __future__ import annotations

import shlex
import subprocess


def command_tokens(command: str) -> list[str]:
    """Split a command line without eating Windows path separators.

    posix shlex treats backslash as an escape character, so Windows absolute
    paths lose their separators; split non-posix and strip one layer of
    surrounding quotes instead.
    """
    tokens = shlex.split(command, posix=False)
    return [token[1:-1] if len(token) >= 2 and token[0] == token[-1] and token[0] in '"\'' else token
            for token in tokens if token.strip()]


def process_command(pid: int) -> str | None:
    """Best-effort command line of a live process; None once it has exited."""
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            cmdline = psutil.Process(int(pid)).cmdline()
        except psutil.Error:
            return None
        return " ".join(cmdline) or None
    result = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() or None