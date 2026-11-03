"""Safe subprocess execution with incremental Streamlit output."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Callable

MAX_LOG_LINES = 150


def run_command(command: list[str], cwd: Path,
                update: Callable[[str], None],
                environment_overrides: dict[str, str] | None = None,
                ) -> tuple[int, str]:
    """Run a validated argument vector and stream merged stdout/stderr."""
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    if environment_overrides:
        environment.update(environment_overrides)
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in iter(process.stdout.readline, ""):
        lines.append(line.rstrip())
        if len(lines) > MAX_LOG_LINES:
            lines = lines[-MAX_LOG_LINES:]
        update("\n".join(lines))
    while process.poll() is None:
        time.sleep(0.05)
    output = "\n".join(lines)
    return int(process.returncode), output
