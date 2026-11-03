"""Per-stage wall-clock timing for runtime reporting.

Each pipeline script records the duration of its stage into
data/<dataset>/pipeline_timings.json. summarize_runtime.py aggregates these files.

    {"hardware": {...},
     "stages": {"<stage>": {"seconds": 12.3, "updated": "...", "items": {"obj": 4.1}, ...}}}

A stage with `item` (e.g. one object) stores a value per item; the stage total is the
sum of the items, except that an item named "__all__" (a run over every object) takes
precedence over per-object runs, so a later single-object rerun cannot double count.
Recording never raises: a timing problem must not fail a pipeline run.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path

TIMINGS_FILE = "pipeline_timings.json"
ALL_ITEMS = "__all__"


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def hardware() -> dict:
    """Best-effort description of the machine the timings were measured on."""
    info = {"platform": platform.platform(), "logical_cpus": os.cpu_count()}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                info["ram_gib"] = round(int(line.split()[1]) / 1024 ** 2, 1)
                break
    except OSError:
        pass
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        info["gpu"] = gpu.splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return info


def stage_total(entry: dict) -> float:
    items = entry.get("items")
    if not items:
        return float(entry.get("seconds", 0.0))
    if ALL_ITEMS in items:
        return float(items[ALL_ITEMS])
    return float(sum(items.values()))


def record(dataset, stage: str, seconds: float, item: str | None = None, **meta) -> None:
    """Store `seconds` for `stage` (optionally for one `item`) in the dataset's timing file."""
    try:
        path = Path(dataset) / TIMINGS_FILE
        data = _read(path)
        data.setdefault("hardware", None)
        if not data["hardware"]:
            data["hardware"] = hardware()
        entry = data.setdefault("stages", {}).setdefault(stage, {})
        if item is not None:
            entry.setdefault("items", {})[item] = round(float(seconds), 3)
            entry["seconds"] = round(stage_total(entry), 3)
        else:
            entry["seconds"] = round(float(seconds), 3)
        entry["updated"] = datetime.now().isoformat(timespec="seconds")
        entry.update(meta)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(path)
    except Exception as error:  # noqa: BLE001 - timing must never break a pipeline run
        print(f"Warning: could not record timing for {stage}: {error}", flush=True)


def total(dataset, stage: str) -> float:
    """Current recorded total of a stage (0 if absent)."""
    entry = _read(Path(dataset) / TIMINGS_FILE).get("stages", {}).get(stage)
    return stage_total(entry) if entry else 0.0
