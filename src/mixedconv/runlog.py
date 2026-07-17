from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_directory(output_root: str | Path, manifest_path: str | Path) -> tuple[str, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_stem = Path(manifest_path).stem
    run_id = f"{timestamp}-{manifest_stem}-{os.urandom(3).hex()}"
    run_dir = Path(output_root) / run_id
    (run_dir / "images").mkdir(parents=True, exist_ok=False)
    (run_dir / "latents").mkdir()
    return run_id, run_dir


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit(),
    }
    try:
        import torch

        record["torch"] = torch.__version__
        record["cuda_runtime"] = torch.version.cuda
        record["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            record["gpu"] = torch.cuda.get_device_name(0)
            record["gpu_capability"] = list(torch.cuda.get_device_capability(0))
            record["gpu_vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except ImportError:
        record["torch"] = None

    for package in ("accelerate", "diffusers", "transformers", "huggingface_hub"):
        try:
            module = __import__(package)
            record[package] = module.__version__
            distribution_name = "huggingface-hub" if package == "huggingface_hub" else package
            distribution = importlib.metadata.distribution(distribution_name)
            direct_url = distribution.read_text("direct_url.json")
            if direct_url:
                record[f"{package}_direct_url"] = json.loads(direct_url)
        except (ImportError, importlib.metadata.PackageNotFoundError):
            record[package] = None
    return record


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    index_path = Path(path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
