from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PromptCase:
    id: str
    category: str
    seed: int
    prompt: str


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required_sections = {"model", "inference", "artifacts", "protocol"}
    missing = required_sections.difference(config)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")

    inference = config["inference"]
    expected = {
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 52,
        "guidance_scale": 3.5,
        "num_images_per_prompt": 1,
    }
    mismatches = {
        key: (inference.get(key), value)
        for key, value in expected.items()
        if inference.get(key) != value
    }
    if mismatches:
        raise ValueError(f"BF16 v1 protocol mismatch: {mismatches}")

    if config["model"].get("transformer_dtype") != "bfloat16":
        raise ValueError("Step 1 transformer dtype must be bfloat16")
    if inference.get("torch_compile") is not False:
        raise ValueError("torch.compile must remain disabled for the v1 baseline")


def config_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_manifest(path: str | Path) -> list[PromptCase]:
    manifest_path = Path(path)
    cases: list[PromptCase] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                cases.append(PromptCase(**record))
            except (json.JSONDecodeError, TypeError) as error:
                raise ValueError(f"Invalid {manifest_path}:{line_number}: {error}") from error
    validate_manifest(cases, manifest_path)
    return cases


def validate_manifest(cases: list[PromptCase], source: Path | None = None) -> None:
    label = str(source) if source else "manifest"
    if not cases:
        raise ValueError(f"{label} is empty")
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} contains duplicate prompt IDs")
    pairs = [(case.prompt, case.seed) for case in cases]
    if len(pairs) != len(set(pairs)):
        raise ValueError(f"{label} contains duplicate prompt/seed pairs")
    if any(case.seed < 0 for case in cases):
        raise ValueError(f"{label} contains a negative seed")


def assert_disjoint(manifests: dict[str, list[PromptCase]]) -> None:
    names = list(manifests)
    for left_index, left_name in enumerate(names):
        left_prompts = {case.prompt for case in manifests[left_name]}
        for right_name in names[left_index + 1 :]:
            right_prompts = {case.prompt for case in manifests[right_name]}
            overlap = left_prompts.intersection(right_prompts)
            if overlap:
                raise ValueError(
                    f"Prompt leakage between {left_name} and {right_name}: {sorted(overlap)}"
                )
