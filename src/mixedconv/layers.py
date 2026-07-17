from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


BLOCK_PATTERNS = (
    re.compile(r"(?:transformer_blocks|blocks|layers)\.(\d+)"),
    re.compile(r"(?:single_transformer_blocks|single_blocks)\.(\d+)"),
)


@dataclass(frozen=True)
class LayerRecord:
    name: str
    weight_shape: tuple[int, ...]
    block_index: int | None
    functional_type: str
    in_features: int
    out_features: int
    has_bias: bool


def block_index(name: str) -> int | None:
    for pattern in BLOCK_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return None


def functional_type(name: str) -> str:
    lowered = name.lower()
    mappings = (
        (("to_q", "q_proj"), "attention_q"),
        (("to_k", "k_proj"), "attention_k"),
        (("to_v", "v_proj"), "attention_v"),
        (("to_out", "out_proj"), "attention_out"),
        (("w1", "gate_proj"), "swiglu_w1"),
        (("w3", "up_proj"), "swiglu_w3"),
        (("w2", "down_proj"), "swiglu_w2"),
        (("mod", "adaln"), "modulation"),
    )
    for tokens, label in mappings:
        if any(token in lowered for token in tokens):
            return label
    return "other_linear"


def build_linear_registry(module: nn.Module) -> list[LayerRecord]:
    records = []
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            records.append(
                LayerRecord(
                    name=name,
                    weight_shape=tuple(child.weight.shape),
                    block_index=block_index(name),
                    functional_type=functional_type(name),
                    in_features=child.in_features,
                    out_features=child.out_features,
                    has_bias=child.bias is not None,
                )
            )
    return records


def write_registry(path: str | Path, records: list[LayerRecord]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump([asdict(record) for record in records], handle, indent=2)
        handle.write("\n")


class BF16IdentityWrapper(nn.Module):
    """Transparent wrapper used to prove that config-driven swapping is neutral."""

    def __init__(self, inner: nn.Linear):
        super().__init__()
        self.inner = inner

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.inner(inputs)


def _replace_submodule(root: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, replacement)


def apply_precision_map(module: nn.Module, precision_map: dict[str, str]) -> None:
    named_modules = dict(module.named_modules())
    unknown = set(precision_map).difference(named_modules)
    if unknown:
        raise KeyError(f"Precision map contains unknown layers: {sorted(unknown)}")

    for name, precision in precision_map.items():
        target = named_modules[name]
        if not isinstance(target, nn.Linear):
            raise TypeError(f"Target is not nn.Linear: {name}")
        if precision != "bf16":
            raise ValueError(f"Step 1 only supports the bf16 identity wrapper, got {precision}")
        _replace_submodule(module, name, BF16IdentityWrapper(target))


def identity_precision_map(records: list[LayerRecord]) -> dict[str, str]:
    return {record.name: "bf16" for record in records}
