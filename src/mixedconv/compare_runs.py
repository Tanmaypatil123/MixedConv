from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two paired Step 1 runs")
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--require-bit-exact", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_metadata(run_dir: Path) -> dict:
    with (run_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.float().reshape(-1)
    right_flat = right.float().reshape(-1)
    return torch.nn.functional.cosine_similarity(left_flat, right_flat, dim=0).item()


def main() -> None:
    args = parse_args()
    reference = load_metadata(args.reference)
    candidate = load_metadata(args.candidate)

    for key in ("config_sha256", "manifest_sha256", "resolved_model_sha"):
        if reference[key] != candidate[key]:
            raise ValueError(f"Runs are not paired: {key} differs")

    reference_cases = {case["id"]: case for case in reference["cases"]}
    candidate_cases = {case["id"]: case for case in candidate["cases"]}
    if reference_cases.keys() != candidate_cases.keys():
        raise ValueError("Runs contain different prompt IDs")

    comparisons = []
    for case_id in sorted(reference_cases):
        reference_image = np.asarray(Image.open(args.reference / "images" / f"{case_id}.png"))
        candidate_image = np.asarray(Image.open(args.candidate / "images" / f"{case_id}.png"))
        reference_latent = load_file(args.reference / "latents" / f"{case_id}.safetensors")[
            "latents"
        ]
        candidate_latent = load_file(args.candidate / "latents" / f"{case_id}.safetensors")[
            "latents"
        ]
        image_equal = np.array_equal(reference_image, candidate_image)
        latent_equal = torch.equal(reference_latent, candidate_latent)
        comparisons.append(
            {
                "id": case_id,
                "image_bit_exact": image_equal,
                "image_mse": float(
                    np.mean(
                        (reference_image.astype(np.float32) - candidate_image.astype(np.float32))
                        ** 2
                    )
                ),
                "latent_bit_exact": latent_equal,
                "latent_mse": torch.mean(
                    (reference_latent.float() - candidate_latent.float()) ** 2
                ).item(),
                "latent_cosine": cosine_similarity(reference_latent, candidate_latent),
            }
        )

    summary = {
        "reference_run_id": reference["run_id"],
        "candidate_run_id": candidate["run_id"],
        "all_images_bit_exact": all(item["image_bit_exact"] for item in comparisons),
        "all_latents_bit_exact": all(item["latent_bit_exact"] for item in comparisons),
        "cases": comparisons,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    if args.require_bit_exact and not (
        summary["all_images_bit_exact"] and summary["all_latents_bit_exact"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
