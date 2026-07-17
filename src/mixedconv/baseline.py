from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from mixedconv.config import config_hash, load_config, load_manifest
from mixedconv.runlog import (
    append_jsonl,
    environment_record,
    file_sha256,
    new_run_directory,
    utc_now,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paired Krea 2 Raw BF16 references")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--offload",
        choices=("none", "model", "sequential"),
        default=None,
        help="Override config runtime.offload. The cloud baseline should use none.",
    )
    parser.add_argument(
        "--license-accepted",
        action="store_true",
        help="Confirm that you reviewed and accept the Krea 2 Community License.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--identity-swap",
        action="store_true",
        help="Wrap every transformer nn.Linear in the transparent BF16 wrapper.",
    )
    return parser.parse_args()


def configure_numerics(torch: Any, config: dict[str, Any]) -> None:
    inference = config["inference"]
    torch.backends.cuda.matmul.allow_tf32 = bool(inference["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(inference["allow_tf32"])
    torch.use_deterministic_algorithms(bool(inference["deterministic_algorithms"]))


def load_pipeline(config: dict[str, Any], offload: str, local_files_only: bool):
    import torch
    from diffusers import Krea2Pipeline

    model = config["model"]
    pipe = Krea2Pipeline.from_pretrained(
        model["id"],
        revision=model["revision"],
        torch_dtype=torch.bfloat16,
        local_files_only=local_files_only,
    )
    if offload == "none":
        pipe.to(config["runtime"]["device"])
    elif offload == "model":
        pipe.enable_model_cpu_offload()
    else:
        pipe.enable_sequential_cpu_offload()
    pipe.set_progress_bar_config(disable=False)
    return pipe


def resolve_model_sha(model_id: str, revision: str, local_files_only: bool) -> str | None:
    if local_files_only:
        return None
    from huggingface_hub import model_info

    return model_info(model_id, revision=revision).sha


def main() -> None:
    args = parse_args()
    if not args.license_accepted:
        raise SystemExit(
            "Refusing to download or run Krea 2 until --license-accepted is supplied after review."
        )

    import torch
    from safetensors.torch import save_file

    config = load_config(args.config)
    cases = load_manifest(args.manifest)
    offload = args.offload or config["runtime"]["offload"]
    if offload == "none" and not torch.cuda.is_available():
        raise RuntimeError("The no-offload BF16 baseline requires a CUDA GPU")

    configure_numerics(torch, config)
    model_sha = resolve_model_sha(
        config["model"]["id"], config["model"]["revision"], args.local_files_only
    )

    run_id, run_dir = new_run_directory(config["artifacts"]["output_root"], args.manifest)
    started_at = utc_now()
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "started_at": started_at,
        "config_path": str(args.config),
        "config_sha256": config_hash(config),
        "config_file_sha256": file_sha256(args.config),
        "manifest_path": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "model_id": config["model"]["id"],
        "requested_model_revision": config["model"]["revision"],
        "resolved_model_sha": model_sha,
        "offload": offload,
        "identity_swap": args.identity_swap,
        "environment": environment_record(),
        "cases": [],
    }

    pipe = load_pipeline(config, offload, args.local_files_only)
    if args.identity_swap:
        from mixedconv.layers import (
            apply_precision_map,
            build_linear_registry,
            identity_precision_map,
            write_registry,
        )

        registry = build_linear_registry(pipe.transformer)
        write_registry(run_dir / "layer_registry.json", registry)
        apply_precision_map(pipe.transformer, identity_precision_map(registry))
        metadata["registered_linear_layers"] = len(registry)
    inference = config["inference"]
    device = config["runtime"]["device"]
    torch.cuda.reset_peak_memory_stats()
    run_start = time.perf_counter()

    for case in cases:
        final_latents: dict[str, Any] = {}

        def capture_final_latents(_pipe, step_index, _timestep, callback_kwargs):
            if step_index == inference["num_inference_steps"] - 1:
                final_latents["latents"] = callback_kwargs["latents"].detach().to("cpu")
            return callback_kwargs

        generator = torch.Generator(device=device).manual_seed(case.seed)
        case_start = time.perf_counter()
        result = pipe(
            prompt=case.prompt,
            width=inference["width"],
            height=inference["height"],
            num_inference_steps=inference["num_inference_steps"],
            guidance_scale=inference["guidance_scale"],
            num_images_per_prompt=inference["num_images_per_prompt"],
            max_sequence_length=inference["max_sequence_length"],
            generator=generator,
            callback_on_step_end=capture_final_latents,
            callback_on_step_end_tensor_inputs=["latents"],
        )
        elapsed = time.perf_counter() - case_start

        image_path = run_dir / "images" / f"{case.id}.png"
        result.images[0].save(image_path)
        latent_path = run_dir / "latents" / f"{case.id}.safetensors"
        if "latents" not in final_latents:
            raise RuntimeError(f"Final latent callback did not fire for {case.id}")
        save_file(final_latents, latent_path)

        metadata["cases"].append(
            {
                "id": case.id,
                "category": case.category,
                "seed": case.seed,
                "prompt": case.prompt,
                "elapsed_seconds": elapsed,
                "image": str(image_path),
                "image_sha256": file_sha256(image_path),
                "latent": str(latent_path),
                "latent_sha256": file_sha256(latent_path),
            }
        )

    metadata.update(
        {
            "status": "complete",
            "completed_at": utc_now(),
            "total_elapsed_seconds": time.perf_counter() - run_start,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        }
    )
    write_json(run_dir / "metadata.json", metadata)
    append_jsonl(
        config["artifacts"]["append_only_index"],
        {
            "run_id": run_id,
            "status": metadata["status"],
            "started_at": started_at,
            "completed_at": metadata["completed_at"],
            "config_sha256": metadata["config_sha256"],
            "manifest_sha256": metadata["manifest_sha256"],
            "resolved_model_sha": model_sha,
            "run_directory": str(run_dir),
        },
    )
    print(f"run_id={run_id}")
    print(f"metadata={run_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
