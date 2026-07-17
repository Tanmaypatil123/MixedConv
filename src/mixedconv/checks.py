from __future__ import annotations

import argparse
from pathlib import Path

from mixedconv.config import assert_disjoint, config_hash, load_config, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen Step 1 protocol")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    protocol = config["protocol"]
    manifests = {
        "sanity": load_manifest(protocol["sanity_manifest"]),
        "calibration": load_manifest(protocol["calibration_manifest"]),
        "dev": load_manifest(protocol["dev_manifest"]),
    }
    assert_disjoint(manifests)

    expected_counts = {"sanity": 8, "calibration": 8, "dev": 32}
    actual_counts = {name: len(cases) for name, cases in manifests.items()}
    if actual_counts != expected_counts:
        raise ValueError(f"Manifest counts differ: expected={expected_counts}, actual={actual_counts}")

    print(f"config_sha256={config_hash(config)}")
    for name, count in actual_counts.items():
        print(f"{name}_prompts={count}")
    print("protocol=valid")


if __name__ == "__main__":
    main()
