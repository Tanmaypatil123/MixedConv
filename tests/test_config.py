from pathlib import Path

from mixedconv.config import assert_disjoint, load_config, load_manifest


ROOT = Path(__file__).parents[1]


def test_frozen_config_and_manifests() -> None:
    config = load_config(ROOT / "configs" / "experiment_config.yaml")
    protocol = config["protocol"]
    manifests = {
        "sanity": load_manifest(ROOT / protocol["sanity_manifest"]),
        "calibration": load_manifest(ROOT / protocol["calibration_manifest"]),
        "dev": load_manifest(ROOT / protocol["dev_manifest"]),
    }
    assert_disjoint(manifests)
    assert {name: len(cases) for name, cases in manifests.items()} == {
        "sanity": 8,
        "calibration": 8,
        "dev": 32,
    }
