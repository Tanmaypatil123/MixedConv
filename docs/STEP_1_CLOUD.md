# Step 1 cloud runbook

This runbook is for the full-CUDA BF16 reference on the RTX PRO 6000 cloud machine. Run every command from the repository root. Do not begin Step 2 until all acceptance criteria at the bottom pass.

## 1. Review the model license

Read the Krea 2 Community License linked from:

<https://huggingface.co/krea/Krea-2-Raw>

The generation command requires `--license-accepted` as an explicit confirmation. This flag records intent but does not replace compliance with the license or acceptable-use policy.

## 2. Create an isolated environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

The Diffusers dependency comes from its official Git repository because Krea's model card currently requires the source version. The run metadata captures its resolved Git commit from `direct_url.json`.

## 3. Validate the protocol before downloading weights

```bash
python -m mixedconv.checks --config configs/experiment_config.yaml
pytest -q
```

Expected manifest counts are 8 sanity, 8 calibration, and 32 development prompts. These sets must remain disjoint.

## 4. Generate the sanity reference

```bash
python -m mixedconv.baseline \
  --config configs/experiment_config.yaml \
  --manifest data/prompts/sanity.jsonl \
  --offload none \
  --license-accepted
```

Inspect all eight images. Raw is a mid-training checkpoint, so imperfect aesthetics are not automatically a failure. Stop for obvious corruption, blank images, severe numerical artifacts, or inconsistent dimensions.

## 5. Measure identical-run noise

Repeat the command above without changing anything, then compare the two produced run directories:

```bash
python -m mixedconv.compare_runs runs/<FIRST_RUN> runs/<SECOND_RUN>
```

Store the comparison output. If results are not bit-exact, repeat until there are five runs; their variation defines the minimum detectable effect for subsequent experiments.

## 6. Prove the identity swap

Generate the same sanity manifest with every transformer linear wrapped in the BF16 identity wrapper:

```bash
python -m mixedconv.baseline \
  --config configs/experiment_config.yaml \
  --manifest data/prompts/sanity.jsonl \
  --offload none \
  --identity-swap \
  --license-accepted
```

Compare it against the first accepted baseline and require bit-exact output:

```bash
python -m mixedconv.compare_runs \
  runs/<REFERENCE_RUN> \
  runs/<IDENTITY_SWAP_RUN> \
  --require-bit-exact \
  --output runs/identity_swap_comparison.json
```

The identity-swap run also writes `layer_registry.json`, which becomes the shared layer vocabulary for later sensitivity and allocation work.

## 7. Generate the development reference

After sanity, repeatability, and identity swapping pass:

```bash
python -m mixedconv.baseline \
  --config configs/experiment_config.yaml \
  --manifest data/prompts/dev.jsonl \
  --offload none \
  --license-accepted
```

Copy `resolved_model_sha` from the accepted run metadata into a new frozen configuration revision. Never silently edit the configuration used by an existing run.

## Acceptance criteria

- Eight sanity images have been visually inspected.
- The exact model snapshot, package versions, Git commit, GPU, timings, and peak memory are recorded.
- Numerical repeatability has been measured; any nonzero variation is documented as the noise floor.
- The identity-wrapper run is bit-exact to the unwrapped reference.
- The 32-prompt BF16 development reference, images, and final latents are stored durably.
- `layer_registry.json` exists and contains every target transformer `nn.Linear`.
