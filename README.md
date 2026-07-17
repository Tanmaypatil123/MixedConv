# MixedConv

Research code for mixed-precision post-training quantization of Krea 2 Raw.

The project follows the ordered plan in `Krea2_Mixed_Precision_Project_Guide.pdf`. We are currently working on **Step 1: BF16 baseline and frozen evaluation harness**. Quantization is deliberately out of scope until the baseline is reproducible.

## Step 1 status

- [x] Frozen candidate experiment configuration
- [x] Disjoint sanity, calibration, and development prompt manifests
- [x] Prompt-to-seed pairing
- [x] Append-only run metadata and environment capture
- [x] Full-CUDA BF16 generation entry point for the cloud GPU
- [x] Linear-layer registry and BF16 identity-wrapper foundation
- [ ] Review and accept the Krea 2 Community License
- [ ] Install the GPU environment
- [ ] Resolve and record the exact model snapshot SHA
- [ ] Generate 8 sanity images
- [ ] Generate the 32-prompt BF16 development reference
- [ ] Repeat identical runs and measure the numerical noise floor
- [ ] Run the identity-swap equivalence test on Krea 2

## Setup

Python 3.11 or 3.12 is recommended. From PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"
```

The model is governed by the Krea 2 Community License. Review it on the model page before downloading weights:

<https://huggingface.co/krea/Krea-2-Raw>

## Commands

Validate the frozen config and manifests without loading the model:

```powershell
.venv\Scripts\python -m mixedconv.checks --config configs/experiment_config.yaml
```

Run the eight-image sanity set on the cloud GPU:

```powershell
.venv\Scripts\python -m mixedconv.baseline `
  --config configs/experiment_config.yaml `
  --manifest data/prompts/sanity.jsonl `
  --offload none `
  --license-accepted
```

Run the complete 32-prompt development reference on a large GPU:

```powershell
.venv\Scripts\python -m mixedconv.baseline `
  --config configs/experiment_config.yaml `
  --manifest data/prompts/dev.jsonl `
  --offload none `
  --license-accepted
```

The detailed cloud sequence and Step 1 acceptance criteria are in [`docs/STEP_1_CLOUD.md`](docs/STEP_1_CLOUD.md).

Every invocation creates a new folder below `runs/` and appends one record to `runs/index.jsonl`. Generated data and model caches are ignored by Git.

## Frozen-baseline rule

After the first accepted BF16 reference is produced, do not edit the experiment config or prompt manifests. If a numerical setting must change, create a new config version and regenerate every baseline comparison.
