# DeepCyTOF Module

## What this module does

Wraps DeepCyTOF training/inference for benchmark execution.

The local wrapper now consumes preprocessing outputs as-is and skips the prior
log/standard-scaling feature transforms inside the runner.

- Entrypoint wrapper: `entrypoint_deepcytof.py`
- Pipeline runner: `deepcytof_pipeline/run_deepcytof.py`
- Local runner: `run_deepcytof.sh`
- Output: `deepcytof_predicted_labels.tar.gz`

## Run locally

```bash
bash models/deepcytof/run_deepcytof.sh
```

## Run as part of benchmark

Defined in `benchmark/Clustering_conda.yml` analysis stage, executed by:

```bash
just benchmark
```

## What `run_deepcytof.sh` needs

- Preprocessing outputs at `models/deepcytof/out/data/data_preprocessing/default`
- Either:
  - `models/deepcytof/.venv/bin/python`, or
  - conda env `deepcytof_rocm` created from `models/deepcytof/deepcytof.yml`
- TensorFlow/Keras stack and numeric dependencies
- ROCm-related env setup if running GPU path (`HSA_OVERRIDE_GFX_VERSION`,
  `LD_LIBRARY_PATH`)
