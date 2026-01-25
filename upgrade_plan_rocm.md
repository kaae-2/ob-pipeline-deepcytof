# ROCm Upgrade Plan (DeepCyTOF)

This plan upgrades the existing `deepcytof_legacy` environment in place for ROCm
GPU support. No code changes are included yet.

## Steps

1) Backup the current environment (rollback safety)

```bash
conda env export -n deepcytof_legacy > /home/kaae2/code/ob/models/ob-pipeline-deepcytof/deepcytof_legacy.freeze.yml
```

2) Update the environment to a modern Python and ROCm TensorFlow

- Choose Python 3.10 or 3.11.
- Remove TF1.15 + standalone `keras`.
- Install `tensorflow-rocm` via pip inside the same env name.

```bash
conda activate deepcytof_legacy
conda install -y python=3.10 pip
pip uninstall -y tensorflow keras
pip install tensorflow-rocm
```

3) Verify ROCm GPU visibility

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

4) Run DeepCyTOF unchanged (smoke test)

```bash
python /home/kaae2/code/ob/models/ob-pipeline-deepcytof/entrypoint_deepcytof.py \
  --name deepcytof \
  --output_dir /tmp/deepcytof_test \
  --data.train_matrix <train_matrix.tar.gz> \
  --data.train_labels <train_labels.tar.gz> \
  --data.test_matrix <test_matrix.tar.gz>
```

5) Decision point

- If it runs, GPU acceleration is enabled.
- If it fails due to TF2/Keras compatibility, decide on minimal import updates
  (`keras` -> `tf.keras`, `tf.compat.v1` shims).
