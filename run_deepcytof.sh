#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
python_cmd=("${script_dir}/.venv/bin/python")
if [ ! -x "${python_cmd[0]}" ]; then
  if command -v conda >/dev/null 2>&1; then
    if conda env list | awk '{print $1}' | grep -Fxq "deepcytof_legacy"; then
      python_cmd=(conda run -n deepcytof_legacy python)
    else
      echo "ERROR: deepcytof_legacy conda env not found. Create it with:" >&2
      echo "  conda env create -f ${script_dir}/deepcytof.yml -n deepcytof_legacy" >&2
      exit 1
    fi
  else
    echo "ERROR: conda not found and no ${script_dir}/.venv/bin/python available." >&2
    exit 1
  fi
fi

"${python_cmd[@]}" "${script_dir}/entrypoint_deepcytof.py" \
  --name "deepcytof" \
  --output_dir "${script_dir}/out/data/analysis/default/deepcytof" \
  --data.train_matrix "${script_dir}/out/data/data_preprocessing/default/data_import.train.matrix.tar.gz" \
  --data.train_labels "${script_dir}/out/data/data_preprocessing/default/data_import.train.labels.tar.gz" \
  --data.test_matrix "${script_dir}/out/data/data_preprocessing/default/data_import.test.matrices.tar.gz"
