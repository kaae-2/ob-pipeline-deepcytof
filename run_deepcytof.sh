#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
model_name="deepcytof"
conda_env="deepcytof_legacy"

train_matrix="${script_dir}/out/data/data_preprocessing/default/data_import.train.matrix.tar.gz"
train_labels="${script_dir}/out/data/data_preprocessing/default/data_import.train.labels.tar.gz"
test_matrix="${script_dir}/out/data/data_preprocessing/default/data_import.test.matrices.tar.gz"

for required_file in "$train_matrix" "$train_labels" "$test_matrix"; do
  if [ ! -f "$required_file" ]; then
    echo "ERROR: missing input file: ${required_file}" >&2
    exit 1
  fi
done

python_cmd=("${script_dir}/.venv/bin/python")
if [ ! -x "${python_cmd[0]}" ]; then
  if command -v conda >/dev/null 2>&1; then
    if conda env list | awk '{print $1}' | grep -Fxq "${conda_env}"; then
      python_cmd=(conda run --no-capture-output -n "${conda_env}" python)
    else
      echo "ERROR: ${conda_env} conda env not found. Create it with:" >&2
      echo "  conda env create -f ${script_dir}/deepcytof.yml -n ${conda_env}" >&2
      exit 1
    fi
  else
    echo "ERROR: conda not found and no ${script_dir}/.venv/bin/python available." >&2
    exit 1
  fi
fi

output_dir="${script_dir}/out/data/analysis/default/${model_name}"

cmd=(
  "${python_cmd[@]}" "${script_dir}/entrypoint_deepcytof.py"
  --name "${model_name}"
  --output_dir "${output_dir}"
  --data.train_matrix "${train_matrix}"
  --data.train_labels "${train_labels}"
  --data.test_matrix "${test_matrix}"
)

"${cmd[@]}"
