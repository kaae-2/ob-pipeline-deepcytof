import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import tarfile
import shutil
import os


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[deepcytof] {timestamp} {message}", flush=True)


def run_tf_probe(env: dict) -> None:
    probe_cmd = [
        sys.executable,
        "-c",
        "import tensorflow as tf; print('TF GPUs:', tf.config.list_physical_devices('GPU'))",
    ]
    log("Running TensorFlow GPU probe")
    try:
        result = subprocess.run(
            probe_cmd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        log(f"TensorFlow GPU probe failed: {exc}")
        return

    if result.stdout:
        for line in result.stdout.strip().splitlines():
            log(f"TF probe stdout: {line}")
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            log(f"TF probe stderr: {line}")
    log(f"TensorFlow GPU probe exit code: {result.returncode}")

def extract_if_tar(file_path, extract_to):
    """Snakemake-safe extraction: returns absolute path to the extracted file."""
    path = Path(file_path).resolve()
    if any(path.name.endswith(ext) for ext in ['.tar.gz', '.tar', '.tgz']):
        print(f"Snakemake Job: Extracting {path.name}...", flush=True)
        with tarfile.open(path, "r:*") as tar:
            try:
                tar.extractall(path=extract_to, filter="data")
            except TypeError:
                tar.extractall(path=extract_to)
            # Find all relevant files
            extracted_files = [f for f in extract_to.glob("**/*") if f.suffix in ['.csv', '.txt'] and f.is_file()]
            if extracted_files:
                # Sort to ensure deterministic behavior in Snakemake
                extracted_files.sort()
                return str(extracted_files[0].resolve())
    return str(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--data.train_matrix", dest="train_matrix", type=str, required=True)
    parser.add_argument("--data.train_labels", dest="train_labels", type=str, required=True)
    parser.add_argument("--data.test_matrix", dest="test_matrix", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_extract = output_dir / f"tmp_{args.name}_extracted"
    if tmp_extract.exists():
        log(f"Removing existing temp directory: {tmp_extract}")
        shutil.rmtree(tmp_extract)
    tmp_extract.mkdir(parents=True, exist_ok=True)

    log("Preparing inputs")
    log(f"Input train matrix: {args.train_matrix}")
    log(f"Input train labels: {args.train_labels}")
    log(f"Input test matrix: {args.test_matrix}")
    train_x_csv = extract_if_tar(args.train_matrix, tmp_extract)
    train_y_csv = extract_if_tar(args.train_labels, tmp_extract)
    test_x_csv = str(Path(args.test_matrix).resolve())
    log(f"Resolved train matrix: {train_x_csv}")
    log(f"Resolved train labels: {train_y_csv}")
    log(f"Resolved test matrix: {test_x_csv}")

    repo_root = Path(__file__).resolve().parent
    run_script = repo_root / "deepcytof_pipeline" / "run_deepcytof.py"
    output_file = output_dir / f"{args.name}_predicted_labels.tar.gz"

    cmd = [
        sys.executable, "-u", str(run_script), # -u forces unbuffered output
        "--train_x", train_x_csv,
        "--train_y", train_y_csv,
        "--test_x", test_x_csv,
        "--output_file", str(output_file),
        "--dataset_name", args.name
    ]

    log(f"Snakemake Rule Start: {args.name}")
    log("Launching DeepCyTOF pipeline")
    log(f"Command: {' '.join(cmd)}")
    
    # Real-time output streaming
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "2")
    env.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    env.setdefault("TF_NUM_INTEROP_THREADS", "1")
    env.setdefault("DEEPCYTOF_DAE_EPOCHS", "1")
    env.setdefault("DEEPCYTOF_DAE_BATCH_SIZE", "2048")
    env.setdefault("DEEPCYTOF_CLF_EPOCHS", "1")
    env.setdefault("DEEPCYTOF_CLF_BATCH_SIZE", "2048")
    env.setdefault("DEEPCYTOF_PRED_BATCH_SIZE", "1024")
    env.setdefault("DEEPCYTOF_MAX_PRED_CELLS", "20000")
    env.setdefault("DEEPCYTOF_PRED_SEED", "42")
    env.setdefault("DEEPCYTOF_TRAIN_LOG_EVERY", "0")
    env.setdefault("DEEPCYTOF_SKIP_MMD", "1")
    env.setdefault("DEEPCYTOF_PRED_LOG", "1")
    log(
        "CPU overrides: CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL="
        f"{env['TF_CPP_MIN_LOG_LEVEL']}"
    )
    log(
        "DeepCyTOF tuning: DAE_EPOCHS="
        f"{env['DEEPCYTOF_DAE_EPOCHS']} DAE_BATCH_SIZE="
        f"{env['DEEPCYTOF_DAE_BATCH_SIZE']} CLF_EPOCHS="
        f"{env['DEEPCYTOF_CLF_EPOCHS']} CLF_BATCH_SIZE="
        f"{env['DEEPCYTOF_CLF_BATCH_SIZE']} PRED_BATCH_SIZE="
        f"{env['DEEPCYTOF_PRED_BATCH_SIZE']} MAX_PRED_CELLS="
        f"{env['DEEPCYTOF_MAX_PRED_CELLS']} SKIP_MMD="
        f"{env['DEEPCYTOF_SKIP_MMD']}"
    )
    run_tf_probe(env)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    output_lines = []
    status_prefixes = (
        "--- Preparing",
        "--- Processing",
        "DeepCyTOF complete",
    )
    if process.stdout is not None:
        for line in process.stdout:
            output_lines.append(line)
            if line.startswith(status_prefixes):
                print(f"[{args.name}] {line}", end='', flush=True)

    process.wait()
    log(f"Pipeline process exited with code {process.returncode}")

    if process.returncode != 0:
        sys.stderr.write("".join(output_lines))
        print(f"ERROR: {args.name} failed with exit code {process.returncode}", flush=True)
        sys.exit(process.returncode)

    shutil.rmtree(tmp_extract)
    log("Pipeline completed")
    log(f"SUCCESS: {args.name} finished")

if __name__ == "__main__":
    main()
