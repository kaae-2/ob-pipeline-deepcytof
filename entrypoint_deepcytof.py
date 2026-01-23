import argparse
from pathlib import Path
import subprocess
import sys
import tarfile
import shutil
import os

def extract_if_tar(file_path, extract_to):
    """Snakemake-safe extraction: returns absolute path to the extracted file."""
    path = Path(file_path).resolve()
    if any(path.name.endswith(ext) for ext in ['.tar.gz', '.tar', '.tgz']):
        print(f"Snakemake Job: Extracting {path.name}...", flush=True)
        with tarfile.open(path, "r:*") as tar:
            tar.extractall(path=extract_to)
            # Find all relevant files
            extracted_files = [f for f in extract_to.glob("**/*") if f.suffix in ['.csv', '.txt'] and f.is_file()]
            if extracted_files:
                # Sort to ensure deterministic behavior in Snakemake
                extracted_files.sort()
                print(f"DEBUG: Found {len(extracted_files)} files. Using: {extracted_files[0].name}", flush=True)
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
    if tmp_extract.exists(): shutil.rmtree(tmp_extract)
    tmp_extract.mkdir(parents=True, exist_ok=True)

    train_x_csv = extract_if_tar(args.train_matrix, tmp_extract)
    train_y_csv = extract_if_tar(args.train_labels, tmp_extract)
    test_x_csv = extract_if_tar(args.test_matrix, tmp_extract)

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

    print(f"Snakemake Rule Start: {args.name}", flush=True)
    
    # Real-time output streaming
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        bufsize=1
    )

    for line in process.stdout:
        print(f"[{args.name}] {line}", end='', flush=True)

    process.wait()

    if process.returncode != 0:
        print(f"ERROR: {args.name} failed with exit code {process.returncode}", flush=True)
        sys.exit(process.returncode)

    shutil.rmtree(tmp_extract)
    print(f"SUCCESS: {args.name} finished.", flush=True)

if __name__ == "__main__":
    main()