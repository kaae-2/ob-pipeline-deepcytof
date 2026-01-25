#!/usr/bin/env python3
import os
import sys
import tarfile
import tempfile
import re
import contextlib
import io
import pandas as pd
import argparse
from pathlib import Path

# Prevent Matplotlib from hanging on font-cache building
import matplotlib
matplotlib.use('Agg')

# Fix for Protobuf legacy issues
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from deepcytof_core import DeepCyTOFRunner


def log_gpu_status():
    try:
        import tensorflow as tf
    except Exception as exc:
        print(f"--- TensorFlow import failed: {exc}", flush=True)
        return

    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception as exc:
        print(f"--- TensorFlow GPU detection failed: {exc}", flush=True)
        return

    if gpus:
        names = ", ".join([gpu.name for gpu in gpus])
        print(f"--- TensorFlow GPUs detected: {names}", flush=True)
    else:
        print("--- TensorFlow GPUs detected: none (CPU-only)", flush=True)

def extract_first_csv_from_tar(path_str, temp_dir):
    path = Path(path_str).resolve()
    if path.suffix in ['.csv', '.txt']:
        return str(path)
    
    with tarfile.open(path, "r:*") as tar:
        member = next((m for m in tar.getmembers() if m.name.endswith('.csv')), None)
        if not member:
            raise ValueError(f"No CSV found in {path_str}")
        tar.extract(member, path=temp_dir)
        return str(Path(temp_dir) / member.name)


def extract_sample_number(sample_name):
    base = os.path.basename(sample_name)
    while True:
        root, ext = os.path.splitext(base)
        if not ext:
            break
        base = root
    match = re.search(r"(\d+)(?!.*\d)", base)
    if match:
        return match.group(1)
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_x', required=True)
    parser.add_argument('--train_y', required=True)
    parser.add_argument('--test_x', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--dataset_name', default='cytof_data')
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        print("--- Preparing Training Data ---", flush=True)
        train_x_csv = extract_first_csv_from_tar(args.train_x, tmpdir)
        train_y_csv = extract_first_csv_from_tar(args.train_y, tmpdir)

        log_gpu_status()
        
        runner = DeepCyTOFRunner(dataset_name=args.dataset_name, output_dir=tmpdir)
        with contextlib.redirect_stdout(io.StringIO()):
            runner.train(train_x_csv, train_y_csv)

        print("--- Processing Test Samples ---", flush=True)
        prediction_files = []
        test_path = Path(args.test_x)
        tar_test = None

        if test_path.suffix in ['.csv', '.txt']:
            test_list = [test_path]
            is_tar = False
        else:
            is_tar = True
            tar_test = tarfile.open(test_path, "r:*")
            test_list = [
                m for m in tar_test.getmembers() if m.name.endswith('.csv') and m.isfile()
            ]

        print(f"--- Processing {len(test_list)} samples ---", flush=True)
        for idx, item in enumerate(test_list, start=1):
            item_name = item.name if is_tar else item.name
            
            if is_tar:
                assert tar_test is not None
                assert isinstance(item, tarfile.TarInfo)
                tar_test.extract(item, path=tmpdir)
                sample_path = os.path.join(tmpdir, item.name)
            else:
                sample_path = str(item)

            with contextlib.redirect_stdout(io.StringIO()):
                predictions = runner.predict(sample_path)
            
            sample_number = extract_sample_number(item_name) or str(idx)
            out_name = f"{args.dataset_name}-prediction-{sample_number}.csv"
            out_path = os.path.join(tmpdir, out_name)

            pd.Series(predictions).to_csv(out_path, index=False, header=False)
            
            prediction_files.append(out_path)

        if is_tar:
            assert tar_test is not None
            tar_test.close()

        with tarfile.open(args.output_file, "w:gz") as tar_out:
            for pf in prediction_files:
                tar_out.add(pf, arcname=os.path.basename(pf))

    print(f"DeepCyTOF complete. Results saved to {args.output_file}", flush=True)

if __name__ == "__main__":
    main()
