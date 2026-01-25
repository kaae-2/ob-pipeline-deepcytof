#!/usr/bin/env python3
import os
import sys
import tarfile
import tempfile
import re
import contextlib
import io
import time
from datetime import datetime
import pandas as pd
import argparse
from pathlib import Path

# Prevent Matplotlib from hanging on font-cache building
import matplotlib
matplotlib.use('Agg')

# Fix for Protobuf legacy issues
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "16")

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from deepcytof_core import DeepCyTOFRunner


def log_ts(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"--- {timestamp} {message}", flush=True)


def log_tf_devices() -> None:
    log_ts(
        "GPU env: ROCR_VISIBLE_DEVICES="
        f"{os.environ.get('ROCR_VISIBLE_DEVICES')} HSA_OVERRIDE_GFX_VERSION="
        f"{os.environ.get('HSA_OVERRIDE_GFX_VERSION')}"
    )
    try:
        import tensorflow as tf
    except Exception as exc:
        log_ts(f"TensorFlow import failed: {exc}")
        return

    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception as exc:
        log_ts(f"TensorFlow GPU probe failed: {exc}")
        return

    log_ts(f"TensorFlow GPUs: {gpus}")

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

    log_tf_devices()

    with tempfile.TemporaryDirectory() as tmpdir:
        print("--- Preparing Training Data ---", flush=True)
        log_ts("Starting training data extraction")
        extract_start = time.time()
        train_x_csv = extract_first_csv_from_tar(args.train_x, tmpdir)
        log_ts(f"Extracted train_x in {time.time() - extract_start:.2f}s")
        extract_start = time.time()
        train_y_csv = extract_first_csv_from_tar(args.train_y, tmpdir)
        log_ts(f"Extracted train_y in {time.time() - extract_start:.2f}s")

        runner = DeepCyTOFRunner(dataset_name=args.dataset_name, output_dir=tmpdir)
        log_ts("Starting model training")
        train_start = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            runner.train(train_x_csv, train_y_csv)
        log_ts(f"Model training completed in {time.time() - train_start:.2f}s")

        print("--- Processing Test Samples ---", flush=True)
        log_ts("Starting test sample processing")
        test_start = time.time()
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
        with tarfile.open(args.output_file, "w:gz") as tar_out:
            for idx, item in enumerate(test_list, start=1):
                item_name = item.name if is_tar else item.name
                log_ts(f"Predicting sample {idx}/{len(test_list)}: {os.path.basename(item_name)}")

                if is_tar:
                    assert tar_test is not None
                    assert isinstance(item, tarfile.TarInfo)
                    sample_file = tar_test.extractfile(item)
                    if sample_file is None:
                        continue
                    try:
                        df_x = pd.read_csv(sample_file, header=None, dtype="float32")
                    finally:
                        sample_file.close()
                else:
                    df_x = pd.read_csv(str(item), header=None, dtype="float32")

                log_ts(
                    "Sample data shape: "
                    f"{df_x.shape} dtypes={list(df_x.dtypes.unique())}"
                )

                try:
                    pred_log = os.getenv("DEEPCYTOF_PRED_LOG", "0") == "1"
                    if pred_log:
                        log_ts(
                            "Prediction logging enabled for sample: "
                            f"{os.path.basename(item_name)}"
                        )
                        predictions = runner.predict_df(
                            df_x, sample_name=os.path.basename(item_name)
                        )
                    else:
                        with contextlib.redirect_stdout(io.StringIO()):
                            predictions = runner.predict_df(
                                df_x, sample_name=os.path.basename(item_name)
                            )
                except Exception as exc:
                    log_ts(
                        "Prediction failed for sample: "
                        f"{os.path.basename(item_name)} error={exc}"
                    )
                    raise

                log_ts(f"Finished sample {idx}/{len(test_list)}: {os.path.basename(item_name)}")

                sample_number = extract_sample_number(item_name) or str(idx)
                out_name = f"{args.dataset_name}-prediction-{sample_number}.csv"

                csv_buffer = io.StringIO()
                pd.Series(predictions).to_csv(csv_buffer, index=False, header=False)
                data = csv_buffer.getvalue().encode("utf-8")
                tar_info = tarfile.TarInfo(name=os.path.basename(out_name))
                tar_info.size = len(data)
                tar_out.addfile(tar_info, io.BytesIO(data))

        if is_tar:
            assert tar_test is not None
            tar_test.close()

        log_ts(f"Test sample processing completed in {time.time() - test_start:.2f}s")

    print(f"DeepCyTOF complete. Results saved to {args.output_file}", flush=True)

if __name__ == "__main__":
    main()
