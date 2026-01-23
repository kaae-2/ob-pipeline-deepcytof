#!/usr/bin/env python3
import os
import sys
import tarfile
import tempfile
import pandas as pd
import argparse
import gzip
from pathlib import Path

# Prevent Matplotlib from hanging on font-cache building
import matplotlib
matplotlib.use('Agg')

# Fix for Protobuf legacy issues
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from deepcytof_core import DeepCyTOFRunner

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
        
        runner = DeepCyTOFRunner(dataset_name=args.dataset_name, output_dir=tmpdir)
        runner.train(train_x_csv, train_y_csv)

        print("--- Processing Test Samples ---", flush=True)
        prediction_files = []
        test_path = Path(args.test_x)

        if test_path.suffix in ['.csv', '.txt']:
            test_list = [test_path]
            is_tar = False
        else:
            is_tar = True
            tar_test = tarfile.open(test_path, "r:*")
            test_list = [m for m in tar_test.getmembers() if m.name.endswith('.csv') and m.isfile()]

        for item in test_list:
            item_name = item.name if is_tar else item.name
            print(f"--- Starting Sample: {item_name} ---", flush=True)
            
            if is_tar:
                tar_test.extract(item, path=tmpdir)
                sample_path = os.path.join(tmpdir, item.name)
            else:
                sample_path = str(item)

            predictions = runner.predict(sample_path)
            
            out_name = os.path.basename(item_name).replace(".csv", ".predictions.csv.gz")
            out_path = os.path.join(tmpdir, out_name)
            
            with gzip.open(out_path, "wt") as f:
                pd.Series(predictions).to_csv(f, index=False, header=False)
            
            prediction_files.append(out_path)
            print(f"--- Finished Sample: {item_name} ---", flush=True)

        if is_tar:
            tar_test.close()

        with tarfile.open(args.output_file, "w:gz") as tar_out:
            for pf in prediction_files:
                tar_out.add(pf, arcname=os.path.basename(pf))

    print(f"DeepCyTOF complete. Results saved to {args.output_file}", flush=True)

if __name__ == "__main__":
    main()