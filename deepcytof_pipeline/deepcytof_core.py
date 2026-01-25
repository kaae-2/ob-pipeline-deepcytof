#!/usr/bin/env python3
import os
import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# --- KERAS/TF SHIM ---
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TF noise

from tensorflow import keras
K = keras.backend
initializations = keras.initializers

# --- LEGACY IMPORTS ---
current_dir = Path(__file__).resolve().parent
paths_to_add = [str(current_dir), str(current_dir / "Util")]
for p in paths_to_add:
    if p not in sys.path:
        sys.path.insert(0, p)

from Util import denoisingAutoEncoder as dae
from Util import DataHandler as dh
from Util import feedforwadClassifier as net
from Util import MMDNet as mmd

class Sample:
    def __init__(self, X, y=None, cell_ids=None):
        self.X = X
        self.y = y
        self.cell_ids = cell_ids

class DeepCyTOFRunner:
    def __init__(self, dataset_name, output_dir):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.encoder = LabelEncoder()
        self.model = None
        self.dae_model = None
        self.preprocessor = None
        self.marker_names = None
        self.skip_mmd = os.environ.get("DEEPCYTOF_SKIP_MMD", "1") == "1"

    def train(self, x_path, y_path):
        def log_ts(message: str) -> None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[deepcytof-train] {timestamp} {message}", file=sys.stderr, flush=True)

        log_ts("Loading training data")
        load_start = time.time()
        df_x = pd.read_csv(x_path, header=None, dtype=np.float32)
        X = df_x.values

        df_y = pd.read_csv(y_path, header=None)
        y_raw = df_y.iloc[:, 0].values.astype(str)
        y_encoded = self.encoder.fit_transform(y_raw).reshape(-1, 1).astype(np.float32)
        log_ts(f"Loaded training data in {time.time() - load_start:.2f}s")

        target = Sample(X, y_encoded)
        preprocess_start = time.time()
        target = dh.preProcessSamplesCyTOFData(target)
        log_ts(f"Preprocessed training data in {time.time() - preprocess_start:.2f}s")

        log_ts("Training AutoEncoder (DAE) start")
        dae_start = time.time()
        self.dae_model = dae.trainDAE(target, self.output_dir, 0, np.array([0]), 
                                     None, 'CSV', 0.8, True, False, str(self.dataset_name))
        log_ts(f"Training AutoEncoder (DAE) completed in {time.time() - dae_start:.2f}s")
        
        log_ts("Denoising target data start")
        denoise_start = time.time()
        denoise_target = dae.predictDAE(target, self.dae_model, True)
        denoise_target, self.preprocessor = dh.standard_scale(denoise_target)
        log_ts(f"Denoising target data completed in {time.time() - denoise_start:.2f}s")

        log_ts("Training Feedforward Classifier start")
        clf_start = time.time()
        self.model = net.trainClassifier(denoise_target, 'CSV', 0, [12, 6, 3], 
                                        'softplus', 1e-4, str(self.dataset_name))
        self.target_denoised = denoise_target
        log_ts(f"Training Feedforward Classifier completed in {time.time() - clf_start:.2f}s")
        log_ts("Training complete")

    def _predict_array(self, X, sample_name=None):
        import time
        start_time = time.time()
        if sample_name:
            print(f"      -> Loading {sample_name}...", flush=True)
        X = np.asarray(X, dtype=np.float32)
        source = Sample(X)
        source = dh.preProcessSamplesCyTOFData(source)
        
        print(f"      -> Denoising source data...", flush=True)
        denoise_source = dae.predictDAE(source, self.dae_model, True)
        denoise_source, _ = dh.standard_scale(denoise_source, preprocessor=self.preprocessor)
        
        # --- MMD MONITORING ---
        if self.skip_mmd:
            calibrated_source = denoise_source
        else:
            probs = self.model.predict(denoise_source.X)
            init_preds = np.argmax(probs, axis=1)
            print(f"      -> Starting MMD Calibration at {time.strftime('%H:%M:%S')}...", flush=True)
            print(f"      -> Sample size: {len(X)} cells. Large samples (>10k) will take time.", flush=True)
            calibrated_source = mmd.calibrate(self.target_denoised, denoise_source, 
                                             0, init_preds, str(self.dataset_name))
            mmd_end = time.time()
            print(f"      -> MMD Calibration finished in {round(mmd_end - start_time, 2)} seconds.", flush=True)
        
        print(f"      -> Running final classification...", flush=True)
        final_probs = self.model.predict(calibrated_source.X)
        final_idx = np.argmax(final_probs, axis=1)
        
        return self.encoder.inverse_transform(final_idx)

    def predict(self, x_path):
        df_x = pd.read_csv(x_path, header=None, dtype=np.float32)
        sample_name = os.path.basename(x_path)
        return self._predict_array(df_x.values, sample_name=sample_name)

    def predict_df(self, df_x, sample_name=None):
        return self._predict_array(df_x.values, sample_name=sample_name)
