import os
import threading
from typing import Iterable, List
from datetime import datetime

import numpy as np
import lightgbm as lgb

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "models/model.txt")


class ModelService:
    """
    Simple LightGBM model service.
    - On first use or startup, loads a model from DEFAULT_MODEL_PATH.
    - If missing, trains a tiny demo model and saves it.
    - Exposes training from a CSV file on demand or via scheduler.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = model_path
        self._booster: lgb.Booster | None = None
        self._lock = threading.Lock()

    def _save_booster(self, booster: lgb.Booster) -> str:
        """Save the model to a timestamped path and update the latest pointer.

        Returns the full path of the timestamped model file.
        """
        ts = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        dir_ = os.path.dirname(self.model_path)
        os.makedirs(dir_, exist_ok=True)
        ts_path = os.path.join(dir_, f"model_{ts}.txt")
        booster.save_model(ts_path)
        # Update latest model pointer (model_path). Try symlink else copy
        try:
            if os.path.islink(self.model_path) or os.path.exists(self.model_path):
                try:
                    os.remove(self.model_path)
                except OSError:
                    pass
            os.symlink(os.path.basename(ts_path), self.model_path)
        except OSError:
            # Fallback: save/copy as model_path directly
            booster.save_model(self.model_path)
        self._booster = booster
        return ts_path

    def load_or_train(self) -> None:
        with self._lock:
            if self._booster is not None:
                return
            if os.path.exists(self.model_path):
                self._booster = lgb.Booster(model_file=self.model_path)
                return

            # Train a tiny binary classifier on synthetic data
            rng = np.random.default_rng(42)
            n_samples, n_features = 200, 6
            X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
            # Create a simple non-linear decision boundary
            y = (
                X[:, 0]
                + 0.5 * X[:, 1]
                - 0.2 * X[:, 2]
                + rng.normal(scale=0.3, size=n_samples)
                > 0
            ).astype(int)

            train_data = lgb.Dataset(X, label=y)
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbose": -1,
            }
            booster = lgb.train(params, train_data, num_boost_round=20)
            self._save_booster(booster)

    def train_from_file(self, data_path: str) -> str:
        """
        Train a LightGBM model from CSV at data_path and save it with a timestamp.

        CSV format: numeric columns, last column is the binary target (0/1),
        preceding columns are features. A header row is allowed and will be skipped.

        Returns the timestamped model path.
        """
        with self._lock:
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Training data not found: {data_path}")

            # Load CSV: allow header, comma-separated
            try:
                data = np.genfromtxt(
                    data_path, delimiter=",", dtype=np.float32, skip_header=1
                )
            except ValueError:
                # If no header present
                data = np.loadtxt(data_path, delimiter=",", dtype=np.float32)

            if data.ndim == 1:
                raise ValueError(
                    "Training data must have at least 2 columns (features + target)"
                )
            if data.shape[1] < 2:
                raise ValueError(
                    "Training data must have at least 2 columns (features + target)"
                )

            X = data[:, :-1]
            y = data[:, -1].astype(int)

            train_data = lgb.Dataset(X, label=y)
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbose": -1,
            }
            booster = lgb.train(params, train_data, num_boost_round=100)
            ts_path = self._save_booster(booster)
            return ts_path

    def predict(self, features: Iterable[Iterable[float]]) -> List[float]:
        if self._booster is None:
            self.load_or_train()
        assert self._booster is not None

        X = np.asarray(list(features), dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        preds = self._booster.predict(X)
        # For binary objective, predictions are probabilities for class 1
        return preds.tolist()
