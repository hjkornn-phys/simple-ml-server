import os
import threading
from typing import Iterable, List

import numpy as np
import lightgbm as lgb

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "models/model.txt")


class ModelService:
    """
    Simple LightGBM model service.
    - On first use or startup, loads a model from DEFAULT_MODEL_PATH.
    - If missing, trains a tiny demo model and saves it.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = model_path
        self._booster: lgb.Booster | None = None
        self._lock = threading.Lock()

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

            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            booster.save_model(self.model_path)
            self._booster = booster

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
