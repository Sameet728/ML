"""
models/ann_model.py
===================
Artificial Neural Network (Multi-Layer Perceptron) model.
Used to capture highly non-linear feature interactions in the ensemble.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from utils.logger import log

class ANNModel:
    def __init__(self, params: dict = None):
        """
        Initialize ANN with robust default hyperparameters.
        """
        # Hidden layer sizes: 128 -> 64 -> 32
        # Early stopping prevents overfitting
        default_params = {
            "hidden_layer_sizes": (128, 64, 32),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.001,           # L2 penalty
            "batch_size": 256,
            "learning_rate": "adaptive",
            "max_iter": 500,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 15,
            "random_state": 42
        }
        
        if params:
            default_params.update(params)
            
        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(**default_params))
        ])

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train the ANN model."""
        self.model.fit(X_train, y_train)
        log.info(f"ANNModel trained: {len(X_train)} samples")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities (returns [P(0), P(1)] array)."""
        return self.model.predict_proba(X)
