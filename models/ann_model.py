"""
models/ann_model.py
====================
PyTorch MLP — scaffolded, disabled by default (cfg.enable_ann = False).
Same interface as other models.

Architecture:
  Input → BN → Linear(256) → ReLU → Dropout(0.3)
         → Linear(128) → ReLU → Dropout(0.2)
         → Linear(64)  → ReLU → Dropout(0.1)
         → Linear(1)   → Sigmoid
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from utils.logger import log


class ANNModel:
    """
    PyTorch MLP wrapper (disabled by default).
    Enable via config: cfg.enable_ann = True
    """

    def __init__(self, input_dim: int = None, hidden: list = None, dropout: float = 0.3):
        self.input_dim = input_dim
        self.hidden    = hidden or [256, 128, 64]
        self.dropout   = dropout
        self._model    = None
        self._scaler   = None
        self._feature_names: list = []
        self._trained  = False

    def _build_model(self):
        try:
            import torch
            import torch.nn as nn

            class MLP(nn.Module):
                def __init__(self, in_dim, hidden, dropout):
                    super().__init__()
                    layers = [nn.BatchNorm1d(in_dim)]
                    prev = in_dim
                    for i, h in enumerate(hidden):
                        layers += [nn.Linear(prev, h), nn.ReLU(),
                                   nn.Dropout(dropout * (1 - i * 0.1))]
                        prev = h
                    layers += [nn.Linear(prev, 1), nn.Sigmoid()]
                    self.net = nn.Sequential(*layers)

                def forward(self, x):
                    return self.net(x)

            return MLP(self.input_dim, self.hidden, self.dropout)
        except ImportError:
            raise ImportError("PyTorch not installed. pip install torch")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series    = None,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        **kwargs,
    ) -> "ANNModel":
        """Train the MLP."""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            from sklearn.preprocessing import StandardScaler
        except ImportError as e:
            log.error(f"ANN dependencies missing: {e}")
            return self

        self._feature_names = list(X_train.columns)
        self.input_dim = len(self._feature_names)

        # Scale
        self._scaler = StandardScaler()
        X_tr = self._scaler.fit_transform(X_train.values).astype(np.float32)
        y_tr = y_train.values.astype(np.float32)

        dataset   = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr).unsqueeze(1))
        loader    = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self._model = self._build_model()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCELoss()

        best_val_loss = np.inf
        patience = 10
        no_improve = 0

        for epoch in range(epochs):
            self._model.train()
            ep_loss = 0
            for xb, yb in loader:
                pred = self._model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()

            if X_val is not None:
                self._model.eval()
                with torch.no_grad():
                    Xv = torch.tensor(
                        self._scaler.transform(X_val.values).astype(np.float32)
                    )
                    yv = torch.tensor(y_val.values.astype(np.float32)).unsqueeze(1)
                    val_loss = criterion(self._model(Xv), yv).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve    = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        log.info(f"ANN early stopping at epoch {epoch+1}")
                        break

        self._trained = True
        log.info(f"ANN trained: {epochs} epochs, {len(X_train):,} samples")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._trained or self._model is None:
            n = len(X)
            return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])
        try:
            import torch
            self._model.eval()
            Xs = self._scaler.transform(X[self._feature_names].values).astype(np.float32)
            with torch.no_grad():
                probs = self._model(torch.tensor(Xs)).numpy().flatten()
            return np.column_stack([1 - probs, probs])
        except Exception as e:
            log.error(f"ANN predict failed: {e}")
            n = len(X)
            return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def feature_importance(self) -> pd.Series:
        """ANN doesn't have native importance — return uniform."""
        return pd.Series(
            1.0 / len(self._feature_names),
            index=self._feature_names,
            name="importance",
        )

    def save(self, path: Path) -> None:
        import joblib
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._model:
            import torch
            torch.save(self._model.state_dict(), path / "ann_weights.pt")
        joblib.dump(self._scaler,        path / "ann_scaler.pkl")
        joblib.dump(self._feature_names, path / "feature_names.pkl")

    def load(self, path: Path) -> "ANNModel":
        import joblib
        path = Path(path)
        self._scaler        = joblib.load(path / "ann_scaler.pkl")
        self._feature_names = joblib.load(path / "feature_names.pkl")
        self.input_dim      = len(self._feature_names)
        try:
            import torch
            self._model = self._build_model()
            self._model.load_state_dict(torch.load(path / "ann_weights.pt"))
            self._trained = True
        except Exception as e:
            log.warning(f"ANN load failed: {e}")
        return self
