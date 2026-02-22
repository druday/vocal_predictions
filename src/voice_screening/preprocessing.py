from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass
class FeaturePreprocessor:
    imputer: SimpleImputer
    scaler: StandardScaler | None

    @classmethod
    def from_config(cls, standardize: bool = True) -> "FeaturePreprocessor":
        return cls(
            imputer=SimpleImputer(strategy="median"),
            scaler=StandardScaler() if standardize else None,
        )

    def fit_transform(self, x: pd.DataFrame) -> np.ndarray:
        arr = self.imputer.fit_transform(_clean_numeric(x))
        if self.scaler is not None:
            arr = self.scaler.fit_transform(arr)
        return np.asarray(arr, dtype=float)

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        arr = self.imputer.transform(_clean_numeric(x))
        if self.scaler is not None:
            arr = self.scaler.transform(arr)
        return np.asarray(arr, dtype=float)


def _clean_numeric(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out
