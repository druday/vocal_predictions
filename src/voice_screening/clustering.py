from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def feature_distance_matrix(x_train: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(x_train, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
    corr = np.clip(corr, -1.0, 1.0)
    dist = 1.0 - np.abs(corr)
    dist = np.nan_to_num(dist, nan=1.0, posinf=1.0, neginf=1.0)
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    return dist


def fit_hierarchical_clustering(dist_matrix: np.ndarray, linkage_method: str = "ward") -> np.ndarray:
    condensed = squareform(dist_matrix, checks=False)
    return linkage(condensed, method=linkage_method)


def cluster_assignments(
    feature_names: list[str],
    linkage_matrix: np.ndarray,
    k: int,
) -> pd.DataFrame:
    labels = fcluster(linkage_matrix, t=k, criterion="maxclust")
    return pd.DataFrame({"feature": feature_names, "cluster": labels}).sort_values("cluster")


def clustering_diagnostics(
    dist_matrix: np.ndarray,
    linkage_matrix: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    condensed = squareform(dist_matrix, checks=False)
    coph_corr = float(cophenet(linkage_matrix, condensed)[0])

    sil = np.nan
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        sil = float(silhouette_score(dist_matrix, labels, metric="precomputed"))

    return {
        "cophenetic_correlation": coph_corr,
        "silhouette_score": sil,
    }


@dataclass
class ClusterAggregator:
    assignments: pd.DataFrame
    method: str = "mean"
    pca_models: dict[int, PCA] = field(default_factory=dict)
    clusters_: list[int] = field(default_factory=list)
    cluster_features_: dict[int, list[str]] = field(default_factory=dict)

    def fit(self, x_train: pd.DataFrame) -> "ClusterAggregator":
        if not {"feature", "cluster"}.issubset(self.assignments.columns):
            raise ValueError("assignments must contain 'feature' and 'cluster' columns")

        available_features = set(x_train.columns)
        mapping: dict[int, list[str]] = {}

        for cluster_id, group in self.assignments.groupby("cluster"):
            feats = [f for f in group["feature"].tolist() if f in available_features]
            if feats:
                mapping[int(cluster_id)] = feats

        if not mapping:
            raise RuntimeError("No cluster features overlap with training dataframe columns.")

        self.cluster_features_ = mapping
        self.clusters_ = sorted(mapping.keys())

        if self.method == "pca":
            self.pca_models = {}
            for cid in self.clusters_:
                feats = mapping[cid]
                if len(feats) == 1:
                    continue
                pca = PCA(n_components=1, random_state=42)
                pca.fit(x_train[feats].to_numpy())
                self.pca_models[cid] = pca

        return self

    def transform(self, x: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        if not self.cluster_features_:
            raise RuntimeError("ClusterAggregator must be fitted before transform.")

        cols: list[np.ndarray] = []
        names: list[str] = []

        for cid in self.clusters_:
            feats = self.cluster_features_[cid]
            block = x[feats].to_numpy()

            if self.method == "mean":
                agg = np.mean(block, axis=1)
            elif self.method == "median":
                agg = np.median(block, axis=1)
            elif self.method == "max":
                agg = np.max(block, axis=1)
            elif self.method == "min":
                agg = np.min(block, axis=1)
            elif self.method == "std":
                agg = np.std(block, axis=1)
            elif self.method == "sum":
                agg = np.sum(block, axis=1)
            elif self.method == "pca":
                if cid in self.pca_models:
                    agg = self.pca_models[cid].transform(block)[:, 0]
                else:
                    agg = block[:, 0]
            else:
                raise ValueError(f"Unsupported aggregation method: {self.method}")

            cols.append(agg.reshape(-1, 1))
            names.append(f"cluster_{cid}")

        out = np.hstack(cols)
        return out, names

    def fit_transform(self, x_train: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        return self.fit(x_train).transform(x_train)
