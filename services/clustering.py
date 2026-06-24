from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans


@dataclass
class ClusterResult:
    labels: list[int]
    info: dict


def _bounded_cluster_count(value: int, n_samples: int) -> int:
    return max(2, min(int(value), n_samples))


def run_clustering(features: pd.DataFrame, method: str, params: dict) -> ClusterResult:
    n_samples = len(features)

    if method == "kmeans":
        clusters = _bounded_cluster_count(params.get("n_clusters", 3), n_samples)
        model = KMeans(n_clusters=clusters, random_state=42, n_init=10)
        labels = model.fit_predict(features)
        return ClusterResult(
            labels=labels.astype(int).tolist(),
            info={
                "method": "K-Means",
                "clusters": clusters,
                "inertia": round(float(model.inertia_), 4),
            },
        )

    if method == "dbscan":
        eps = max(float(params.get("eps", 0.8)), 0.01)
        min_samples = max(2, min(int(params.get("min_samples", 5)), n_samples))
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(features)
        cluster_count = len(set(labels.tolist()) - {-1})
        return ClusterResult(
            labels=labels.astype(int).tolist(),
            info={
                "method": "DBSCAN",
                "eps": eps,
                "min_samples": min_samples,
                "clusters": cluster_count,
                "noise_points": int((labels == -1).sum()),
            },
        )

    if method == "hierarchical":
        clusters = _bounded_cluster_count(params.get("n_clusters", 3), n_samples)
        model = AgglomerativeClustering(n_clusters=clusters)
        labels = model.fit_predict(features)
        return ClusterResult(
            labels=labels.astype(int).tolist(),
            info={"method": "Hierarchical Clustering", "clusters": clusters},
        )

    raise ValueError("Unsupported clustering method.")


def elbow_scores(features: pd.DataFrame, max_k: int = 10) -> list[dict]:
    upper = min(max_k, len(features))
    scores = []
    for k in range(1, upper + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(features)
        scores.append({"k": k, "inertia": round(float(model.inertia_), 4)})
    return scores


def cluster_distribution(labels: list[int]) -> list[dict]:
    counts = pd.Series(labels, dtype="int").astype(str).value_counts().sort_index()
    return [{"label": label, "count": int(count)} for label, count in counts.items()]
