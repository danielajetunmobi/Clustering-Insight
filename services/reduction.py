from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


@dataclass
class ReductionResult:
    x: list[float]
    y: list[float]
    info: dict


def reduce_to_2d(features: pd.DataFrame, method: str = "pca") -> ReductionResult:
    values = features.to_numpy()

    if values.shape[1] == 1:
        return ReductionResult(
            x=values[:, 0].astype(float).round(6).tolist(),
            y=np.zeros(values.shape[0], dtype=float).tolist(),
            info={"method": "Single Feature Projection"},
        )

    if method == "tsne" and len(features) >= 4:
        perplexity = max(2, min(30, (len(features) - 1) // 3))
        model = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="random",
            learning_rate=200.0,
            random_state=42,
        )
        coords = model.fit_transform(values)
        return ReductionResult(
            x=coords[:, 0].astype(float).round(6).tolist(),
            y=coords[:, 1].astype(float).round(6).tolist(),
            info={"method": "t-SNE", "perplexity": int(perplexity)},
        )

    model = PCA(n_components=2, random_state=42)
    coords = model.fit_transform(values)
    variance = model.explained_variance_ratio_
    return ReductionResult(
        x=coords[:, 0].astype(float).round(6).tolist(),
        y=coords[:, 1].astype(float).round(6).tolist(),
        info={
            "method": "PCA",
            "explained_variance": [round(float(item), 4) for item in variance],
        },
    )
