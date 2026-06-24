from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class AnomalyResult:
    labels: list[int]
    scores: list[float]
    info: dict


def detect_anomalies(features: pd.DataFrame, contamination: float = 0.08) -> AnomalyResult:
    contamination = max(0.01, min(float(contamination), 0.45))
    model = IsolationForest(contamination=contamination, random_state=42)
    labels = model.fit_predict(features)
    scores = model.decision_function(features)
    anomaly_count = int((labels == -1).sum())
    return AnomalyResult(
        labels=labels.astype(int).tolist(),
        scores=[round(float(score), 6) for score in scores],
        info={"method": "Isolation Forest", "contamination": contamination, "anomalies": anomaly_count},
    )
