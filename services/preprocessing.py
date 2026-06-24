from __future__ import annotations

from dataclasses import dataclass


import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessingResult:
    features: pd.DataFrame
    source_rows: pd.DataFrame
    notes: list[str]



def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that can reasonably be used for numeric analysis."""
    numeric = df.select_dtypes(include="number").columns.tolist()
    converted = []
    for column in df.columns:
        if column in numeric :
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().sum() >= max(2, len(df) // 2):
            converted.append(column)
    return numeric + converted


def dataframe_profile(df: pd.DataFrame) -> dict:
    missing = df.isna().sum()
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "total_missing": int(missing.sum()),
        "dtypes": [{"column": col, "dtype": str(dtype)} for col, dtype in df.dtypes.items()],
        "missing": [
            {"column": col, "missing": int(count)}
            for col, count in missing.items()
            if int(count) > 0
        ],
    }


def preprocess_dataframe(
    df: pd.DataFrame,
    columns: list[str],
    missing_strategy: str = "fill_mean",
    scale: bool = True,
) -> PreprocessingResult:
    if not columns:
        raise ValueError("Select at least one numeric column.")

    working = df.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    notes: list[str] = []

    if missing_strategy == "drop":
        before = len(working)
        mask = working.notna().all(axis=1)
        working = working.loc[mask].copy()
        source_rows = df.loc[mask].copy()
        dropped = before - len(working)
        notes.append(f"Dropped {dropped} row(s) with missing selected values.")
    elif missing_strategy == "fill_mean":
        source_rows = df.copy()
        means = working.mean(numeric_only=True)
        working = working.fillna(means).fillna(0)
        notes.append("Filled missing selected values with column means.")
    else:
        raise ValueError("Unsupported missing-value strategy.")

    working = working.reset_index(drop=True)
    source_rows = source_rows.reset_index(drop=True)

    empty_or_constant = [col for col in working.columns if working[col].nunique(dropna=False) <= 1]
    if empty_or_constant:
        notes.append(
            "Constant columns kept for export but ignored by some models: "
            + ", ".join(empty_or_constant)
            + "."
        )

    if len(working) < 2:
        raise ValueError("At least two valid rows are required for analysis.")

    if scale:
        scaler = StandardScaler()
        values = scaler.fit_transform(working)
        working = pd.DataFrame(values, columns=working.columns)
        notes.append("Applied standard feature scaling.")
    else:
        notes.append("Used raw numeric values without scaling.")

    return PreprocessingResult(features=working, source_rows=source_rows, notes=notes)
