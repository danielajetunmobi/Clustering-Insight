from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from werkzeug.utils import secure_filename

from config import Config

if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"

from services.anomaly import detect_anomalies
from services.clustering import cluster_distribution, elbow_scores, run_clustering
from services.preprocessing import  dataframe_profile, numeric_columns, preprocess_dataframe
from services.reduction import reduce_to_2d


app = Flask(__name__)
app.config.from_object(Config)


def ensure_directories() -> None:
    app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)
    app.config["RESULT_FOLDER"].mkdir(parents=True, exist_ok=True)
    app.config["DATASET_FOLDER"].mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


def dataset_path(dataset_id: str) -> Path:
    return app.config["UPLOAD_FOLDER"] / f"{dataset_id}.csv"


def dataset_meta_path(dataset_id: str) -> Path:
    return app.config["UPLOAD_FOLDER"] / f"{dataset_id}.json"


def result_path(run_id: str) -> Path:
    return app.config["RESULT_FOLDER"] / f"{run_id}.csv"


def result_meta_path(run_id: str) -> Path:
    return app.config["RESULT_FOLDER"] / f"{run_id}.json"


def result_plot_path(run_id: str) -> Path:
    return app.config["RESULT_FOLDER"] / f"{run_id}.html"


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    if not path.exists():
        abort(404)
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset(dataset_id: str) -> tuple[pd.DataFrame, dict]:
    path = dataset_path(dataset_id)
    if not path.exists():
        abort(404)
    return pd.read_csv(path), read_json(dataset_meta_path(dataset_id))


def register_dataset(source: Path, display_name: str) -> str:
    ensure_directories()
    dataset_id = uuid4().hex
    target = dataset_path(dataset_id)
    shutil.copy2(source, target)
    save_json(
        dataset_meta_path(dataset_id),
        {
            "id": dataset_id,
            "name": display_name,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        },
    )
    return dataset_id


def upload_dataset(file_storage) -> str:
    ensure_directories()
    original = secure_filename(file_storage.filename or "dataset.csv")
    dataset_id = uuid4().hex
    file_storage.save(dataset_path(dataset_id))
    save_json(
        dataset_meta_path(dataset_id),
        {
            "id": dataset_id,
            "name": original,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        },
    )
    return dataset_id


def sample_datasets() -> list[dict]:
    ensure_directories()
    samples = []
    for csv_file in sorted(app.config["DATASET_FOLDER"].glob("*.csv")):
        samples.append(
            {
                "key": csv_file.stem,
                "name": csv_file.stem.replace("_", " ").title(),
                "filename": csv_file.name,
            }
        )
    return samples


def dataframe_table(df: pd.DataFrame, rows: int = 50) -> str:
    table_df = df.head(rows).copy()
    return table_df.to_html(
        classes="table table-sm table-hover align-middle data-table",
        index=False,
        border=0,
        escape=True,
    )


@app.route("/")
def index():
    return render_template("index.html", samples=sample_datasets())


@app.post("/upload")
def upload():
    file = request.files.get("dataset")
    if not file or not file.filename:
        flash("Choose a CSV file to upload.", "warning")
        return redirect(url_for("index"))
    if not allowed_file(file.filename):
        flash("Only CSV files are supported.", "danger")
        return redirect(url_for("index"))

    dataset_id = upload_dataset(file)
    return redirect(url_for("preview", dataset_id=dataset_id))


@app.get("/sample/<sample_key>")
def load_sample(sample_key: str):
    sample_file = app.config["DATASET_FOLDER"] / f"{secure_filename(sample_key)}.csv"
    if not sample_file.exists():
        abort(404)
    dataset_id = register_dataset(sample_file, sample_file.stem.replace("_", " ").title())
    return redirect(url_for("preview", dataset_id=dataset_id))


@app.get("/preview/<dataset_id>")
def preview(dataset_id: str):
    df, meta = load_dataset(dataset_id)
    profile = dataframe_profile(df)
    numeric = numeric_columns(df)
    stats_html = df.describe(include="all").fillna("").to_html(
        classes="table table-sm table-hover align-middle data-table",
        border=0,
        escape=True,
    )
    return render_template(
        "preview.html",
        dataset_id=dataset_id,
        meta=meta,
        profile=profile,
        numeric_columns=numeric,
        table_html=dataframe_table(df),
        stats_html=stats_html,
    )


@app.route("/analyze/<dataset_id>", methods=["GET", "POST"])
def analyze(dataset_id: str):
    df, meta = load_dataset(dataset_id)
    numeric = numeric_columns(df)
    if not numeric:
        flash("This dataset does not contain usable numeric columns.", "danger")
        return redirect(url_for("preview", dataset_id=dataset_id))

    if request.method == "GET":
        return render_template("analyze.html", dataset_id=dataset_id, meta=meta, numeric_columns=numeric)

    selected_columns = request.form.getlist("columns")
    missing_strategy = request.form.get("missing_strategy", "fill_mean")
    scale = request.form.get("scale") == "on"
    cluster_method = request.form.get("cluster_method", "kmeans")
    reduction_method = request.form.get("reduction_method", "pca")
    anomaly_enabled = request.form.get("anomaly_enabled") == "on"

    try:
        prepared = preprocess_dataframe(df, selected_columns, missing_strategy, scale)
        clusters = run_clustering(
            prepared.features,
            cluster_method,
            {
                "n_clusters": int(request.form.get("n_clusters", 3)),
                "eps": float(request.form.get("eps", 0.8)),
                "min_samples": int(request.form.get("min_samples", 5)),
            },
        )
        reduction = reduce_to_2d(prepared.features, reduction_method)
        elbow = elbow_scores(prepared.features) if cluster_method == "kmeans" else []
        if anomaly_enabled:
            anomalies = detect_anomalies(prepared.features, float(request.form.get("contamination", 0.08)))
        else:
            anomalies = None
    except ValueError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("analyze", dataset_id=dataset_id))

    run_id = uuid4().hex
    result_df = prepared.source_rows.copy()
    result_df["cluster"] = clusters.labels
    result_df["component_x"] = reduction.x
    result_df["component_y"] = reduction.y
    if anomalies:
        result_df["anomaly"] = ["anomaly" if label == -1 else "normal" for label in anomalies.labels]
        result_df["anomaly_score"] = anomalies.scores
    else:
        result_df["anomaly"] = "not_run"

    result_df.to_csv(result_path(run_id), index=False)

    points = []
    for index, row in result_df.iterrows():
        points.append(
            {
                "row": int(index + 1),
                "x": float(row["component_x"]),
                "y": float(row["component_y"]),
                "cluster": str(row["cluster"]),
                "anomaly": str(row["anomaly"]),
            }
        )

    payload = {
        "run_id": run_id,
        "dataset": meta,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "selected_columns": selected_columns,
        "preprocessing": {"scale": scale, "missing_strategy": missing_strategy, "notes": prepared.notes},
        "cluster_info": clusters.info,
        "reduction_info": reduction.info,
        "anomaly_info": anomalies.info if anomalies else {"method": "Not run", "anomalies": 0},
        "metrics": {
            "rows": int(len(result_df)),
            "selected_columns": int(len(selected_columns)),
            "clusters": int(len(set(clusters.labels) - {-1})),
            "anomalies": int((result_df["anomaly"] == "anomaly").sum()),
        },
        "charts": {
            "points": points,
            "distribution": cluster_distribution(clusters.labels),
            "elbow": elbow,
        },
    }
    save_json(result_meta_path(run_id), payload)
    return redirect(url_for("results", run_id=run_id))


@app.get("/results/<run_id>")
def results(run_id: str):
    payload = read_json(result_meta_path(run_id))
    df = pd.read_csv(result_path(run_id))
    return render_template(
        "results.html",
        payload=payload,
        chart_json=json.dumps(payload["charts"]),
        table_html=dataframe_table(df),
    )


@app.get("/download/<run_id>/csv")
def download_csv(run_id: str):
    path = result_path(run_id)
    if not path.exists():
        abort(404)
    return send_file(path, as_attachment=True, download_name=f"clusterinsight-{run_id}.csv")


@app.get("/download/<run_id>/plot")
def download_plot(run_id: str):
    payload = read_json(result_meta_path(run_id))
    html = render_template("plot_export.html", payload=payload, chart_json=json.dumps(payload["charts"]))
    path = result_plot_path(run_id)
    path.write_text(html, encoding="utf-8")
    return send_file(path, as_attachment=True, download_name=f"clusterinsight-plot-{run_id}.html")


@app.errorhandler(404)
def not_found(_error):
    return render_template("error.html", title="Not Found", message="The requested item was not found."), 404


@app.errorhandler(413)
def file_too_large(_error):
    return render_template("error.html", title="File Too Large", message="Upload a CSV smaller than 16 MB."), 413


if __name__ == "__main__":
    ensure_directories()
    app.run(debug=True)
