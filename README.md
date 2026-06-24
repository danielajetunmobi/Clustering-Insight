# ClusterInsight

ClusterInsight is a Flask web application for unsupervised dataset analysis. It accepts CSV uploads, previews the dataset, runs clustering and anomaly detection, renders Plotly.js charts, and exports the analyzed result as CSV or standalone chart HTML.

## Features

- Upload a CSV file or start from included Iris and Mall Customers samples.
- Preview the first 50 rows, dataset shape, missing values, data types, and descriptive statistics.
- Select numeric features, handle missing values, and apply feature scaling.
- Run K-Means, DBSCAN, or hierarchical clustering.
- Project data into 2D with PCA or t-SNE.
- Detect anomalies with Isolation Forest.
- View Plotly.js scatter, cluster distribution, and K-Means elbow charts.
- Download clustered data as CSV and export the charts as HTML.

## Run Locally

Create a virtual environment, install dependencies, and start the Flask app:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open the local URL printed by Flask, usually `http://127.0.0.1:5000`.

## Project Structure

```text
ClusterInsight/
|-- app.py
|-- config.py
|-- requirements.txt
|-- datasets/
|-- services/
|   |-- preprocessing.py
|   |-- clustering.py
|   |-- reduction.py
|   |-- anomaly.py
|-- static/
|   |-- css/
|   |-- js/
|-- templates/
|-- uploads/
```

## Notes

Uploaded datasets and analysis outputs are stored in `uploads/`. The folder is ignored by git except for placeholders.
