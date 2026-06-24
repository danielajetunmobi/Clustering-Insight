from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


class Config:
    SECRET_KEY = "clusterinsight-dev-key"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    RESULT_FOLDER = UPLOAD_FOLDER / "results"
    DATASET_FOLDER = BASE_DIR / "datasets"
    ALLOWED_EXTENSIONS = {"csv"}
