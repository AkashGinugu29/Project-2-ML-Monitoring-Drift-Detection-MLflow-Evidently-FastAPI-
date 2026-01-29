import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", os.path.join(BASE_DIR, "mlruns"))
MODEL_NAME = os.getenv("MODEL_NAME", "monitoring_model")

DATA_REF_DIR = os.path.join(BASE_DIR, "data", "reference")
DATA_CUR_DIR = os.path.join(BASE_DIR, "data", "current")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
