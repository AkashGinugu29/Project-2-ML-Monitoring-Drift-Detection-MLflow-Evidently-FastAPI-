import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

from config import MODEL_PATH, REPORTS_DIR
from drift import main as generate_drift_report

app = FastAPI(title="ML Monitoring API", version="1.0.0")

class PredictRequest(BaseModel):
    x1: float
    x2: float

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run: python src/train.py")
    return joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        model = load_model()
        X = pd.DataFrame([{"x1": req.x1, "x2": req.x2}])
        yhat = float(model.predict(X)[0])
        return {"prediction": yhat}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/drift-report", response_class=HTMLResponse)
def drift_report():
    try:
        # Generate fresh report each time (simple demo approach)
        generate_drift_report()

        path = os.path.join(REPORTS_DIR, "drift_report.html")
        if not os.path.exists(path):
            raise FileNotFoundError("Drift report not found.")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
