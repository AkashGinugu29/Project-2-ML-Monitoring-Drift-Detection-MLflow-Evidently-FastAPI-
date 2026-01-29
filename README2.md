# ML Monitoring & Drift Detection (MLflow + Evidently + FastAPI)

A production-style ML monitoring project that tracks model versions, logs inference metadata, and detects data drift after deployment.

## Features
- Train and log models, parameters, and metrics using MLflow
- FastAPI inference service with request and prediction logging
- Reference vs current data drift detection using Evidently
- Generates an HTML drift report and exposes it via an API endpoint

## Quickstart
1) Install dependencies:
pip install -r requirements.txt

2) Train and log model:
python src/train.py

3) Run API:
uvicorn src.api:app --reload

4) Generate drift report:
python src/drift.py

## Endpoints
- GET /health
- POST /predict
- GET /drift-report

