# Customer Churn Prediction — End-to-End MLOps Pipeline

A production-grade machine learning pipeline for predicting customer churn in a streaming service. Built with a focus on modularity, reproducibility, and real-world MLOps practices.

[![Build and Push Docker Image](https://github.com/hdsd1007/customer-churn/actions/workflows/deploy.yml/badge.svg)](https://github.com/hdsd1007/customer-churn/actions/workflows/deploy.yml)
[![Docker](https://img.shields.io/badge/Docker-himanshudhurve96%2Fchurn--app--demo-blue)](https://hub.docker.com/r/himanshudhurve96/churn-app-demo)

---

## What This Project Does

Given a streaming service customer's profile — subscription type, payment method, viewing habits, support tickets — the model predicts whether that customer is likely to churn. The output is a churn probability and a binary classification at a tuned threshold of 0.30, optimised for recall (catching as many churners as possible).

---

## Architecture

```
Raw CSV
  │
  ▼
load_data.py          → loads and validates the CSV
  │
  ▼
preprocess.py         → cleaning only (strip headers, drop IDs, coerce numerics)
  │
  ▼
build_features.py     → encoding (binary, one-hot, bool→int, cardinality guard)
  │
  ▼
tune.py (optional)    → Optuna hyperparameter search (30 trials, 3-fold CV on recall)
  │
  ▼
train.py              → fits XGBClassifier with best params + scale_pos_weight
  │
  ▼
evaluate.py           → predict_proba + threshold → recall, precision, f1, roc_auc
  │
  ▼
MLflow                → tracks params, metrics, artifacts per run
  │
  ▼
Model Registry        → register best run, promote to @production alias
  │
  ▼
inference.py          → loads model, transforms single-row input, returns probability
  │
  ▼
FastAPI + Gradio      → REST API (/predict) + Web UI (/ui)
  │
  ▼
Docker                → containerised, self-contained image
  │
  ▼
GitHub Actions        → CI/CD: push to main → build → push to Docker Hub
```

---

## Dataset

- **Size:** 243,787 rows × 21 columns
- **Target:** `Churn` (binary, ~18% positive — imbalanced)
- **Features:** Account age, subscription type, payment method, viewing hours, support tickets, device registered, genre preference, and more

---

## Tech Stack

| Category | Tool |
|---|---|
| Language | Python 3.12 |
| ML | XGBoost, Scikit-learn |
| Hyperparameter Tuning | Optuna |
| Experiment Tracking | MLflow (SQLite backend) |
| Data Manipulation | Pandas, NumPy |
| API | FastAPI + Uvicorn |
| Web UI | Gradio |
| Containerisation | Docker (multi-stage build) |
| CI/CD | GitHub Actions |
| Dependency Management | uv |

---

## Project Structure

```
├── scripts/
│   └── run_pipeline.py        # orchestrates the full pipeline
├── src/
│   ├── data/
│   │   ├── load_data.py       # CSV loading
│   │   └── preprocess.py      # cleaning only
│   ├── features/
│   │   └── build_features.py  # all encoding
│   ├── models/
│   │   ├── train.py           # model training
│   │   ├── tune.py            # Optuna tuning
│   │   └── evaluate.py        # threshold-based evaluation
│   ├── serving/
│   │   ├── inference.py       # prediction pipeline
│   │   └── model/             # model artifacts
│   └── app/
│       └── main.py            # FastAPI + Gradio app
├── artifacts/
│   ├── feature_columns.json   # training feature schema
│   └── preprocessing.pkl      # serving metadata
├── notebooks/
│   ├── EDA.ipynb
│   ├── test_preprocess.ipynb
│   └── test_buildfeatures.ipynb
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── .github/
    └── workflows/
        └── deploy.yml
```

---

## Quickstart

### Run with Docker (recommended)

```bash
docker pull himanshudhurve96/churn-app-demo:latest
docker run -p 8000:8000 himanshudhurve96/churn-app-demo:latest
```

Then open:
- `http://localhost:8000/ui` — Gradio web interface
- `http://localhost:8000/docs` — FastAPI interactive docs
- `http://localhost:8000/predict` — REST endpoint

### Run locally

```bash
# clone and install
git clone https://github.com/hdsd1007/customer-churn.git
cd customer-churn
uv sync

# run pipeline (default params)
python scripts/run_pipeline.py --input data/raw/data.csv

# run pipeline with Optuna tuning
python scripts/run_pipeline.py --input data/raw/data.csv --tune

# start serving
uvicorn src.app.main:app --reload
```

---

## Pipeline Arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | required | Path to raw CSV |
| `--target` | `Churn` | Target column name |
| `--test_size` | `0.2` | Train/test split ratio |
| `--threshold` | `0.30` | Classification threshold |
| `--tune` | `False` | Run Optuna tuning |
| `--experiment` | `churn-detection` | MLflow experiment name |

---

## Model

**Algorithm:** XGBoost (binary classification)

**Class imbalance handling:** `scale_pos_weight = count(0) / count(1) ≈ 4.52`

**Evaluation metric:** `aucpr` (area under precision-recall curve) — chosen over logloss because it directly measures the precision/recall tradeoff and is robust to class imbalance.

**Threshold:** 0.30 (tuned for recall — the cost of missing a churner is higher than the cost of a false alarm)

**Results at threshold 0.30:**

| Metric | Value |
|---|---|
| Recall | 0.933 |
| Precision | 0.228 |
| F1 | 0.366 |
| ROC AUC | ~0.85 |

93% recall means the model catches nearly all churners. The lower precision is an acceptable tradeoff — in a retention campaign, contacting a non-churner costs far less than missing a churner.

---

## API

### POST `/predict`

```json
{
  "AccountAge": 20,
  "MonthlyCharges": 85.0,
  "TotalCharges": 1700.0,
  "ViewingHoursPerWeek": 40.0,
  "AverageViewingDuration": 60.0,
  "ContentDownloadsPerMonth": 2,
  "UserRating": 1.5,
  "SupportTicketsPerMonth": 6,
  "WatchlistSize": 1,
  "PaperlessBilling": "Yes",
  "MultiDeviceAccess": "No",
  "Gender": "Female",
  "ParentalControl": "No",
  "SubtitlesEnabled": "No",
  "SubscriptionType": "Basic",
  "PaymentMethod": "Electronic check",
  "ContentType": "Movies",
  "DeviceRegistered": "Mobile",
  "GenrePreference": "Action"
}
```

**Response:**
```json
{
  "prediction": "Likely to churn",
  "probability": 0.8423,
  "threshold": 0.3
}
```

---

## CI/CD

Every push to `main` triggers the GitHub Actions workflow:

1. Checkout code
2. Login to Docker Hub
3. Build Docker image
4. Push `himanshudhurve96/churn-app-demo:latest` to Docker Hub

---

## Key Design Decisions

**Preprocess vs build_features separation** — `preprocess.py` handles cleaning only. `build_features.py` handles all encoding. This separation ensures each script has one responsibility and prevents the downstream encoding step from receiving already-encoded data.

**Hardcoded transforms in inference.py** — `build_features.py` detects binary columns using `nunique()`, which returns 1 for every column on a single-row inference input. `inference.py` hardcodes the encoding maps so single-row prediction works correctly.

**Threshold at 0.30** — the default `predict` threshold of 0.50 missed too many churners. Lowering to 0.30 increases recall significantly at the cost of precision, which is the right tradeoff for a churn use case.

**scale_pos_weight** — computed from training data only, not the full dataset, to prevent leakage.

---

## Known Limitations

- MLflow model registry uses local SQLite — not suitable for multi-user or cloud environments
- `src/serving/model/` contains model files committed to the repo — acceptable for demo, would use S3/cloud storage in production
- No data validation at inference time beyond type coercion
- No model monitoring or drift detection

---
