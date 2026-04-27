import os
import sys
import json
import logging
import mlflow.pyfunc
import pandas as pd
import numpy as np 

logger = logging.getLogger(__name__)

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE         = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.append(_PROJECT_ROOT)

# ── Load feature schema saved during training ─────────────────────────────────
_FEATURE_FILE = os.path.join(_PROJECT_ROOT, "artifacts", "feature_columns.json")
with open(_FEATURE_FILE) as f:
    FEATURE_COLS = json.load(f)

logger.info("Loaded %d feature columns", len(FEATURE_COLS))

# ── Load model ────────────────────────────────────────────────────────────────
# The following uses the DB when not manually copying the model
# mlflow.set_tracking_uri(f"sqlite:///mlflow.db")
# _MODEL_URI = "models:/Customer-Churn-Model@production"

try:
    # model = mlflow.pyfunc.load_model(_MODEL_URI)
    model = mlflow.pyfunc.load_model(os.path.join(_HERE,"model")) # <- Manually copied the model under serving/model
    logger.info("Model loaded: %s", model)
except Exception as e:
    raise RuntimeError(f"Could not load model from '{model}': {e}")

# ── Transformation constants ──────────────────────────────────────────────────
# These must exactly match what build_features.py produces during training.

# Binary columns — deterministic mappings
_BINARY_MAP = {
    "PaperlessBilling":  {"No": 0, "Yes": 1},
    "MultiDeviceAccess": {"No": 0, "Yes": 1},
    "Gender":            {"Female": 0, "Male": 1},
    "ParentalControl":   {"No": 0, "Yes": 1},
    "SubtitlesEnabled":  {"No": 0, "Yes": 1},
}

# Multi-category OHE columns — categories kept after drop_first=True
# Dropped category is first alphabetically (pd.get_dummies default)
_OHE_COLS = {
    "SubscriptionType": ["Premium", "Standard"],
    "PaymentMethod":    ["Credit card", "Electronic check", "Mailed check"],
    "ContentType":      ["Movies", "TV Shows"],
    "DeviceRegistered": ["Mobile", "TV", "Tablet"],
    "GenrePreference":  ["Comedy", "Drama", "Fantasy", "Sci-Fi"],
}

# Numeric columns — passed through as-is
_NUMERIC_COLS = [
    "AccountAge", "MonthlyCharges", "TotalCharges",
    "ViewingHoursPerWeek", "AverageViewingDuration",
    "ContentDownloadsPerMonth", "UserRating",
    "SupportTicketsPerMonth", "WatchlistSize",
]

THRESHOLD = 0.30


def _transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw input into model-ready features.
    Hardcodes encoding so it works on single-row inference
    (build_features uses nunique() which breaks on 1 row).
    """
    out = pd.DataFrame()

    # Step 1: Numeric — coerce to float
    for col in _NUMERIC_COLS:
        out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Step 2: Binary encoding
    for col, mapping in _BINARY_MAP.items():
        out[col] = df[col].map(mapping).fillna(0).astype(int)

    # Step 3: One-hot encoding — manually expand each category
    for col, categories in _OHE_COLS.items():
        val = df[col].iloc[0]
        for cat in categories:
            out[f"{col}_{cat}"] = 1 if val == cat else 0

    # Step 4: Align to exact training column order
    out = out.reindex(columns=FEATURE_COLS, fill_value=0)

    return out


def predict(input_dict: dict) -> dict:
    """
    Raw customer dict → prediction + probability.

    Returns
    -------
    dict with keys: prediction, probability, threshold
    """
    df     = pd.DataFrame([input_dict])
    df_enc = _transform(df)

    native = model._model_impl.xgb_model
    prob   = float(native.predict_proba(df_enc)[:, 1][0])
    
    label  = "Likely to churn" if prob >= THRESHOLD else "Not likely to churn"

    logger.info("Prediction: %s  |  probability: %.4f", label, prob)

    return {
        "prediction":  label,
        "probability": round(prob, 4),
        "threshold":   THRESHOLD,
    }