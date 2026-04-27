# Srcipt for the entire pipele
# Loading -> Pre-Processing -> Feature-Building -> Modeling

import os
import sys
import argparse
import json
import joblib
import mlflow
from xgboost import XGBClassifier
import logging


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Import path for Local Modules 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(project_root)

# Local Modules
from sklearn.model_selection import train_test_split
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.models.tune import tune_model
from src.models.train import train_model
from src.models.evaluate import evaluate_model

# Main function
def main(args):
    # ── MLflow setup ──────────────────────────────────────────────────────────
    # NEW
    #db_path = os.path.join(project_root, "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///mlflow.db")
    mlflow.set_experiment(args.experiment)
 
    with mlflow.start_run(): # ADDED ARTIfact uri
        mlflow.log_param("model", "xgboost")
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("test_size", args.test_size)
 
        # ── Load ──────────────────────────────────────────────────────────────
        logger.info("Loading data from %s", args.input)
        df = load_data(args.input)
        logger.info("Loaded: %s", df.shape)
 
        # ── Preprocess (Clearning Only) ────────────────────────────────────────────────────────
        logger.info("Preprocessing")
        df = preprocess_data(df)
 
        processed_path = os.path.join(project_root, "data", "processed", "churn_processed.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        logger.info("Processed data saved → %s", processed_path)
 
        # ── Feature Engineering (Encoding Only) ───────────────────────────────────────────────
        logger.info("Building features")
        target = args.target
        if target not in df.columns:                          # Fix: was df.colums
            raise ValueError(f"Target column '{target}' not found")
 
        df_enc = build_features(df, target_col=target)
        logger.info("Feature engineering done: %d columns", df_enc.shape[1])
 
        # ── Save feature metadata for serving ─────────────────────────────────
        artifact_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifact_dir, exist_ok=True)
 
        feature_cols = list(df_enc.drop(columns=[target]).columns)
        with open(os.path.join(artifact_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)
 
        preprocessing_artifact = {"feature_columns": feature_cols, "target": target}
        joblib.dump(preprocessing_artifact, os.path.join(artifact_dir, "preprocessing.pkl"))
 
        mlflow.log_artifact(os.path.join(artifact_dir, "preprocessing.pkl"))
        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")
        logger.info("Saved %d feature columns", len(feature_cols))
 
        # ── Train / Test Split ────────────────────────────────────────────────
        X = df_enc.drop(columns=[target])
        y = df_enc[target]
 
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=21
        )
        logger.info("Train: %d rows  |  Test: %d rows", len(X_train), len(X_test))
 
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        logger.info("Class imbalance ratio (scale_pos_weight): %.2f", scale_pos_weight)
 
        # ── Tune (optional) ───────────────────────────────────────────────────
        # Pass --tune flag to run Optuna; otherwise uses sensible defaults
        if args.tune:
            logger.info("Running Optuna tuning")
            best_params = tune_model(X_train, y_train, scale_pos_weight)
            mlflow.log_params(best_params)
        else:
            logger.info("Skipping tuning — using default params")
            best_params = {
                "n_estimators": 495,
                "learning_rate": 0.011,
                "max_depth": 4,
                "subsample": 0.95,
                "colsample_bytree": 0.94,
                "min_child_weight": 9,
                "gamma": 2.47,
                "reg_alpha": 2.26,
                "reg_lambda": 4.97,
            }
 
        # ── Train ─────────────────────────────────────────────────────────────
        logger.info("Training model")
        model = train_model(
            X_train, y_train,
            params=best_params,
            scale_pos_weight=scale_pos_weight,
        )
 
        # ── Evaluate ──────────────────────────────────────────────────────────
        logger.info("Evaluating model")
        metrics = evaluate_model(model, X_test, y_test, threshold=args.threshold)
        mlflow.log_metrics(metrics)
 
        # ── Log model ─────────────────────────────────────────────────────────
        mlflow.xgboost.log_model(model, name="model")
        mlflow.log_artifact(os.path.join(artifact_dir, "preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifact_dir, "feature_columns.json")) 
        logger.info("Pipeline complete")
        
def parse_args():
    parser = argparse.ArgumentParser(description="Churn Detection Pipeline")
    parser.add_argument("--input",       required=True,  help="Path to raw CSV")
    parser.add_argument("--target",      default="Churn", help="Target column name")
    parser.add_argument("--test_size",   default=0.2,    type=float)
    parser.add_argument("--threshold",   default=0.30,   type=float, help="Classification threshold")
    parser.add_argument("--tune",        action="store_true", help="Run Optuna tuning before training")
    parser.add_argument("--experiment",  default="churn-detection")
    parser.add_argument("--mlflow_uri",  default=None)
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
      
