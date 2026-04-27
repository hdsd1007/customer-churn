from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score)
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.30) -> dict:
    """
    Evaluates model using a custom probability threshold.
    Returns a metrics dict that run_pipeline logs to MLflow.
 
    Parameters
    ----------
    model      : Trained XGBClassifier
    X_test     : Test features  (Fix: original was passing y_test into predict)
    y_test     : True labels
    threshold  : Probability cutoff (default 0.30 — tuned for recall)
    """
    probs = model.predict_proba(X_test)[:, 1]          # probability of churn
    preds = (probs >= threshold).astype(int)            # apply threshold
 
    recall    = recall_score(y_test, preds)
    precision = precision_score(y_test, preds)
    f1        = f1_score(y_test, preds)
    roc_auc   = roc_auc_score(y_test, probs)
 
    logger.info("Threshold : %.2f", threshold)
    logger.info("\n%s", classification_report(y_test, preds, digits=3))
    logger.info("Confusion Matrix:\n%s", confusion_matrix(y_test, preds))
 
    return {
        "recall":    recall,
        "precision": precision,
        "f1":        f1,
        "roc_auc":   roc_auc,
        "threshold": threshold,
    }