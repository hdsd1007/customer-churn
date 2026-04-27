import mlflow
import logging
import pandas as pd 
import mlflow.xgboost
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
    scale_pos_weight: float,
) -> XGBClassifier:
    """
    Trains XGBoost with the provided params.
    Does NOT split data — run_pipeline.py owns the split.
    Does NOT start its own mlflow run — run_pipeline.py owns the run.
 
    Parameters
    ----------
    X_train           : Training features
    y_train           : Training target
    params            : Hyperparameters (from tune_model or defaults)
    scale_pos_weight  : Class imbalance ratio
 
    Returns
    -------
    Trained XGBClassifier
    """
    full_params = {
        **params,
        "scale_pos_weight": scale_pos_weight,
        "random_state":     21,
        "n_jobs":           -1,
        "eval_metric":      "aucpr",
    }
 
    model = XGBClassifier(**full_params)
    model.fit(X_train, y_train)
 
    logger.info("Model trained on %d rows, %d features", X_train.shape[0], X_train.shape[1])
    return model
        
        