import optuna
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import logging

optuna.logging.set_verbosity(optuna.logging.WARNING)
 
logger = logging.getLogger(__name__)
 
 
def tune_model(X, y, scale_pos_weight: float, n_trials: int = 30) -> dict:
    """
    Runs Optuna hyperparameter search.
 
    Parameters
    ----------
    X                 : Training features
    y                 : Training target
    scale_pos_weight  : Class imbalance ratio (computed in run_pipeline)
    n_trials          : Number of Optuna trials
 
    Returns
    -------
    dict of best hyperparameters (ready to pass into XGBClassifier)
    """
 
    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 300, 800),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0, 5),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0, 5),
            "scale_pos_weight":  scale_pos_weight,   # Fix: was missing entirely
            "random_state":      42,
            "n_jobs":            -1,
            "eval_metric":       "aucpr",
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=3, scoring="recall")
        return scores.mean()
 
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
 
    logger.info("Best recall (CV): %.4f", study.best_value)   
    logger.info("Best params: %s", study.best_params)
 
    return study.best_params
    
    