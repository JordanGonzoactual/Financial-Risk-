import os
import sys
import logging
import mlflow
from mlflow.models import infer_signature
import optuna
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add project root to path to import custom modules
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


def run_sampling_tuning(X_train, X_val, y_train, y_val, base_params, n_trials, parent_run_id, random_state=42):
    """
    Performs the third step of hyperparameter tuning for sampling parameters.
    """

    def objective(trial: optuna.trial.Trial) -> float:
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True) as run:
            if parent_run_id:
                mlflow.set_tag("mlflow.parentRunId", parent_run_id)
            mlflow.set_tag("optuna_trial_number", trial.number)
            mlflow.set_tag("optimization_step", "sampling")

            params = {
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
                'early_stopping_rounds': 50,
                **base_params
            }
            mlflow.log_params(params)

            # Train the model on the training set and evaluate on the validation set
            model = xgb.XGBRegressor(enable_categorical=True, **params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            # Log the best iteration
            mlflow.log_metric("best_iteration", model.best_iteration)

            # Make predictions and calculate metrics
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            mae = mean_absolute_error(y_val, preds)
            r2 = r2_score(y_val, preds)

            # Log metrics
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

            # Report to Optuna for pruning
            trial.report(rmse, step=0)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return rmse

    # Set up and run Optuna study
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    db_path = os.path.join(project_root, 'db', 'xgboost_tuning.db')
    storage_name = f"sqlite:///{db_path}"
    study_name = "step3_sampling_optimization"
    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(
        direction='minimize', 
        sampler=sampler, 
        pruner=pruner, 
        storage=storage_name, 
        study_name=study_name, 
        load_if_exists=True
    )

    study.optimize(objective, n_trials=n_trials)

    logging.info(f"Best trial for sampling tuning: {study.best_trial.value}")
    logging.info(f"Best parameters for sampling tuning: {study.best_params}")

    # Update the base parameters with the best found parameters
    best_params = base_params.copy()
    best_params.update(study.best_params)
    
    return best_params, study.best_value
