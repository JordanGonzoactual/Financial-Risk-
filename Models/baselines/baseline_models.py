import os
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATA_PATH = 'D:\\Python\\FinancialRisk\\Data\\processed'
RESULTS_PATH = 'D:\\Python\\FinancialRisk\\Results'
RANDOM_STATE = 42

# Ensure results directory exists
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

def load_data(path):
    """Loads pickled training and testing data."""
    with open(os.path.join(path, 'X_train_processed.pkl'), 'rb') as f:
        X_train = pickle.load(f)
    with open(os.path.join(path, 'X_test_processed.pkl'), 'rb') as f:
        X_test = pickle.load(f)
    with open(os.path.join(path, 'y_train.pkl'), 'rb') as f:
        y_train = pickle.load(f)
    with open(os.path.join(path, 'y_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)
    return X_train, X_test, y_train, y_test

def calculate_metrics(y_true, y_pred):
    """Calculates a dictionary of regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

def plot_residuals(y_true, y_pred, model_name):
    """Generates and saves a residuals vs. predicted values plot."""
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs. Predicted Values for {model_name}')
    plot_path = os.path.join(RESULTS_PATH, f'{model_name}_residuals.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def plot_feature_importance(model, feature_names, model_name):
    """Generates and saves a feature importance plot."""
    if not hasattr(model, 'feature_importances_'):
        return None
    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:] # Top 15 features
    plt.figure(figsize=(12, 8))
    plt.title(f'Feature Importance for {model_name}')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plot_path = os.path.join(RESULTS_PATH, f'{model_name}_feature_importance.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def train_and_evaluate():
    """Main function to train models and evaluate."""
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)
    
    models = {
        'XGBoost': xgb.XGBRegressor(random_state=RANDOM_STATE, enable_categorical=True),
        'RandomForest': RandomForestRegressor(random_state=RANDOM_STATE),
        'Lasso': Lasso(random_state=RANDOM_STATE),
        'Ridge': Ridge(random_state=RANDOM_STATE),
        'ElasticNet': ElasticNet(random_state=RANDOM_STATE)
    }

    all_results = {}

    for name, model in models.items():
        print(f'Training {name}...')
        
        # Cross-validation
        cv_scores_rmse = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
        cv_scores_mae = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_scores_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        cv_metrics = {
            'cv_mean_rmse': -np.mean(cv_scores_rmse),
            'cv_std_rmse': np.std(cv_scores_rmse),
            'cv_mean_mae': -np.mean(cv_scores_mae),
            'cv_std_mae': np.std(cv_scores_mae),
            'cv_mean_r2': np.mean(cv_scores_r2),
            'cv_std_r2': np.std(cv_scores_r2)
        }
        
        # Train model on full training data
        model.fit(X_train, y_train)
        
        # Test performance
        y_pred = model.predict(X_test)
        test_metrics = calculate_metrics(y_test, y_pred)
        
        # Generate and save plots
        plot_residuals(y_test, y_pred, name)
        
        if name in ['XGBoost', 'RandomForest']:
            plot_feature_importance(model, X_train.columns, name)
        
        # Store results for comparison table
        all_results[name] = {**cv_metrics, **{f'test_{k}': v for k, v in test_metrics.items()}}

    # Save final results to JSON
    results_df = pd.DataFrame(all_results).T
    print('\n--- Results Comparison ---')
    print(results_df)
    results_df.to_json(os.path.join(RESULTS_PATH, 'baseline_results.json'), orient='index', indent=4)

if __name__ == '__main__':
    train_and_evaluate()

