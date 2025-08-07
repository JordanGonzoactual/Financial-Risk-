import os
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet

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

def train_and_evaluate(sample_size=None):
    """Main function to train models and evaluate."""
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)
    
    # Optional: Sample data for faster training during development
    if sample_size and sample_size < len(X_train):
        print(f"🎯 Sampling {sample_size} records for faster training...")
        sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
        X_train = X_train.iloc[sample_idx]
        y_train = y_train.iloc[sample_idx]
    
    # Configure baseline models with standard CPU settings
    models = {
        'XGBoost': xgb.XGBRegressor(
            random_state=RANDOM_STATE,
            tree_method='hist',
            enable_categorical=True,
            n_jobs=-1  # Use all CPU cores
        ),
        'RandomForest': RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1  # Use all CPU cores for faster training
        ),
        'Lasso': Lasso(random_state=RANDOM_STATE),
        'Ridge': Ridge(random_state=RANDOM_STATE),
        'ElasticNet': ElasticNet(random_state=RANDOM_STATE)
    }

    all_results = {}
    
    # Handle categorical columns for XGBoost compatibility
    print("🔧 Preprocessing data for model compatibility...")
    
    # Convert categorical columns to numeric for better compatibility
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    
    # Convert any remaining categorical columns to numeric
    for col in X_train_processed.columns:
        if X_train_processed[col].dtype.name == 'category':
            print(f"Converting categorical column: {col}")
            X_train_processed[col] = X_train_processed[col].astype('int8')
            X_test_processed[col] = X_test_processed[col].astype('int8')
    
    print(f"\nTraining {len(models)} baseline models...")
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print()

    # Progress bar for models
    model_progress = tqdm(models.items(), desc="Overall Progress", unit="model")
    
    for name, model in model_progress:
        model_progress.set_description(f"Training {name}")
        start_time = time.time()
        
        # Use processed data for all models
        X_train_model = X_train_processed
        X_test_model = X_test_processed
        y_train_model = y_train
        y_test_model = y_test
        
        # Cross-validation with progress
        print(f'\n🔄 Cross-validating {name}...')
        cv_progress = tqdm(total=3, desc="CV Metrics", leave=False)
        
        cv_scores_rmse = cross_val_score(model, X_train_model, y_train_model, cv=5, scoring='neg_root_mean_squared_error')
        cv_progress.update(1)
        
        cv_scores_mae = cross_val_score(model, X_train_model, y_train_model, cv=5, scoring='neg_mean_absolute_error')
        cv_progress.update(1)
        
        cv_scores_r2 = cross_val_score(model, X_train_model, y_train_model, cv=5, scoring='r2')
        cv_progress.update(1)
        cv_progress.close()
        
        cv_metrics = {
            'cv_mean_rmse': -np.mean(cv_scores_rmse),
            'cv_std_rmse': np.std(cv_scores_rmse),
            'cv_mean_mae': -np.mean(cv_scores_mae),
            'cv_std_mae': np.std(cv_scores_mae),
            'cv_mean_r2': np.mean(cv_scores_r2),
            'cv_std_r2': np.std(cv_scores_r2)
        }
        
        # Train model on full training data
        print(f'🏋️ Training {name} on full dataset...')
        model.fit(X_train_model, y_train_model)
        
        # Test performance
        print(f'📊 Evaluating {name} on test set...')
        y_pred = model.predict(X_test_model)
        
        test_metrics = calculate_metrics(y_test_model, y_pred)
        
        # Generate and save plots
        print(f'📈 Generating plots for {name}...')
        plot_residuals(y_test_model, y_pred, name)
        
        # Feature importance for tree-based models
        if name in ['XGBoost', 'RandomForest']:
            plot_feature_importance(model, X_train_model.columns, name)
        
        # Calculate training time
        training_time = time.time() - start_time
        print(f'✅ {name} completed in {training_time:.2f} seconds')
        print(f'   Test RMSE: {test_metrics["rmse"]:.4f}, R²: {test_metrics["r2"]:.4f}')
        
        # Store results for comparison table
        all_results[name] = {**cv_metrics, **{f'test_{k}': v for k, v in test_metrics.items()}}

    model_progress.close()
    
    # Save final results to JSON
    print('\n📊 Generating final results summary...')
    results_df = pd.DataFrame(all_results).T
    print('\n🏆 === BASELINE MODELS COMPARISON ===\n')
    print(results_df.round(4))
    
    print(f'\n💾 Saving results to {RESULTS_PATH}/baseline_results.json')
    results_df.to_json(os.path.join(RESULTS_PATH, 'baseline_results.json'), orient='index', indent=4)
    
    print('\n✨ All baseline models training completed successfully! ✨')

if __name__ == '__main__':
    train_and_evaluate()

