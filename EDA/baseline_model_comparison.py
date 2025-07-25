"""
Baseline Model Comparison Visualization Script

This script creates bar charts comparing different baseline models across various metrics
using data from the baseline_results.json file.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import os

# Set the style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_baseline_results(results_path):
    """Load baseline results from JSON file."""
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        print(f"Error: Could not find {results_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {results_path}")
        return None

def create_model_comparison_charts(results, save_path=None):
    """Create comprehensive bar charts comparing baseline models."""
    
    if not results:
        print("No results data available")
        return
    
    # Extract model names and metrics
    models = list(results.keys())
    
    # Define metrics to plot (focusing on test metrics for final comparison)
    test_metrics = {
        'RMSE': 'test_rmse',
        'MAE': 'test_mae', 
        'R²': 'test_r2',
        'MAPE': 'test_mape'
    }
    
    # CV metrics for comparison
    cv_metrics = {
        'CV RMSE (Mean)': 'cv_mean_rmse',
        'CV MAE (Mean)': 'cv_mean_mae',
        'CV R² (Mean)': 'cv_mean_r2'
    }
    
    # Define colors for each model
    colors = {
        'XGBoost': '#1f77b4',      # Blue
        'RandomForest': '#ff7f0e',  # Orange  
        'Lasso': '#2ca02c',         # Green
        'Ridge': '#d62728',         # Red
        'ElasticNet': '#9467bd'     # Purple
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Baseline Model Performance Comparison', fontsize=20, fontweight='bold', y=0.98)
    
    # Plot 1: Test RMSE and MAE
    ax1 = axes[0, 0]
    x_pos = np.arange(len(models))
    width = 0.35
    
    rmse_values = [results[model]['test_rmse'] for model in models]
    mae_values = [results[model]['test_mae'] for model in models]
    
    bars1 = ax1.bar(x_pos - width/2, rmse_values, width, label='RMSE', alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, mae_values, width, label='MAE', alpha=0.8)
    
    # Color bars by model
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        bar1.set_color(colors[models[i]])
        bar2.set_color(colors[models[i]])
        bar2.set_alpha(0.6)  # Make MAE bars slightly transparent
    
    ax1.set_xlabel('Models', fontweight='bold')
    ax1.set_ylabel('Error Values', fontweight='bold')
    ax1.set_title('Test Set Error Metrics (Lower is Better)', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (rmse, mae) in enumerate(zip(rmse_values, mae_values)):
        ax1.text(i - width/2, rmse + 0.1, f'{rmse:.2f}', ha='center', va='bottom', fontweight='bold')
        ax1.text(i + width/2, mae + 0.1, f'{mae:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Test R² and MAPE
    ax2 = axes[0, 1]
    r2_values = [results[model]['test_r2'] for model in models]
    mape_values = [results[model]['test_mape'] * 100 for model in models]  # Convert to percentage
    
    bars3 = ax2.bar(x_pos - width/2, r2_values, width, label='R² Score', alpha=0.8)
    bars4 = ax2.bar(x_pos + width/2, mape_values, width, label='MAPE (%)', alpha=0.8)
    
    # Color bars by model
    for i, (bar3, bar4) in enumerate(zip(bars3, bars4)):
        bar3.set_color(colors[models[i]])
        bar4.set_color(colors[models[i]])
        bar4.set_alpha(0.6)
    
    ax2.set_xlabel('Models', fontweight='bold')
    ax2.set_ylabel('Score Values', fontweight='bold')
    ax2.set_title('Test Set R² (Higher is Better) & MAPE % (Lower is Better)', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (r2, mape) in enumerate(zip(r2_values, mape_values)):
        ax2.text(i - width/2, r2 + 0.02, f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')
        ax2.text(i + width/2, mape + 0.2, f'{mape:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Cross-Validation RMSE with Error Bars
    ax3 = axes[1, 0]
    cv_rmse_mean = [results[model]['cv_mean_rmse'] for model in models]
    cv_rmse_std = [results[model]['cv_std_rmse'] for model in models]
    
    bars5 = ax3.bar(models, cv_rmse_mean, yerr=cv_rmse_std, capsize=5, alpha=0.8)
    
    # Color bars by model
    for i, bar in enumerate(bars5):
        bar.set_color(colors[models[i]])
    
    ax3.set_xlabel('Models', fontweight='bold')
    ax3.set_ylabel('CV RMSE', fontweight='bold')
    ax3.set_title('Cross-Validation RMSE (Mean ± Std)', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(cv_rmse_mean, cv_rmse_std)):
        ax3.text(i, mean + std + 0.1, f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Model Ranking Heatmap
    ax4 = axes[1, 1]
    
    # Create ranking data (1 = best, 5 = worst for 5 models)
    ranking_data = {}
    
    # For RMSE, MAE, MAPE: lower is better (rank ascending)
    for metric in ['test_rmse', 'test_mae', 'test_mape']:
        values = [results[model][metric] for model in models]
        ranks = pd.Series(values).rank(method='min').tolist()
        ranking_data[metric] = ranks
    
    # For R²: higher is better (rank descending)
    r2_values = [results[model]['test_r2'] for model in models]
    r2_ranks = pd.Series(r2_values).rank(method='min', ascending=False).tolist()
    ranking_data['test_r2'] = r2_ranks
    
    # Create DataFrame for heatmap
    rank_df = pd.DataFrame(ranking_data, index=models)
    rank_df.columns = ['RMSE Rank', 'MAE Rank', 'MAPE Rank', 'R² Rank']
    
    # Create heatmap (lower rank number = better performance = darker color)
    sns.heatmap(rank_df, annot=True, cmap='RdYlGn_r', ax=ax4, 
                cbar_kws={'label': 'Rank (1=Best, 5=Worst)'}, fmt='.0f')
    ax4.set_title('Model Performance Ranking Heatmap', fontweight='bold')
    ax4.set_xlabel('Metrics', fontweight='bold')
    ax4.set_ylabel('Models', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("BASELINE MODEL PERFORMANCE SUMMARY")
    print("="*60)
    
    # Find best model for each metric
    best_models = {}
    for metric_name, metric_key in test_metrics.items():
        if metric_name == 'R²':
            # Higher is better for R²
            best_model = max(models, key=lambda x: results[x][metric_key])
        else:
            # Lower is better for error metrics
            best_model = min(models, key=lambda x: results[x][metric_key])
        best_models[metric_name] = best_model
        
    for metric, model in best_models.items():
        value = results[model][test_metrics[metric]]
        if metric == 'MAPE':
            print(f"Best {metric}: {model} ({value*100:.2f}%)")
        else:
            print(f"Best {metric}: {model} ({value:.4f})")
    
    print("\nOverall Recommendation: XGBoost appears to be the best performing model" if 
          list(best_models.values()).count('XGBoost') >= 2 else 
          f"\nMixed results - consider ensemble or further tuning")

def main():
    """Main function to run the baseline model comparison."""
    # Get the script directory and construct paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    results_file = project_root / "Results" / "baseline_results.json"
    
    # Load results
    print("Loading baseline results...")
    results = load_baseline_results(results_file)
    
    if results:
        print(f"Found results for {len(results)} models: {', '.join(results.keys())}")
        
        # Create output path for saving the plot
        output_path = script_dir / "baseline_model_comparison.png"
        
        # Create comparison charts
        create_model_comparison_charts(results, save_path=output_path)
        
    else:
        print("Failed to load baseline results. Please check the file path and format.")

if __name__ == "__main__":
    main()
