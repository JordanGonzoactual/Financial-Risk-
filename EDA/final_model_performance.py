"""
Final Model Performance Visualization Script

This script generates a bar chart visualizing the performance metrics of the final model
using data from the 'model_metadata.json' file.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_model_metrics(metadata_path):
    """Load performance metrics from the model_metadata.json file."""
    try:
        with open(metadata_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    except FileNotFoundError:
        print(f"Error: Could not find {metadata_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {metadata_path}")
        return None

def create_performance_chart(metrics, model_name, save_path=None):
    """Create a bar chart for the final model's performance metrics."""
    if not metrics:
        print("No metrics data available to plot.")
        return

    metric_keys = {
        'RMSE': 'test_rmse',
        'MAE': 'test_mae',
        'R-squared (R²)': 'test_r2',
    }

    metric_values = [metrics.get(key, 0) for key in metric_keys.values()]
    metric_labels = list(metric_keys.keys())

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax.bar(metric_labels, metric_values, color=colors, alpha=0.8)

    ax.set_title(f'{model_name} - Final Performance Metrics', fontsize=16, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_ylim(0, max(metric_values) * 1.15)
    ax.tick_params(axis='x', rotation=0, labelsize=12)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + max(metric_values) * 0.02, f'{yval:.4f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Chart saved to {save_path}")

    plt.show()

def main():
    """Main function to load metrics and generate the performance chart."""
    script_dir = Path(__file__).parent.parent
    metadata_path = script_dir / "Models" / "trained_models" / "model_metadata.json"
    output_path = Path(__file__).parent / "final_model_performance.png"

    print(f"Loading metrics from {metadata_path}...")
    model_metrics = load_model_metrics(metadata_path)

    if model_metrics:
        model_name = model_metrics.get('model_type', 'Final Model')
        print("Successfully loaded metrics. Generating chart...")
        create_performance_chart(model_metrics, model_name, save_path=output_path)
    else:
        print("Could not generate chart due to missing data.")

if __name__ == "__main__":
    main()
