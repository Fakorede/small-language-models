"""
Visualization utilities for training and evaluation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

import sys
sys.path.append("../..")
import config


def plot_learning_curves(
    train_losses: List[float],
    val_losses: List[float],
    title: str,
    save_path: str
):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
        title: Plot title.
        save_path: Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='x')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    print(f"Loss plot saved to {save_path}")


def plot_metric_comparison(
    metrics: Dict[str, Dict[str, float]],
    metric_name: str,
    title: str,
    save_path: str,
    is_lower_better: bool = True
):
    """
    Plot comparison of a metric across different models.
    
    Args:
        metrics: Dictionary of model_name -> metrics dictionary.
        metric_name: Name of the metric to plot.
        title: Plot title.
        save_path: Path to save the plot.
        is_lower_better: Whether lower values are better for this metric.
    """
    model_names = list(metrics.keys())
    metric_values = [metrics[model][metric_name] for model in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, metric_values)
    
    plt.title(title)
    plt.xlabel('Model')
    plt.ylabel(metric_name)
    plt.grid(axis='y')
    
    # Add value labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.02 * max(metric_values),
            f'{height:.4f}',
            ha='center',
            va='bottom',
            rotation=0
        )
    
    # Highlight best model
    if is_lower_better:
        best_idx = np.argmin(metric_values)
    else:
        best_idx = np.argmax(metric_values)
        
    bars[best_idx].set_color('green')
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    print(f"Metric comparison plot saved to {save_path}")


def plot_all_metrics_comparison(
    metrics: Dict[str, Dict[str, float]],
    title: str,
    save_path: str
):
    """
    Plot comparison of all metrics across different models using a normalized scale.
    
    Args:
        metrics: Dictionary of model_name -> metrics dictionary.
        title: Plot title.
        save_path: Path to save the plot.
    """
    if not metrics:
        print("No metrics to plot")
        return
    
    model_names = list(metrics.keys())
    metric_names = list(metrics[model_names[0]].keys())
    
    # Normalize metrics to 0-1 scale for comparison
    normalized_metrics = {}
    for metric in metric_names:
        values = [metrics[model][metric] for model in model_names]
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val > min_val else 1.0
        
        # For perplexity, lower is better, so invert normalization
        if metric == 'perplexity':
            normalized_metrics[metric] = [(max_val - metrics[model][metric]) / range_val for model in model_names]
        else:
            normalized_metrics[metric] = [(metrics[model][metric] - min_val) / range_val for model in model_names]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(model_names))
    width = 0.8 / len(metric_names)
    
    for i, metric in enumerate(metric_names):
        offset = (i - len(metric_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, normalized_metrics[metric], width, label=metric)
        
        # Add value labels
        for j, bar in enumerate(bars):
            original_value = metrics[model_names[j]][metric]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f'{original_value:.4f}',
                ha='center',
                va='bottom',
                rotation=90,
                fontsize=8
            )
    
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel('Normalized Score (higher is better)')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    print(f"All metrics comparison plot saved to {save_path}")