"""
Visualization functions for training and evaluation metrics.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import logging

from config import PLOTS_DIR

logger = logging.getLogger(__name__)

def plot_training_curves(train_losses: List[float],
                         val_losses: List[float],
                         model_name: str,
                         output_dir: Path = PLOTS_DIR):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        model_name: Name of the model
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training Curves')
    plt.legend()
    plt.grid(True)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = output_dir / f"{model_name}_training_curves.png"
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Training curves saved to {output_path}")

def plot_comparison_metrics(metrics: Dict[str, Dict[str, float]],
                            metric_name: str,
                            title: str,
                            ylabel: str,
                            output_dir: Path = PLOTS_DIR):
    """
    Plot comparison of a specific metric across different models.
    
    Args:
        metrics: Dictionary of metrics for each model
        metric_name: Name of the metric to compare
        title: Title of the plot
        ylabel: Label for the y-axis
        output_dir: Directory to save the plot
    """
    model_names = list(metrics.keys())
    metric_values = [metrics[model][metric_name] for model in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, metric_values)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{value:.2f}', ha='center', va='bottom')
    
    plt.xlabel('Model')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = output_dir / f"comparison_{metric_name}.png"
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Comparison plot saved to {output_path}")

def plot_all_metrics(metrics_path: str,
                     output_dir: Path = PLOTS_DIR):
    """
    Generate all comparison plots from a metrics JSON file.
    
    Args:
        metrics_path: Path to the metrics JSON file
        output_dir: Directory to save the plots
    """
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Plot perplexity comparison
    plot_comparison_metrics(
        metrics=metrics,
        metric_name='perplexity',
        title='Model Comparison: Perplexity',
        ylabel='Perplexity (lower is better)',
        output_dir=output_dir
    )
    
    # Plot BLEU score comparison
    plot_comparison_metrics(
        metrics=metrics,
        metric_name='bleu_score',
        title='Model Comparison: BLEU Score',
        ylabel='BLEU Score (higher is better)',
        output_dir=output_dir
    )
    
    # Create normalized metrics chart (for comparing multiple metrics on one scale)
    model_names = list(metrics.keys())
    metric_names = ['perplexity', 'bleu_score']
    
    # Normalize metrics (invert perplexity so lower is better becomes higher is better)
    normalized_metrics = {}
    
    for metric_name in metric_names:
        values = [metrics[model][metric_name] for model in model_names]
        if metric_name == 'perplexity':
            # Invert perplexity: lower is better, so take 1/perplexity
            # First add a small epsilon to prevent division by zero
            values = [1 / (v + 1e-10) for v in values]
        
        # Min-max normalization to [0, 1]
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val != min_val else 1
        normalized = [(v - min_val) / range_val for v in values]
        
        for model, value in zip(model_names, normalized):
            if model not in normalized_metrics:
                normalized_metrics[model] = {}
            normalized_metrics[model][metric_name] = value
    
    # Plot normalized metrics
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(model_names))
    width = 0.35
    n_metrics = len(metric_names)
    
    for i, metric_name in enumerate(metric_names):
        offset = (i - n_metrics / 2 + 0.5) * width
        values = [normalized_metrics[model][metric_name] for model in model_names]
        bars = plt.bar(x + offset, values, width, label=metric_name)
        
        # Add value labels
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Model')
    plt.ylabel('Normalized Score (higher is better)')
    plt.title('Model Comparison: Normalized Metrics')
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = output_dir / "comparison_normalized_metrics.png"
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Normalized metrics comparison plot saved to {output_path}")