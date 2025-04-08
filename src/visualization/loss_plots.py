"""
Visualization utilities for training and evaluation results.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from typing import List, Dict, Any, Tuple


def plot_loss_curves(
    train_losses: List[float], 
    val_losses: List[float], 
    title: str, 
    save_path: str
) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Loss plot saved to {save_path}")


def plot_all_models_loss(
    losses: Dict[str, Tuple[List[float], List[float]]],
    save_dir: str
) -> None:
    """
    Plot losses for all models.
    
    Args:
        losses: Dictionary mapping model names to (train_losses, val_losses)
        save_dir: Directory to save plots
    """
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot individual model losses
    for model_name, (train_losses, val_losses) in losses.items():
        save_path = os.path.join(save_dir, f"{model_name.lower()}_loss.png")
        plot_loss_curves(
            train_losses, 
            val_losses, 
            f"{model_name} Model Loss", 
            save_path
        )
    
    # Plot all models training losses on one graph
    plt.figure(figsize=(12, 6))
    for model_name, (train_losses, _) in losses.items():
        plt.plot(train_losses, label=f'{model_name} Training', marker='o')
    
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "all_training_loss.png"), dpi=300)
    plt.close()
    
    # Plot all models validation losses on one graph
    plt.figure(figsize=(12, 6))
    for model_name, (_, val_losses) in losses.items():
        plt.plot(val_losses, label=f'{model_name} Validation', marker='s')
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "all_validation_loss.png"), dpi=300)
    plt.close()


def plot_metrics_comparison(
    metrics: Dict[str, Dict[str, float]],
    save_dir: str
) -> None:
    """
    Plot comparison of metrics across models.
    
    Args:
        metrics: Dictionary with metrics for each model
        save_dir: Directory to save plots
    """
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a DataFrame for easier plotting
    models = list(next(iter(metrics.values())).keys())
    df_data = {}
    
    for metric_name, metric_values in metrics.items():
        df_data[metric_name] = [metric_values[model] for model in models]
    
    df = pd.DataFrame(df_data, index=models)
    
    # Plot perplexity (lower is better)
    if 'perplexity' in metrics:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=df.index, y='perplexity', data=df, palette='Blues_d')
        ax.set_title('Perplexity Comparison (Lower is Better)', fontsize=16)
        ax.set_ylabel('Perplexity', fontsize=14)
        ax.set_xlabel('Model', fontsize=14)
        
        # Add value labels on top of bars
        for i, v in enumerate(df['perplexity']):
            ax.text(i, v + 1, f'{v:.2f}', ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "perplexity_comparison.png"), dpi=300)
        plt.close()
    
    # Plot BLEU score (higher is better)
    if 'bleu' in metrics:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=df.index, y='bleu', data=df, palette='Greens_d')
        ax.set_title('BLEU Score Comparison (Higher is Better)', fontsize=16)
        ax.set_ylabel('BLEU Score', fontsize=14)
        ax.set_xlabel('Model', fontsize=14)
        
        # Add value labels on top of bars
        for i, v in enumerate(df['bleu']):
            ax.text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "bleu_comparison.png"), dpi=300)
        plt.close()
    
    # Combined normalized metrics (higher is better for both)
    plt.figure(figsize=(12, 8))
    
    # Create a copy of the DataFrame for normalized values
    norm_df = df.copy()
    
    # Normalize perplexity (lower is better, so invert)
    if 'perplexity' in norm_df:
        max_perplexity = norm_df['perplexity'].max()
        norm_df['normalized_perplexity'] = 1 - (norm_df['perplexity'] / max_perplexity)
    
    # Normalize BLEU score (higher is better)
    if 'bleu' in norm_df:
        max_bleu = norm_df['bleu'].max()
        norm_df['normalized_bleu'] = norm_df['bleu'] / max_bleu
    
    # Melt the DataFrame for easier plotting
    plot_cols = ['normalized_perplexity', 'normalized_bleu'] 
    plot_cols = [col for col in plot_cols if col in norm_df.columns]
    
    if plot_cols:
        melted_df = pd.melt(
            norm_df.reset_index(), 
            id_vars='index', 
            value_vars=plot_cols, 
            var_name='Metric', 
            value_name='Normalized Value'
        )
        
        # Create grouped bar chart
        plt.figure(figsize=(12, 7))
        ax = sns.barplot(x='Metric', y='Normalized Value', hue='index', data=melted_df, palette='viridis')
        ax.set_title('Normalized Performance Metrics (Higher is Better)', fontsize=16)
        ax.set_ylabel('Normalized Value', fontsize=14)
        ax.set_xlabel('Metric', fontsize=14)
        ax.legend(title='Model')
        
        # Format x-axis labels
        metric_labels = {
            'normalized_perplexity': 'Perplexity\n(Normalized, Higher is Better)',
            'normalized_bleu': 'BLEU Score\n(Normalized)'
        }
        ax.set_xticklabels([metric_labels[col] for col in plot_cols])
        
        # Add value labels on top of bars
        for i, container in enumerate(ax.containers):
            for j, v in enumerate(container):
                ax.text(v.get_x() + v.get_width()/2, v.get_height() + 0.02, 
                        f'{v.get_height():.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "normalized_comparison.png"), dpi=300)
        plt.close()