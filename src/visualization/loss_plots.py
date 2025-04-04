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

