import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

import config
from src.models.base_model import BaseTextGenerationModel


class ModelTrainer:
    """Trainer class for text generation models."""

    def __init__(
            self,
            model: BaseTextGenerationModel,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            learning_rate: float = 0.001,
            model_save_path: str = None,
            device: torch.device = torch.device("cpu")
    ):
        """
        Initialize the trainer.

        Args:
            model: Model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            learning_rate: Learning rate
            model_save_path: Path to save the model
            device: Device to train on
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model_save_path = model_save_path
        self.device = device

        # Set up loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=config.LR_SCHEDULER_FACTOR, patience=config.LR_SCHEDULER_PATIENCE
        )

    def train_epoch(self) -> float:
        """
        Train the model for one epoch.

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0

        for batch in self.train_dataloader:
            self.optimizer.zero_grad()

            input_ids = batch["input_ids"]
            target_ids = batch["target_ids"]

            # Forward pass
            logits, _, _ = self.model(input_ids)

            # Compute loss
            loss = 0
            for i in range(logits.size(0)):
                # Only consider the tokens corresponding to the target
                pred = logits[i, :len(target_ids[i]), :]
                target = target_ids[i]
                loss += self.criterion(pred, target)

            loss /= logits.size(0)  # Average over batch

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_dataloader)

    def validate(self) -> float:
        """
        Validate the model.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch["input_ids"]
                target_ids = batch["target_ids"]

                # Forward pass
                logits, _, _ = self.model(input_ids)

                # Compute loss
                loss = 0
                for i in range(logits.size(0)):
                    pred = logits[i, :len(target_ids[i]), :]
                    target = target_ids[i]
                    loss += self.criterion(pred, target)

                loss /= logits.size(0)
                total_loss += loss.item()

        return total_loss / len(self.val_dataloader)

    def train(
            self,
            num_epochs: int,
            early_stopping_patience: int = 3
    ) -> Tuple[List[float], List[float]]:
        """
        Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train for
            early_stopping_patience: Number of epochs to wait before early stopping

        Returns:
            Tuple of (train_losses, val_losses)
        """
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            # Training phase
            start_time = time.time()
            train_loss = self.train_epoch()
            train_losses.append(train_loss)

            # Validation phase
            val_loss = self.validate()
            val_losses.append(val_loss)

            # Calculate epoch time
            epoch_time = time.time() - start_time

            # Print epoch results
            print(f"Epoch {epoch + 1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Time: {epoch_time:.2f}s")

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if self.model_save_path:
                    # Make sure directory exists
                    os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
                    # Save the model
                    torch.save(self.model.state_dict(), self.model_save_path)
                    print(f"Model saved to {self.model_save_path}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Load the best model if saved
        if self.model_save_path and os.path.exists(self.model_save_path):
            self.model.load_state_dict(torch.load(self.model_save_path))
            print(f"Loaded best model from {self.model_save_path}")

        return train_losses, val_losses