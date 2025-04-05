"""
Training loop implementation for language models.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any, Union

import sys
sys.path.append("../..")
import config
from src.visualization.loss_plots import plot_learning_curves


class Trainer:
    """Trainer class for language models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = config.LEARNING_RATE,
        weight_decay: float = config.WEIGHT_DECAY,
        clip_grad_norm: float = config.CLIP_GRAD_NORM,
        patience: int = config.PATIENCE,
        lr_patience: int = config.LR_PATIENCE,
        lr_factor: float = config.LR_FACTOR,
        model_type: str = 'model'
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train.
            train_dataloader: DataLoader for training.
            val_dataloader: DataLoader for validation.
            device: Device to train on.
            learning_rate: Learning rate.
            weight_decay: Weight decay for regularization.
            clip_grad_norm: Maximum norm for gradient clipping.
            patience: Patience for early stopping.
            lr_patience: Patience for learning rate scheduler.
            lr_factor: Factor for learning rate scheduler.
            model_type: Type of model (used for saving).
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_grad_norm = clip_grad_norm
        self.patience = patience
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.model_type = model_type
        
        # Move model to device
        self.model.to(device)
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=lr_factor,
            patience=lr_patience,
            verbose=True
        )
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Initialize history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model = None
        self.patience_counter = 0
        
    def save_model(self, path: Optional[str] = None):
        """
        Save the model.
        
        Args:
            path: Path to save the model. If None, uses default path.
        """
        if path is None:
            if self.model_type == 'rnn':
                path = config.RNN_MODEL_PATH
            elif self.model_type == 'lstm':
                path = config.LSTM_MODEL_PATH
            elif self.model_type == 'transformer':
                path = config.TRANSFORMER_MODEL_PATH
            else:
                path = os.path.join(config.MODELS_DIR, f"{self.model_type}_model.pt")
                
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def train_epoch(self) -> float:
        """
        Train the model for one epoch.
        
        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device) if "attention_mask" in batch else None
            labels = batch["labels"].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = self.model(input_ids, attention_mask)
            
            # Reshape outputs for loss computation
            # outputs: [batch_size, seq_len, vocab_size]
            # labels: [batch_size, seq_len]
            batch_size, seq_len = labels.shape
            outputs = outputs.view(batch_size * seq_len, -1)
            labels = labels.view(-1)
            
            # Compute loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.clip_grad_norm
                )
            
            # Update parameters
            self.optimizer.step()
            
            # Update total loss
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
        
        # Compute average loss
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        
        return avg_loss
    
    def validate(self) -> float:
        """
        Validate the model.
        
        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device) if "attention_mask" in batch else None
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                outputs, _ = self.model(input_ids, attention_mask)
                
                # Reshape outputs for loss computation
                batch_size, seq_len = labels.shape
                outputs = outputs.view(batch_size * seq_len, -1)
                labels = labels.view(-1)
                
                # Compute loss
                loss = self.criterion(outputs, labels)
                
                # Update total loss
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        # Compute average loss
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        
        return avg_loss
    
    def train(self, num_epochs: int = config.NUM_EPOCHS) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train for.
            
        Returns:
            Dictionary with training and validation losses.
        """
        print(f"Starting training for {self.model_type} model on {self.device}...")
        print(f"Training for {num_epochs} epochs")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train for one epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Print losses
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Check if this is the best model so far
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model = self.model.state_dict().copy()
                self.patience_counter = 0
                
                # Save best model
                self.save_model()
            else:
                self.patience_counter += 1
                print(f"Patience counter: {self.patience_counter}/{self.patience}")
                
                # Check for early stopping
                if self.patience_counter >= self.patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
        
        # If we didn't save any model (unlikely), save the last one
        if self.best_model is None:
            self.save_model()
        else:
            # Load the best model
            self.model.load_state_dict(self.best_model)
        
        # Calculate training time
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        
        # Plot learning curves
        plot_path = os.path.join(config.PLOTS_DIR, f"{self.model_type}_loss.png")
        plot_learning_curves(
            self.train_losses,
            self.val_losses,
            f"{self.model_type.upper()} Model Training Curve",
            plot_path
        )
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }