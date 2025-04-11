"""
Training loop implementation for language models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from tqdm import tqdm
import time

from src.models.base_model import BaseModel
from config import (
    DEVICE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS,
    GRADIENT_CLIP_VAL, EARLY_STOPPING_PATIENCE, MODELS_DIR
)

logger = logging.getLogger(__name__)

class Trainer:
    """
    Trainer class for training and evaluating language models.
    """
    def __init__(self, 
                 model: BaseModel,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 learning_rate: float = LEARNING_RATE,
                 weight_decay: float = WEIGHT_DECAY,
                 num_epochs: int = NUM_EPOCHS,
                 gradient_clip_val: float = GRADIENT_CLIP_VAL,
                 patience: int = EARLY_STOPPING_PATIENCE,
                 model_dir: Path = MODELS_DIR,
                 model_name: str = None):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            num_epochs: Number of training epochs
            gradient_clip_val: Gradient clipping value
            patience: Patience for early stopping
            model_dir: Directory to save model checkpoints
            model_name: Name of the model for saving checkpoints
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.gradient_clip_val = gradient_clip_val
        self.patience = patience
        self.model_dir = Path(model_dir)
        self.model_name = model_name or model.__class__.__name__.lower()
        
        # Make sure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Set up optimizer and loss function
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=patience // 2,
            verbose=True
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token (0)
        
        # Track metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
        # Move model to device
        self.model.to(DEVICE)
        
        logger.info(f"Initialized trainer for {self.model_name}")
        logger.info(f"Model has {model.get_model_size()} parameters")
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs.
        
        Returns:
            Dictionary of metrics (train_loss, val_loss)
        """
        logger.info(f"Starting training for {self.model_name}")
        
        for epoch in range(self.num_epochs):
            start_time = time.time()
            
            # Training
            train_loss = self._train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self._validate_epoch()
            self.val_losses.append(val_loss)
            
            # Calculate perplexity
            train_ppl = np.exp(train_loss)
            val_ppl = np.exp(val_loss)
            
            # Logging
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} | "
                        f"Time: {epoch_time:.2f}s | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Train PPL: {train_ppl:.2f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val PPL: {val_ppl:.2f}")
            
            # Learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                
                # Save model checkpoint
                self._save_checkpoint(val_loss=val_loss, epoch=epoch)
                logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1
                logger.info(f"No improvement for {self.epochs_without_improvement} epochs")
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        logger.info(f"Training completed. Best model from epoch {self.best_epoch+1} "
                    f"with validation loss: {self.best_val_loss:.4f}")
        
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        }
    
    def _train_epoch(self) -> float:
        """
        Train the model for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        # Progress bar
        progress_bar = tqdm(self.train_dataloader, desc="Training", leave=False)
        
        for batch in progress_bar:
            # Get batch data
            input_ids = batch['input_ids'].to(DEVICE)
            target_ids = batch['target_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(input_ids, attention_mask)
            
            # Reshape for cross-entropy
            # From (batch_size, seq_len, vocab_size) to (batch_size * seq_len, vocab_size)
            logits = logits.reshape(-1, logits.size(-1))
            targets = target_ids.reshape(-1)
            
            # Calculate loss
            loss = self.criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            
            # Update weights
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item() * targets.size(0)
            total_tokens += (targets != 0).sum().item()  # Exclude padding tokens
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average loss
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        
        return avg_loss
    
    def _validate_epoch(self) -> float:
        """
        Validate the model for one epoch.
        
        Returns:
            Average validation loss for the epoch
        """
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            # Progress bar
            progress_bar = tqdm(self.val_dataloader, desc="Validation", leave=False)
            
            for batch in progress_bar:
                # Get batch data
                input_ids = batch['input_ids'].to(DEVICE)
                target_ids = batch['target_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                
                # Forward pass
                logits, _ = self.model(input_ids, attention_mask)
                
                # Reshape for cross-entropy
                logits = logits.reshape(-1, logits.size(-1))
                targets = target_ids.reshape(-1)
                
                # Calculate loss
                loss = self.criterion(logits, targets)
                
                # Track loss
                total_loss += loss.item() * targets.size(0)
                total_tokens += (targets != 0).sum().item()  # Exclude padding tokens
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average loss
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        
        return avg_loss
    
    def _save_checkpoint(self, val_loss: float, epoch: int):
        """
        Save model checkpoint.
        
        Args:
            val_loss: Validation loss
            epoch: Current epoch
        """
        checkpoint_path = self.model_dir / f"{self.model_name}_best.pt"
        self.model.save(checkpoint_path)
        
        # Save metadata about the checkpoint
        metadata_path = self.model_dir / f"{self.model_name}_best_metadata.pt"
        torch.save({
            'epoch': epoch,
            'val_loss': val_loss,
            'val_ppl': np.exp(val_loss),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, metadata_path)