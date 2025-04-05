"""
RNN model implementation for text generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

import sys
sys.path.append("../..")
import config
from src.models.base_model import BaseLanguageModel


class RNNModel(BaseLanguageModel):
    """Vanilla RNN-based language model."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = config.EMBEDDING_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        num_layers: int = config.NUM_LAYERS,
        dropout: float = config.DROPOUT,
        pad_idx: int = 0
    ):
        """
        Initialize the RNN model.
        
        Args:
            vocab_size: Size of the vocabulary.
            embedding_dim: Dimension of the token embeddings.
            hidden_dim: Dimension of the hidden state.
            num_layers: Number of RNN layers.
            dropout: Dropout probability.
            pad_idx: Index of the padding token.
        """
        super().__init__(vocab_size, embedding_dim, hidden_dim, pad_idx, dropout)
        
        self.num_layers = num_layers
        
        # RNN layers
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Initialize weights with a small value to avoid exploding gradients
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the model weights."""
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the RNN model.
        
        Args:
            input_ids: Tensor of token ids [batch_size, seq_len].
            attention_mask: Tensor indicating which tokens to attend to [batch_size, seq_len].
            hidden: Initial hidden state [num_layers, batch_size, hidden_dim].
            
        Returns:
            output: Tensor of token logits [batch_size, seq_len, vocab_size].
            hidden: Updated hidden state [num_layers, batch_size, hidden_dim].
        """
        # Get batch size and sequence length
        batch_size, seq_len = input_ids.shape
        
        # Initialize hidden state if not provided
        if hidden is None:
            device = input_ids.device
            hidden = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_dim,
                device=device
            )
        
        # Embed the input tokens
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        # Pass through RNN
        output, hidden = self.rnn(embedded, hidden)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Project to vocabulary size
        output = self.output_layer(output)
        
        return output, hidden
    
    def save(self, path: str = config.RNN_MODEL_PATH):
        """
        Save the model to the specified path.
        
        Args:
            path: Path to save the model.
        """
        # Save additional parameters specific to RNN
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'pad_idx': self.pad_idx,
        }
        
        torch.save(checkpoint, path)
    
    @classmethod
    def load(cls, path: str = config.RNN_MODEL_PATH, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Load the model from the specified path.
        
        Args:
            path: Path to load the model from.
            device: Device to load the model to.
            
        Returns:
            The loaded model.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
            
        checkpoint = torch.load(path, map_location=device)
        
        # Create a new model instance
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers'],
            pad_idx=checkpoint['pad_idx']
        )
        
        # Load the saved state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device
        model.to(device)
        
        return model