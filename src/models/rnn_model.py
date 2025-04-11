"""
RNN model implementation.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional

from src.models.base_model import BaseModel
from src.data.tokenizer import Tokenizer
from config import DEVICE, DROPOUT, NUM_LAYERS

class RNNModel(BaseModel):
    """
    Vanilla RNN-based language model.
    """
    def __init__(self, 
                 vocab_size: int, 
                 embedding_dim: int, 
                 hidden_dim: int,
                 num_layers: int = NUM_LAYERS,
                 dropout: float = DROPOUT,
                 tokenizer: Tokenizer = None):
        """
        Initialize the RNN model.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of the embedding layer
            hidden_dim: Dimension of the hidden layers
            num_layers: Number of RNN layers
            dropout: Dropout probability
            tokenizer: Tokenizer instance for encoding/decoding text
        """
        super().__init__(vocab_size, embedding_dim, hidden_dim, tokenizer)
        
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # RNN layers
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Move model to device
        self.to(DEVICE)
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                temperature: float = 1.0,
                hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the RNN model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            temperature: Temperature for sampling (not used in forward pass)
            hidden_state: Optional hidden state from previous call
            
        Returns:
            Tuple of (logits, hidden_state)
        """
        # Get embeddings
        embeddings = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        
        # Apply RNN
        rnn_output, hidden = self.rnn(embeddings, hidden_state)  # (batch_size, seq_len, hidden_dim)
        
        # Apply dropout
        rnn_output = self.dropout(rnn_output)
        
        # Project to vocabulary
        logits = self.output_layer(rnn_output)  # (batch_size, seq_len, vocab_size)
        
        return logits, hidden
    
    def get_model_size(self) -> int:
        """
        Get the number of parameters in the model.
        
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters())