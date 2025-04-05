"""
Transformer model implementation for text generation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, Union

import os
import sys
sys.path.append("../..")
import config
from src.models.base_model import BaseLanguageModel


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model.
    Based on the paper "Attention Is All You Need".
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = config.MAX_SEQ_LENGTH):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and store
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input embeddings.
        
        Args:
            x: Input embeddings [batch_size, seq_len, embedding_dim]
            
        Returns:
            Embeddings with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerModel(BaseLanguageModel):
    """Transformer-based language model."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = config.EMBEDDING_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        nhead: int = config.TRANSFORMER_NHEAD,
        num_layers: int = config.NUM_LAYERS,
        dim_feedforward: int = config.TRANSFORMER_DIM_FEEDFORWARD,
        dropout: float = config.DROPOUT,
        pad_idx: int = 0,
        max_seq_length: int = config.MAX_SEQ_LENGTH
    ):
        """
        Initialize the Transformer model.
        
        Args:
            vocab_size: Size of the vocabulary.
            embedding_dim: Dimension of the token embeddings.
            hidden_dim: Dimension of the hidden state.
            nhead: Number of attention heads.
            num_layers: Number of transformer layers.
            dim_feedforward: Dimension of the feedforward network.
            dropout: Dropout probability.
            pad_idx: Index of the padding token.
            max_seq_length: Maximum sequence length.
        """
        super().__init__(vocab_size, embedding_dim, hidden_dim, pad_idx, dropout)
        
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.max_seq_length = max_seq_length
        
        # Position encoding layer
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, max_seq_length)
        
        # Project embedding dim to hidden dim if they differ
        self.input_projection = nn.Linear(embedding_dim, hidden_dim) if embedding_dim != hidden_dim else nn.Identity()
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_layers
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the model weights."""
        # Initialize embedding and output layer
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.output_layer.weight, mean=0, std=0.02)
        nn.init.zeros_(self.output_layer.bias)
        
        # Initialize transformer weights (already done by PyTorch)
    
    def _generate_square_subsequent_mask(self, sz: int, device: str) -> torch.Tensor:
        """
        Generate a square mask for the sequence to prevent the model 
        from looking at future tokens during training.
        
        Args:
            sz: Size of the square mask.
            device: Device to create the mask on.
            
        Returns:
            A square mask of shape [sz, sz]
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        hidden: Optional[Any] = None  # Not used in Transformer but kept for API consistency
    ) -> Tuple[torch.Tensor, None]:
        """
        Forward pass through the Transformer model.
        
        Args:
            input_ids: Tensor of token ids [batch_size, seq_len].
            attention_mask: Tensor indicating which tokens to attend to [batch_size, seq_len].
            hidden: Not used, kept for API consistency with RNN models.
            
        Returns:
            output: Tensor of token logits [batch_size, seq_len, vocab_size].
            hidden: None (kept for API consistency).
        """
        # Get the masks
        device = input_ids.device
        seq_len = input_ids.size(1)
        
        # Create causal mask for autoregressive property
        mask = self._generate_square_subsequent_mask(seq_len, device)
        
        # Create padding mask if attention_mask is provided
        src_key_padding_mask = None
        if attention_mask is not None:
            # Invert attention mask for transformer
            src_key_padding_mask = (attention_mask == 0)
        
        # Embed the input tokens
        embedded = self.embedding(input_ids) * math.sqrt(self.embedding_dim)
        
        # Add positional encoding
        embedded = self.pos_encoder(embedded)
        
        # Project to hidden dim if needed
        embedded = self.input_projection(embedded)
        
        # Pass through transformer encoder
        output = self.transformer_encoder(
            src=embedded,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Apply dropout
        output = self.dropout(output)
        
        # Project to vocabulary size
        output = self.output_layer(output)
        
        return output, None  # None to maintain API consistency
    
    def prompt(
        self,
        prompt_text: str,
        max_length: int = config.MAX_GEN_LENGTH,
        temperature: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt_text: The text prompt to start generation from.
            max_length: Maximum number of tokens to generate.
            temperature: Temperature for sampling.
            device: Device to run the model on.
            
        Returns:
            The generated text.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Call set_tokenizer() first.")
        
        # Move model to the specified device
        self.to(device)
        self.eval()
        
        # Tokenize the prompt
        if prompt_text.startswith("<bos>"):
            # Prompt already has BOS token
            prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        else:
            # Add BOS token
            prompt_tokens = [self.tokenizer.bos_id()] + self.tokenizer.encode(prompt_text, add_special_tokens=False)
        
        # Convert to tensor
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
        
        # Generate text
        with torch.no_grad():
            # Track all generated tokens (including initial prompt)
            all_tokens = prompt_tokens.copy()
            
            # Generate tokens one by one
            for _ in range(max_length):
                # If sequence gets too long, truncate from the beginning
                if len(all_tokens) > self.max_seq_length:
                    input_ids = torch.tensor([all_tokens[-self.max_seq_length:]], dtype=torch.long).to(device)
                else:
                    input_ids = torch.tensor([all_tokens], dtype=torch.long).to(device)
                
                # Forward pass
                logits, _ = self.forward(input_ids)
                
                # Get the logits for the next token
                next_token_logits = logits[0, -1, :]
                
                # Sample next token
                next_token = self._sample_next_token(next_token_logits, temperature)
                
                # Add to generated tokens
                all_tokens.append(next_token)
                
                # Stop if EOS token is generated
                if next_token == self.tokenizer.eos_id():
                    break
        
        # Decode generated tokens
        generated_text = self.tokenizer.decode(all_tokens)
        
        return generated_text
    
    def save(self, path: str = config.TRANSFORMER_MODEL_PATH):
        """
        Save the model to the specified path.
        
        Args:
            path: Path to save the model.
        """
        # Save additional parameters specific to Transformer
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'dim_feedforward': self.dim_feedforward,
            'pad_idx': self.pad_idx,
            'max_seq_length': self.max_seq_length,
        }
        
        torch.save(checkpoint, path)
    
    @classmethod
    def load(cls, path: str = config.TRANSFORMER_MODEL_PATH, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
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
            nhead=checkpoint['nhead'],
            num_layers=checkpoint['num_layers'],
            dim_feedforward=checkpoint['dim_feedforward'],
            pad_idx=checkpoint['pad_idx'],
            max_seq_length=checkpoint['max_seq_length'],
        )
        
        # Load the saved state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device
        model.to(device)
        
        return model