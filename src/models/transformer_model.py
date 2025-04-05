"""
Transformer model implementation for text generation.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Any

from src.models.base_model import BaseTextGenerationModel


class PositionalEncoding(nn.Module):
    """Positional encoding for the Transformer model."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize the positional encoding.
        
        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input.
        
        Args:
            x: Input tensor [seq_len, batch_size, embedding_dim]
            
        Returns:
            Output with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class TransformerModel(BaseTextGenerationModel):
    """Transformer model for text generation."""
    
    def __init__(
        self, 
        vocab_size: int, 
        embedding_dim: int, 
        hidden_dim: int, 
        nhead: int = 4, 
        num_layers: int = 2, 
        dropout: float = 0.2
    ):
        """
        Initialize the Transformer model.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of token embeddings
            hidden_dim: Dimension of feed forward network
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super(TransformerModel, self).__init__(vocab_size, embedding_dim, hidden_dim)
        
        # Transformer specific layers
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.fc = nn.Linear(embedding_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate a square mask for the sequence.
        
        Args:
            sz: Size of the square mask
            
        Returns:
            Mask tensor
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Any] = None, 
        temperature: float = 1.0,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, None, torch.Tensor]:
        """
        Forward pass of the Transformer model.
        
        Args:
            x: Input tensor of token indices [batch_size, seq_len]
            hidden: Not used in Transformer but kept for compatibility
            temperature: Temperature for sampling (1.0 means greedy)
            mask: Optional attention mask
            
        Returns:
            Tuple of (logits, None, next_token)
        """
        # Apply embedding and scale
        embedded = self.embedding(x) * np.sqrt(self.embedding.embedding_dim)
        
        # Apply positional encoding (expects [seq_len, batch_size, embedding_dim])
        embedded = self.pos_encoder(embedded.transpose(0, 1)).transpose(0, 1)
        embedded = self.dropout(embedded)
        
        # Create attention mask if None
        if mask is None:
            mask = self._generate_square_subsequent_mask(x.size(1)).to(x.device)
        
        # Apply transformer encoder
        output = self.transformer_encoder(embedded, mask)
        output = self.dropout(output)
        logits = self.fc(output)
        
        # Apply temperature scaling for sampling
        if temperature != 1.0:
            logits = logits / temperature
        
        # Sample the next token
        if temperature == 1.0 or not self.training:
            # Take the highest probability token (required for undergrads)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
        else:
            # Sample based on probabilities (required for grad students)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze(-1)
        
        return logits, None, next_token
    
    def prompt(
        self, 
        tokenizer, 
        prompt_text: str, 
        max_seq_length: int = 100, 
        temperature: float = 1.0,
        max_context_length: int = 512
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            tokenizer: SentencePiece tokenizer
            prompt_text: Text prompt to start generation
            max_seq_length: Maximum number of tokens to generate
            temperature: Temperature for sampling (1.0 means greedy)
            max_context_length: Maximum context length to use
            
        Returns:
            Generated text
        """
        self.eval()
        device = next(self.parameters()).device
        prompt_ids = tokenizer.encode(prompt_text, out_type=int)
        
        # Store generated tokens (initialize with prompt)
        generated_ids = list(prompt_ids)
        
        # Generate tokens autoregressively
        with torch.no_grad():
            for _ in range(max_seq_length):
                # Make sure input doesn't exceed max context length
                if len(generated_ids) > max_context_length:
                    context_ids = generated_ids[-max_context_length:]
                else:
                    context_ids = generated_ids
                
                input_ids = torch.tensor([context_ids], dtype=torch.long).to(device)
                
                # Forward pass
                logits, _, next_token = self.forward(input_ids, temperature=temperature)
                
                # Get the next token
                next_token_id = next_token.item()
                generated_ids.append(next_token_id)
                
                # Check for EOS token
                if next_token_id == tokenizer.eos_id():
                    break
        
        # Remove prompt from the generated text
        response_ids = generated_ids[len(prompt_ids):]
        
        # Decode the generated tokens
        generated_text = tokenizer.decode(response_ids)
        
        return generated_text