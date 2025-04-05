"""
Base class for all text generation models.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any

import sys
sys.path.append("../..")
import config
from src.data.tokenizer import SPTokenizer


class BaseLanguageModel(nn.Module):
    """Base class for all language models."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        pad_idx: int = 0,
        dropout: float = 0.2
    ):
        """
        Initialize the base language model.
        
        Args:
            vocab_size: Size of the vocabulary.
            embedding_dim: Dimension of the token embeddings.
            hidden_dim: Dimension of the hidden state.
            pad_idx: Index of the padding token.
            dropout: Dropout probability.
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx
        
        # Token embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )
        
        # Output layer to predict next token probabilities
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Tokenizer reference (to be set later)
        self.tokenizer = None
    
    def forward(self, input_ids, attention_mask=None, hidden=None):
        """
        Forward pass (to be implemented by child classes).
        
        Args:
            input_ids: Tensor of token ids of shape [batch_size, seq_len].
            attention_mask: Tensor indicating which tokens to attend to.
            hidden: Initial hidden state (model-specific).
            
        Returns:
            output: Tensor of token probabilities of shape [batch_size, seq_len, vocab_size].
            hidden: Updated hidden state (model-specific).
        """
        raise NotImplementedError("Forward method must be implemented by subclasses")
    
    def _sample_next_token(self, logits: torch.Tensor, temperature: float = 1.0) -> int:
        """
        Sample the next token from the logits distribution.
        
        Args:
            logits: Logits for the next token.
            temperature: Temperature for sampling (1.0 means no temperature).
            
        Returns:
            The sampled token id.
        """
        if temperature == 0.0 or temperature == 1.0:
            # Use argmax for temperature 0 or 1
            return torch.argmax(logits, dim=-1).item()
        else:
            # Apply temperature scaling and sample
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).item()
    
    def set_tokenizer(self, tokenizer: SPTokenizer):
        """Set the tokenizer for the model."""
        self.tokenizer = tokenizer
    
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
            # Initialize hidden state if necessary
            hidden = None
            
            # Track all generated tokens (including initial prompt)
            all_tokens = input_ids.tolist()[0]
            
            # Keep only the last token for generation
            curr_input = input_ids[:, -1].unsqueeze(1)
            
            # Generate tokens one by one
            for _ in range(max_length):
                # Forward pass
                logits, hidden = self.forward(curr_input, hidden=hidden)
                
                # Get the logits for the next token
                next_token_logits = logits[:, -1, :]
                
                # Sample next token
                next_token = self._sample_next_token(next_token_logits[0], temperature)
                
                # Add to generated tokens
                all_tokens.append(next_token)
                
                # Update input for next step
                curr_input = torch.tensor([[next_token]], dtype=torch.long).to(device)
                
                # Stop if EOS token is generated
                if next_token == self.tokenizer.eos_id():
                    break
        
        # Decode generated tokens
        generated_text = self.tokenizer.decode(all_tokens)
        
        return generated_text
    
    def save(self, path: str):
        """
        Save the model to the specified path.
        
        Args:
            path: Path to save the model.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'pad_idx': self.pad_idx,
        }, path)
        
    @classmethod
    def load(cls, path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
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
            pad_idx=checkpoint['pad_idx']
        )
        
        # Load the saved state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device
        model.to(device)
        
        return model