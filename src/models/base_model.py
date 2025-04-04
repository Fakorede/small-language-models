import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Any


class BaseTextGenerationModel(nn.Module):
    """Base class for text generation models."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        """
        Initializes the base model.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of token embeddings
            hidden_dim: Dimension of hidden states
        """
        super(BaseTextGenerationModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Common layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)  # Default dropout rate

    def forward(
            self,
            x: torch.Tensor,
            hidden: Optional[Any] = None,
            temperature: float = 1.0
    ) -> Tuple[torch.Tensor, Any, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of token indices [batch_size, seq_len]
            hidden: Optional hidden state
            temperature: Temperature for sampling (1.0 means greedy)

        Returns:
            Tuple of (logits, hidden_state, next_token)
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def prompt(
            self,
            tokenizer,
            prompt_text: str,
            max_seq_length: int = 100,
            temperature: float = 1.0
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            tokenizer: SentencePiece tokenizer
            prompt_text: Text prompt to start generation
            max_seq_length: Maximum number of tokens to generate
            temperature: Temperature for sampling (1.0 means greedy)

        Returns:
            Generated text
        """
        raise NotImplementedError("Subclasses must implement prompt method")