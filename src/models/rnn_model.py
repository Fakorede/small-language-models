import torch
import torch.nn as nn
from typing import Tuple, Optional

from src.models.base_model import BaseTextGenerationModel


class RNNModel(BaseTextGenerationModel):
    """RNN model for text generation."""

    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            hidden_dim: int,
            num_layers: int = 2,
            dropout: float = 0.2
    ):
        """
        Initialize the RNN model.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of token embeddings
            hidden_dim: Dimension of hidden states
            num_layers: Number of RNN layers
            dropout: Dropout rate
        """
        super(RNNModel, self).__init__(vocab_size, embedding_dim, hidden_dim)

        # RNN specific layers
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            hidden: Optional[torch.Tensor] = None,
            temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the RNN model.

        Args:
            x: Input tensor of token indices [batch_size, seq_len]
            hidden: Optional hidden state [num_layers, batch_size, hidden_dim]
            temperature: Temperature for sampling (1.0 means greedy)

        Returns:
            Tuple of (logits, hidden_state, next_token)
        """
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.rnn(embedded, hidden)
        output = self.dropout(output)
        logits = self.fc(output)

        # Apply temperature scaling for sampling
        if temperature != 1.0:
            logits = logits / temperature

        # Sample the next token
        # take the highest probability token
        if temperature == 1.0 or not self.training:
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
        else:
            # or sample based on probabilities
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze(-1)

        return logits, hidden, next_token

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
        self.eval()
        device = next(self.parameters()).device
        prompt_ids = tokenizer.encode(prompt_text, out_type=int)
        input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(device)

        # Initialize hidden state
        hidden = None

        # Store generated tokens
        generated_ids = []

        # Generate tokens auto regressively
        with torch.no_grad():
            # First pass with the full prompt
            logits, hidden, next_token = self.forward(input_ids, hidden, temperature)
            next_token_id = next_token.item()
            generated_ids.append(next_token_id)

            # Continue generating one token at a time
            for _ in range(max_seq_length - 1):
                # Create a new tensor with just the last predicted token
                # Make it [batch_size, sequence_length] which is [1, 1]
                input_ids = torch.tensor([[next_token_id]], dtype=torch.long).to(device)

                # Forward pass
                logits, hidden, next_token = self.forward(input_ids, hidden, temperature)

                # Get the next token
                next_token_id = next_token.item()
                generated_ids.append(next_token_id)

                # Check for EOS token
                if next_token_id == tokenizer.eos_id():
                    break

        # Decode the generated tokens
        generated_text = tokenizer.decode(generated_ids)

        return generated_text