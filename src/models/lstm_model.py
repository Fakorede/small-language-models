"""
LSTM model implementation for text generation.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Any

from src.models.base_model import BaseTextGenerationModel


class LSTMModel(BaseTextGenerationModel):
    """LSTM model for text generation."""
    
    def __init__(
        self, 
        vocab_size: int, 
        embedding_dim: int, 
        hidden_dim: int, 
        num_layers: int = 2, 
        dropout: float = 0.2
    ):
        """
        Initialize the LSTM model.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of token embeddings
            hidden_dim: Dimension of hidden states
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMModel, self).__init__(vocab_size, embedding_dim, hidden_dim)
        
        # LSTM specific layers
        self.lstm = nn.LSTM(
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
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass of the LSTM model.
        
        Args:
            x: Input tensor of token indices [batch_size, seq_len]
            hidden: Optional tuple of (hidden_state, cell_state)
            temperature: Temperature for sampling (1.0 means greedy)
            
        Returns:
            Tuple of (logits, (hidden_state, cell_state), next_token)
        """
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.lstm(embedded, hidden)
        output = self.dropout(output)
        logits = self.fc(output)
        
        # Apply temperature scaling for sampling
        if temperature != 1.0:
            logits = logits / temperature
        
        # Sample the next token
        # For undergrads, just take the highest probability token
        if temperature == 1.0 or not self.training:
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
        else:
            # Sample based on probabilities (for grad students)
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
        
        # Generate tokens autoregressively
        with torch.no_grad():
            for _ in range(max_seq_length):
                # Forward pass
                logits, hidden, next_token = self.forward(input_ids, hidden, temperature)
                
                # Get the next token
                next_token_id = next_token.item()
                generated_ids.append(next_token_id)
                
                # Check for EOS token
                if next_token_id == tokenizer.eos_id():
                    break
                
                # Prepare next input (only use the last predicted token)
                input_ids = next_token.unsqueeze(0)
        
        # Decode the generated tokens
        generated_text = tokenizer.decode(generated_ids)
        
        return generated_text