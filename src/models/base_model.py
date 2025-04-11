"""
Base class for all language models.
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any
import logging

from src.data.tokenizer import Tokenizer
from config import DEVICE, MAX_GENERATION_LENGTH, TEMPERATURE

logger = logging.getLogger(__name__)

class BaseModel(nn.Module):
    """
    Base class for all language models.
    """
    def __init__(self, 
                 vocab_size: int, 
                 embedding_dim: int, 
                 hidden_dim: int,
                 tokenizer: Tokenizer):
        """
        Initialize the base model.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of the embedding layer
            hidden_dim: Dimension of the hidden layers
            tokenizer: Tokenizer instance for encoding/decoding text
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tokenizer = tokenizer
        
        # Common layers for all models
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                temperature: float = 1.0) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            temperature: Temperature for sampling (higher = more random)
            
        Returns:
            Tuple of (logits, hidden_state)
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _select_next_token(self, 
                          logits: torch.Tensor, 
                          temperature: float = 1.0) -> int:
        """
        Select the next token based on the logits.
        
        Args:
            logits: Logits from the model
            temperature: Temperature for sampling (higher = more random)
            
        Returns:
            The selected token ID
        """
        if temperature == 0.0:
            # Greedy selection
            return torch.argmax(logits, dim=-1).item()
        else:
            # Apply temperature scaling
            logits = logits / temperature
            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)
            # Sample from the distribution
            return torch.multinomial(probs, 1).item()
    
    def generate(self, 
                prompt: str, 
                max_length: int = MAX_GENERATION_LENGTH,
                temperature: float = TEMPERATURE) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The input prompt
            max_length: Maximum number of tokens to generate
            temperature: Temperature for sampling (higher = more random)
            
        Returns:
            The generated text
        """
        self.eval()
        
        with torch.no_grad():
            # Check if prompt has BOS marker
            if prompt.startswith("<bos>"):
                prompt = prompt[5:]  # Remove <bos> tag
                # Encode with BOS token
                tokens = [self.tokenizer.bos_id] + self.tokenizer.encode(prompt)
            else:
                # Regular encoding
                tokens = self.tokenizer.encode(prompt)
            
            token_tensor = torch.tensor([tokens], device=DEVICE)
            
            # Generate tokens
            generated_tokens = tokens.copy()
            hidden = None
            
            for _ in range(max_length):
                # Create attention mask (all 1's since we're processing all tokens)
                attention_mask = torch.ones_like(token_tensor, dtype=torch.bool)
                
                # Forward pass
                # For RNN and LSTM, we need to reuse the hidden state
                if hidden is None:
                    logits, hidden = self.forward(token_tensor, attention_mask, temperature)
                else:
                    # For subsequent tokens, we only need to feed the last generated token
                    # but we still need to use the updated hidden state
                    last_token = token_tensor[:, -1:]
                    logits, hidden = self.forward(last_token, None, temperature, hidden_state=hidden)
                
                # Get the next token logits (last token in the sequence)
                next_token_logits = logits[0, -1, :]
                
                # Apply temperature and sample
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                else:
                    # Greedy sampling
                    next_token = torch.argmax(next_token_logits).item()
                
                # Append to the generated tokens
                generated_tokens.append(next_token)
                
                # Check if we've generated an EOS token
                if next_token == self.tokenizer.eos_id:
                    break
                
                # Update input for next iteration (add the new token)
                token_tensor = torch.tensor([[next_token]], device=DEVICE)
            
            # Only decode the newly generated tokens (excluding the prompt)
            prompt_len = len(tokens)
            new_tokens = generated_tokens[prompt_len:]
            
            # Debug information
            print(f"Generated {len(new_tokens)} new tokens")
            if len(new_tokens) > 0:
                print(f"First few token IDs: {new_tokens[:10]}")
            else:
                print("No tokens were generated")
            
            # Decode the generated tokens
            generated_text = self.tokenizer.decode(new_tokens)
            
            # Remove EOS token from output if present
            eos_token = self.tokenizer.id_to_token(self.tokenizer.eos_id)
            if generated_text.endswith(eos_token):
                generated_text = generated_text[:-len(eos_token)]
                
            return generated_text
    
    def save(self, path: str):
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
        }, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, tokenizer: Tokenizer) -> 'BaseModel':
        """
        Load a model from disk.
        
        Args:
            path: Path to load the model from
            tokenizer: Tokenizer instance
            
        Returns:
            The loaded model
        """
        checkpoint = torch.load(path, map_location=DEVICE)
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            tokenizer=tokenizer
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()
        logger.info(f"Model loaded from {path}")
        return model