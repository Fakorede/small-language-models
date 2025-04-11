"""
Transformer model implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

from src.models.base_model import BaseModel
from src.data.tokenizer import Tokenizer
from config import (
    DEVICE, DROPOUT, NUM_LAYERS, MAX_SEQ_LENGTH,
    TRANSFORMER_HEADS
)

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = MAX_SEQ_LENGTH):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_cached_len = max_len

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, embedding_dim]
        """
        seq_len = x.size(1)
        
        # If sequence is longer than our cached positions, compute the additional positions
        if seq_len > self.max_cached_len:
            # Compute positional encodings for the extra positions
            extra_len = seq_len - self.max_cached_len
            extra_pe = torch.zeros(1, extra_len, self.d_model, device=x.device)
            position = torch.arange(self.max_cached_len, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() * (-math.log(10000.0) / self.d_model))
            
            # Apply sine to even indices
            extra_pe[0, :, 0::2] = torch.sin(position * div_term)
            # Apply cosine to odd indices
            extra_pe[0, :, 1::2] = torch.cos(position * div_term)
            
            # Concatenate with cached positional encodings
            pos_encoding = torch.cat([self.pe, extra_pe], dim=1)[:, :seq_len]
        else:
            # Use cached positional encodings
            pos_encoding = self.pe[:, :seq_len]
        
        x = x + pos_encoding
        return self.dropout(x)

class TransformerModel(BaseModel):
    """
    Transformer-based language model.
    """
    def __init__(self, 
                 vocab_size: int, 
                 embedding_dim: int, 
                 hidden_dim: int,
                 num_layers: int = NUM_LAYERS,
                 num_heads: int = TRANSFORMER_HEADS,
                 dropout: float = DROPOUT,
                 tokenizer: Tokenizer = None):
        """
        Initialize the Transformer model.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of the embedding layer
            hidden_dim: Dimension of the hidden layers
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            tokenizer: Tokenizer instance for encoding/decoding text
        """
        super().__init__(vocab_size, embedding_dim, hidden_dim, tokenizer)
        
        self.model_type = 'Transformer'
        
        # Token embedding and positional encoding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        
        # Make sure embedding_dim is divisible by num_heads
        if embedding_dim % num_heads != 0:
            raise ValueError(f"Embedding dimension ({embedding_dim}) must be divisible by number of heads ({num_heads})")
        
        # Transformer encoder layer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_layers,
            norm=nn.LayerNorm(embedding_dim)
        )
        
        # Output projection
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        
        # Initialize parameters using Xavier/Glorot initialization
        self._init_weights()
        
        # Move model to device
        self.to(DEVICE)
    
    def _init_weights(self):
        """Initialize weights for the model."""
        # Initialize embedding weights
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        
        # Initialize output layer
        nn.init.normal_(self.output_layer.weight, mean=0, std=0.02)
        nn.init.zeros_(self.output_layer.bias)
    
    def _generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence to mask future positions."""
        # Create a mask with ones in the upper triangular part
        mask = torch.triu(torch.ones(sz, sz, device=DEVICE), diagonal=1)
        # Convert to Boolean mask where True means to mask
        mask = mask.bool()
        # Convert to float mask where -inf means to mask, 0.0 means to keep
        mask = torch.zeros_like(mask, dtype=torch.float, device=DEVICE).masked_fill(mask, float('-inf'))
        return mask
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                temperature: float = 1.0,
                hidden_state = None) -> Tuple[torch.Tensor, None]:
        """
        Forward pass of the Transformer model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            temperature: Temperature for sampling (not used in forward pass)
            hidden_state: Not used for Transformer but included for compatibility
            
        Returns:
            Tuple of (logits, None)
        """
        # Get embedding
        x = self.embedding(input_ids) * math.sqrt(self.embedding_dim)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create padding mask if provided
        src_key_padding_mask = None
        if attention_mask is not None:
            # Invert attention mask (1 = keep, 0 = mask out)
            src_key_padding_mask = ~attention_mask
        
        # Create causal attention mask for autoregressive generation
        # This mask ensures that positions can only attend to previous positions
        seq_len = input_ids.size(1)
        src_mask = self._generate_square_subsequent_mask(seq_len)
        
        # Pass through encoder
        try:
            output = self.transformer_encoder(
                x, 
                mask=src_mask,
                src_key_padding_mask=src_key_padding_mask if src_key_padding_mask is not None else None
            )
        except Exception as e:
            print(f"Error in transformer encoder: {e}")
            print(f"Input shape: {x.shape}")
            print(f"Mask shape: {src_mask.shape}")
            if src_key_padding_mask is not None:
                print(f"Key padding mask shape: {src_key_padding_mask.shape}")
            raise
        
        # Project to vocab size
        logits = self.output_layer(output)
        
        return logits, None  # Transformer doesn't have a hidden state like RNN/LSTM
    
    def generate(self, 
                prompt: str, 
                max_length: int = MAX_SEQ_LENGTH,
                temperature: float = 1.0) -> str:
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
        
        # Process the prompt
        if prompt.startswith("<bos>"):
            prompt = prompt[5:]  # Remove <bos> tag
            # Encode with BOS token
            input_tokens = [self.tokenizer.bos_id] + self.tokenizer.encode(prompt)
        else:
            # Regular encoding
            input_tokens = self.tokenizer.encode(prompt)
        
        # Convert to tensor and move to device
        input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(DEVICE)
        
        # Store original prompt tokens to exclude them from the output
        prompt_len = len(input_tokens)
        generated_tokens = input_tokens.copy()
        
        # Generate tokens
        with torch.no_grad():
            for _ in range(max_length):
                # Create attention mask (all 1's)
                attention_mask = torch.ones_like(input_tensor, dtype=torch.bool)
                
                # Get predictions
                logits, _ = self.forward(input_tensor, attention_mask)
                
                # Get next token probabilities (last position)
                next_token_logits = logits[0, -1, :].float()
                
                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Convert to probabilities
                next_token_probs = F.softmax(next_token_logits, dim=0)
                
                # Sample from the distribution or take the argmax
                if temperature > 0 and temperature != 1.0:
                    next_token_id = torch.multinomial(next_token_probs, 1).item()
                else:
                    # Greedy decoding
                    next_token_id = torch.argmax(next_token_probs).item()
                
                # For debugging: print token probabilities for first 5 tokens
                if len(generated_tokens) == prompt_len:
                    print("Top 5 token probabilities for first generated token:")
                    top_probs, top_indices = torch.topk(next_token_probs, 5)
                    for i, (idx, prob) in enumerate(zip(top_indices.tolist(), top_probs.tolist())):
                        token = self.tokenizer.id_to_token(idx)
                        print(f"  {i+1}. Token: '{token}', ID: {idx}, Probability: {prob:.4f}")
                
                # Append the new token
                generated_tokens.append(next_token_id)
                
                # Print the token being generated (for debugging)
                token_str = self.tokenizer.id_to_token(next_token_id)
                print(f"Generated token: '{token_str}' (ID: {next_token_id})")
                
                # Break if EOS token
                if next_token_id == self.tokenizer.eos_id:
                    break
                
                # Update input tensor for next iteration (add the new token)
                # Either append to the sequence or just use the new token depending on implementation
                input_tensor = torch.cat([
                    input_tensor, 
                    torch.tensor([[next_token_id]], dtype=torch.long, device=DEVICE)
                ], dim=1)
        
        # Get only newly generated tokens (exclude prompt)
        new_tokens = generated_tokens[prompt_len:]
        print(f"Generated {len(new_tokens)} new tokens")
        
        # Decode the tokens
        if new_tokens:
            generated_text = self.tokenizer.decode(new_tokens)
            
            # Remove EOS token if present
            eos_token = self.tokenizer.id_to_token(self.tokenizer.eos_id)
            if generated_text.endswith(eos_token):
                generated_text = generated_text[:-len(eos_token)]
            
            return generated_text
        else:
            print("Warning: No tokens were generated!")
            return ""
    
    def get_model_size(self) -> int:
        """
        Get the number of parameters in the model.
        
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters())