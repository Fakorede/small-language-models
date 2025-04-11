"""
Tokenizer implementation using SentencePiece BPE tokenization
"""
import os
import json
import config
import sentencepiece as spm
from pathlib import Path
import logging
from typing import List, Union, Optional

from config import TOKENIZER_MODEL_PREFIX, VOCAB_SIZE, RAW_DATA_DIR

logger = logging.getLogger(__name__)

class Tokenizer:
    """
    A wrapper around the SentencePiece tokenizer for BPE-based subword tokenization.
    """
    def __init__(self, model_path: Union[str, Path] = TOKENIZER_MODEL_PREFIX):
        """
        Initialize the tokenizer from a trained model or train a new one if model doesn't exist.
        
        Args:
            model_path: Path to the trained SentencePiece model
        """
        self.model_path = Path(model_path)
        self.sp_model = None
        
        if self.model_path.exists():
            self.load()
        else:
            logger.info(f"Tokenizer model not found at {model_path}. You need to train it first.")
    
    def train(self, vocab_size: int = VOCAB_SIZE, input_files: Optional[List[Path]] = None):
        """
        Train a new SentencePiece tokenizer model.
        
        Args:
            vocab_size: Size of the vocabulary
            input_files: List of text files to use for training the tokenizer
        """
        # if input_files is None:
        #     # Use all text files in the raw data directory
        #     input_files = list(RAW_DATA_DIR.glob("*.txt"))
        
        text_data = self.read_text_files(RAW_DATA_DIR)
        temp_file = f"{self.model_path}.txt"
        with open(temp_file, "w", encoding="utf-8") as f:
            for text in text_data:
                f.write(text + '\n')

        # Make sure parent directory exists
        # os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Train the SentencePiece model
        spm.SentencePieceTrainer.train(
            input=str(temp_file),
            model_prefix=str(self.model_path),
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            normalization_rule_name="nmt_nfkc_cf"
        )
        
        # Remove temporary file
        # temp_file.unlink()
        
        # Load the trained model
        self.load()
        logger.info(f"Tokenizer trained and saved to {self.model_path}")
    
    def load(self):
        """Load the SentencePiece model."""
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(f"{self.model_path}.model")
        logger.info(f"Loaded tokenizer from {self.model_path}")
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token IDs
        """
        if self.sp_model is None:
            raise ValueError("Tokenizer model not loaded. Call load() or train() first.")
        
        return self.sp_model.encode(text)
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        if self.sp_model is None:
            raise ValueError("Tokenizer model not loaded. Call load() or train() first.")
        
        return self.sp_model.decode(token_ids)
    
    def get_vocab_size(self) -> int:
        """
        Get the vocabulary size.
        
        Returns:
            Size of the vocabulary
        """
        if self.sp_model is None:
            raise ValueError("Tokenizer model not loaded. Call load() or train() first.")
        
        return self.sp_model.get_piece_size()
    
    def id_to_token(self, token_id: int) -> str:
        """
        Convert a token ID to its string representation.
        
        Args:
            token_id: The token ID
            
        Returns:
            The string representation of the token
        """
        if self.sp_model is None:
            raise ValueError("Tokenizer model not loaded. Call load() or train() first.")
        
        return self.sp_model.id_to_piece(token_id)
    
    def token_to_id(self, token: str) -> int:
        """
        Convert a token string to its ID.
        
        Args:
            token: The token string
            
        Returns:
            The token ID
        """
        if self.sp_model is None:
            raise ValueError("Tokenizer model not loaded. Call load() or train() first.")
        
        return self.sp_model.piece_to_id(token)
    
    def read_text_files(self, raw_dir: str) -> List[str]:
        """
        Read all text files from the raw directory.

        Args:
            raw_dir: Directory containing raw text files

        Returns:
            List of text contents
        """
        text_data = []
        for filename in os.listdir(raw_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(raw_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    text_data.append(file.read())
        return text_data
    
    @property
    def pad_id(self) -> int:
        """Get the ID of the padding token."""
        return 0
    
    @property
    def unk_id(self) -> int:
        """Get the ID of the unknown token."""
        return 1
    
    @property
    def bos_id(self) -> int:
        """Get the ID of the beginning-of-sequence token."""
        return 2
    
    @property
    def eos_id(self) -> int:
        """Get the ID of the end-of-sequence token."""
        return 3