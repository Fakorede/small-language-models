"""
Tokenizer implementation using SentencePiece BPE.
"""

import os
import glob
import sentencepiece as spm
from typing import List, Union, Dict, Any
import torch

import sys
sys.path.append("../..")
import config


class SPTokenizer:
    """
    SentencePiece BPE tokenizer implementation for the text generation task.
    """
    def __init__(self, vocab_size: int = config.VOCAB_SIZE):
        """
        Initialize the tokenizer.
        
        Args:
            vocab_size: Size of the vocabulary.
        """
        self.vocab_size = vocab_size
        self.model_prefix = os.path.join(config.TOKENIZER_DIR, f"spm_bpe_{vocab_size}")
        self.tokenizer = None
        
        # Special tokens
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        
    def train(self, raw_data_dir: str = config.RAW_DATA_DIR, 
              character_coverage: float = 0.9995,
              model_type: str = "bpe"):
        """
        Train the tokenizer on the raw data.
        
        Args:
            raw_data_dir: Directory containing raw text files.
            character_coverage: Character coverage in SentencePiece.
            model_type: Type of model ('bpe' or 'unigram').
        """
        # Prepare a temporary merged corpus file
        temp_corpus_file = os.path.join(config.TOKENIZER_DIR, "corpus.txt")
        
        # Get all text files in the raw data directory
        text_files = glob.glob(os.path.join(raw_data_dir, "*.txt"))
        
        # Merge all text files into a single corpus
        with open(temp_corpus_file, 'w', encoding='utf-8') as outfile:
            for file_path in text_files:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
                    outfile.write("\n")
                    
        # Train the SentencePiece tokenizer
        spm.SentencePieceTrainer.train(
            input=temp_corpus_file,
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size,
            character_coverage=character_coverage,
            model_type=model_type,
            normalization_rule_name="nmt_nfkc_cf",
            user_defined_symbols=[self.pad_token, self.bos_token, self.eos_token],
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3
        )
        
        # Load the trained tokenizer
        self.load()
        
        # Clean up the temporary corpus file
        if os.path.exists(temp_corpus_file):
            os.remove(temp_corpus_file)
            
    def load(self, model_path: str = None):
        """
        Load the tokenizer from a SentencePiece model file.
        
        Args:
            model_path: Path to the SentencePiece model file.
                        If None, uses the default path.
        """
        if model_path is None:
            model_path = f"{self.model_prefix}.model"
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SentencePiece model not found at {model_path}")
            
        self.tokenizer = spm.SentencePieceProcessor(model_file=model_path)
        
    def save(self, path: str = None):
        """
        Save the tokenizer configuration.
        
        Args:
            path: Path to save the configuration.
        """
        # The SentencePiece model file is already saved during training
        pass
        
    def encode(self, text: Union[str, List[str]], add_special_tokens: bool = False) -> Union[List[int], List[List[int]]]:
        """
        Encode text to token ids.
        
        Args:
            text: Text to encode (string or list of strings).
            add_special_tokens: Whether to add special tokens.
            
        Returns:
            List of token ids.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load() or train() first.")
            
        if isinstance(text, str):
            if add_special_tokens:
                return [self.tokenizer.bos_id] + self.tokenizer.encode(text) + [self.tokenizer.eos_id]
            else:
                return self.tokenizer.encode(text)
        else:
            if add_special_tokens:
                return [[self.tokenizer.bos_id] + self.tokenizer.encode(t) + [self.tokenizer.eos_id] for t in text]
            else:
                return [self.tokenizer.encode(t) for t in text]
            
    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """
        Decode token ids to text.
        
        Args:
            token_ids: List of token ids.
            
        Returns:
            Decoded text.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load() or train() first.")
            
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
            
        return self.tokenizer.decode(token_ids)
    
    def decode_batch(self, token_ids_batch: Union[List[List[int]], torch.Tensor]) -> List[str]:
        """
        Decode a batch of token ids to text.
        
        Args:
            token_ids_batch: Batch of token ids.
            
        Returns:
            List of decoded texts.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load() or train() first.")
            
        if isinstance(token_ids_batch, torch.Tensor):
            token_ids_batch = token_ids_batch.cpu().tolist()
            
        return [self.tokenizer.decode(ids) for ids in token_ids_batch]
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        if self.tokenizer is not None:
            return self.tokenizer.get_piece_size()
        return self._vocab_size
        
    @vocab_size.setter
    def vocab_size(self, size: int):
        """Set the vocabulary size."""
        self._vocab_size = size
        
    @property
    def pad_id(self) -> int:
        """Get the pad token id."""
        if self.tokenizer is not None:
            return self.tokenizer.pad_id()
        return 0
        
    @property
    def bos_id(self) -> int:
        """Get the beginning-of-sequence token id."""
        if self.tokenizer is not None:
            return self.tokenizer.bos_id
        return 1
        
    @property
    def eos_id(self) -> int:
        """Get the end-of-sequence token id."""
        if self.tokenizer is not None:
            return self.tokenizer.eos_id
        return 2
        
    @property
    def unk_id(self) -> int:
        """Get the unknown token id."""
        if self.tokenizer is not None:
            return self.tokenizer.unk_id()
        return 3