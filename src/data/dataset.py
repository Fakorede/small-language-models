"""
Dataset and dataloader implementations for the text generation task.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union

import sys
sys.path.append("../..")
import config
from src.data.tokenizer import SPTokenizer


class TextGenerationDataset(Dataset):
    """
    Dataset for text generation tasks with prompt-completion pairs.
    """
    def __init__(
        self,
        file_path: str,
        tokenizer: SPTokenizer,
        max_length: int = config.MAX_SEQ_LENGTH
    ):
        """
        Initialize the dataset.
        
        Args:
            file_path: Path to the JSONL file with prompt-completion pairs.
            tokenizer: Tokenizer instance.
            max_length: Maximum sequence length.
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        self._load_data()
        
    def _load_data(self):
        """Load data from the JSONL file."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.samples.append(sample)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample.
            
        Returns:
            Dictionary with input_ids, attention_mask, and labels.
        """
        sample = self.samples[idx]
        prompt = sample["prompt"]
        completion = sample["completion"]
        
        # Check if prompt already contains <bos> token
        if prompt.startswith("<bos>"):
            # Already has BOS token
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            # Add BOS token
            prompt_tokens = [self.tokenizer.bos_id] + self.tokenizer.encode(prompt, add_special_tokens=False)
        
        # Check if completion needs EOS token
        if completion.endswith("<eos>"):
            # Already has EOS token
            completion_tokens = self.tokenizer.encode(completion, add_special_tokens=False)
        else:
            # Add EOS token
            completion_tokens = self.tokenizer.encode(completion, add_special_tokens=False) + [self.tokenizer.eos_id]
        
        # Combine prompt and completion tokens
        input_ids = prompt_tokens + completion_tokens
        
        # Create labels (shifted right, -100 for prompt tokens to ignore them in loss)
        labels = [-100] * len(prompt_tokens) + completion_tokens
        
        # Truncate if needed
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)
        
        # Pad if necessary
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_id()] * padding_length
            labels = labels + [-100] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


def create_dataloaders(
    train_file: str = config.TRAIN_FILE,
    test_file: str = config.TEST_FILE,
    tokenizer: SPTokenizer = None,
    batch_size: int = config.BATCH_SIZE,
    max_length: int = config.MAX_SEQ_LENGTH,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and testing dataloaders.
    
    Args:
        train_file: Path to the training JSONL file.
        test_file: Path to the testing JSONL file.
        tokenizer: Tokenizer instance.
        batch_size: Batch size.
        max_length: Maximum sequence length.
        num_workers: Number of workers for data loading.
        
    Returns:
        Tuple of (train_dataloader, test_dataloader).
    """
    # Initialize tokenizer if not provided
    if tokenizer is None:
        tokenizer = SPTokenizer()
        tokenizer_model_path = f"{tokenizer.model_prefix}.model"
        
        if os.path.exists(tokenizer_model_path):
            print(f"Loading existing tokenizer from {tokenizer_model_path}")
            tokenizer.load()
        else:
            print("Training new tokenizer...")
            tokenizer.train()
            print(f"Tokenizer trained and saved to {tokenizer_model_path}")
    
    # Create datasets
    train_dataset = TextGenerationDataset(train_file, tokenizer, max_length)
    test_dataset = TextGenerationDataset(test_file, tokenizer, max_length)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_dataloader, test_dataloader