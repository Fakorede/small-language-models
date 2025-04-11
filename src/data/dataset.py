"""
Dataset and dataloader implementations
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

from config import BATCH_SIZE, MAX_SEQ_LENGTH
from src.data.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

class TextCompletionDataset(Dataset):
    """
    Dataset for text completion tasks, loading data from JSONL files.
    """
    def __init__(self, 
                 file_path: Path, 
                 tokenizer: Tokenizer,
                 max_seq_length: int = MAX_SEQ_LENGTH):
        """
        Initialize dataset from a JSONL file with prompt-completion pairs.
        
        Args:
            file_path: Path to the JSONL file
            tokenizer: Tokenizer instance for encoding text
            max_seq_length: Maximum sequence length for truncation
        """
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples = []
        
        self._load_data()
    
    def _load_data(self):
        """Load and process data from the JSONL file."""
        logger.info(f"Loading data from {self.file_path}")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line.strip())
                
                # Combine prompt and completion for training
                prompt = example['prompt']
                completion = example['completion']
                
                # Special handling for examples with BOS/EOS tags
                if prompt.startswith("<bos>"):
                    prompt = prompt[5:]  # Remove <bos> tag
                    # Add BOS token at the beginning
                    prompt_tokens = [self.tokenizer.bos_id] + self.tokenizer.encode(prompt)
                else:
                    # Regular encoding
                    prompt_tokens = self.tokenizer.encode(prompt)
                
                if completion.endswith("<eos>"):
                    completion = completion[:-5]  # Remove <eos> tag
                    # Add EOS token at the end
                    completion_tokens = self.tokenizer.encode(completion) + [self.tokenizer.eos_id]
                else:
                    # Regular encoding
                    completion_tokens = self.tokenizer.encode(completion)
                
                # Combine prompt and completion tokens
                input_tokens = prompt_tokens + completion_tokens
                
                # Truncate if necessary
                if len(input_tokens) > self.max_seq_length:
                    input_tokens = input_tokens[:self.max_seq_length]
                
                # Create input and target sequences for next-token prediction
                # For language modeling, target is input shifted by one position
                x = input_tokens[:-1]
                y = input_tokens[1:]
                
                self.examples.append({
                    'input_ids': x,
                    'target_ids': y,
                    'prompt': prompt,
                    'completion': completion,
                    'length': len(x)
                })
        
        logger.info(f"Loaded {len(self.examples)} examples from {self.file_path}")
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a dataset example by index."""
        return self.examples[idx]


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader to create batches.
    
    Args:
        batch: List of dataset examples
        
    Returns:
        Dictionary of batched tensors
    """
    # Get max length in the batch
    max_len = max([example['length'] for example in batch])
    
    # Initialize tensors
    input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    target_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    
    # Fill in the tensors
    for i, example in enumerate(batch):
        seq_len = example['length']
        input_ids[i, :seq_len] = torch.tensor(example['input_ids'], dtype=torch.long)
        target_ids[i, :seq_len] = torch.tensor(example['target_ids'], dtype=torch.long)
        attention_mask[i, :seq_len] = 1
    
    return {
        'input_ids': input_ids,
        'target_ids': target_ids,
        'attention_mask': attention_mask,
        'prompt': [example['prompt'] for example in batch],
        'completion': [example['completion'] for example in batch]
    }


def create_dataloaders(
    train_file: Path,
    test_file: Path,
    tokenizer: Tokenizer,
    batch_size: int = BATCH_SIZE,
    max_seq_length: int = MAX_SEQ_LENGTH
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and test dataloaders.
    
    Args:
        train_file: Path to the training file
        test_file: Path to the test file
        tokenizer: Tokenizer instance
        batch_size: Batch size
        max_seq_length: Maximum sequence length
        
    Returns:
        Tuple of (train_dataloader, test_dataloader)
    """
    # Create datasets
    train_dataset = TextCompletionDataset(train_file, tokenizer, max_seq_length)
    test_dataset = TextCompletionDataset(test_file, tokenizer, max_seq_length)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_dataloader, test_dataloader