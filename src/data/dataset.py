import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
import os
from typing import Dict, List, Any, Tuple


class TextGenerationDataset(Dataset):
    """Dataset and dataloader implementations for text generation tasks."""

    def __init__(self, data: List[Dict[str, str]], tokenizer):
        """
        Initialize the dataset.

        Args:
            data: List of dictionaries with 'prompt' and 'completion' keys
            tokenizer: SentencePiece tokenizer
        """
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.

        Args:
            idx: Index of the item

        Returns:
            Dictionary with 'input_ids', 'target_ids', and 'input_len' keys
        """
        item = self.data[idx]
        prompt = item["prompt"]
        completion = item["completion"]

        # Tokenize the prompt and completion
        prompt_ids = self.tokenizer.encode(prompt, out_type=int)
        completion_ids = self.tokenizer.encode(completion, out_type=int)

        # Convert to tensors
        input_ids = torch.tensor(prompt_ids, dtype=torch.long)
        target_ids = torch.tensor(completion_ids, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "input_len": len(input_ids)
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for padding sequences.

    Args:
        batch: List of dictionaries returned by __getitem__
        device: Device to move tensors to

    Returns:
        Dictionary with padded sequences
    """
    input_ids = [item["input_ids"] for item in batch]
    target_ids = [item["target_ids"] for item in batch]
    input_lens = [item["input_len"] for item in batch]

    # Pad sequences
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    target_ids = pad_sequence(target_ids, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids.to(device),
        "target_ids": target_ids.to(device),
        "input_lens": torch.tensor(input_lens, dtype=torch.long).to(device)
    }


def read_jsonl_file(file_path: str) -> List[Dict[str, str]]:
    """
    Read data from JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of dictionaries with 'prompt' and 'completion' keys
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def create_dataloaders(
        train_data: List[Dict[str, str]],
        test_data: List[Dict[str, str]],
        tokenizer,
        batch_size: int,
        device: torch.device,
        val_split: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        train_data: Training data
        test_data: Testing data
        tokenizer: SentencePiece tokenizer
        batch_size: Batch size
        device: Device to move tensors to
        val_split: Fraction of training data to use for validation

    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    import random

    # Create a copy of train_data to avoid modifying the original
    train_data = train_data.copy()

    # Shuffle and split training data
    random.shuffle(train_data)
    val_size = int(val_split * len(train_data))
    val_data = train_data[:val_size]
    train_data = train_data[val_size:]

    # Create datasets
    train_dataset = TextGenerationDataset(train_data, tokenizer)
    val_dataset = TextGenerationDataset(val_data, tokenizer)
    test_dataset = TextGenerationDataset(test_data, tokenizer)

    # Create dataloaders with custom collate function
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, device)
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, device)
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, device)
    )

    return train_dataloader, val_dataloader, test_dataloader