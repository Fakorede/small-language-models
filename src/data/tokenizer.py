"""
Tokenizer implementation for text generation.
"""
import os
import sentencepiece as spm
from typing import List, Optional


def read_text_files(raw_dir: str) -> List[str]:
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


def train_tokenizer(
        text_data: List[str],
        vocab_size: int,
        model_prefix: str
) -> spm.SentencePieceProcessor:
    """
    Train a SentencePiece BPE tokenizer.

    Args:
        text_data: List of text contents
        vocab_size: Size of the vocabulary
        model_prefix: Prefix for tokenizer model files

    Returns:
        Trained SentencePiece tokenizer
    """
    # Write text to a temporary file for SentencePiece
    with open(f"{model_prefix}.txt", 'w', encoding='utf-8') as f:
        for text in text_data:
            f.write(text + '\n')

    # Train the tokenizer
    spm.SentencePieceTrainer.Train(
        input=f"{model_prefix}.txt",
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=1.0,
        normalization_rule_name='nmt_nfkc',
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )

    # Load the trained tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(f"{model_prefix}.model")
    return sp


def load_or_train_tokenizer(
        raw_dir: str,
        vocab_size: int,
        model_prefix: str,
        force_retrain: bool = False
) -> spm.SentencePieceProcessor:
    """
    Load an existing tokenizer if available, or train a new one.

    Args:
        raw_dir: Directory containing raw text files
        vocab_size: Size of the vocabulary
        model_prefix: Prefix for tokenizer model files
        force_retrain: Force retraining even if a model exists

    Returns:
        SentencePiece tokenizer
    """
    model_path = f"{model_prefix}.model"

    # Check if model already exists
    if os.path.exists(model_path) and not force_retrain:
        print(f"Loading existing tokenizer from {model_path}")
        sp = spm.SentencePieceProcessor()
        sp.Load(model_path)
        return sp

    # Train new tokenizer
    print(f"Training new tokenizer with vocab size {vocab_size}")
    text_data = read_text_files(raw_dir)
    return train_tokenizer(text_data, vocab_size, model_prefix)