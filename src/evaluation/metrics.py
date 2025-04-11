"""
Evaluation metrics for language models.
"""
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from typing import Dict, List, Union
import logging
import nltk
import math
from torch.utils.data import DataLoader
import json

from src.models.base_model import BaseModel
from config import DEVICE

logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=False)

# Make sure it's actually downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    raise RuntimeError("Failed to download NLTK punkt tokenizer. Please run 'python -m nltk.downloader punkt' manually.")

def simple_tokenize(text):
    """
    A simple tokenizer that splits on whitespace and punctuation.
    Fallback in case NLTK isn't available.
    
    Args:
        text: The text to tokenize
        
    Returns:
        List of tokens
    """
    # Replace punctuation with spaces and split
    for char in ',.!?;:()[]{}"\'-':
        text = text.replace(char, f' {char} ')
    return [token for token in text.split() if token]

def modified_bleu_score(reference, candidate, max_n=4, weights=None):
    """
    Calculate a simplified BLEU score based on n-gram precision.
    
    Args:
        reference: List of tokens from the reference text
        candidate: List of tokens from the candidate (generated) text
        max_n: Maximum n-gram size to consider
        weights: Weights for each n-gram precision (default: equal weights)
        
    Returns:
        Modified BLEU score
    """
    if not candidate:
        return 0.0
    
    # Default to equal weights if not provided
    if weights is None:
        weights = [1.0/max_n] * max_n
    
    # Calculate brevity penalty
    bp = min(1.0, np.exp(1 - len(reference)/max(len(candidate), 1)))
    
    precisions = []
    for n in range(1, max_n + 1):
        # Create n-grams
        ref_ngrams = Counter()
        cand_ngrams = Counter()
        
        # Handle the case where n is greater than the length of the sequence
        if len(reference) >= n and len(candidate) >= n:
            # Create n-grams for reference
            for i in range(len(reference) - n + 1):
                ngram = tuple(reference[i:i+n])
                ref_ngrams[ngram] += 1
            
            # Create n-grams for candidate
            for i in range(len(candidate) - n + 1):
                ngram = tuple(candidate[i:i+n])
                cand_ngrams[ngram] += 1
            
            # Count matches (clipped by reference count)
            matches = sum(min(cand_ngrams[ngram], ref_ngrams[ngram]) for ngram in cand_ngrams)
            total = max(len(candidate) - n + 1, 1)  # Total number of n-grams in candidate
            
            # Calculate precision for this n-gram size
            precisions.append(matches / total if total > 0 else 0.0001)  # Small value to avoid log(0)
        else:
            precisions.append(0.0001)  # Small value for n-grams larger than sequence
    
    # Calculate weighted geometric mean of precisions
    log_precisions = [weight * np.log(precision) for weight, precision in zip(weights, precisions)]
    weighted_precision = np.exp(sum(log_precisions))
    
    # Final BLEU score
    bleu = bp * weighted_precision
    
    return bleu

def calculate_perplexity(model: BaseModel, 
                         dataloader: DataLoader) -> float:
    """
    Calculate perplexity on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        
    Returns:
        Perplexity score
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Get batch data
            input_ids = batch['input_ids'].to(DEVICE)
            target_ids = batch['target_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            
            # Forward pass
            logits, _ = model(input_ids, attention_mask)
            
            # Reshape for cross-entropy
            logits = logits.reshape(-1, logits.size(-1))
            targets = target_ids.reshape(-1)
            
            # Calculate loss
            loss = criterion(logits, targets)
            
            # Track loss
            total_loss += loss.item()
            total_tokens += (targets != 0).sum().item()  # Exclude padding tokens
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)
    
    return perplexity

def calculate_bleu_score(model: BaseModel, 
                         dataloader: DataLoader,
                         max_samples: int = 100) -> float:
    """
    Calculate BLEU score on a dataset using the modified BLEU implementation.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        BLEU score
    """
    model.eval()
    bleu_scores = []
    samples_processed = 0
    
    # Choose tokenization function - try NLTK first, fall back to simple tokenizer
    tokenize_func = None
    try:
        # Test if NLTK tokenizer works
        nltk.word_tokenize("test sentence")
        tokenize_func = nltk.word_tokenize
        logger.info("Using NLTK word tokenizer for BLEU score calculation")
    except (LookupError, ImportError):
        tokenize_func = simple_tokenize
        logger.warning("NLTK tokenizer not available, using fallback simple tokenizer")
    
    for batch in dataloader:
        if samples_processed >= max_samples:
            break
        
        prompts = batch['prompt']
        completions = batch['completion']
        
        for prompt, reference in zip(prompts, completions):
            if samples_processed >= max_samples:
                break
            
            # Generate text
            generated = model.generate(prompt, max_length=50, temperature=0.0)  # Use greedy decoding
            
            # Tokenize reference and candidate
            reference_tokens = tokenize_func(reference)
            candidate_tokens = tokenize_func(generated)
            
            # Calculate modified BLEU score
            score = modified_bleu_score(
                reference=reference_tokens,
                candidate=candidate_tokens,
                max_n=1,  # Consider up to 4-grams
                weights=[0.25, 0.25, 0.25, 0.25]  # Equal weights for 1-gram to 4-gram
            )
            
            bleu_scores.append(score)
            samples_processed += 1
    
    # Calculate average BLEU score
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    
    return avg_bleu

def evaluate_model(model: BaseModel, 
                   test_dataloader: DataLoader,
                   prompt_examples: List[str] = None) -> Dict[str, Union[float, List[str]]]:
    """
    Comprehensive evaluation of a model.
    
    Args:
        model: The model to evaluate
        test_dataloader: DataLoader for test data
        prompt_examples: Optional list of prompts to generate text for
        
    Returns:
        Dictionary of evaluation metrics and generated examples
    """
    logger.info(f"Evaluating model: {model.__class__.__name__}")
    
    # Calculate perplexity
    perplexity = calculate_perplexity(model, test_dataloader)
    logger.info(f"Perplexity: {perplexity:.2f}")
    
    # Calculate BLEU score using our modified implementation
    bleu = calculate_bleu_score(model, test_dataloader)
    logger.info(f"Modified BLEU score: {bleu:.4f}")
    
    # Generate examples
    generated_examples = []
    if prompt_examples:
        for prompt in prompt_examples:
            generated_text = model.generate(prompt, temperature=0.8)
            generated_examples.append({
                'prompt': prompt,
                'generated': generated_text
            })
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Generated: {generated_text}")
    
    # Return all metrics
    return {
        'perplexity': perplexity,
        'bleu_score': bleu,
        'generated_examples': generated_examples
    }

def save_metrics(metrics_dict: Dict[str, Dict], 
                 output_path: str):
    """
    Save evaluation metrics to a JSON file.
    
    Args:
        metrics_dict: Dictionary of metrics for each model
        output_path: Path to save the metrics
    """
    # Ensure all values are JSON serializable
    for model_name, metrics in metrics_dict.items():
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_dict[model_name][key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                metrics_dict[model_name][key] = value.cpu().tolist()
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    logger.info(f"Metrics saved to {output_path}")