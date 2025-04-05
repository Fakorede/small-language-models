"""
Evaluation metrics for the text generation models.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional, Any
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


def compute_perplexity(model: nn.Module, 
                       dataloader: torch.utils.data.DataLoader, 
                       device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> float:
    """
    Compute perplexity on a dataset.
    
    Args:
        model: Model to evaluate.
        dataloader: DataLoader for evaluation.
        device: Device to run the evaluation on.
        
    Returns:
        Perplexity score (lower is better).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs, _ = model(input_ids, attention_mask)
            
            # Reshape for cross entropy
            outputs = outputs.view(-1, outputs.shape[-1])
            labels = labels.view(-1)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Count tokens
            non_pad_mask = (labels != -100)
            num_tokens = non_pad_mask.sum().item()
            
            total_loss += loss.item()
            total_tokens += num_tokens
    
    # Compute average loss
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    
    # Compute perplexity
    perplexity = math.exp(avg_loss)
    
    return perplexity


def compute_bleu(model: nn.Module, 
                dataloader: torch.utils.data.DataLoader,
                tokenizer,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                max_samples: int = 100) -> float:
    """
    Compute BLEU score on a dataset.
    
    Args:
        model: Model to evaluate.
        dataloader: DataLoader for evaluation.
        tokenizer: Tokenizer for decoding/encoding.
        device: Device to run the evaluation on.
        max_samples: Maximum number of samples to evaluate.
        
    Returns:
        BLEU score (higher is better).
    """
    model.eval()
    references = []
    hypotheses = []
    
    # Set the tokenizer for the model
    model.set_tokenizer(tokenizer)
    
    with torch.no_grad():
        sample_count = 0
        for batch in dataloader:
            if sample_count >= max_samples:
                break
                
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"]
            
            # Loop through batch
            for i in range(min(len(input_ids), max_samples - sample_count)):
                # Extract prompt (until first non -100 label)
                prompt_ids = []
                for j, label in enumerate(labels[i]):
                    if label.item() == -100:
                        prompt_ids.append(input_ids[i][j].item())
                    else:
                        break
                        
                # Skip if no prompt found
                if not prompt_ids:
                    continue
                    
                # Get actual prompt text
                prompt_text = tokenizer.decode(prompt_ids)
                
                # Generate text from the model
                generated_text = model.prompt(prompt_text, device=device)
                
                # Extract reference text (only from the non -100 labels)
                reference_ids = [label.item() for label in labels[i] if label.item() != -100]
                reference_text = tokenizer.decode(reference_ids)
                
                # Tokenize for BLEU
                reference_tokens = nltk.word_tokenize(reference_text)
                hypothesis_tokens = nltk.word_tokenize(generated_text)
                
                # Add to lists
                references.append([reference_tokens])
                hypotheses.append(hypothesis_tokens)
                
                sample_count += 1
    
    # Compute BLEU score
    smoothie = SmoothingFunction().method1
    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
    
    return bleu_score


def evaluate_model(model: nn.Module,
                  test_dataloader: torch.utils.data.DataLoader,
                  tokenizer,
                  device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, float]:
    """
    Evaluate a model using multiple metrics.
    
    Args:
        model: Model to evaluate.
        test_dataloader: DataLoader for testing.
        tokenizer: Tokenizer for decoding/encoding.
        device: Device to run the evaluation on.
        
    Returns:
        Dictionary of metrics.
    """
    # Compute perplexity
    perplexity = compute_perplexity(model, test_dataloader, device)
    
    # Compute BLEU score
    bleu_score = compute_bleu(model, test_dataloader, tokenizer, device)
    
    # Return metrics
    return {
        "perplexity": perplexity,
        "bleu_score": bleu_score
    }


def generate_responses(model: nn.Module,
                       prompts: List[str],
                       tokenizer,
                       max_length: int = 100,
                       temperature: float = 1.0,
                       device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> List[str]:
    """
    Generate responses for a list of prompts.
    
    Args:
        model: Model to generate responses with.
        prompts: List of prompt texts.
        tokenizer: Tokenizer for decoding/encoding.
        max_length: Maximum length of generated sequences.
        temperature: Temperature for sampling.
        device: Device to run the model on.
        
    Returns:
        List of generated responses.
    """
    model.eval()
    model.set_tokenizer(tokenizer)
    
    responses = []
    
    for prompt in prompts:
        response = model.prompt(prompt, max_length=max_length, temperature=temperature, device=device)
        responses.append(response)
    
    return responses