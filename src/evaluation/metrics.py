"""
Evaluation metrics for text generation models.
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu
from typing import List, Dict, Tuple, Any

from src.models.base_model import BaseTextGenerationModel


def calculate_perplexity(
    model: BaseTextGenerationModel, 
    dataloader: DataLoader, 
    tokenizer: Any
) -> float:
    """
    Calculate perplexity on the dataset.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader with test data
        tokenizer: SentencePiece tokenizer
        
    Returns:
        Perplexity score (lower is better)
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"]
            target_ids = batch["target_ids"]
            
            # Forward pass
            logits, _, _ = model(input_ids)
            
            # Calculate loss
            for i in range(logits.size(0)):
                pred = logits[i, :len(target_ids[i]), :].view(-1, len(tokenizer))
                target = target_ids[i].view(-1)
                loss = criterion(pred, target)
                total_loss += loss.item()
                total_tokens += target.size(0)
    
    # Perplexity = exp(average negative log-likelihood)
    perplexity = np.exp(total_loss / total_tokens)
    return perplexity


def calculate_bleu(
    model: BaseTextGenerationModel, 
    dataloader: DataLoader, 
    tokenizer: Any
) -> float:
    """
    Calculate BLEU score for the model.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader with test data
        tokenizer: SentencePiece tokenizer
        
    Returns:
        BLEU score (higher is better)
    """
    model.eval()
    references = []
    hypotheses = []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"]
            target_ids = batch["target_ids"]
            
            for i in range(input_ids.size(0)):
                # Get the prompt
                prompt = tokenizer.decode(input_ids[i].tolist())
                
                # Get the reference (ground truth)
                reference = [tokenizer.decode(target_ids[i].tolist())]
                references.append([reference])
                
                # Generate text with the model
                generated = model.prompt(tokenizer, prompt)
                hypotheses.append(generated)
    
    # Calculate BLEU score
    bleu_score = corpus_bleu(references, hypotheses)
    return bleu_score


def generate_text_samples(
    model: BaseTextGenerationModel, 
    tokenizer: Any, 
    prompts: List[str], 
    max_length: int = 100, 
    temperature: float = 1.0
) -> Dict[str, str]:
    """
    Generate text samples from a list of prompts.
    
    Args:
        model: Model to generate text
        tokenizer: SentencePiece tokenizer
        prompts: List of text prompts
        max_length: Maximum length of generated text
        temperature: Temperature for sampling
        
    Returns:
        Dictionary mapping prompts to generated text
    """
    model.eval()
    results = {}
    
    for prompt in prompts:
        generated_text = model.prompt(
            tokenizer, 
            prompt, 
            max_seq_length=max_length, 
            temperature=temperature
        )
        results[prompt] = generated_text
    
    return results


def compare_models(
    models: Dict[str, BaseTextGenerationModel], 
    test_dataloader: DataLoader, 
    tokenizer: Any, 
    prompts: List[str]
) -> Dict[str, Any]:
    """
    Compare multiple models on various metrics.
    
    Args:
        models: Dictionary mapping model names to model instances
        test_dataloader: Data loader with test data
        tokenizer: SentencePiece tokenizer
        prompts: List of prompts for text generation
        
    Returns:
        Dictionary with comparison results
    """
    results = {
        "perplexity": {},
        "bleu": {},
        "generated_text": {}
    }
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # Calculate perplexity
        perplexity = calculate_perplexity(model, test_dataloader, tokenizer)
        results["perplexity"][name] = perplexity
        print(f"  Perplexity: {perplexity:.2f}")
        
        # Calculate BLEU score
        bleu = calculate_bleu(model, test_dataloader, tokenizer)
        results["bleu"][name] = bleu
        print(f"  BLEU Score: {bleu:.4f}")
        
        # Generate text samples
        # generated = generate_text_samples(model, tokenizer, prompts)
        # results["generated_text"][name] = generated
        
        # print("  Generated text samples:")
        # for prompt, text in generated.items():
        #     print(f"    Prompt: {prompt}")
        #     print(f"    Generated: {text}")
        #     print()
    
    return results