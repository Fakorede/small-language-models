"""
Main entry point for the text generation project.
"""

import os
import sys
import argparse
import torch
import numpy as np
import random
from typing import Dict, List, Optional, Any, Union

import config
from src.data.tokenizer import SPTokenizer
from src.data.dataset import create_dataloaders
from src.models.rnn_model import RNNModel
from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_model, generate_responses
from src.visualization.loss_plots import plot_metric_comparison, plot_all_metrics_comparison


def set_seed(seed: int = config.SEED):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model(model_type: str, vocab_size: int) -> torch.nn.Module:
    """
    Get the appropriate model based on the model type.
    
    Args:
        model_type: Type of model ('rnn', 'lstm', or 'transformer').
        vocab_size: Size of the vocabulary.
        
    Returns:
        The instantiated model.
    """
    if model_type == 'rnn':
        model = RNNModel(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT
        )
    elif model_type == 'lstm':
        model = LSTMModel(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT
        )
    elif model_type == 'transformer':
        model = TransformerModel(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            nhead=config.TRANSFORMER_NHEAD,
            num_layers=config.NUM_LAYERS,
            dim_feedforward=config.TRANSFORMER_DIM_FEEDFORWARD,
            dropout=config.DROPOUT
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def load_or_train_model(
    model_type: str,
    tokenizer: SPTokenizer,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    train: bool = False,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> torch.nn.Module:
    """
    Load a pre-trained model or train a new one.
    
    Args:
        model_type: Type of model ('rnn', 'lstm', or 'transformer').
        tokenizer: Tokenizer instance.
        train_dataloader: DataLoader for training.
        val_dataloader: DataLoader for validation.
        train: Whether to train the model even if a pre-trained one exists.
        device: Device to train on.
        
    Returns:
        The loaded or trained model.
    """
    # Get model paths
    if model_type == 'rnn':
        model_path = config.RNN_MODEL_PATH
        model_class = RNNModel
    elif model_type == 'lstm':
        model_path = config.LSTM_MODEL_PATH
        model_class = LSTMModel
    elif model_type == 'transformer':
        model_path = config.TRANSFORMER_MODEL_PATH
        model_class = TransformerModel
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Check if model exists and load it
    if os.path.exists(model_path) and not train:
        print(f"Loading pre-trained {model_type} model from {model_path}")
        model = model_class.load(model_path, device=device)
        model.set_tokenizer(tokenizer)
        return model
    
    # Create new model
    model = get_model(model_type, tokenizer.vocab_size)
    model.set_tokenizer(tokenizer)
    
    # Train the model
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        model_type=model_type
    )
    
    trainer.train(num_epochs=config.NUM_EPOCHS)
    
    return model


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Text Generation with RNN, LSTM, and Transformer models")
    
    parser.add_argument("--train", action="store_true", help="Train the models")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the models")
    parser.add_argument("--generate", action="store_true", help="Generate text from the models")
    parser.add_argument("--model_type", type=int, default=0, 
                        help="Model type: 0=all, 1=RNN, 2=LSTM, 3=Transformer")
    parser.add_argument("--prompt", type=str, default="Which do you prefer? Dogs or cats?",
                        help="Prompt for text generation")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for text generation sampling")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum length for generated text")
    
    args = parser.parse_args()
    
    # If no action is specified, print help
    if not (args.train or args.evaluate or args.generate):
        parser.print_help()
        return
    
    # Set random seed for reproducibility
    set_seed(config.SEED)
    
    # Map model_type to model names
    model_types = {
        0: ['rnn', 'lstm', 'transformer'],
        1: ['rnn'],
        2: ['lstm'],
        3: ['transformer']
    }
    model_types = model_types.get(args.model_type, ['rnn', 'lstm', 'transformer'])
    
    # Initialize tokenizer
    tokenizer = SPTokenizer()
    tokenizer_model_path = f"{tokenizer.model_prefix}.model"
    
    if os.path.exists(tokenizer_model_path):
        print(f"Loading existing tokenizer from {tokenizer_model_path}")
        tokenizer.load()
    else:
        print("Training new tokenizer...")
        tokenizer.train()
        print(f"Tokenizer trained and saved to {tokenizer_model_path}")
    
    # Create dataloaders
    train_dataloader, test_dataloader = create_dataloaders(
        tokenizer=tokenizer,
        batch_size=config.BATCH_SIZE
    )
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Process each model type
    models = {}
    for model_type in model_types:
        print(f"\nProcessing {model_type.upper()} model")
        
        # Load or train model
        model = load_or_train_model(
            model_type=model_type,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            val_dataloader=test_dataloader,
            train=args.train,
            device=device
        )
        
        models[model_type] = model
    
    # Evaluate models
    if args.evaluate:
        metrics = {}
        
        for model_type, model in models.items():
            print(f"\nEvaluating {model_type.upper()} model")
            metrics[model_type] = evaluate_model(
                model=model,
                test_dataloader=test_dataloader,
                tokenizer=tokenizer,
                device=device
            )
            print(f"Metrics for {model_type.upper()} model: {metrics[model_type]}")
        
        # Plot perplexity comparison
        perplexity_plot_path = os.path.join(config.PLOTS_DIR, "perplexity_comparison.png")
        plot_metric_comparison(
            metrics=metrics,
            metric_name="perplexity",
            title="Perplexity Comparison",
            save_path=perplexity_plot_path,
            is_lower_better=True
        )
        
        # Plot BLEU score comparison
        bleu_plot_path = os.path.join(config.PLOTS_DIR, "bleu_comparison.png")
        plot_metric_comparison(
            metrics=metrics,
            metric_name="bleu_score",
            title="BLEU Score Comparison",
            save_path=bleu_plot_path,
            is_lower_better=False
        )
        
        # Plot normalized metrics comparison
        metrics_plot_path = os.path.join(config.PLOTS_DIR, "all_metrics_comparison.png")
        plot_all_metrics_comparison(
            metrics=metrics,
            title="All Metrics Comparison (Normalized)",
            save_path=metrics_plot_path
        )
        
    # Generate text
    if args.generate:
        # Use standard prompt if not specified
        prompts = [
            args.prompt,
            "Which do you prefer? Dogs or cats?"
        ]
        
        for model_type, model in models.items():
            print(f"\nGenerating text with {model_type.upper()} model")
            responses = generate_responses(
                model=model,
                prompts=prompts,
                tokenizer=tokenizer,
                max_length=args.max_length,
                temperature=args.temperature,
                device=device
            )
            
            for i, (prompt, response) in enumerate(zip(prompts, responses)):
                print(f"\nPrompt {i+1}: {prompt}")
                print(f"Response: {response}")

if __name__ == "__main__":
    main()