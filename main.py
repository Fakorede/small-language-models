#!/usr/bin/env python
"""
Main entry point script for training, evaluating and generating text from language models.
"""
import argparse
import logging
import nltk
import os
import sys
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Import project modules
from config import (
    DATA_DIR, MODELS_DIR, PLOTS_DIR, TRAIN_FILE, TEST_FILE,
    EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, VOCAB_SIZE,
    ALL_MODELS, RNN_MODEL, LSTM_MODEL, TRANSFORMER_MODEL,
    MODEL_NAMES, DEVICE, MAX_GENERATION_LENGTH, TEMPERATURE, TOKENIZER_MODEL_PREFIX
)
from src.data.tokenizer import Tokenizer
from src.data.dataset import create_dataloaders
from src.models.rnn_model import RNNModel
from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_model, save_metrics
from src.visualization.loss_plots import plot_training_curves, plot_all_metrics

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Language Model Training and Evaluation")
    
    # Operation modes
    parser.add_argument("--train", action="store_true", help="Train the model(s)")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model(s)")
    parser.add_argument("--generate", action="store_true", help="Generate text from a model")
    
    # Model selection
    parser.add_argument("--model_type", type=int, default=0, choices=[0, 1, 2, 3],
                       help="Model type: 0=All, 1=RNN, 2=LSTM, 3=Transformer")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, default="",
                       help="Prompt for text generation")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                       help="Temperature for sampling")
    parser.add_argument("--max_length", type=int, default=MAX_GENERATION_LENGTH,
                       help="Maximum length of generated text")
    
    return parser.parse_args()

def create_model(model_type: int, tokenizer: Tokenizer) -> torch.nn.Module:
    """
    Create a model of the specified type.
    
    Args:
        model_type: Type of model to create
        tokenizer: Tokenizer instance
        
    Returns:
        The created model
    """
    vocab_size = tokenizer.get_vocab_size()
    
    if model_type == RNN_MODEL:
        logger.info("Creating RNN model")
        return RNNModel(
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            tokenizer=tokenizer
        )
    elif model_type == LSTM_MODEL:
        logger.info("Creating LSTM model")
        return LSTMModel(
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            tokenizer=tokenizer
        )
    elif model_type == TRANSFORMER_MODEL:
        logger.info("Creating Transformer model")
        return TransformerModel(
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            tokenizer=tokenizer
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")

def train_models(model_types: List[int], tokenizer: Tokenizer):
    """
    Train the specified models.
    
    Args:
        model_types: List of model types to train
        tokenizer: Tokenizer instance
    """
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        train_file=TRAIN_FILE,
        test_file=TEST_FILE,
        tokenizer=tokenizer
    )
    
    # Train each model
    for model_type in model_types:
        model = create_model(model_type, tokenizer)
        model_name = MODEL_NAMES[model_type]
        
        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            model_name=model_name
        )
        
        # Train model
        metrics = trainer.train()
        
        # Plot training curves
        plot_training_curves(
            train_losses=metrics['train_loss'],
            val_losses=metrics['val_loss'],
            model_name=model_name
        )
        
        logger.info(f"Finished training {model_name}")

def evaluate_models(model_types: List[int], tokenizer: Tokenizer):
    """
    Evaluate the specified models.
    
    Args:
        model_types: List of model types to evaluate
        tokenizer: Tokenizer instance
    """
    # Create dataloaders
    _, test_dataloader = create_dataloaders(
        train_file=TRAIN_FILE,
        test_file=TEST_FILE,
        tokenizer=tokenizer
    )
    
    # Example prompts for generation
    prompts = [
        "Which do you prefer? Dogs or cats?",
        "Once upon a time in a land far away",
        "The quick brown fox jumps over"
    ]
    
    # Evaluate each model
    all_metrics = {}
    
    for model_type in model_types:
        model_name = MODEL_NAMES[model_type]
        model_path = MODELS_DIR / f"{model_name}_best.pt"
        
        if not model_path.exists():
            logger.warning(f"Model {model_name} not found at {model_path}, skipping evaluation")
            continue
        
        # Load model
        if model_type == RNN_MODEL:
            model = RNNModel.load(model_path, tokenizer)
        elif model_type == LSTM_MODEL:
            model = LSTMModel.load(model_path, tokenizer)
        elif model_type == TRANSFORMER_MODEL:
            model = TransformerModel.load(model_path, tokenizer)
        
        # Evaluate model
        metrics = evaluate_model(model, test_dataloader, prompts)
        all_metrics[model_name] = metrics
        
        logger.info(f"Finished evaluating {model_name}")
    
    # Save metrics
    if all_metrics:
        metrics_path = MODELS_DIR / "evaluation_metrics.json"
        save_metrics(all_metrics, metrics_path)
        
        # Generate comparison plots
        plot_all_metrics(metrics_path)
    else:
        logger.warning("No models were evaluated")

def generate_text(model_type: int, prompt: str, temperature: float, max_length: int, tokenizer: Tokenizer):
    """
    Generate text from the specified model.
    
    Args:
        model_type: Type of model to use
        prompt: Prompt for text generation
        temperature: Temperature for sampling
        max_length: Maximum length of generated text
        tokenizer: Tokenizer instance
    """
    model_name = MODEL_NAMES[model_type]
    model_path = MODELS_DIR / f"{model_name}_best.pt"
    
    if not model_path.exists():
        logger.error(f"Model {model_name} not found at {model_path}")
        return
    
    # Load model
    if model_type == RNN_MODEL:
        model = RNNModel.load(model_path, tokenizer)
    elif model_type == LSTM_MODEL:
        model = LSTMModel.load(model_path, tokenizer)
    elif model_type == TRANSFORMER_MODEL:
        model = TransformerModel.load(model_path, tokenizer)
    
    # Generate text
    generated_text = model.generate(prompt, max_length, temperature)
    
    print("\nGenerated Text:")
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Check that at least one operation is specified
    if not (args.train or args.evaluate or args.generate):
        logger.error("Please specify at least one operation: --train, --evaluate, or --generate")
        sys.exit(1)
    
    # Ensure directories exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = Tokenizer(TOKENIZER_MODEL_PREFIX)
    
    # Check if tokenizer needs to be trained
    if not tokenizer.model_path.exists():
        logger.info("Training tokenizer")
        tokenizer.train()
    
    # Determine which models to process
    if args.model_type == ALL_MODELS:
        model_types = [RNN_MODEL, LSTM_MODEL, TRANSFORMER_MODEL]
    else:
        model_types = [args.model_type]
    
    # Perform operations
    if args.train:
        train_models(model_types, tokenizer)
    
    if args.evaluate:
        try:
            nltk.download('punkt', quiet=False)
        except Exception as e:
            logger.warning(f"Warning: Failed to download NLTK data: {e}")
            logger.warning("BLEU score calculation may use fallback tokenizer")

        evaluate_models(model_types, tokenizer)
    
    if args.generate:
        if not args.prompt:
            logger.error("Please specify a prompt with --prompt")
            sys.exit(1)
        
        if args.model_type == ALL_MODELS:
            logger.error("Please specify a single model type for text generation")
            sys.exit(1)
        
        generate_text(
            model_type=args.model_type,
            prompt=args.prompt,
            temperature=args.temperature,
            max_length=args.max_length,
            tokenizer=tokenizer
        )

if __name__ == "__main__":
    main()