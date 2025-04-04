"""
Main script for text generation model training and evaluation.
"""
import os
import torch
import random
import numpy as np
import argparse
from typing import Dict, Any, List, Tuple

# Import project modules
import config
from src.data.tokenizer import load_or_train_tokenizer
from src.data.dataset import create_dataloaders, read_jsonl_file
from src.models.rnn_model import RNNModel
from src.training.trainer import ModelTrainer


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    """Ensure all required directories exist."""
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.TOKENIZER_MODEL_PREFIX), exist_ok=True)


def train_models() -> Tuple[Dict[str, Any], Dict[str, Tuple[List[float], List[float]]]]:
    """
    Train all models.

    Returns:
        Tuple of (models_dict, losses_dict)
    """
    # Load and prepare data
    print("\nLoading and preparing data...")

    # Train or load tokenizer
    tokenizer = load_or_train_tokenizer(
        config.RAW_DIR,
        config.VOCAB_SIZE,
        config.TOKENIZER_MODEL_PREFIX
    )

    # Read training and testing data
    train_data = read_jsonl_file(config.TRAIN_FILE)
    test_data = read_jsonl_file(config.TEST_FILE)

    # Create dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        train_data,
        test_data,
        tokenizer,
        config.BATCH_SIZE,
        config.DEVICE,
        config.TRAIN_VAL_SPLIT
    )

    # Initialize models
    print("\nInitializing models...")

    rnn_model = RNNModel(
        config.VOCAB_SIZE,
        config.EMBEDDING_DIM,
        config.HIDDEN_DIM,
        num_layers=config.RNN_LAYERS,
        dropout=config.DROPOUT
    ).to(config.DEVICE)


    # Define model save paths
    rnn_save_path = os.path.join(config.MODEL_DIR, "rnn_model.pt")

    # Train RNN model
    print("\nTraining RNN model...")
    rnn_trainer = ModelTrainer(
        rnn_model,
        train_dataloader,
        val_dataloader,
        learning_rate=config.LEARNING_RATE,
        model_save_path=rnn_save_path,
        device=config.DEVICE
    )
    rnn_train_losses, rnn_val_losses = rnn_trainer.train(
        config.NUM_EPOCHS,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE
    )

    # Create models and losses dictionaries
    models = {
        'RNN': rnn_model,
    }

    losses = {
        'RNN': (rnn_train_losses, rnn_val_losses),
    }

    return models, losses #, test_dataloader, tokenizer


def load_trained_models() -> Tuple[Dict[str, Any], torch.utils.data.DataLoader, Any]:
    """
    Load pre-trained models.

    Returns:
        Tuple of (models_dict, test_dataloader, tokenizer)
    """
    # Load tokenizer
    tokenizer = load_or_train_tokenizer(
        config.RAW_DIR,
        config.VOCAB_SIZE,
        config.TOKENIZER_MODEL_PREFIX
    )

    # Read testing data
    test_data = read_jsonl_file(config.TEST_FILE)

    # Create test dataloader (with dummy train/val data)
    _, _, test_dataloader = create_dataloaders(
        test_data,  # We're not using train data here, but the function requires it
        test_data,
        tokenizer,
        config.BATCH_SIZE,
        config.DEVICE,
        config.TRAIN_VAL_SPLIT
    )

    # Initialize models
    rnn_model = RNNModel(
        config.VOCAB_SIZE,
        config.EMBEDDING_DIM,
        config.HIDDEN_DIM,
        num_layers=config.RNN_LAYERS,
        dropout=config.DROPOUT
    ).to(config.DEVICE)

    # Define model paths
    rnn_save_path = os.path.join(config.MODEL_DIR, "rnn_model.pt")

    # Load trained weights if available
    if os.path.exists(rnn_save_path):
        rnn_model.load_state_dict(torch.load(rnn_save_path, map_location=config.DEVICE))
        print(f"Loaded RNN model from {rnn_save_path}")
    else:
        print(f"Warning: RNN model file not found at {rnn_save_path}")


    # Set models to evaluation mode
    rnn_model.eval()

    # Create models dictionary
    models = {
        'RNN': rnn_model,
    }

    return models, test_dataloader, tokenizer  # !/usr/bin/env python


def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Text Generation with RNN, LSTM, and Transformer')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models')
    parser.add_argument('--generate', action='store_true', help='Generate text')
    parser.add_argument('--prompt', type=str, help='Text prompt for generation')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum length to generate')
    parser.add_argument('--seed', type=int, default=config.RANDOM_SEED, help='Random seed')
    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Ensure required directories exist
    ensure_dirs()

    # Print device information
    print(f"Using device: {config.DEVICE}")

    # Train or load models
    if args.train:
        models, losses = train_models()

        # Plot loss curves
        print("\nPlotting loss curves...")

        # Evaluate models if requested
        if args.evaluate:
            print("\nEvaluating models...")

    # Only evaluate pre-trained models
    elif args.evaluate:
        models, test_dataloader, tokenizer = load_trained_models()
        print("\nEvaluating models...")

    # Generate text from a prompt
    elif args.generate:
        if not args.prompt:
            print("Error: Please provide a prompt with --prompt")
            return

        models, _, tokenizer = load_trained_models()

        print(f"\nGenerating text with prompt: '{args.prompt}'")
        print(f"Temperature: {args.temperature}, Max Length: {args.max_length}")

        for name, model in models.items():
            generated = model.prompt(
                tokenizer,
                args.prompt,
                max_seq_length=args.max_length,
                temperature=args.temperature
            )
            print(f"\n{name} Model:")
            print(f"'{generated}'")

    # If no action specified, print help
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
