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
from src.models.lstm_model import LSTMModel
from src.models.rnn_model import RNNModel
from src.models.transformer_model import TransformerModel
from src.training.trainer import ModelTrainer
from src.visualization.loss_plots import plot_loss_curves


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


def train_models(model_type=0) -> Tuple[Dict[str, Any], Dict[str, Tuple[List[float], List[float]]]]:
    """
    Train models based on the specified model type.

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

    models = {}
    losses = {}

    # Train models based on the model type
    if model_type == 0 or model_type == 1:
        # Train RNN model
        print("\nTraining RNN model...")
        rnn_model = RNNModel(
            config.VOCAB_SIZE, 
            config.EMBEDDING_DIM, 
            config.HIDDEN_DIM,
            num_layers=config.RNN_LAYERS,
            dropout=config.DROPOUT
        ).to(config.DEVICE)
        
        rnn_save_path = os.path.join(config.MODEL_DIR, "rnn_model.pt")
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
        
        models['RNN'] = rnn_model
        losses['RNN'] = (rnn_train_losses, rnn_val_losses)

    if model_type == 0 or model_type == 2:
        # Train LSTM model
        print("\nTraining LSTM model...")
        lstm_model = LSTMModel(
            config.VOCAB_SIZE, 
            config.EMBEDDING_DIM, 
            config.HIDDEN_DIM,
            num_layers=config.LSTM_LAYERS,
            dropout=config.DROPOUT
        ).to(config.DEVICE)
        
        lstm_save_path = os.path.join(config.MODEL_DIR, "lstm_model.pt")
        lstm_trainer = ModelTrainer(
            lstm_model,
            train_dataloader,
            val_dataloader,
            learning_rate=config.LEARNING_RATE,
            model_save_path=lstm_save_path,
            device=config.DEVICE
        )
        lstm_train_losses, lstm_val_losses = lstm_trainer.train(
            config.NUM_EPOCHS,
            early_stopping_patience=config.EARLY_STOPPING_PATIENCE
        )
        
        models['LSTM'] = lstm_model
        losses['LSTM'] = (lstm_train_losses, lstm_val_losses)

    if model_type == 0 or model_type == 3:
        # Train Transformer model
        print("\nTraining Transformer model...")
        transformer_model = TransformerModel(
            config.VOCAB_SIZE, 
            config.EMBEDDING_DIM, 
            config.HIDDEN_DIM,
            nhead=config.TRANSFORMER_HEADS,
            num_layers=config.TRANSFORMER_LAYERS,
            dropout=config.DROPOUT
        ).to(config.DEVICE)
        
        transformer_save_path = os.path.join(config.MODEL_DIR, "transformer_model.pt")
        transformer_trainer = ModelTrainer(
            transformer_model,
            train_dataloader,
            val_dataloader,
            learning_rate=config.LEARNING_RATE,
            model_save_path=transformer_save_path,
            device=config.DEVICE
        )
        transformer_train_losses, transformer_val_losses = transformer_trainer.train(
            config.NUM_EPOCHS,
            early_stopping_patience=config.EARLY_STOPPING_PATIENCE
        )

        models['Transformer'] = transformer_model
        losses['Transformer'] = (transformer_train_losses, transformer_val_losses)

    return models, losses #, test_dataloader, tokenizer


def load_trained_models(model_type=0) -> Tuple[Dict[str, Any], torch.utils.data.DataLoader, Any]:
    """
    Load pre-trained models based on the specified model type.

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

    # Initialize models dictionary
    models = {}

    # Load RNN model
    if model_type == 0 or model_type == 1:
        rnn_model = RNNModel(
            config.VOCAB_SIZE, 
            config.EMBEDDING_DIM, 
            config.HIDDEN_DIM,
            num_layers=config.RNN_LAYERS,
            dropout=config.DROPOUT
        ).to(config.DEVICE)
        
        rnn_save_path = os.path.join(config.MODEL_DIR, "rnn_model.pt")
        if os.path.exists(rnn_save_path):
            rnn_model.load_state_dict(torch.load(rnn_save_path, map_location=config.DEVICE))
            print(f"Loaded RNN model from {rnn_save_path}")
            models['RNN'] = rnn_model
        else:
            print(f"Warning: RNN model file not found at {rnn_save_path}")    

    # Load LSTM model
    if model_type == 0 or model_type == 2:
        lstm_model = LSTMModel(
            config.VOCAB_SIZE, 
            config.EMBEDDING_DIM, 
            config.HIDDEN_DIM,
            num_layers=config.LSTM_LAYERS,
            dropout=config.DROPOUT
        ).to(config.DEVICE)
        
        lstm_save_path = os.path.join(config.MODEL_DIR, "lstm_model.pt")
        if os.path.exists(lstm_save_path):
            lstm_model.load_state_dict(torch.load(lstm_save_path, map_location=config.DEVICE))
            print(f"Loaded LSTM model from {lstm_save_path}")
            models['LSTM'] = lstm_model
        else:
            print(f"Warning: LSTM model file not found at {lstm_save_path}")
    
    # Load Transformer model
    if model_type == 0 or model_type == 3:
        transformer_model = TransformerModel(
            config.VOCAB_SIZE, 
            config.EMBEDDING_DIM, 
            config.HIDDEN_DIM,
            nhead=config.TRANSFORMER_HEADS,
            num_layers=config.TRANSFORMER_LAYERS,
            dropout=config.DROPOUT
        ).to(config.DEVICE)
        
        transformer_save_path = os.path.join(config.MODEL_DIR, "transformer_model.pt")
        if os.path.exists(transformer_save_path):
            transformer_model.load_state_dict(torch.load(transformer_save_path, map_location=config.DEVICE))
            print(f"Loaded Transformer model from {transformer_save_path}")
            models['Transformer'] = transformer_model
        else:
            print(f"Warning: Transformer model file not found at {transformer_save_path}")

    # Set models to evaluation mode
    for model in models.values():
        model.eval()

    return models, test_dataloader, tokenizer


def test_token_generation(model, tokenizer, device):
    """Test token generation directly."""
    # Create a simple input
    test_text = "the cat sat on the"
    test_ids = tokenizer.encode(test_text, out_type=int)
    input_tensor = torch.tensor([test_ids], dtype=torch.long).to(device)
    
    print(f"\nTesting with input: '{test_text}'")
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Run forward pass
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(input_tensor)
    
    # Get predictions for last token
    last_token_logits = logits[0, -1, :]
    
    # Get top 10 predictions
    probs = torch.softmax(last_token_logits, dim=-1)
    top_k = torch.topk(probs, 10)
    
    print("\nTop 10 next token predictions:")
    for i, (idx, prob) in enumerate(zip(top_k.indices.cpu().numpy(), top_k.values.cpu().numpy())):
        try:
            token_text = tokenizer.IdToPiece(int(idx))
            print(f"{i+1}. Token ID {idx}: '{token_text}' (Prob: {prob:.4f})")
        except Exception as e:
            print(f"{i+1}. Token ID {idx}: [Error decoding token: {e}] (Prob: {prob:.4f})")

    generated = model.prompt(tokenizer, test_text, config.DEFAULT_MAX_GEN_LENGTH, config.DEFAULT_TEMPERATURE)
    print("CHECKED!")


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
    parser.add_argument('--model_type', type=int, default=config.MODEL_TYPE, help='Model type (0=all, 1=rnn, 2=lstm, 3=transformer)')
    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Ensure required directories exist
    ensure_dirs()

    # Print device information
    print(f"Using device: {config.DEVICE}")

    # Validate model_type
    if args.model_type not in [0, 1, 2, 3]:
        print(f"Error: Invalid model_type {args.model_type}. Must be 0 (all), 1 (RNN), 2 (LSTM), or 3 (Transformer).")
        return

    # Train or load models
    if args.train:
        models, losses = train_models(args.model_type) # models, losses, test_dataloader, tokenizer = train_models(args.model_type)

        # Plot loss curves
        print("\nPlotting loss curves...")
        # plot_all_models_loss(losses, config.PLOT_DIR)
        # save_path = os.path.join(config.PLOT_DIR, "transformer_loss.png")
        # transformer_train_losses, transformer_val_losses = losses['Transformer']

        # plot_loss_curves(
        #     transformer_train_losses,
        #     transformer_val_losses,
        #     "Transformer Model Loss",
        #     save_path
        # )

        # Evaluate models if requested
        if args.evaluate:
            print("\nEvaluating models...")
            # evaluate_models(models, test_dataloader, tokenizer)

    # Only evaluate pre-trained models
    elif args.evaluate:
        models, test_dataloader, tokenizer = load_trained_models(args.model_type)
        print("\nEvaluating models...")
        # evaluate_models(models, test_dataloader, tokenizer)

    # Generate text from a prompt
    elif args.generate:
        if not args.prompt:
            print("Error: Please provide a prompt with --prompt")
            return

        models, _, tokenizer = load_trained_models(args.model_type)

        if not models:
            print("Error: No models loaded. Make sure the model files exist.")
            return

        # test_sentences = [
        #     "The quick brown fox jumps over the lazy dog.",
        #     "Hello, world! How are you doing today?",
        #     "Once upon a time in a land far, far away."
        # ]

        # for sentence in test_sentences:
        #     encoded = tokenizer.encode(sentence, out_type=int)
        #     decoded = tokenizer.decode(encoded)
        #     print(f"Original: '{sentence}'")
        #     print(f"Encoded: {encoded}")
        #     print(f"Decoded: '{decoded}'")
        #     print()

        # print("\nTokenizer information:")
        # print(f"Pad ID: {tokenizer.pad_id()}")
        # print(f"Unknown ID: {tokenizer.unk_id()}")
        # print(f"BOS ID: {tokenizer.bos_id()}")
        # print(f"EOS ID: {tokenizer.eos_id()}")
        # print(f"Vocab size: {tokenizer.get_piece_size()}")

        # # Test token generation directly
        # print("\nTesting token generation with RNN model")
        # test_token_generation(models['RNN'], tokenizer, config.DEVICE)

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
