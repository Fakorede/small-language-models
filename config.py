"""
Configuration file for the text generation project.
Contains all hyperparameters and settings.
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
TOKENIZER_DIR = os.path.join(BASE_DIR, "models", "tokenizer")

# Data Configuration
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
TEST_FILE = os.path.join(DATA_DIR, "test.jsonl")
VOCAB_SIZE = 10000
MAX_SEQ_LENGTH = 512

# Common Model Configuration
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.2

# RNN Configuration
RNN_MODEL_PATH = os.path.join(MODELS_DIR, "rnn_model.pt")

# LSTM Configuration
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model.pt")

# Transformer Configuration
TRANSFORMER_MODEL_PATH = os.path.join(MODELS_DIR, "transformer_model.pt")
TRANSFORMER_NHEAD = 4  # Number of attention heads
TRANSFORMER_DIM_FEEDFORWARD = 1024  # Hidden dimension of the feedforward network

# Training Configuration
BATCH_SIZE = 128
NUM_EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
CLIP_GRAD_NORM = 1.0
PATIENCE = 5  # Early stopping patience
LR_PATIENCE = 2  # Learning rate scheduler patience
LR_FACTOR = 0.5  # Learning rate scheduler factor
SEED = 42

# Generation Configuration
TEMPERATURE = 1.0  # 1.0 means no temperature (just argmax)
MAX_GEN_LENGTH = 100  # Maximum tokens to generate

# Create necessary directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TOKENIZER_DIR, exist_ok=True)