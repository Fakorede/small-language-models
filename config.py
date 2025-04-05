"""
Configuration settings for the text generation project.
"""
import os
import torch

# Paths
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
TEST_FILE = os.path.join(DATA_DIR, "test.jsonl")
MODEL_DIR = "models"
PLOT_DIR = "plots"
TOKENIZER_MODEL_PREFIX = os.path.join(DATA_DIR, "tokenizer")

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing
VOCAB_SIZE = 10000
MAX_SEQ_LENGTH = 512
TRAIN_VAL_SPLIT = 0.2  # 10% of training data used for validation

# Model parameters
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
RNN_LAYERS = 3
LSTM_LAYERS = 2
TRANSFORMER_LAYERS = 2
TRANSFORMER_HEADS = 4
DROPOUT = 0.2

# Training parameters
BATCH_SIZE = 128
NUM_EPOCHS = 30
LEARNING_RATE = 0.01
EARLY_STOPPING_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 1

# Generation parameters
DEFAULT_MAX_GEN_LENGTH = 100
DEFAULT_TEMPERATURE = 0.8

# Random seeds for reproducibility
RANDOM_SEED = 42

MODEL_TYPE = "transformer"  # Options: "rnn", "lstm", "transformer", "all"
