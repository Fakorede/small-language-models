"""
Configuration parameters for the language modeling project.
"""
import os
from pathlib import Path
import torch

# Project paths
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"

# Ensure directories exist
for dir_path in [MODELS_DIR, PLOTS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Data processing parameters
TOKENIZER_MODEL_PREFIX = os.path.join(DATA_DIR, "tokenizer")
VOCAB_SIZE = 10000
MAX_SEQ_LENGTH = 512
TRAIN_FILE = DATA_DIR / "train.jsonl"
TEST_FILE = DATA_DIR / "test.jsonl"

# Model parameters
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.2
TRANSFORMER_HEADS = 8

# Training parameters
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 3
GRADIENT_CLIP_VAL = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Generation parameters
TEMPERATURE = 0.8
MAX_GENERATION_LENGTH = 100

# Model type constants
ALL_MODELS = 0
RNN_MODEL = 1
LSTM_MODEL = 2
TRANSFORMER_MODEL = 3

# Model names
MODEL_NAMES = {
    RNN_MODEL: "rnn_model",
    LSTM_MODEL: "lstm_model",
    TRANSFORMER_MODEL: "transformer_model"
}

# Evaluation metrics
METRICS = ["perplexity", "bleu_score"]