# small-language-models

This project implements text generation models using RNN, LSTM, and Transformer architectures in PyTorch. It trains these models on a small text dataset from Project Gutenberg and evaluates their performance using perplexity and BLEU score metrics

## Key Features

- Modular implementation of RNN, LSTM, and Transformer models
- BPE tokenization using SentencePiece
- Training pipeline with early stopping and learning rate scheduling
- Evaluation using perplexity and BLEU score metrics
- Text generation with temperature sampling
- Visualization of results and model comparisons
- Command-line interface for training, evaluation, and generation

## Repository Structure

```
project/
├── config.py                 # Configuration parameters and hyperparameters
├── main.py                   # Main entry point script
├── requirements.txt          # Project dependencies
├── data/                     # Data directory
│   ├── raw/                  # Raw text files
│   ├── train.jsonl           # Training data
│   └── test.jsonl            # Testing data
├── models/                   # Directory for saved models
├── plots/                    # Directory for output plots
└── src/                      # Source code directory
    ├── data/                 # Data handling modules
    │   ├── dataset.py        # Dataset and dataloader implementations
    │   └── tokenizer.py      # Tokenizer implementation
    ├── models/               # Model implementations
    │   ├── base_model.py     # Base class for all models
    │   ├── rnn_model.py      # RNN model implementation
    │   ├── lstm_model.py     # LSTM model implementation
    │   └── transformer_model.py  # Transformer model implementation
    ├── training/             # Training modules
    │   └── trainer.py        # Training loop implementation
    ├── evaluation/           # Evaluation modules
    │   └── metrics.py        # Evaluation metrics
    └── visualization/        # Visualization modules
        └── loss_plots.py     # Training/validation loss plots
```

## Installation

```shell
# create virtual environment
$ conda create -n slm python=3.10 -y

# activate environment
$ conda activate slm

# install packages
$ pip install -r requirements.txt
```

## Dataset

The dataset consists of classic literature texts from Project Gutenberg, including works such as:

- Alice in Wonderland
- Art of War
- Dracula
- Frankenstein
- Great Gatsby
- and more...

The data is organized into:

- `data/raw/`: Original text files
- `train.jsonl`: Training examples in JSON format with prompt and completion pairs
- `test.jsonl`: Testing examples in the same format


## Usage

### 1. Training the Models

To train all three models (RNN, LSTM, and Transformer):

```bash
# Train all models
python main.py --train --model_type 0

# Train only the RNN model
python main.py --train --model_type 1

# Train only the LSTM model
python main.py --train --model_type 2

# Train only the Transformer model
python main.py --train --model_type 3
```

This will:
1. Load and tokenize the data
2. Train a BPE tokenizer with vocabulary size 10,000
3. Train each model for up to 30 epochs with early stopping
4. Save model checkpoints to the `models/` directory
5. Generate training/validation loss plots

### 2. Evaluating the Models

To evaluate the trained models on the test set:

```bash
# Evaluate all the models
python main.py --evaluate --model_type 0
```

This will:
1. Load the trained models
2. Calculate perplexity and BLEU score for each model
3. Generate comparison visualizations
4. Print a summary of results

### 3. Generating Text

To generate text from trained models:

```bash
# Generate text using only the Transformer model
python main.py --generate --prompt "Your prompt here" --model_type 3 --temperature 0.8 --max_length 100
```

Additional arguments:
- `--temperature`: Controls randomness in generation (default: 1.0)
- `--max_length`: Maximum number of tokens to generate (default: 100)
- `--seed`: Random seed for reproducibility (default: 42)

### 4. Combined Operations

You can also combine operations:

```bash
python main.py --train --evaluate
```

## Configuration

Model hyperparameters and training settings can be modified in the `config.py` file:

```python
# Model parameters
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
RNN_LAYERS = 2
LSTM_LAYERS = 2
TRANSFORMER_LAYERS = 2
TRANSFORMER_HEADS = 4
DROPOUT = 0.2

# Training parameters
BATCH_SIZE = 128
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 3
```

## Results

After training and evaluation, you'll find:

1. Trained model checkpoints in the `models/` directory
2. Visualization plots in the `plots/` directory:
   - Training and validation loss curves
   - Perplexity comparison
   - BLEU score comparison
   - Normalized metrics comparison

## Extending the Project

The modular structure makes it easy to extend this project:

1. **Add new models**: Create a new model file in `src/models/` that inherits from `BaseTextGenerationModel`
2. **Add new metrics**: Implement additional evaluation metrics in `src/evaluation/metrics.py`
3. **Try different datasets**: Update the data loading and preprocessing in `src/data/`

## Acknowledgments

- The dataset is derived from public domain texts from Project Gutenberg
- The implementation is built with PyTorch
