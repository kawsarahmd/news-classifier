# Bangla News Classification Benchmark

A comprehensive benchmarking framework for Bangla news classification using deep learning models. This framework compares the performance of various neural network architectures on the Bangla news categorization task.

## Models

This benchmark includes 5 state-of-the-art models:

1. **BiLSTM** - Bidirectional LSTM with word embeddings
2. **CNN-LSTM** - Hybrid model combining CNN feature extraction with LSTM
3. **Attention-BiLSTM** - BiLSTM with attention mechanism
4. **XLM-RoBERTa** - Multilingual transformer (xlm-roberta-base)
5. **BanglaBERT** - Bangla-specific BERT (csebuetnlp/banglabert)

## Dataset

**Source**: `kawsarahmd/bangla-news-category-plos-one` (HuggingFace Datasets)

The dataset is automatically downloaded and split into:
- Training set (80%)
- Validation set (10%)
- Test set (10%)

Label encoding is performed alphabetically for consistency across runs.

## Project Structure

```
bangla_news_benchmark/
├── data/
│   ├── __init__.py
│   └── load_dataset.py          # Dataset loading and preprocessing
├── models/
│   ├── __init__.py
│   ├── bilstm.py                # BiLSTM model
│   ├── cnn_lstm.py              # CNN-LSTM model
│   ├── attention_bilstm.py      # Attention-BiLSTM model
│   ├── xlmr.py                  # XLM-RoBERTa model
│   └── banglabert.py            # BanglaBERT model
├── utils/
│   ├── __init__.py
│   ├── metrics.py               # Evaluation metrics
│   └── training_utils.py        # Training utilities
├── benchmark.py                 # Main benchmarking script
├── results.csv                  # Results template
└── README.md                    # This file
```

## Requirements

Install required dependencies:

```bash
pip install torch torchvision torchaudio
pip install transformers datasets accelerate
pip install scikit-learn pandas numpy tqdm
```

For GPU support (recommended):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Run Full Benchmark

Train and evaluate all models:

```bash
cd bangla_news_benchmark
python benchmark.py
```

### Run Specific Models

Train only selected models:

```bash
# Train only LSTM-based models
python benchmark.py --models bilstm cnn-lstm attention-bilstm

# Train only transformer models
python benchmark.py --models xlm-roberta banglabert

# Train a single model
python benchmark.py --models banglabert
```

### Set Random Seed

For reproducibility:

```bash
python benchmark.py --seed 42
```

### Train Individual Models

Each model can be trained independently:

```bash
# BiLSTM
python -m models.bilstm

# CNN-LSTM
python -m models.cnn_lstm

# Attention-BiLSTM
python -m models.attention_bilstm

# XLM-RoBERTa
python -m models.xlmr

# BanglaBERT
python -m models.banglabert
```

## Model Configurations

### LSTM Models (BiLSTM, CNN-LSTM, Attention-BiLSTM)

- **Framework**: Pure PyTorch
- **Device**: Automatic GPU detection
- **Batch size**: 32
- **Epochs**: 20 (with early stopping)
- **Optimizer**: Adam (lr=0.001)
- **Early stopping**: Patience=5 epochs
- **Features**:
  - Train/validation loss tracking
  - Best model saving
  - Automatic early stopping

### Transformer Models (XLM-RoBERTa, BanglaBERT)

- **Framework**: HuggingFace Transformers + Accelerate
- **Mixed Precision**: FP16 (via Accelerate)
- **Multi-GPU**: Automatic (via Accelerate)
- **Batch size**: 16
- **Epochs**: 5 (with early stopping)
- **Optimizer**: AdamW (lr=2e-5, weight_decay=0.01)
- **Scheduler**: Linear warmup (10% steps)
- **Early stopping**: Patience=3 epochs
- **Features**:
  - Distributed training support
  - Gradient accumulation
  - Mixed precision training

## Evaluation Metrics

All models are evaluated using:

- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted average precision
- **Recall**: Weighted average recall
- **F1-Score**: Weighted average F1-score
- **Classification Report**: Per-class metrics

## Results

Results are saved in two formats:

1. **results.csv**: Summary with all metrics
2. **results_detailed.csv**: Pandas-friendly format for analysis

The benchmark also prints a leaderboard sorted by F1-score:

```
================================================================================
BANGLA NEWS CLASSIFICATION BENCHMARK - LEADERBOARD
================================================================================
Rank   Model                Accuracy     Precision    Recall       F1-Score
--------------------------------------------------------------------------------
1      BanglaBERT           0.XXXX       0.XXXX       0.XXXX       0.XXXX
2      XLM-RoBERTa          0.XXXX       0.XXXX       0.XXXX       0.XXXX
...
================================================================================
```

## Model Checkpoints

Trained models are saved in the `checkpoints/` directory:

- `checkpoints/bilstm_best.pt`
- `checkpoints/cnn_lstm_best.pt`
- `checkpoints/attention_bilstm_best.pt`
- `checkpoints/xlmr_best/` (HuggingFace format)
- `checkpoints/banglabert_best/` (HuggingFace format)

## Hardware Requirements

### Minimum
- CPU: 4+ cores
- RAM: 16GB
- Storage: 10GB

### Recommended (for faster training)
- GPU: NVIDIA T4 or better
- VRAM: 16GB+ (for transformer models with batch size 16)
- RAM: 32GB
- Storage: 20GB (for cached datasets and models)

### Kaggle T4x2 GPUs
This framework is optimized for Kaggle T4x2 GPUs and will automatically utilize:
- Multi-GPU training (via Accelerate)
- Mixed precision (FP16)
- Distributed data loading

## Reproducibility

To ensure reproducible results:

1. Fixed random seeds (PyTorch, NumPy, Python)
2. Deterministic CUDA operations
3. Stratified train/val/test splits
4. Alphabetically sorted label encoding

## Extending the Framework

### Adding a New Model

1. Create a new file in `models/` (e.g., `models/new_model.py`)
2. Implement the model class and training function
3. Add the model to `benchmark.py` in the `all_models` dictionary
4. Update `models/__init__.py`

### Customizing Hyperparameters

Edit the `config` dictionary in `benchmark.py` for each model:

```python
'bilstm': {
    'name': 'BiLSTM',
    'train_fn': train_bilstm,
    'data_prep': lambda: dataset.prepare_for_lstm(),
    'config': {
        'batch_size': 64,        # Increase batch size
        'hidden_dim': 512,       # Larger hidden dimension
        'num_epochs': 30,        # More epochs
        # ... other parameters
    }
}
```


