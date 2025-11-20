"""
Dataset loader for Bangla News Classification
"""
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import numpy as np


class BanglaNewsDataset:
    """Load and preprocess Bangla news dataset from HuggingFace"""

    def __init__(self, dataset_name="kawsarahmd/bangla-news-category-plos-one"):
        """
        Load dataset and create label mappings

        Args:
            dataset_name: HuggingFace dataset identifier
        """
        print(f"Loading dataset: {dataset_name}")
        self.dataset = load_dataset(dataset_name)

        # Create label encoding (sorted for consistency)
        all_labels = self.dataset['train']['category']
        unique_labels = sorted(list(set(all_labels)))

        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.num_labels = len(unique_labels)

        print(f"Found {self.num_labels} categories: {unique_labels}")

    def get_splits(self, val_size=0.1, test_size=0.1, seed=42):
        """
        Create train/val/test splits

        Args:
            val_size: Validation split ratio
            test_size: Test split ratio
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # First split: train+val vs test
        train_val = self.dataset['train'].train_test_split(
            test_size=test_size,
            seed=seed,
            stratify_by_column='category'
        )

        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train_val_split = train_val['train'].train_test_split(
            test_size=val_ratio,
            seed=seed,
            stratify_by_column='category'
        )

        train_data = train_val_split['train']
        val_data = train_val_split['test']
        test_data = train_val['test']

        print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

        return train_data, val_data, test_data

    def prepare_for_lstm(self, tokenizer_name="google/muril-base-cased", max_length=256):
        """
        Prepare dataset for LSTM models using BERT tokenizer

        Args:
            tokenizer_name: Pre-trained tokenizer to use
            max_length: Maximum sequence length

        Returns:
            Dictionary containing tokenized datasets and vocab size
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        train_data, val_data, test_data = self.get_splits()

        def tokenize_function(examples):
            # Tokenize and encode labels
            tokens = tokenizer(
                examples['content'],
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors=None
            )
            tokens['labels'] = [self.label2id[cat] for cat in examples['category']]
            return tokens

        train_dataset = train_data.map(tokenize_function, batched=True, remove_columns=train_data.column_names)
        val_dataset = val_data.map(tokenize_function, batched=True, remove_columns=val_data.column_names)
        test_dataset = test_data.map(tokenize_function, batched=True, remove_columns=test_data.column_names)

        train_dataset.set_format(type='torch')
        val_dataset.set_format(type='torch')
        test_dataset.set_format(type='torch')

        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset,
            'vocab_size': tokenizer.vocab_size,
            'num_labels': self.num_labels,
            'label2id': self.label2id,
            'id2label': self.id2label
        }

    def prepare_for_transformers(self, model_name, max_length=256):
        """
        Prepare dataset for transformer models

        Args:
            model_name: Model identifier for tokenizer
            max_length: Maximum sequence length

        Returns:
            Dictionary containing tokenized datasets
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_data, val_data, test_data = self.get_splits()

        def tokenize_function(examples):
            tokens = tokenizer(
                examples['content'],
                padding='max_length',
                truncation=True,
                max_length=max_length,
            )
            tokens['labels'] = [self.label2id[cat] for cat in examples['category']]
            return tokens

        train_dataset = train_data.map(tokenize_function, batched=True, remove_columns=train_data.column_names)
        val_dataset = val_data.map(tokenize_function, batched=True, remove_columns=val_data.column_names)
        test_dataset = test_data.map(tokenize_function, batched=True, remove_columns=test_data.column_names)

        train_dataset.set_format(type='torch')
        val_dataset.set_format(type='torch')
        test_dataset.set_format(type='torch')

        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset,
            'num_labels': self.num_labels,
            'label2id': self.label2id,
            'id2label': self.id2label,
            'tokenizer': tokenizer
        }


if __name__ == "__main__":
    # Test dataset loading
    dataset = BanglaNewsDataset()

    # Test LSTM preparation
    lstm_data = dataset.prepare_for_lstm()
    print(f"\nLSTM data prepared:")
    print(f"  Vocab size: {lstm_data['vocab_size']}")
    print(f"  Train samples: {len(lstm_data['train'])}")

    # Test transformer preparation
    transformer_data = dataset.prepare_for_transformers("xlm-roberta-base")
    print(f"\nTransformer data prepared:")
    print(f"  Train samples: {len(transformer_data['train'])}")
