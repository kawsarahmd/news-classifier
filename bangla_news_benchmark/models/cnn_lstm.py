"""
CNN-LSTM hybrid model for Bangla news classification
TPU and GPU compatible
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import compute_metrics, get_classification_report
from utils.training_utils import (
    set_seed, EarlyStopping, save_checkpoint, load_checkpoint,
    count_parameters, get_device, optimizer_step, mark_step, AverageMeter
)


class CNNLSTMClassifier(nn.Module):
    """
    CNN-LSTM hybrid model for text classification
    CNN extracts local features, LSTM captures sequential dependencies
    """

    def __init__(self, vocab_size, embedding_dim=128, num_filters=100,
                 filter_sizes=[3, 4, 5], lstm_hidden_dim=128, num_layers=1,
                 num_classes=5, dropout=0.3):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            num_filters: Number of filters for each filter size
            filter_sizes: List of filter sizes for CNN
            lstm_hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(CNNLSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Convolutional layers with different filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                     out_channels=num_filters,
                     kernel_size=fs)
            for fs in filter_sizes
        ])

        # LSTM layer
        cnn_output_dim = num_filters * len(filter_sizes)
        self.lstm = nn.LSTM(
            cnn_output_dim,
            lstm_hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden_dim * 2, num_classes)  # *2 for bidirectional

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask (optional)

        Returns:
            Logits [batch_size, num_classes]
        """
        # Embedding
        embedded = self.embedding(input_ids)  # [batch, seq_len, emb_dim]

        # CNN expects [batch, channels, seq_len]
        embedded_t = embedded.permute(0, 2, 1)  # [batch, emb_dim, seq_len]

        # Apply convolutions and ReLU activation
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded_t))  # [batch, num_filters, seq_len - k + 1]
            conv_outputs.append(conv_out)

        # Concatenate along the filter dimension
        # First, pad to same length
        max_len = max([co.size(2) for co in conv_outputs])
        padded_convs = []
        for co in conv_outputs:
            if co.size(2) < max_len:
                padding = max_len - co.size(2)
                co = torch.nn.functional.pad(co, (0, padding))
            padded_convs.append(co)

        # Concatenate
        concat_conv = torch.cat(padded_convs, dim=1)  # [batch, num_filters * len(filter_sizes), seq_len]

        # Prepare for LSTM: [batch, seq_len, features]
        lstm_input = concat_conv.permute(0, 2, 1)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(lstm_input)

        # Use last hidden states from both directions
        hidden_fwd = hidden[-2, :, :]
        hidden_bwd = hidden[-1, :, :]
        hidden_concat = torch.cat([hidden_fwd, hidden_bwd], dim=1)

        # Dropout and classification
        dropped = self.dropout(hidden_concat)
        logits = self.fc(dropped)

        return logits


def train_cnn_lstm(data_dict, config):
    """
    Train CNN-LSTM model

    Args:
        data_dict: Dictionary containing train/val/test datasets
        config: Configuration dictionary

    Returns:
        Dictionary containing metrics and model path
    """
    # Set seed for reproducibility
    set_seed(config.get('seed', 42))

    # Get device
    device = get_device()

    # Create dataloaders
    train_loader = DataLoader(
        data_dict['train'],
        batch_size=config.get('batch_size', 32),
        shuffle=True
    )
    val_loader = DataLoader(
        data_dict['val'],
        batch_size=config.get('batch_size', 32),
        shuffle=False
    )
    test_loader = DataLoader(
        data_dict['test'],
        batch_size=config.get('batch_size', 32),
        shuffle=False
    )

    # Initialize model
    model = CNNLSTMClassifier(
        vocab_size=data_dict['vocab_size'],
        embedding_dim=config.get('embedding_dim', 128),
        num_filters=config.get('num_filters', 100),
        filter_sizes=config.get('filter_sizes', [3, 4, 5]),
        lstm_hidden_dim=config.get('lstm_hidden_dim', 128),
        num_layers=config.get('num_layers', 1),
        num_classes=data_dict['num_labels'],
        dropout=config.get('dropout', 0.3)
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 0.001))

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.get('patience', 5),
        mode='min'
    )

    # Training loop
    num_epochs = config.get('num_epochs', 20)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print(f"\nTraining CNN-LSTM for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer_step(optimizer, device)  # TPU-compatible optimizer step
            mark_step(device)  # Mark step for TPU (no-op on GPU/CPU)

            train_loss_meter.update(loss.item(), input_ids.size(0))
            pbar.set_postfix({'loss': f'{train_loss_meter.avg:.4f}'})

        # Validation phase
        model.eval()
        val_loss_meter = AverageMeter()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                val_loss_meter.update(loss.item(), input_ids.size(0))
                all_preds.append(outputs.cpu())
                all_labels.append(labels.cpu())

        # Compute validation metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        val_metrics = compute_metrics(all_preds, all_labels)

        train_losses.append(train_loss_meter.avg)
        val_losses.append(val_loss_meter.avg)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss_meter.avg:.4f}")
        print(f"  Val Loss: {val_loss_meter.avg:.4f}")
        print(f"  Val Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

        # Save best model
        if val_loss_meter.avg < best_val_loss:
            best_val_loss = val_loss_meter.avg
            save_checkpoint(
                model, optimizer, epoch, val_loss_meter.avg,
                config.get('model_save_path', 'checkpoints/cnn_lstm_best.pt'),
                device=device  # Pass device for TPU compatibility
            )
            print(f"  → Saved best model (val_loss: {best_val_loss:.4f})")

        # Early stopping check
        if early_stopping(val_loss_meter.avg):
            if not early_stopping.early_stop:
                pass  # New best
        else:
            if early_stopping.early_stop:
                print(f"\nEarly stopping triggered after epoch {epoch+1}")
                break

    # Load best model and evaluate on test set
    print("\nLoading best model for final evaluation...")
    load_checkpoint(config.get('model_save_path', 'checkpoints/cnn_lstm_best.pt'), model)

    model.eval()
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            test_preds.append(outputs.cpu())
            test_labels.append(labels.cpu())

    test_preds = torch.cat(test_preds, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    # Final metrics
    test_metrics = compute_metrics(test_preds, test_labels)
    print("\n" + "="*50)
    print("CNN-LSTM Test Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print("="*50)

    # Classification report
    report = get_classification_report(test_preds, test_labels, data_dict['id2label'])
    print("\nClassification Report:")
    print(report)

    return {
        'model_name': 'CNN-LSTM',
        'test_metrics': test_metrics,
        'model_path': config.get('model_save_path', 'checkpoints/cnn_lstm_best.pt'),
        'train_losses': train_losses,
        'val_losses': val_losses
    }


if __name__ == "__main__":
    from data.load_dataset import BanglaNewsDataset

    # Load dataset
    dataset = BanglaNewsDataset()
    data_dict = dataset.prepare_for_lstm()

    # Configuration
    config = {
        'batch_size': 32,
        'embedding_dim': 128,
        'num_filters': 100,
        'filter_sizes': [3, 4, 5],
        'lstm_hidden_dim': 128,
        'num_layers': 1,
        'dropout': 0.3,
        'lr': 0.001,
        'num_epochs': 20,
        'patience': 5,
        'seed': 42,
        'model_save_path': 'checkpoints/cnn_lstm_best.pt'
    }

    # Train model
    results = train_cnn_lstm(data_dict, config)
    print("\nFinal Results:", results['test_metrics'])
