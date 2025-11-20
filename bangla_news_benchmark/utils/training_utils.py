"""
Common training utilities and helper functions
"""
import torch
import numpy as np
import random
import os
from pathlib import Path


def set_seed(seed=42):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve
    """

    def __init__(self, patience=5, min_delta=0.0, mode='min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for loss, 'max' for metrics like accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
        else:
            self.is_better = lambda current, best: current > best + min_delta

    def __call__(self, current_score):
        """
        Check if training should stop

        Args:
            current_score: Current validation score

        Returns:
            True if new best score, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            return True

        if self.is_better(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save model checkpoint

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load model checkpoint

    Args:
        filepath: Path to checkpoint
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state into

    Returns:
        Tuple of (epoch, loss)
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch'], checkpoint['loss']


def count_parameters(model):
    """
    Count trainable parameters in a model

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    """
    Get available device (GPU if available, else CPU)

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_time(seconds):
    """
    Format seconds into human-readable time

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


if __name__ == "__main__":
    # Test utilities
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # Test early stopping
    early_stopping = EarlyStopping(patience=3, mode='min')
    losses = [0.5, 0.4, 0.38, 0.39, 0.40, 0.41, 0.42]

    for epoch, loss in enumerate(losses):
        is_best = early_stopping(loss)
        print(f"Epoch {epoch}: Loss={loss:.2f}, Is Best={is_best}, Early Stop={early_stopping.early_stop}")

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break
