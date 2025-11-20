"""
BanglaBERT model for Bangla news classification using Accelerate
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import compute_metrics, get_classification_report
from utils.training_utils import set_seed, EarlyStopping, AverageMeter


def train_banglabert(data_dict, config):
    """
    Train BanglaBERT model with Accelerate

    Args:
        data_dict: Dictionary containing train/val/test datasets
        config: Configuration dictionary

    Returns:
        Dictionary containing metrics and model path
    """
    # Set seed for reproducibility
    set_seed(config.get('seed', 42))

    # Initialize Accelerator for multi-GPU and FP16 support
    accelerator = Accelerator(mixed_precision='fp16')

    # Print device info
    print(f"Using device: {accelerator.device}")
    print(f"Number of processes: {accelerator.num_processes}")
    print(f"Mixed precision: {accelerator.mixed_precision}")

    # Create dataloaders
    train_loader = DataLoader(
        data_dict['train'],
        batch_size=config.get('batch_size', 16),
        shuffle=True
    )
    val_loader = DataLoader(
        data_dict['val'],
        batch_size=config.get('batch_size', 16),
        shuffle=False
    )
    test_loader = DataLoader(
        data_dict['test'],
        batch_size=config.get('batch_size', 16),
        shuffle=False
    )

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        "csebuetnlp/banglabert",
        num_labels=data_dict['num_labels'],
        id2label=data_dict['id2label'],
        label2id=data_dict['label2id']
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 2e-5),
        weight_decay=config.get('weight_decay', 0.01)
    )

    # Learning rate scheduler
    num_epochs = config.get('num_epochs', 5)
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Prepare everything with Accelerator
    model, optimizer, train_loader, val_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader, scheduler
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.get('patience', 3),
        mode='min'
    )

    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print(f"\nTraining BanglaBERT for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not accelerator.is_local_main_process)
        for batch in pbar:
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )

            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            scheduler.step()

            train_loss_meter.update(loss.item(), batch['input_ids'].size(0))
            pbar.set_postfix({'loss': f'{train_loss_meter.avg:.4f}'})

        # Validation phase
        model.eval()
        val_loss_meter = AverageMeter()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )

                loss = outputs.loss
                logits = outputs.logits

                # Gather predictions and labels from all processes
                all_logits = accelerator.gather(logits)
                all_batch_labels = accelerator.gather(batch['labels'])

                val_loss_meter.update(loss.item(), batch['input_ids'].size(0))
                all_preds.append(all_logits.cpu())
                all_labels.append(all_batch_labels.cpu())

        # Compute validation metrics (only on main process)
        if accelerator.is_local_main_process:
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
                # Save using accelerator to handle distributed training
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    config.get('model_save_path', 'checkpoints/banglabert_best'),
                    save_function=accelerator.save
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

        # Wait for all processes
        accelerator.wait_for_everyone()

        # Check early stopping on all processes
        if early_stopping.early_stop:
            break

    # Load best model and evaluate on test set
    if accelerator.is_local_main_process:
        print("\nLoading best model for final evaluation...")

    # Load best model
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model = AutoModelForSequenceClassification.from_pretrained(
        config.get('model_save_path', 'checkpoints/banglabert_best')
    )
    model = accelerator.prepare(unwrapped_model)

    model.eval()
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", disable=not accelerator.is_local_main_process):
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )

            logits = outputs.logits
            labels = batch['labels']

            # Gather from all processes
            all_logits = accelerator.gather(logits)
            all_batch_labels = accelerator.gather(labels)

            test_preds.append(all_logits.cpu())
            test_labels.append(all_batch_labels.cpu())

    # Final metrics (only on main process)
    if accelerator.is_local_main_process:
        test_preds = torch.cat(test_preds, dim=0)
        test_labels = torch.cat(test_labels, dim=0)

        test_metrics = compute_metrics(test_preds, test_labels)
        print("\n" + "="*50)
        print("BanglaBERT Test Results:")
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
            'model_name': 'BanglaBERT',
            'test_metrics': test_metrics,
            'model_path': config.get('model_save_path', 'checkpoints/banglabert_best'),
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    else:
        # Return dummy results for non-main processes
        return {
            'model_name': 'BanglaBERT',
            'test_metrics': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
            'model_path': config.get('model_save_path', 'checkpoints/banglabert_best'),
            'train_losses': [],
            'val_losses': []
        }


if __name__ == "__main__":
    from data.load_dataset import BanglaNewsDataset

    # Load dataset
    dataset = BanglaNewsDataset()
    data_dict = dataset.prepare_for_transformers("csebuetnlp/banglabert")

    # Configuration
    config = {
        'batch_size': 16,
        'lr': 2e-5,
        'weight_decay': 0.01,
        'num_epochs': 5,
        'patience': 3,
        'seed': 42,
        'model_save_path': 'checkpoints/banglabert_best'
    }

    # Train model
    results = train_banglabert(data_dict, config)
    if results:
        print("\nFinal Results:", results['test_metrics'])
