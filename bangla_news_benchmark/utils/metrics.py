"""
Metrics computation for model evaluation
"""
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


def compute_metrics(predictions, labels, id2label=None):
    """
    Compute comprehensive evaluation metrics

    Args:
        predictions: Model predictions (logits or class indices)
        labels: Ground truth labels
        id2label: Optional mapping from indices to label names

    Returns:
        Dictionary containing metrics
    """
    # Convert to numpy if needed
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    # Get predicted classes if logits provided
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        pred_classes = np.argmax(predictions, axis=1)
    else:
        pred_classes = predictions

    # Compute metrics
    accuracy = accuracy_score(labels, pred_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred_classes, average='weighted', zero_division=0
    )

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }

    return metrics


def get_classification_report(predictions, labels, id2label):
    """
    Generate detailed classification report

    Args:
        predictions: Model predictions
        labels: Ground truth labels
        id2label: Mapping from indices to label names

    Returns:
        Classification report string
    """
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        pred_classes = np.argmax(predictions, axis=1)
    else:
        pred_classes = predictions

    target_names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(labels, pred_classes, target_names=target_names, zero_division=0)

    return report


def get_confusion_matrix(predictions, labels):
    """
    Compute confusion matrix

    Args:
        predictions: Model predictions
        labels: Ground truth labels

    Returns:
        Confusion matrix (numpy array)
    """
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        pred_classes = np.argmax(predictions, axis=1)
    else:
        pred_classes = predictions

    cm = confusion_matrix(labels, pred_classes)
    return cm


def evaluate_model(model, dataloader, device, return_predictions=False):
    """
    Evaluate a PyTorch model on a dataloader

    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        return_predictions: Whether to return predictions and labels

    Returns:
        Tuple of (loss, predictions, labels) if return_predictions=True
        Otherwise just loss
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_predictions.append(outputs.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(dataloader)

    if return_predictions:
        predictions = torch.cat(all_predictions, dim=0)
        labels = torch.cat(all_labels, dim=0)
        return avg_loss, predictions, labels
    else:
        return avg_loss


if __name__ == "__main__":
    # Test metrics computation
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred_logits = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.1, 0.2, 0.7],
        [0.9, 0.05, 0.05],
        [0.2, 0.6, 0.2],
        [0.1, 0.1, 0.8]
    ])

    metrics = compute_metrics(y_pred_logits, y_true)
    print("Metrics:", metrics)

    id2label = {0: "Sports", 1: "Politics", 2: "Entertainment"}
    report = get_classification_report(y_pred_logits, y_true, id2label)
    print("\nClassification Report:")
    print(report)
