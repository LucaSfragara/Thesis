"""
Evaluation utilities for trained probes.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm


@torch.no_grad()
def evaluate_probe(
    probe: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    return_predictions: bool = False
) -> Dict[str, float]:
    """
    Evaluate a trained probe on a dataset.

    Args:
        probe: Trained probe model
        dataloader: DataLoader with evaluation data
        device: Device to run evaluation on
        return_predictions: Whether to return predictions and labels

    Returns:
        metrics: Dictionary of evaluation metrics (accuracy, precision, recall, f1)
        If return_predictions=True, also returns (predictions, labels)
    """
    if device is None:
        device = next(probe.parameters()).device

    probe.eval()

    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc='Evaluating'):
        hidden_states = batch['hidden_states'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        logits = probe(hidden_states)

        # Get predictions
        if logits.dim() == 3:
            # Sequence output
            pred = logits.argmax(dim=-1)
            mask = labels != -100

            # Flatten and filter valid positions
            pred_flat = pred.reshape(-1)[mask.reshape(-1)]
            labels_flat = labels.reshape(-1)[mask.reshape(-1)]
        else:
            # Single position output
            pred_flat = logits.argmax(dim=-1)
            labels_flat = labels

        all_preds.append(pred_flat.cpu())
        all_labels.append(labels_flat.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Compute metrics
    metrics = compute_probe_metrics(all_labels, all_preds)

    if return_predictions:
        return metrics, (all_preds, all_labels)
    else:
        return metrics


def compute_probe_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Compute classification metrics for probe evaluation.

    Args:
        labels: True labels
        predictions: Predicted labels
        average: Averaging method for multi-class metrics ('macro', 'micro', 'weighted')

    Returns:
        metrics: Dictionary with accuracy, precision, recall, f1-score
    """
    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average=average,
        zero_division=0
    )

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }

    return metrics


def compute_per_class_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class metrics for detailed analysis.

    Args:
        labels: True labels
        predictions: Predicted labels
        class_names: Optional list of class names

    Returns:
        per_class_metrics: Dictionary mapping class -> metrics
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        average=None,
        zero_division=0
    )

    num_classes = len(precision)
    if class_names is None:
        class_names = [f'class_{i}' for i in range(num_classes)]

    per_class_metrics = {}
    for i, class_name in enumerate(class_names[:num_classes]):
        per_class_metrics[class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }

    return per_class_metrics


def compute_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    normalize: Optional[str] = None
) -> np.ndarray:
    """
    Compute confusion matrix for probe predictions.

    Args:
        labels: True labels
        predictions: Predicted labels
        normalize: Normalization method ('true', 'pred', 'all', or None)

    Returns:
        conf_matrix: Confusion matrix [num_classes, num_classes]
    """
    conf_matrix = confusion_matrix(labels, predictions, normalize=normalize)
    return conf_matrix


@torch.no_grad()
def probe_layer_comparison(
    probe_dict: Dict[int, nn.Module],
    dataloader: DataLoader,
    device: Optional[torch.device] = None
) -> Dict[int, Dict[str, float]]:
    """
    Compare probe performance across different layers.

    Args:
        probe_dict: Dictionary mapping layer_id -> trained probe
        dataloader: DataLoader with evaluation data (should provide layer-specific data)
        device: Device to run evaluation on

    Returns:
        layer_metrics: Dictionary mapping layer_id -> metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    layer_metrics = {}

    for layer_id, probe in probe_dict.items():
        print(f"Evaluating layer {layer_id}...")
        metrics = evaluate_probe(probe, dataloader, device=device)
        layer_metrics[layer_id] = metrics

    return layer_metrics


def analyze_probe_selectivity(
    probe: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None
) -> Dict[str, any]:
    """
    Analyze selectivity of probe (how confident it is in predictions).

    Args:
        probe: Trained probe model
        dataloader: DataLoader with evaluation data
        device: Device to run evaluation on

    Returns:
        selectivity_stats: Dictionary with confidence statistics
    """
    if device is None:
        device = next(probe.parameters()).device

    probe.eval()

    all_probs = []
    all_correct = []

    for batch in tqdm(dataloader, desc='Analyzing selectivity'):
        hidden_states = batch['hidden_states'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        logits = probe(hidden_states)
        probs = torch.softmax(logits, dim=-1)

        # Get max probability and prediction
        if logits.dim() == 3:
            max_probs, preds = probs.max(dim=-1)
            mask = labels != -100

            # Flatten and filter
            max_probs_flat = max_probs.reshape(-1)[mask.reshape(-1)]
            preds_flat = preds.reshape(-1)[mask.reshape(-1)]
            labels_flat = labels.reshape(-1)[mask.reshape(-1)]
        else:
            max_probs_flat, preds_flat = probs.max(dim=-1)
            labels_flat = labels

        correct = (preds_flat == labels_flat).float()

        all_probs.append(max_probs_flat.cpu())
        all_correct.append(correct.cpu())

    # Concatenate
    all_probs = torch.cat(all_probs).numpy()
    all_correct = torch.cat(all_correct).numpy()

    # Compute statistics
    selectivity_stats = {
        'mean_confidence': float(np.mean(all_probs)),
        'mean_confidence_correct': float(np.mean(all_probs[all_correct == 1])),
        'mean_confidence_incorrect': float(np.mean(all_probs[all_correct == 0])),
        'median_confidence': float(np.median(all_probs)),
        'confidence_std': float(np.std(all_probs)),
    }

    return selectivity_stats


def compute_probe_control_task(
    probe: nn.Module,
    control_dataloader: DataLoader,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Evaluate probe on a control task (e.g., random labels) to check for memorization.

    A good probe should perform poorly on random labels, indicating it's learning
    meaningful patterns rather than memorizing.

    Args:
        probe: Trained probe model
        control_dataloader: DataLoader with random/shuffled labels
        device: Device to run evaluation on

    Returns:
        control_metrics: Performance on control task
    """
    return evaluate_probe(probe, control_dataloader, device=device)


def layer_wise_accuracy_curve(
    layer_probes: Dict[int, nn.Module],
    dataloader: DataLoader,
    device: Optional[torch.device] = None
) -> Tuple[List[int], List[float]]:
    """
    Compute accuracy curve across layers.

    Args:
        layer_probes: Dictionary mapping layer_id -> probe
        dataloader: Evaluation data
        device: Device to use

    Returns:
        (layer_ids, accuracies): Lists for plotting accuracy vs layer
    """
    layer_ids = sorted(layer_probes.keys())
    accuracies = []

    for layer_id in layer_ids:
        probe = layer_probes[layer_id]
        metrics = evaluate_probe(probe, dataloader, device=device)
        accuracies.append(metrics['accuracy'])

    return layer_ids, accuracies
