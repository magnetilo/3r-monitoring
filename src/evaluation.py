from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any, Union
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def read_labels_file(file_path: Path) -> Dict[str, List[str]]:
    """
    Read a labels file and return a dictionary mapping PMIDs to labels.
    
    Args:
        file_path: Path to the file with format "PMID\tlabel1,label2,..."
    
    Returns:
        Dictionary mapping PMIDs to lists of labels
    """
    labels_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                pmid, labels_str = parts
                labels = labels_str.split(',') if labels_str else []
                labels_dict[pmid] = labels
    return labels_dict


def evaluate_multilabel(
    true_labels_path: Path,
    pred_labels_path: Path,
    all_labels: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multilabel classification results.
    
    Args:
        true_labels_path: Path to the file with true labels
        pred_labels_path: Path to the file with predicted labels
        all_labels: List of all possible labels
    
    Returns:
        Dictionary with precision, recall, and F1 score for each label
        and micro/macro averages
    """
    # Read the label files
    true_labels_dict = read_labels_file(true_labels_path)
    pred_labels_dict = read_labels_file(pred_labels_path)
    
    # Get the set of PMIDs in both files
    common_pmids = set(true_labels_dict.keys()) & set(pred_labels_dict.keys())
    
    if not common_pmids:
        raise ValueError("No common PMIDs found between true and predicted labels")
    
    # Convert to binary matrices for evaluation
    y_true = []
    y_pred = []
    
    for pmid in common_pmids:
        true_label_set = set(true_labels_dict[pmid])
        pred_label_set = set(pred_labels_dict[pmid])
        
        # Convert to binary vectors
        true_binary = [1 if label in true_label_set else 0 for label in all_labels]
        pred_binary = [1 if label in pred_label_set else 0 for label in all_labels]
        
        y_true.append(true_binary)
        y_pred.append(pred_binary)
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    results = {}
    
    # Per-label metrics
    for i, label in enumerate(all_labels):
        label_true = y_true[:, i]
        label_pred = y_pred[:, i]
        
        # Skip labels with no positive samples in the ground truth
        if sum(label_true) == 0:
            results[label] = {
                'precision': float('nan'),
                'recall': float('nan'),
                'f1_score': float('nan'),
                'support': 0
            }
            continue
        
        precision = precision_score(label_true, label_pred, zero_division=0)
        recall = recall_score(label_true, label_pred, zero_division=0)
        f1 = f1_score(label_true, label_pred, zero_division=0)
        support = sum(label_true)
        
        results[label] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support
        }
    
    # Micro average (across all instances and labels)
    micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    micro_recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    results['micro_avg'] = {
        'precision': micro_precision,
        'recall': micro_recall,
        'f1_score': micro_f1,
        'support': y_true.sum()
    }
    
    # Macro average (average of per-label metrics)
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    results['macro_avg'] = {
        'precision': macro_precision,
        'recall': macro_recall,
        'f1_score': macro_f1,
        'support': y_true.sum()
    }
    
    # Weighted average (weighted by support)
    weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    results['weighted_avg'] = {
        'precision': weighted_precision,
        'recall': weighted_recall,
        'f1_score': weighted_f1,
        'support': y_true.sum()
    }
    
    # Subset accuracy (exact match)
    subset_accuracy = accuracy_score(y_true, y_pred)
    results['subset_accuracy'] = {
        'accuracy': subset_accuracy,
        'support': len(y_true)
    }
    
    # Hamming loss (fraction of incorrect labels)
    hamming_loss = (y_true != y_pred).mean()
    results['hamming_loss'] = {
        'loss': hamming_loss,
        'support': y_true.size
    }
    
    return results


def print_evaluation_results(results: Dict[str, Dict[str, float]]) -> None:
    """
    Print evaluation results in a formatted table.
    
    Args:
        results: Dictionary with evaluation results
    """
    print("\nEvaluation Results:")
    print("=" * 80)
    print(f"{'Label':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 80)
    
    # Print per-label metrics
    for label, metrics in results.items():
        if label not in ['micro_avg', 'macro_avg', 'weighted_avg', 'subset_accuracy', 'hamming_loss']:
            print(f"{label:<20} {metrics.get('precision', float('nan')):9.4f} {metrics.get('recall', float('nan')):9.4f} {metrics.get('f1_score', float('nan')):9.4f} {metrics.get('support', 0):<10}")
    
    print("-" * 80)
    
    # Print average metrics
    for avg_type in ['micro_avg', 'macro_avg', 'weighted_avg']:
        if avg_type in results:
            metrics = results[avg_type]
            print(f"{avg_type:<20} {metrics.get('precision', float('nan')):9.4f} {metrics.get('recall', float('nan')):9.4f} {metrics.get('f1_score', float('nan')):9.4f} {metrics.get('support', 0):<10}")
    
    # Print accuracy metrics
    if 'subset_accuracy' in results:
        print(f"{'Subset Accuracy':<20} {results['subset_accuracy'].get('accuracy', float('nan')):9.4f} {'':<10} {'':<10} {results['subset_accuracy'].get('support', 0):<10}")
    
    if 'hamming_loss' in results:
        print(f"{'Hamming Loss':<20} {results['hamming_loss'].get('loss', float('nan')):9.4f} {'':<10} {'':<10} {results['hamming_loss'].get('support', 0):<10}")
    
    print("=" * 80)


def calculate_confusion_matrix(
    true_labels_path: Path,
    pred_labels_path: Path,
    label: str
) -> Tuple[int, int, int, int]:
    """
    Calculate the confusion matrix for a specific label.
    
    Args:
        true_labels_path: Path to the file with true labels
        pred_labels_path: Path to the file with predicted labels
        label: Label to calculate the confusion matrix for
    
    Returns:
        Tuple of (true positives, false positives, false negatives, true negatives)
    """
    # Read the label files
    true_labels_dict = read_labels_file(true_labels_path)
    pred_labels_dict = read_labels_file(pred_labels_path)
    
    # Get the set of PMIDs in both files
    common_pmids = set(true_labels_dict.keys()) & set(pred_labels_dict.keys())
    
    if not common_pmids:
        raise ValueError("No common PMIDs found between true and predicted labels")
    
    # Calculate confusion matrix
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    tn = 0  # True negatives
    
    for pmid in common_pmids:
        true_labels = true_labels_dict[pmid]
        pred_labels = pred_labels_dict[pmid]
        
        true_positive = label in true_labels
        pred_positive = label in pred_labels
        
        if true_positive and pred_positive:
            tp += 1
        elif not true_positive and pred_positive:
            fp += 1
        elif true_positive and not pred_positive:
            fn += 1
        else:  # not true_positive and not pred_positive
            tn += 1
    
    return (tp, fp, fn, tn)