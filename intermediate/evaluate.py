#!/usr/bin/env python3
"""
Evaluation script for computing metrics on predictions
Calculates Dice scores, sensitivity, and specificity for each class
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from typing import Dict, List, Tuple


def compute_dice(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Dice coefficient"""
    intersection = np.sum(pred * target)
    return (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)


def compute_sensitivity(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute sensitivity (recall)"""
    true_positive = np.sum(pred * target)
    false_negative = np.sum((1 - pred) * target)
    return (true_positive + smooth) / (true_positive + false_negative + smooth)


def compute_specificity(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute specificity"""
    true_negative = np.sum((1 - pred) * (1 - target))
    false_positive = np.sum(pred * (1 - target))
    return (true_negative + smooth) / (true_negative + false_positive + smooth)


def compute_metrics_for_class(pred: np.ndarray, target: np.ndarray, class_id: int) -> Dict[str, float]:
    """Compute all metrics for a specific class"""
    pred_binary = (pred == class_id).astype(np.float32)
    target_binary = (target == class_id).astype(np.float32)
    
    # Skip if class not present in target
    if target_binary.sum() == 0:
        return None
    
    metrics = {
        'dice': compute_dice(pred_binary, target_binary),
        'sensitivity': compute_sensitivity(pred_binary, target_binary),
        'specificity': compute_specificity(pred_binary, target_binary),
        'volume_pred': pred_binary.sum(),
        'volume_true': target_binary.sum(),
        'volume_diff': pred_binary.sum() - target_binary.sum()
    }
    
    return metrics


def evaluate_case(pred_path: str, label_path: str) -> Dict[str, Dict[str, float]]:
    """Evaluate a single case"""
    # Load volumes
    pred = nib.load(pred_path).get_fdata().astype(np.int32)
    label = nib.load(label_path).get_fdata().astype(np.int32)
    
    # Ensure same shape
    assert pred.shape == label.shape, f"Shape mismatch: {pred.shape} vs {label.shape}"
    
    # Class names
    class_names = {
        1: 'psma_tumor',
        2: 'psma_normal',
        3: 'fdg_tumor',
        4: 'fdg_normal'
    }
    
    # Compute metrics for each class
    results = {}
    for class_id, class_name in class_names.items():
        metrics = compute_metrics_for_class(pred, label, class_id)
        if metrics is not None:
            results[class_name] = metrics
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate segmentation predictions')
    parser.add_argument('--predictions_dir', type=str, required=True,
                        help='Directory containing prediction files')
    parser.add_argument('--labels_dir', type=str, required=True,
                        help='Directory containing ground truth labels')
    parser.add_argument('--output_file', type=str, default='metrics.csv',
                        help='Output CSV file for metrics')
    parser.add_argument('--json_output', type=str, default='metrics.json',
                        help='Output JSON file for detailed metrics')
    
    args = parser.parse_args()
    
    # Get list of prediction files
    pred_files = [f for f in os.listdir(args.predictions_dir) if f.endswith('.nii.gz')]
    pred_files.sort()
    
    print(f"Found {len(pred_files)} prediction files")
    
    # Evaluate each case
    all_results = {}
    class_metrics = {
        'psma_tumor': {'dice': [], 'sensitivity': [], 'specificity': []},
        'psma_normal': {'dice': [], 'sensitivity': [], 'specificity': []},
        'fdg_tumor': {'dice': [], 'sensitivity': [], 'specificity': []},
        'fdg_normal': {'dice': [], 'sensitivity': [], 'specificity': []}
    }
    
    for pred_file in tqdm(pred_files, desc='Evaluating'):
        case_id = pred_file.replace('.nii.gz', '')
        
        # Find corresponding label file
        label_file = f"{case_id}.nii.gz"
        label_path = os.path.join(args.labels_dir, label_file)
        
        if not os.path.exists(label_path):
            print(f"Warning: Label not found for {case_id}")
            continue
        
        pred_path = os.path.join(args.predictions_dir, pred_file)
        
        # Evaluate
        try:
            case_results = evaluate_case(pred_path, label_path)
            all_results[case_id] = case_results
            
            # Accumulate metrics
            for class_name in class_metrics.keys():
                if class_name in case_results:
                    for metric in ['dice', 'sensitivity', 'specificity']:
                        class_metrics[class_name][metric].append(
                            case_results[class_name][metric]
                        )
        except Exception as e:
            print(f"Error evaluating {case_id}: {str(e)}")
            continue
    
    # Compute summary statistics
    summary = {}
    rows = []
    
    for class_name, metrics in class_metrics.items():
        if len(metrics['dice']) > 0:
            summary[class_name] = {
                'dice_mean': np.mean(metrics['dice']),
                'dice_std': np.std(metrics['dice']),
                'dice_median': np.median(metrics['dice']),
                'sensitivity_mean': np.mean(metrics['sensitivity']),
                'sensitivity_std': np.std(metrics['sensitivity']),
                'specificity_mean': np.mean(metrics['specificity']),
                'specificity_std': np.std(metrics['specificity']),
                'n_cases': len(metrics['dice'])
            }
            
            rows.append({
                'Class': class_name,
                'Dice (mean±std)': f"{summary[class_name]['dice_mean']:.4f}±{summary[class_name]['dice_std']:.4f}",
                'Dice (median)': f"{summary[class_name]['dice_median']:.4f}",
                'Sensitivity': f"{summary[class_name]['sensitivity_mean']:.4f}±{summary[class_name]['sensitivity_std']:.4f}",
                'Specificity': f"{summary[class_name]['specificity_mean']:.4f}±{summary[class_name]['specificity_std']:.4f}",
                'N Cases': summary[class_name]['n_cases']
            })
    
    # Calculate overall metrics
    all_dice = []
    for class_name in ['psma_tumor', 'fdg_tumor']:  # Focus on tumor classes
        if class_name in class_metrics and len(class_metrics[class_name]['dice']) > 0:
            all_dice.extend(class_metrics[class_name]['dice'])
    
    if all_dice:
        rows.append({
            'Class': 'Average (Tumors)',
            'Dice (mean±std)': f"{np.mean(all_dice):.4f}±{np.std(all_dice):.4f}",
            'Dice (median)': f"{np.median(all_dice):.4f}",
            'Sensitivity': '-',
            'Specificity': '-',
            'N Cases': len(all_dice)
        })
    
    # Save results
    df = pd.DataFrame(rows)
    df.to_csv(args.output_file, index=False)
    print(f"\nMetrics saved to {args.output_file}")
    
    # Save detailed JSON
    detailed_results = {
        'per_case': all_results,
        'summary': summary
    }
    
    with open(args.json_output, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"Detailed metrics saved to {args.json_output}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    # Print key metrics
    if 'psma_tumor' in summary:
        print(f"\nPSMA Tumor Dice: {summary['psma_tumor']['dice_mean']:.4f} ± {summary['psma_tumor']['dice_std']:.4f}")
    if 'fdg_tumor' in summary:
        print(f"FDG Tumor Dice: {summary['fdg_tumor']['dice_mean']:.4f} ± {summary['fdg_tumor']['dice_std']:.4f}")
    if all_dice:
        print(f"Average Tumor Dice: {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")


if __name__ == '__main__':
    main()
