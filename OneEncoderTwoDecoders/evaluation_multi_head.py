import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from os.path import join
import nnunet_config_paths as cfg
from typing import List, Tuple, Dict, Union
from DEEP_PSMA_Infer import run_inference, MultiHeadPredictor

def dice_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Dice score between binary masks
    """
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-5)

def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate precision
    """
    true_positives = np.sum(y_true * y_pred)
    predicted_positives = np.sum(y_pred)
    return true_positives / (predicted_positives + 1e-5)

def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate recall (sensitivity)
    """
    true_positives = np.sum(y_true * y_pred)
    actual_positives = np.sum(y_true)
    return true_positives / (actual_positives + 1e-5)

def evaluate_predictions(pred_dir: str, gt_dir: str, output_csv: str, label_value: int = 1) -> pd.DataFrame:
    """
    Evaluate predictions against ground truth
    
    Args:
        pred_dir: Directory with prediction files
        gt_dir: Directory with ground truth files
        output_csv: Path to save results
        label_value: Label value to evaluate (default: 1 for TTB)
        
    Returns:
        DataFrame with evaluation metrics
    """
    metrics = []
    
    # Find all prediction files
    for pred_file in os.listdir(pred_dir):
        if not pred_file.endswith('.nii.gz'):
            continue
            
        case = pred_file.split('.')[0]
        gt_file = join(gt_dir, f"{case}.nii.gz")
        
        if not os.path.exists(gt_file):
            print(f"Ground truth not found for {case}, skipping...")
            continue
            
        # Load prediction and ground truth
        pred = sitk.ReadImage(join(pred_dir, pred_file))
        gt = sitk.ReadImage(gt_file)
        
        # Convert to numpy and binarize
        pred_np = (sitk.GetArrayFromImage(pred) == label_value).astype(np.int8)
        gt_np = (sitk.GetArrayFromImage(gt) == label_value).astype(np.int8)
        
        # Calculate metrics
        dice = dice_score(gt_np, pred_np)
        prec = precision(gt_np, pred_np)
        rec = recall(gt_np, pred_np)
        
        # Add to results
        metrics.append({
            'case': case,
            'dice': dice,
            'precision': prec,
            'recall': rec,
            'gt_volume': np.sum(gt_np),
            'pred_volume': np.sum(pred_np)
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(metrics)
    
    if len(df) > 0:
        # Add summary row with means
        means = df.mean(numeric_only=True)
        means['case'] = 'MEAN'
        df = df.append(means, ignore_index=True)
        
        # Save to CSV
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
        
        # Print summary
        print(f"Mean Dice: {means['dice']:.4f}")
        print(f"Mean Precision: {means['precision']:.4f}")
        print(f"Mean Recall: {means['recall']:.4f}")
    else:
        print("No valid cases found for evaluation")
    
    return df

def evaluate_multi_head_model(fold: int = 0):
    """
    Evaluate both heads of the early fusion model
    
    Args:
        fold: Which fold to evaluate
    """
    task_id = cfg.dataset_dictionary['EarlyFusion']
    model_dir = join(cfg.nn_results_dir, f"Dataset{task_id}_EarlyFusion/MultiHeadnnUNetTrainer__nnUNetPlans__3d_fullres")
    
    # Create a directory for predictions
    pred_dir = f"predictions_earlyfusion_fold{fold}_evaluation"
    os.makedirs(pred_dir, exist_ok=True)
    
    # Setup the predictor
    predictor = MultiHeadPredictor(model_folder=model_dir, folds=fold)
    
    # Prepare output directories for both heads
    psma_output_dir = join(pred_dir, "psma")
    fdg_output_dir = join(pred_dir, "fdg")
    os.makedirs(psma_output_dir, exist_ok=True)
    os.makedirs(fdg_output_dir, exist_ok=True)
    
    # Get validation cases
    validation_csv = join(cfg.nn_preprocessed_dir, f"Dataset{task_id}_EarlyFusion/splits_final.json")
    if os.path.exists(validation_csv):
        import json
        splits = json.load(open(validation_csv))
        val_cases = splits[str(fold)]['val']
    else:
        # Default to all cases in the test data directory
        val_cases = [f.replace('.nii.gz', '') for f in os.listdir(join(cfg.in_dir, 'PSMA', 'CT'))]
    
    # Process each validation case
    for case in val_cases:
        print(f"Processing case {case}...")
        
        # Paths to input images
        psma_pet_path = join(cfg.in_dir, 'PSMA', 'PET', f"{case}.nii.gz")
        psma_ct_path = join(cfg.in_dir, 'PSMA', 'CT', f"{case}.nii.gz")
        fdg_pet_path = join(cfg.in_dir, 'FDG', 'PET', f"{case}.nii.gz")
        fdg_ct_path = join(cfg.in_dir, 'FDG', 'CT', f"{case}.nii.gz")
        
        # Get thresholds
        with open(join(cfg.in_dir, 'PSMA', 'thresholds', f"{case}.json")) as f:
            psma_threshold = json.load(f)['suv_threshold']
            
        with open(join(cfg.in_dir, 'FDG', 'thresholds', f"{case}.json")) as f:
            fdg_threshold = json.load(f)['suv_threshold']
        
        # Run inference with both heads
        psma_pred, fdg_pred, _, _ = run_inference(
            psma_pet_path=psma_pet_path,
            psma_ct_path=psma_ct_path,
            fdg_pet_path=fdg_pet_path,
            fdg_ct_path=fdg_ct_path,
            fold=fold,
            psma_suv_threshold=psma_threshold,
            fdg_suv_threshold=fdg_threshold,
            output_type='both',
            return_sitk=True
        )
        
        # Save predictions
        sitk.WriteImage(psma_pred, join(psma_output_dir, f"{case}.nii.gz"))
        sitk.WriteImage(fdg_pred, join(fdg_output_dir, f"{case}.nii.gz"))
    
    # Evaluate PSMA predictions
    print("\nEvaluating PSMA predictions:")
    evaluate_predictions(
        pred_dir=psma_output_dir,
        gt_dir=join(cfg.in_dir, 'PSMA', 'TTB'),
        output_csv=join(pred_dir, "psma_metrics.csv"),
        label_value=1
    )
    
    # Evaluate FDG predictions
    print("\nEvaluating FDG predictions:")
    evaluate_predictions(
        pred_dir=fdg_output_dir,
        gt_dir=join(cfg.in_dir, 'FDG', 'TTB'),
        output_csv=join(pred_dir, "fdg_metrics.csv"),
        label_value=1
    )
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    # Default to evaluating fold 0
    evaluate_multi_head_model(fold=0)
