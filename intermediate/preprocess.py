#!/usr/bin/env python3
"""
Preprocessing script to create paired PSMA/FDG dataset
Combines PSMA and FDG PET/CT into 4-channel volumes with 5-class labels
"""

import os
import json
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
import SimpleITK as sitk
from typing import Dict, Tuple


def resample_to_reference(moving: sitk.Image, reference: sitk.Image, 
                          interpolator=sitk.sitkLinear, default_value=0) -> sitk.Image:
    """Resample moving image to reference space"""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetOutputSpacing(reference.GetSpacing())
    resampler.SetOutputOrigin(reference.GetOrigin())
    resampler.SetOutputDirection(reference.GetDirection())
    resampler.SetSize(reference.GetSize())
    return resampler.Execute(moving)


def load_threshold(threshold_file: str, default: float = 3.0) -> float:
    """Load SUV threshold from JSON file"""
    if not os.path.exists(threshold_file):
        return default
    
    try:
        with open(threshold_file, 'r') as f:
            data = json.load(f)
        
        # Try different possible keys
        for key in ['threshold', 'suv_threshold', 'SUV_threshold', 'value']:
            if key in data:
                return float(data[key])
        
        # Handle liver-based thresholds
        if 'liver_mean' in data and 'liver_sd' in data:
            return float(data['liver_mean']) + 2.0 * float(data['liver_sd'])
        
        return default
    except:
        return default


def create_paired_case(case_id: str, input_dir: str, output_dir: str) -> bool:
    """Process a single case to create paired 4-channel input"""
    
    # Define input paths
    psma_pet_path = os.path.join(input_dir, 'PSMA', 'PET', f'{case_id}.nii.gz')
    psma_ct_path = os.path.join(input_dir, 'PSMA', 'CT', f'{case_id}.nii.gz')
    psma_ttb_path = os.path.join(input_dir, 'PSMA', 'TTB', f'{case_id}.nii.gz')
    psma_thr_path = os.path.join(input_dir, 'PSMA', 'thresholds', f'{case_id}.json')
    
    fdg_pet_path = os.path.join(input_dir, 'FDG', 'PET', f'{case_id}.nii.gz')
    fdg_ct_path = os.path.join(input_dir, 'FDG', 'CT', f'{case_id}.nii.gz')
    fdg_ttb_path = os.path.join(input_dir, 'FDG', 'TTB', f'{case_id}.nii.gz')
    fdg_thr_path = os.path.join(input_dir, 'FDG', 'thresholds', f'{case_id}.json')
    
    # Check if all files exist
    required_files = [
        psma_pet_path, psma_ct_path, psma_ttb_path,
        fdg_pet_path, fdg_ct_path, fdg_ttb_path
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Missing file: {file_path}")
            return False
    
    try:
        # Load images using SimpleITK for resampling
        psma_pet = sitk.ReadImage(psma_pet_path)
        psma_ct = sitk.ReadImage(psma_ct_path)
        psma_ttb = sitk.ReadImage(psma_ttb_path)
        
        fdg_pet = sitk.ReadImage(fdg_pet_path)
        fdg_ct = sitk.ReadImage(fdg_ct_path)
        fdg_ttb = sitk.ReadImage(fdg_ttb_path)
        
        # Load thresholds
        psma_threshold = load_threshold(psma_thr_path, default=3.0)
        fdg_threshold = load_threshold(fdg_thr_path, default=2.5)
        
        # Use PSMA PET as reference space
        reference = psma_pet
        
        # Resample everything to PSMA PET space
        fdg_pet_resampled = resample_to_reference(fdg_pet, reference, sitk.sitkLinear, 0)
        fdg_ct_resampled = resample_to_reference(fdg_ct, reference, sitk.sitkLinear, -1000)
        fdg_ttb_resampled = resample_to_reference(fdg_ttb, reference, sitk.sitkNearestNeighbor, 0)
        
        psma_ct_resampled = resample_to_reference(psma_ct, reference, sitk.sitkLinear, -1000)
        psma_ttb_resampled = resample_to_reference(psma_ttb, reference, sitk.sitkNearestNeighbor, 0)
        
        # Convert to numpy arrays
        psma_pet_arr = sitk.GetArrayFromImage(psma_pet).astype(np.float32)
        psma_ct_arr = sitk.GetArrayFromImage(psma_ct_resampled).astype(np.float32)
        psma_ttb_arr = sitk.GetArrayFromImage(psma_ttb_resampled).astype(np.uint8)
        
        fdg_pet_arr = sitk.GetArrayFromImage(fdg_pet_resampled).astype(np.float32)
        fdg_ct_arr = sitk.GetArrayFromImage(fdg_ct_resampled).astype(np.float32)
        fdg_ttb_arr = sitk.GetArrayFromImage(fdg_ttb_resampled).astype(np.uint8)
        
        # Normalize PET by SUV threshold
        psma_pet_norm = psma_pet_arr / psma_threshold
        fdg_pet_norm = fdg_pet_arr / fdg_threshold
        
        # Create physiological uptake masks (PET >= 1.0 after normalization, excluding tumor)
        psma_normal = (psma_pet_norm >= 1.0) & (psma_ttb_arr == 0)
        fdg_normal = (fdg_pet_norm >= 1.0) & (fdg_ttb_arr == 0)
        
        # Create 5-class label
        label = np.zeros_like(psma_ttb_arr, dtype=np.uint8)
        label[psma_ttb_arr > 0] = 1  # PSMA tumor
        label[psma_normal] = 2  # PSMA normal
        label[fdg_ttb_arr > 0] = 3  # FDG tumor
        label[fdg_normal & (label == 0)] = 4  # FDG normal (only where not already labeled)
        
        # Save outputs
        img_dir = os.path.join(output_dir, 'imagesTr')
        lbl_dir = os.path.join(output_dir, 'labelsTr')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        
        # Get affine from reference
        affine = np.eye(4)
        affine[:3, :3] = np.array(reference.GetDirection()).reshape(3, 3) * np.array(reference.GetSpacing())
        affine[:3, 3] = np.array(reference.GetOrigin())
        
        # Save 4 channels separately (nnUNet style)
        nib.save(nib.Nifti1Image(psma_pet_norm, affine), 
                 os.path.join(img_dir, f'{case_id}_0000.nii.gz'))
        nib.save(nib.Nifti1Image(psma_ct_arr, affine), 
                 os.path.join(img_dir, f'{case_id}_0001.nii.gz'))
        nib.save(nib.Nifti1Image(fdg_pet_norm, affine), 
                 os.path.join(img_dir, f'{case_id}_0002.nii.gz'))
        nib.save(nib.Nifti1Image(fdg_ct_arr, affine), 
                 os.path.join(img_dir, f'{case_id}_0003.nii.gz'))
        
        # Save label
        nib.save(nib.Nifti1Image(label, affine), 
                 os.path.join(lbl_dir, f'{case_id}.nii.gz'))
        
        return True
        
    except Exception as e:
        print(f"Error processing {case_id}: {str(e)}")
        return False


def create_dataset_json(output_dir: str, case_ids: list) -> None:
    """Create dataset.json file for the paired dataset"""
    
    dataset_dict = {
        "channel_names": {
            "0": "PSMA_PET",
            "1": "PSMA_CT", 
            "2": "FDG_PET",
            "3": "FDG_CT"
        },
        "labels": {
            "background": 0,
            "PSMA_tumor": 1,
            "PSMA_normal": 2,
            "FDG_tumor": 3,
            "FDG_normal": 4
        },
        "numTraining": len(case_ids),
        "file_ending": ".nii.gz",
        "training": []
    }
    
    for case_id in case_ids:
        dataset_dict["training"].append({
            "image": f"imagesTr/{case_id}_0000.nii.gz",
            "label": f"labelsTr/{case_id}.nii.gz"
        })
    
    with open(os.path.join(output_dir, 'dataset.json'), 'w') as f:
        json.dump(dataset_dict, f, indent=2)


def create_splits_json(case_ids: list, output_file: str, n_folds: int = 5) -> None:
    """Create cross-validation splits"""
    
    # Shuffle with fixed seed for reproducibility
    np.random.seed(42)
    shuffled_ids = case_ids.copy()
    np.random.shuffle(shuffled_ids)
    
    # Create folds
    fold_size = len(shuffled_ids) // n_folds
    splits = {}
    
    for fold in range(n_folds):
        start_idx = fold * fold_size
        if fold == n_folds - 1:
            # Last fold gets remaining cases
            end_idx = len(shuffled_ids)
        else:
            end_idx = start_idx + fold_size
        
        splits[f'fold_{fold}'] = shuffled_ids[start_idx:end_idx]
    
    # Save splits
    with open(output_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"Created {n_folds}-fold splits with {len(case_ids)} cases")
    for fold in range(n_folds):
        print(f"  Fold {fold}: {len(splits[f'fold_{fold}'])} cases")


def main():
    parser = argparse.ArgumentParser(description='Preprocess DEEP-PSMA data into paired dataset')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Root directory containing PSMA and FDG subdirectories')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for paired dataset')
    parser.add_argument('--create_splits', action='store_true',
                        help='Create cross-validation splits')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of cross-validation folds')
    
    args = parser.parse_args()
    
    # Get list of cases (from PSMA CT directory)
    psma_ct_dir = os.path.join(args.input_dir, 'PSMA', 'CT')
    if not os.path.exists(psma_ct_dir):
        raise ValueError(f"PSMA CT directory not found: {psma_ct_dir}")
    
    case_ids = [f.replace('.nii.gz', '') for f in os.listdir(psma_ct_dir) 
                if f.endswith('.nii.gz')]
    case_ids.sort()
    
    print(f"Found {len(case_ids)} cases to process")
    
    # Process each case
    successful_cases = []
    failed_cases = []
    
    for case_id in tqdm(case_ids, desc='Processing cases'):
        if create_paired_case(case_id, args.input_dir, args.output_dir):
            successful_cases.append(case_id)
        else:
            failed_cases.append(case_id)
    
    print(f"\nProcessing complete:")
    print(f"  Successful: {len(successful_cases)} cases")
    print(f"  Failed: {len(failed_cases)} cases")
    
    if failed_cases:
        print(f"  Failed cases: {failed_cases}")
    
    # Create dataset.json
    create_dataset_json(args.output_dir, successful_cases)
    print(f"Created dataset.json with {len(successful_cases)} cases")
    
    # Create splits if requested
    if args.create_splits:
        splits_file = os.path.join(args.output_dir, 'splits.json')
        create_splits_json(successful_cases, splits_file, args.n_folds)
        print(f"Created splits file: {splits_file}")


if __name__ == '__main__':
    main()
