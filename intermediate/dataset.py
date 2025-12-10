#!/usr/bin/env python3
"""
Dataset for paired PSMA/FDG PET/CT data
Handles loading, normalization, and augmentation
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from typing import Dict, List, Optional, Tuple


class PairedPETCTDataset(Dataset):
    """
    Dataset for paired PSMA/FDG PET/CT volumes
    
    Expected structure:
    root/
        imagesTr/
            case_0000.nii.gz (PSMA PET)
            case_0001.nii.gz (PSMA CT)
            case_0002.nii.gz (FDG PET)
            case_0003.nii.gz (FDG CT)
        labelsTr/
            case.nii.gz (5-class segmentation)
        dataset.json
    """
    
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 fold: Optional[int] = None,
                 splits_file: Optional[str] = None,
                 patch_size: Optional[Tuple[int, int, int]] = (96, 160, 96),
                 samples_per_epoch: int = 250,
                 foreground_ratio: float = 0.33,
                 augment: bool = True,
                 normalize_ct: bool = True,
                 normalize_pet_to_suv: bool = True,
                 cache_data: bool = False):
        
        self.root_dir = root_dir
        self.split = split
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch if split == 'train' else 1
        self.foreground_ratio = foreground_ratio
        self.augment = augment and split == 'train'
        self.normalize_ct = normalize_ct
        self.normalize_pet_to_suv = normalize_pet_to_suv
        self.cache_data = cache_data
        self.cache = {}
        
        # Load dataset metadata
        with open(os.path.join(root_dir, 'dataset.json'), 'r') as f:
            self.metadata = json.load(f)
        
        # Get case IDs for this split
        self.case_ids = self._get_split_cases(fold, splits_file)
        
        # Load thresholds if available
        self.thresholds = self._load_thresholds()
        
        print(f"Dataset initialized: {len(self.case_ids)} cases for {split}")
        
    def _get_split_cases(self, fold: Optional[int], splits_file: Optional[str]) -> List[str]:
        """Get case IDs for current split/fold"""
        all_cases = [item['image'].split('/')[-1].replace('_0000.nii.gz', '')
                     for item in self.metadata['training']]
        
        if splits_file and os.path.exists(splits_file):
            with open(splits_file, 'r') as f:
                splits = json.load(f)
            
            if fold is not None:
                if self.split == 'train':
                    cases = []
                    for f in range(5):
                        if f != fold:
                            cases.extend(splits[f'fold_{f}'])
                    return cases
                else:
                    return splits[f'fold_{fold}']
        
        # Default split (80/20)
        random.seed(42)
        random.shuffle(all_cases)
        split_idx = int(0.8 * len(all_cases))
        
        if self.split == 'train':
            return all_cases[:split_idx]
        else:
            return all_cases[split_idx:]
    
    def _load_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load SUV thresholds for normalization"""
        thresholds = {}
        
        # Try to load from standard locations
        psma_thr_dir = os.path.join(os.path.dirname(self.root_dir), 'PSMA', 'thresholds')
        fdg_thr_dir = os.path.join(os.path.dirname(self.root_dir), 'FDG', 'thresholds')
        
        for case_id in self.case_ids:
            thresholds[case_id] = {'psma': 3.0, 'fdg': 2.5}  # Defaults
            
            # Load PSMA threshold
            psma_file = os.path.join(psma_thr_dir, f'{case_id}.json')
            if os.path.exists(psma_file):
                try:
                    with open(psma_file, 'r') as f:
                        data = json.load(f)
                        thresholds[case_id]['psma'] = data.get('threshold', 3.0)
                except:
                    pass
            
            # Load FDG threshold
            fdg_file = os.path.join(fdg_thr_dir, f'{case_id}.json')
            if os.path.exists(fdg_file):
                try:
                    with open(fdg_file, 'r') as f:
                        data = json.load(f)
                        thresholds[case_id]['fdg'] = data.get('threshold', 2.5)
                except:
                    pass
        
        return thresholds
    
    def _load_volume(self, case_id: str) -> Dict[str, np.ndarray]:
        """Load all volumes for a case"""
        if self.cache_data and case_id in self.cache:
            return self.cache[case_id]
        
        img_dir = os.path.join(self.root_dir, 'imagesTr')
        lbl_dir = os.path.join(self.root_dir, 'labelsTr')
        
        # Load images
        psma_pet = nib.load(os.path.join(img_dir, f'{case_id}_0000.nii.gz')).get_fdata().astype(np.float32)
        psma_ct = nib.load(os.path.join(img_dir, f'{case_id}_0001.nii.gz')).get_fdata().astype(np.float32)
        fdg_pet = nib.load(os.path.join(img_dir, f'{case_id}_0002.nii.gz')).get_fdata().astype(np.float32)
        fdg_ct = nib.load(os.path.join(img_dir, f'{case_id}_0003.nii.gz')).get_fdata().astype(np.float32)
        label = nib.load(os.path.join(lbl_dir, f'{case_id}.nii.gz')).get_fdata().astype(np.int64)
        
        # Normalize PET to SUV threshold
        if self.normalize_pet_to_suv:
            psma_thr = self.thresholds[case_id]['psma']
            fdg_thr = self.thresholds[case_id]['fdg']
            psma_pet = psma_pet / psma_thr
            fdg_pet = fdg_pet / fdg_thr
        
        # Normalize CT (z-score with clipping)
        if self.normalize_ct:
            psma_ct = self._normalize_ct(psma_ct)
            fdg_ct = self._normalize_ct(fdg_ct)
        
        data = {
            'psma_pet': psma_pet,
            'psma_ct': psma_ct,
            'fdg_pet': fdg_pet,
            'fdg_ct': fdg_ct,
            'label': label
        }
        
        if self.cache_data:
            self.cache[case_id] = data
        
        return data
    
    def _normalize_ct(self, ct: np.ndarray) -> np.ndarray:
        """Normalize CT with robust z-score"""
        # Clip to [-1000, 1000] HU
        ct = np.clip(ct, -1000, 1000)
        
        # Compute statistics on non-air voxels
        mask = ct > -500
        if mask.any():
            mean = ct[mask].mean()
            std = ct[mask].std()
            if std > 1e-6:
                ct = (ct - mean) / std
            else:
                ct = ct - mean
        
        return ct
    
    def _random_crop(self, volumes: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract random patch from volumes"""
        label = volumes['label']
        shape = label.shape
        
        if self.patch_size is None:
            return volumes
        
        # Ensure patch fits
        patch_size = [min(p, s) for p, s in zip(self.patch_size, shape)]
        
        # Decide if foreground-focused
        if random.random() < self.foreground_ratio and (label > 0).any():
            # Sample around foreground
            fg_coords = np.argwhere(label > 0)
            center = fg_coords[random.randint(0, len(fg_coords) - 1)]
            
            # Calculate crop boundaries
            starts = []
            for i in range(3):
                start = center[i] - patch_size[i] // 2
                start = max(0, min(start, shape[i] - patch_size[i]))
                starts.append(start)
        else:
            # Random crop
            starts = [random.randint(0, s - p) for s, p in zip(shape, patch_size)]
        
        # Apply crop to all volumes
        slices = tuple(slice(s, s + p) for s, p in zip(starts, patch_size))
        cropped = {}
        for key, vol in volumes.items():
            cropped[key] = vol[slices]
        
        return cropped
    
    def _augment(self, volumes: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply data augmentation"""
        if not self.augment:
            return volumes
        
        # Random flips
        for axis in range(3):
            if random.random() < 0.5:
                for key in volumes:
                    volumes[key] = np.flip(volumes[key], axis=axis).copy()
        
        # Random rotation (90 degree increments)
        if random.random() < 0.3:
            k = random.randint(1, 3)
            axes = random.choice([(0, 1), (0, 2), (1, 2)])
            for key in volumes:
                volumes[key] = np.rot90(volumes[key], k=k, axes=axes).copy()
        
        # Intensity augmentation for PET
        if random.random() < 0.3:
            scale = random.uniform(0.9, 1.1)
            shift = random.uniform(-0.1, 0.1)
            for key in ['psma_pet', 'fdg_pet']:
                volumes[key] = volumes[key] * scale + shift
        
        return volumes
    
    def __len__(self):
        if self.split == 'train':
            return self.samples_per_epoch
        else:
            return len(self.case_ids)
    
    def __getitem__(self, idx):
        # Select case
        if self.split == 'train':
            case_id = random.choice(self.case_ids)
        else:
            case_id = self.case_ids[idx % len(self.case_ids)]
        
        # Load volume
        volumes = self._load_volume(case_id)
        
        # Extract patch (training) or keep full (validation)
        if self.split == 'train' and self.patch_size is not None:
            volumes = self._random_crop(volumes)
            volumes = self._augment(volumes)
        
        # Convert to tensors
        psma = torch.stack([
            torch.from_numpy(volumes['psma_pet']).float(),
            torch.from_numpy(volumes['psma_ct']).float()
        ])
        
        fdg = torch.stack([
            torch.from_numpy(volumes['fdg_pet']).float(),
            torch.from_numpy(volumes['fdg_ct']).float()
        ])
        
        label = torch.from_numpy(volumes['label']).long()
        
        return {
            'psma': psma,
            'fdg': fdg,
            'label': label,
            'case_id': case_id
        }
