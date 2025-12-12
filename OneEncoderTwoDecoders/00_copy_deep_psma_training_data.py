import os
import shutil
from os.path import join


raw_root  = '/well/papiez/users/kqe223/DEEP-PSMA_CHALLENGE_DATA/New/early_fusion_v2/DEEP-PSMA_CHALLENGE_DATA/CHALLENGE_DATA'
data_root = '/well/papiez/users/kqe223/DEEP-PSMA_CHALLENGE_DATA/New/early_fusion_v2/data'

# create folder structure (correctly substitute tracer and modal)
dir_structure = [
    f"{data_root}/{tracer}/{modal}" 
    for tracer in ['PSMA', 'FDG'] 
    for modal in ['CT', 'PET', 'TTB', 'thresholds']
]
for path in dir_structure:
    os.makedirs(path, exist_ok=True)

# copy files from raw to data
for case in os.listdir(raw_root):
    for tracer in ['PSMA', 'FDG']:
        src_dir = join(raw_root, case, tracer)
        if not os.path.isdir(src_dir):
            continue
        shutil.copy(
            join(src_dir, 'CT.nii.gz'),
            join(data_root, tracer, 'CT', f"{case}.nii.gz")
        )
        shutil.copy(
            join(src_dir, 'PET.nii.gz'),
            join(data_root, tracer, 'PET', f"{case}.nii.gz")
        )
        shutil.copy(
            join(src_dir, 'TTB.nii.gz'),
            join(data_root, tracer, 'TTB', f"{case}.nii.gz")
        )
        shutil.copy(
            join(src_dir, 'threshold.json'),
            join(data_root, tracer, 'thresholds', f"{case}.json")
        )