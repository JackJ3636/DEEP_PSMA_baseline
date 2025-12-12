#!/usr/bin/env python3
import os
import json
import SimpleITK as sitk
from os.path import join
import nnunet_configs_paths_early_fusion as nnunet_config_paths

# -----------------------------------------------------------------------------
# Paths & folders
# -----------------------------------------------------------------------------
input_dataset_folder = '/well/papiez/users/kqe223/DEEP-PSMA_CHALLENGE_DATA/New/data4'  # top-level data folder
raw_root             = nnunet_config_paths.nn_raw_dir                           # nnU-Net raw data root
dataset_name         = 'Dataset805_EarlyFusion'                                  # folder name under raw_root
output_folder        = join(raw_root, dataset_name)

images_tr_folder = join(output_folder, 'imagesTr')
labels_tr_folder = join(output_folder, 'labelsTr')

os.makedirs(images_tr_folder, exist_ok=True)
os.makedirs(labels_tr_folder, exist_ok=True)

# -----------------------------------------------------------------------------
# Loop over cases
# -----------------------------------------------------------------------------
for case in sorted(os.listdir(join(input_dataset_folder, 'PSMA', 'CT'))):
    print(f'Processing {case}…')

    # Read modalities
    ct_psma  = sitk.ReadImage(join(input_dataset_folder, 'PSMA', 'CT', case))
    pet_psma = sitk.ReadImage(join(input_dataset_folder, 'PSMA', 'PET', case))
    ct_fdg   = sitk.ReadImage(join(input_dataset_folder, 'FDG', 'CT', case))
    pet_fdg  = sitk.ReadImage(join(input_dataset_folder, 'FDG', 'PET', case))
    lbl      = sitk.ReadImage(join(input_dataset_folder, 'PSMA', 'TTB', case))

    # Load per-case SUV thresholds
    with open(join(input_dataset_folder, 'PSMA', 'thresholds', case.replace('.nii.gz', '.json')), 'r') as f:
        suv_thr_psma = json.load(f)['suv_threshold']
    with open(join(input_dataset_folder, 'FDG', 'thresholds', case.replace('.nii.gz', '.json')), 'r') as f:
        suv_thr_fdg  = json.load(f)['suv_threshold']

    # Normalize PET volumes
    pet_psma_norm = sitk.Cast(pet_psma / suv_thr_psma, sitk.sitkFloat32)
    pet_fdg_norm  = sitk.Cast(pet_fdg  / suv_thr_fdg,  sitk.sitkFloat32)

    # Resample everything into PSMA-PET space
    def resample_to_ref(img, ref):
        return sitk.Resample(
            img, ref,
            sitk.TranslationTransform(3),
            sitk.sitkLinear,
            0.0  # background fill
        )

    ct_psma_rs = resample_to_ref(ct_psma,      pet_psma)
    ct_fdg_rs  = resample_to_ref(ct_fdg,       pet_psma)
    pet_fdg_rs = resample_to_ref(pet_fdg_norm, pet_psma)

    # Write out the 4 channels
    base = case.replace('.nii.gz', '')
    sitk.WriteImage(pet_psma_norm, join(images_tr_folder, f"{base}_0000.nii.gz"))
    sitk.WriteImage(pet_fdg_rs,    join(images_tr_folder, f"{base}_0001.nii.gz"))
    sitk.WriteImage(ct_psma_rs,    join(images_tr_folder, f"{base}_0002.nii.gz"))
    sitk.WriteImage(ct_fdg_rs,     join(images_tr_folder, f"{base}_0003.nii.gz"))
    # And the label
    sitk.WriteImage(lbl,           join(labels_tr_folder,  case))

# -----------------------------------------------------------------------------
# Write dataset.json in nnU-Net v2 format
# -----------------------------------------------------------------------------
dataset_json = {
    "name":        dataset_name,
    "description": "Early-fusion PSMA + FDG PET/CT – 4 channels",
    "channel_names": {
        "0": "PSMA_PET",
        "1": "FDG_PET",
        "2": "PSMA_CT",
        "3": "FDG_CT"
    },
    "labels": {
        "background": 0,
        "ttb":        1,
        "norm":       2
    },
    "numTraining": len(os.listdir(labels_tr_folder)),
    "file_ending": ".nii.gz"
}

with open(join(output_folder, 'dataset.json'), 'w') as f:
    json.dump(dataset_json, f, indent=4)

# -----------------------------------------------------------------------------
# Run nnU-Net plan & preprocess
# -----------------------------------------------------------------------------
dataset_id = 805  # numeric ID matching Dataset805_EarlyFusion
cmd = (
    f"nnUNetv2_plan_and_preprocess "
    f"-d {dataset_id} "
    f"-c 3d_fullres "
    f"--verify_dataset_integrity "
    f"--verbose"
)
print("Calling:", cmd)
os.system(cmd)
