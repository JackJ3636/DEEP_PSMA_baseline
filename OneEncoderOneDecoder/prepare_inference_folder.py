import os
import SimpleITK as sitk
import json
from os.path import join

input_dataset_folder = '/well/papiez/users/kqe223/DEEP-PSMA_CHALLENGE_DATA/New/data4'
inference_output_folder = '/well/papiez/users/kqe223/DEEP-PSMA_CHALLENGE_DATA/New/inference_input'

os.makedirs(inference_output_folder, exist_ok=True)

for case in os.listdir(join(input_dataset_folder, 'PSMA', 'CT')):
    print(f'Preparing inference case: {case}')

    ct_psma = sitk.ReadImage(join(input_dataset_folder, 'PSMA', 'CT', case))
    ct_fdg = sitk.ReadImage(join(input_dataset_folder, 'FDG', 'CT', case))
    pet_psma = sitk.ReadImage(join(input_dataset_folder, 'PSMA', 'PET', case))
    pet_fdg = sitk.ReadImage(join(input_dataset_folder, 'FDG', 'PET', case))

    with open(join(input_dataset_folder, 'PSMA', 'thresholds', case.replace('.nii.gz', '.json')), 'r') as f:
        suv_threshold_psma = json.load(f)['suv_threshold']
    with open(join(input_dataset_folder, 'FDG', 'thresholds', case.replace('.nii.gz', '.json')), 'r') as f:
        suv_threshold_fdg = json.load(f)['suv_threshold']

    pet_psma_rescaled = pet_psma / suv_threshold_psma
    pet_fdg_rescaled = pet_fdg / suv_threshold_fdg

    # Resample all images to PSMA PET space using identity transform
    reference = pet_psma_rescaled
    identity = sitk.Transform(3, sitk.sitkIdentity)

    ct_psma_resampled = sitk.Resample(ct_psma, reference, identity, sitk.sitkLinear, -1000)
    ct_fdg_resampled = sitk.Resample(ct_fdg, reference, identity, sitk.sitkLinear, -1000)
    pet_fdg_resampled = sitk.Resample(pet_fdg_rescaled, reference, identity, sitk.sitkLinear, 0)

    # Write each channel as a separate file, as nnU-Net expects for inference
    base = case.replace('.nii.gz', '')
    sitk.WriteImage(pet_psma_rescaled, join(inference_output_folder, f"{base}_0000.nii.gz"))
    sitk.WriteImage(pet_fdg_resampled, join(inference_output_folder, f"{base}_0001.nii.gz"))
    sitk.WriteImage(ct_psma_resampled, join(inference_output_folder, f"{base}_0002.nii.gz"))
    sitk.WriteImage(ct_fdg_resampled, join(inference_output_folder, f"{base}_0003.nii.gz"))

print("Inference folder prepared at:", inference_output_folder)