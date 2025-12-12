import os
import SimpleITK as sitk
import json
import nnunet_configs_paths_early_fusion as nnunet_config_paths
from os.path import join
import shutil

input_dataset_folder = '/well/papiez/users/kqe223/DEEP-PSMA_CHALLENGE_DATA/New/data4'

training_output_location = nnunet_config_paths.nn_raw_dir
output_folder = join(training_output_location, 'Dataset805_EarlyFusion')

os.makedirs(join(output_folder, 'imagesTr'), exist_ok=True)
os.makedirs(join(output_folder, 'labelsTr'), exist_ok=True)

for case in os.listdir(join(input_dataset_folder, 'PSMA', 'CT')):
    print(f'Processing case: {case}')

    ct_psma = sitk.ReadImage(join(input_dataset_folder, 'PSMA', 'CT', case))
    ct_fdg = sitk.ReadImage(join(input_dataset_folder, 'FDG', 'CT', case))
    pet_psma = sitk.ReadImage(join(input_dataset_folder, 'PSMA', 'PET', case))
    pet_fdg = sitk.ReadImage(join(input_dataset_folder, 'FDG', 'PET', case))
    ttb_label = sitk.ReadImage(join(input_dataset_folder, 'PSMA', 'TTB', case))

    with open(join(input_dataset_folder, 'PSMA', 'thresholds', case.replace('.nii.gz', '.json')), 'r') as f:
        suv_threshold_psma = json.load(f)['suv_threshold']
    with open(join(input_dataset_folder, 'FDG', 'thresholds', case.replace('.nii.gz', '.json')), 'r') as f:
        suv_threshold_fdg = json.load(f)['suv_threshold']

    pet_psma_rescaled = pet_psma / suv_threshold_psma
    pet_fdg_rescaled = pet_fdg / suv_threshold_fdg

    # Resample both CTs to PSMA PET space
    ct_psma_resampled = sitk.Resample(ct_psma, pet_psma, sitk.TranslationTransform(3), sitk.sitkLinear, -1000)
    ct_fdg_resampled = sitk.Resample(ct_fdg, pet_psma, sitk.TranslationTransform(3), sitk.sitkLinear, -1000)

    sitk.WriteImage(pet_psma_rescaled, join(output_folder, 'imagesTr', case.replace('.nii.gz', '_0000.nii.gz')))
    sitk.WriteImage(pet_fdg_rescaled, join(output_folder, 'imagesTr', case.replace('.nii.gz', '_0001.nii.gz')))
    sitk.WriteImage(ct_psma_resampled, join(output_folder, 'imagesTr', case.replace('.nii.gz', '_0002.nii.gz')))
    sitk.WriteImage(ct_fdg_resampled, join(output_folder, 'imagesTr', case.replace('.nii.gz', '_0003.nii.gz')))
    sitk.WriteImage(ttb_label, join(output_folder, 'labelsTr', case))

json_dict = {
    'channel_names': {'0': 'PSMA_PET', '1': 'FDG_PET', '2': 'PSMA_CT', '3': 'FDG_CT'},
    'labels': {'0': 'background', '1': 'ttb', '2': 'norm'},
    'numTraining': len(os.listdir(join(output_folder, 'labelsTr'))),
    'file_ending': '.nii.gz'
}

with open(join(output_folder, 'dataset.json'), 'w') as f:
    json.dump(json_dict, f, indent=4)

os.system('nnUNetv2_plan_and_preprocess -d 805 -c 3d_fullres --verify_dataset_integrity')
