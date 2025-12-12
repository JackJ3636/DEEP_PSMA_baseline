import os
import SimpleITK as sitk
import shutil
import numpy as np
import subprocess
import time

nn_predict_exe = 'nnUNetv2_predict'  # Ensure this executable is in your PATH


def expand_contract_label(label, distance=5.0):
    """Expand or contract sitk label image by indicated distance."""
    label_single = sitk.BinaryThreshold(label, lowerThreshold=1, upperThreshold=10000, insideValue=1)
    distance_filter = sitk.SignedMaurerDistanceMapImageFilter()
    distance_filter.SetUseImageSpacing(True)
    distance_filter.SquaredDistanceOff()
    dmap = distance_filter.Execute(label_single)
    new_label_ar = (sitk.GetArrayFromImage(dmap) <= distance).astype(np.int16)
    new_label = sitk.GetImageFromArray(new_label_ar)
    new_label.CopyInformation(label)
    return new_label


def run_inference(pet_psma, pet_fdg, ct, tracer='EarlyFusion', output_fname='earlyfusion_inferred.nii.gz',
                  return_ttb_sitk=False, temp_dir='temp', fold='all', expansion_radius=7.0):

    start_time = time.time()

    pt_psma_suv = sitk.ReadImage(pet_psma) if isinstance(pet_psma, str) else pet_psma
    pt_fdg_suv = sitk.ReadImage(pet_fdg) if isinstance(pet_fdg, str) else pet_fdg
    ct_img = sitk.ReadImage(ct) if isinstance(ct, str) else ct

    os.makedirs(temp_dir, exist_ok=True)

    # Clear temporary directories
    for f in os.listdir(temp_dir):
        path = os.path.join(temp_dir, f)
        shutil.rmtree(path) if os.path.isdir(path) else os.unlink(path)

    # Write input images with proper channel numbering
    input_folder = os.path.join(temp_dir, 'nn_input')
    os.makedirs(input_folder, exist_ok=True)

    sitk.WriteImage(pt_psma_suv, os.path.join(input_folder, 'earlyfusion_0000.nii.gz'))  # PSMA PET
    sitk.WriteImage(pt_fdg_suv, os.path.join(input_folder, 'earlyfusion_0001.nii.gz'))   # FDG PET
    sitk.WriteImage(ct_img, os.path.join(input_folder, 'earlyfusion_0002.nii.gz'))       # CT

    output_folder = os.path.join(temp_dir, 'nn_output')

    # Call nnU-Net
    call = f"{nn_predict_exe} -i {input_folder} -o {output_folder} -d 805 -c 3d_fullres"
    if fold != 'all':
        call += f" -f {fold}"

    print(f"Running nnU-Net prediction with command: {call}")

    result = subprocess.run(call, shell=True, capture_output=True)
    print(result.stdout.decode(), result.stderr.decode())

    pred_label = sitk.ReadImage(os.path.join(output_folder, 'earlyfusion.nii.gz'))

    pred_ttb_ar = (sitk.GetArrayFromImage(pred_label) == 1).astype(np.int8)
    pred_norm_ar = (sitk.GetArrayFromImage(pred_label) == 2).astype(np.int8)

    pred_ttb_label = sitk.GetImageFromArray(pred_ttb_ar)
    pred_ttb_label.CopyInformation(pred_label)

    # Expand predicted TTB region
    pred_ttb_expanded = expand_contract_label(pred_ttb_label, distance=expansion_radius)
    pred_ttb_expanded_ar = sitk.GetArrayFromImage(pred_ttb_expanded)

    # Final refined output (excluding physiological/normal regions)
    output_ar = np.logical_and(pred_ttb_expanded_ar > 0, pred_norm_ar == 0).astype(np.int8)

    output_label = sitk.GetImageFromArray(output_ar)
    output_label.CopyInformation(pred_label)

    sitk.WriteImage(output_label, output_fname)

    print('Inference and post-processing completed in:', round(time.time() - start_time, 2), 'seconds')

    if return_ttb_sitk:
        return output_label
