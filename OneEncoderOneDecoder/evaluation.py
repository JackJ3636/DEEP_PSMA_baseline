import os
import SimpleITK as sitk
import numpy as np
import cc3d
from functools import partial

def con_comp(seg_array):
    connectivity = 18
    return cc3d.connected_components(seg_array, connectivity=connectivity)

def false_pos_pix(gt_array, pred_array):
    pred_conn_comp = con_comp(pred_array)
    false_pos = sum(np.isin(pred_conn_comp, idx).sum() for idx in range(1, pred_conn_comp.max() + 1) 
                    if np.sum(gt_array[pred_conn_comp == idx]) == 0)
    return false_pos

def false_neg_pix(gt_array, pred_array):
    gt_conn_comp = con_comp(gt_array)
    false_neg = sum(np.isin(gt_conn_comp, idx).sum() for idx in range(1, gt_conn_comp.max() + 1) 
                    if np.sum(pred_array[gt_conn_comp == idx]) == 0)
    return false_neg

def dice_score(mask1, mask2):
    overlap = np.sum(mask1 * mask2)
    total = np.sum(mask1) + np.sum(mask2)
    return 2 * overlap / total if total else 1.0

distance_map = partial(sitk.SignedMaurerDistanceMap, squaredDistance=False, useImageSpacing=True)

def get_surface_dice(gold, prediction):
    gold_surface = sitk.LabelContour(gold == 1, False)
    prediction_surface = sitk.LabelContour(prediction == 1, False)

    prediction_distance_map = sitk.Abs(distance_map(prediction_surface))
    gold_distance_map = sitk.Abs(distance_map(gold_surface))

    gold_to_prediction = sitk.GetArrayViewFromImage(prediction_distance_map)[sitk.GetArrayViewFromImage(gold_surface) == 1]

    matching_surface_voxels = np.sum(gold_to_prediction == 0)
    gold_surface_voxels = np.sum(sitk.GetArrayViewFromImage(gold_surface) == 1)
    prediction_surface_voxels = np.sum(sitk.GetArrayViewFromImage(prediction_surface) == 1)

    surface_dice = (2. * matching_surface_voxels) / (gold_surface_voxels + prediction_surface_voxels) if (gold_surface_voxels + prediction_surface_voxels) else 1.0
    return surface_dice

def read_label(label_path):
    label = sitk.ReadImage(label_path)
    voxel_volume = np.prod(label.GetSpacing()) / 1000.
    return sitk.GetArrayFromImage(label), voxel_volume

def score_labels(gt_path, pred_path, pet_path):
    gt_ar, voxel_volume = read_label(gt_path)
    pred_ar, _ = read_label(pred_path)
    pet_ar, _ = read_label(pet_path)

    dice = dice_score(gt_ar, pred_ar)
    fp_vol = false_pos_pix(gt_ar, pred_ar) * voxel_volume
    fn_vol = false_neg_pix(gt_ar, pred_ar) * voxel_volume
    surf_dice = get_surface_dice(sitk.ReadImage(gt_path), sitk.ReadImage(pred_path))
    suv_mean_ratio = pet_ar[pred_ar > 0].mean() / pet_ar[gt_ar > 0].mean() if np.any(gt_ar > 0) and np.any(pred_ar > 0) else np.nan
    ttb_vol_ratio = pred_ar.sum() / gt_ar.sum() if np.sum(gt_ar) > 0 else np.nan

    return dice, fp_vol, fn_vol, surf_dice, suv_mean_ratio, ttb_vol_ratio

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python evaluation.py <labels_folder> <predictions_folder> <pet_folder>")
        sys.exit(1)

    labels_folder = sys.argv[1]
    predictions_folder = sys.argv[2]
    pet_folder = sys.argv[3]

    print("case,dice,fp_vol,fn_vol,surface_dice,suv_mean_ratio,ttb_vol_ratio")
    for case in sorted(os.listdir(labels_folder)):
        if not case.endswith(".nii.gz"):
            continue
        gt_path = os.path.join(labels_folder, case)
        pred_path = os.path.join(predictions_folder, case)
        pet_path = os.path.join(pet_folder, case)  # Adjust if PET naming differs

        if not (os.path.exists(gt_path) and os.path.exists(pred_path) and os.path.exists(pet_path)):
            print(f"{case},MISSING_FILE")
            continue

        try:
            dice, fp_vol, fn_vol, surf_dice, suv_mean_ratio, ttb_vol_ratio = score_labels(gt_path, pred_path, pet_path)
            print(f"{case},{dice:.4f},{fp_vol:.2f},{fn_vol:.2f},{surf_dice:.4f},{suv_mean_ratio:.4f},{ttb_vol_ratio:.4f}")
        except Exception as e:
            print(f"{case},ERROR,{e}")
