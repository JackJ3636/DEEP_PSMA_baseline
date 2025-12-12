import os, json, pandas as pd, SimpleITK as sitk
from os.path import join
import nnunet_config_paths as cfg
from DEEP_PSMA_Infer import run_inference
from evaluation import dice_score

def baseline_dice(gt, pr):
    import numpy as np
    return 2*np.sum((gt>0)&(pr>0))/(np.sum(gt>0)+np.sum(pr>0))

for tracer, task_id in cfg.dataset_dictionary.items():
    df_val = pd.read_csv(join('data',tracer,'validation_folds.csv'))
    out    = []
    for _, row in df_val.iterrows():
        case = row['case']; fold = int(row['val_fold'])
        pt, ct = join('data',tracer,'PET',f"{case}.nii.gz"), join('data',tracer,'CT', f"{case}.nii.gz")
        gt     = sitk.ReadImage(join('data',tracer,'TTB',f"{case}.nii.gz"))
        thr    = json.load(open(join('data',tracer,'thresholds',f"{case}.json")))['suv_threshold']
        nn_pred, bs_pred = run_inference(pt, ct, tracer, fold, thr, return_sitk=True)
        nn_arr  = sitk.GetArrayFromImage(nn_pred)
        bs_arr  = sitk.GetArrayFromImage(bs_pred)
        gt_arr  = sitk.GetArrayFromImage(gt)
        out.append({
            'case': case,
            'fold': fold,
            'nn_dice': dice_score(gt_arr, nn_arr),
            'baseline_dice': baseline_dice(gt_arr, bs_arr)
        })
    pd.DataFrame(out).to_csv(join('data',tracer,'inferred_dice_scores.csv'), index=False)