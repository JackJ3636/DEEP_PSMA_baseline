import os
from os.path import join
import nnunet_configs_paths_early_fusion as nnunet_config_paths

task_number = '805'
folds = ['0', '1', '2', '3', '4']

for fold in folds:
    final_result_fname = join(
        nnunet_config_paths.nn_results_dir,
        f'Dataset{task_number}_EarlyFusion',
        'nnUNetTrainer__nnUNetPlans__3d_fullres',
        f'fold_{fold}',
        'checkpoint_final.pth'
    )

    if not os.path.exists(final_result_fname):
        os.system(f'nnUNetv2_train {task_number} 3d_fullres {fold}')
    else:
        print(f'Model already trained for fold {fold}')
