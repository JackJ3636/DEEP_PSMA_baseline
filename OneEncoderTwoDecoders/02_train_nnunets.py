import os
from os.path import join
import nnunet_config_paths as cfg

# Set nnUNet environment variables properly
os.environ['nnUNet_raw'] = cfg.nn_raw_dir
os.environ['nnUNet_preprocessed'] = cfg.nn_preprocessed_dir  
os.environ['nnUNet_results'] = cfg.nn_results_dir

# Train the early fusion model with dual decoder using our custom trainer
task_id = cfg.dataset_dictionary['EarlyFusion']

for fold in ['0', '1', '2', '3', '4']:
    ckpt = join(cfg.nn_results_dir,
                f"Dataset{task_id}_EarlyFusion/MultiHeadnnUNetTrainer__nnUNetPlans__3d_fullres/fold_{fold}/checkpoint_final.pth")
    if not os.path.isfile(ckpt):
        print(f"Training early fusion model for fold {fold} with multi-head trainer...")
        # Use our custom trainer
        os.system(f"nnUNetv2_train {task_id} 3d_fullres {fold} -tr MultiHeadnnUNetTrainer")