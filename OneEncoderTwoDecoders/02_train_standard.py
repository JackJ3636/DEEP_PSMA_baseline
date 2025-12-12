import os
from os.path import join
import nnunet_config_paths as cfg

# Set environment variables properly
os.environ['nnUNet_raw'] = cfg.nn_raw_dir
os.environ['nnUNet_preprocessed'] = cfg.nn_preprocessed_dir  
os.environ['nnUNet_results'] = cfg.nn_results_dirs
from os.path import join
import nnunet_config_paths as cfg

# Set nnUNet environment variables properly
os.environ['nnUNet_raw'] = cfg.nn_raw_data_dir
os.environ['nnUNet_preprocessed'] = cfg.nn_preprocessed_data_dir  
os.environ['nnUNet_results'] = cfg.nn_results_dir

print(f"nnUNet_raw: {os.environ['nnUNet_raw']}")
print(f"nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
print(f"nnUNet_results: {os.environ['nnUNet_results']}")

# Train with standard nnUNet trainer first to test
task_id = cfg.dataset_dictionary['EarlyFusion']

print(f"Starting training for Dataset{task_id}_EarlyFusion with standard nnUNetTrainer for fold 0...")
# Use standard trainer first
os.system(f"nnUNetv2_train {task_id} 3d_fullres 0")
