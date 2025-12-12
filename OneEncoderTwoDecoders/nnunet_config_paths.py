# nnunet_config_paths.py
import os

# Base directories for nnU-Net data (all under this project root to avoid permission issues)
project_root = os.path.dirname(os.path.abspath(__file__))
nn_raw_dir          = os.path.join(project_root, 'nnUNet_raw_data')
nn_preprocessed_dir = os.path.join(project_root, 'nnUNet_preprocessed')
nn_results_dir      = os.path.join(project_root, 'nnUNet_results')

# Map your tracers to nnU-Net dataset IDs
dataset_dictionary = {'EarlyFusion': 805}