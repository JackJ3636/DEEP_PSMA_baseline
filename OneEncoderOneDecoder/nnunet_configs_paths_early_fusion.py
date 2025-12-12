import os
from os.path import join

nnunet_data_folder = join(os.path.dirname(__file__), 'data', 'nnUNet_data')

nn_raw_dir = join(nnunet_data_folder, 'raw')
nn_preprocessed_dir = join(nnunet_data_folder, 'preprocessed')
nn_results_dir = join(nnunet_data_folder, 'results')

for path in [nn_raw_dir, nn_preprocessed_dir, nn_results_dir]:
    os.makedirs(path, exist_ok=True)

os.environ["nnUNet_raw"] = nn_raw_dir
os.environ["nnUNet_preprocessed"] = nn_preprocessed_dir
os.environ["nnUNet_results"] = nn_results_dir

dataset_dictionary = {'EarlyFusion': 805}
