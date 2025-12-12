import json, os, pandas as pd
from os.path import join
import nnunet_configs_paths_early_fusion as nnunet_config_paths

input_dataset_folder = 'data'
training_output_location = nnunet_config_paths.nn_preprocessed_dir
output_folder = join(training_output_location, 'Dataset805_EarlyFusion')

with open(join(output_folder, "splits_final.json"), 'r') as file:
    splits_final = json.load(file)

df = pd.DataFrame(columns=['case', 'val_fold'])
for idx, split in enumerate(splits_final):
    for case in split['val']:
        df.loc[len(df)] = [case, idx]

df.to_csv(join(input_dataset_folder, 'EarlyFusion_validation_folds.csv'))
