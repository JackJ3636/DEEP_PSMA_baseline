import json, pandas as pd
from os.path import join
import nnunet_config_paths as cfg

for tracer, task_id in cfg.dataset_dictionary.items():
    pre_dir = join(cfg.nn_preprocessed_dir, f"Dataset{task_id}_{tracer}_PET")
    splits  = json.load(open(join(pre_dir,'splits_final.json')))
    records = []
    for fold, s in enumerate(splits):
        for case in s['val']:
            records.append({'case':case,'val_fold':fold})
    df = pd.DataFrame(records)
    df.to_csv(join('data',tracer,'validation_folds.csv'), index=False)