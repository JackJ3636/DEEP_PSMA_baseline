#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import SimpleITK as sitk

# ---- nnU-Net v2 folder configuration ----
os.environ['nnUNet_raw']          = "/well/papiez/users/kqe223/DEEP-PSMA_CHALLENGE_DATA/New/early_fusion/data/nnUNet_data/raw"
os.environ['nnUNet_preprocessed'] = "/well/papiez/users/kqe223/DEEP-PSMA_CHALLENGE_DATA/New/early_fusion/data/nnUNet_data/preprocessed"
os.environ['nnUNet_results']      = "/well/papiez/users/kqe223/DEEP-PSMA_CHALLENGE_DATA/New/early_fusion/data/nnUNet_data/results"

# -----------------------------------------------------------------------------
# Configuration — edit these to match your environment
# -----------------------------------------------------------------------------
TEST_INPUT_FOLDER = '/well/papiez/users/kqe223/DEEP-PSMA_CHALLENGE_DATA/New/inference_input'
OUTPUT_DIR        = Path('temp/nn_output')
EXPECTED_FILENAME = 'earlyfusion.nii.gz'

# -----------------------------------------------------------------------------
# Prepare output directory
# -----------------------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 1) Run nnU-Net inference
# -----------------------------------------------------------------------------
dataset_id = 805
config_id  = '3d_fullres'

cmd = (
    f"nnUNetv2_predict "
    f"-i \"{TEST_INPUT_FOLDER}\" "
    f"-o \"{OUTPUT_DIR}\" "
    f"-d {dataset_id} "
    f"-c {config_id} "
    f"--disable_tta "
    f"--save_probabilities "
    f"--verbose"
)

print("Running inference:\n ", cmd)
ret = os.system(cmd)
if ret != 0:
    print("ERROR: nnUNetv2_predict failed with exit code", ret, file=sys.stderr)
    sys.exit(1)

# -----------------------------------------------------------------------------
# 2) Locate and verify the output file
# -----------------------------------------------------------------------------
candidates = list(OUTPUT_DIR.glob("*.nii.gz"))
if not candidates:
    print(f"ERROR: no .nii.gz files found in {OUTPUT_DIR}", file=sys.stderr)
    sys.exit(1)

# pick the expected filename if present
out_path = next((p for p in candidates if p.name == EXPECTED_FILENAME), candidates[0])
if out_path.name != EXPECTED_FILENAME:
    print(f"WARNING: {EXPECTED_FILENAME} not found, using {out_path.name}")

print("Reading output from:", out_path)

# -----------------------------------------------------------------------------
# 3) Read with SimpleITK
# -----------------------------------------------------------------------------
try:
    prediction = sitk.ReadImage(str(out_path))
    print("Successfully loaded prediction:")
    print("  Size:   ", prediction.GetSize())
    print("  Spacing:", prediction.GetSpacing())
    print("  Origin: ", prediction.GetOrigin())
except RuntimeError as e:
    print(f"ERROR: Failed to read {out_path}:", e, file=sys.stderr)
    sys.exit(1)

