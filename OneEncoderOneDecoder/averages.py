import pandas as pd

df = pd.read_csv("/well/papiez/users/kqe223/DEEP-PSMA_CHALLENGE_DATA/earlyfusion_metrics.csv")

# Remove rows with errors or nan in dice (or any metric you want)
df_clean = df.dropna(subset=["dice"])

# Compute mean and std for each metric
metrics = ["dice", "fp_vol", "fn_vol", "surface_dice", "suv_mean_ratio", "ttb_vol_ratio"]
for tracer, group in df_clean.groupby("tracer"):
    print(f"\nTracer: {tracer}")
    for metric in metrics:
        mean = group[metric].mean()
        std = group[metric].std()
        print(f"  {metric}: mean={mean:.4f}, std={std:.4f}")