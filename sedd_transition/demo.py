
import numpy as np
import pandas as pd
from pathlib import Path
from binning import bin_continuous, ensure_int_categories
from transitions import build_M_R_A_all, mh_calibrate, stationary_error
from scheduling import cosine_schedule, gamma_teleport_schedule
from utils import mat_to_df

def make_synth(N: int=5000, seed: int=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(40, 12, N), 18, 75)
    bmi = np.clip(rng.normal(24, 4, N), 16, 38)
    treat = rng.choice([0,1,2], size=N, p=[0.45, 0.35, 0.20])
    logit = -1.1 + 0.028*(age-40) - 0.035*(bmi-24) + 0.45*(treat==2) - 0.15*(treat==1)
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=N) < prob).astype(int)
    return pd.DataFrame({"age": age, "bmi": bmi, "treat": treat, "y": y})

def run_demo(output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df = make_synth()
    # binning
    df["age_bin"], _ = bin_continuous(df["age"], n_bins=10, method="quantile")
    df["bmi_bin"], _ = bin_continuous(df["bmi"], n_bins=8, method="quantile")
    df["treat_cat"], _ = ensure_int_categories(df["treat"])
    feature_bins = {"age_bin": 10, "bmi_bin": 8, "treat_cat": 3}

    Ms, Rs, As, Pis = build_M_R_A_all(df, y_col="y", feature_bins=feature_bins, alpha_m=1.0, alpha_r=1.0)
    A_mh = {feat: mh_calibrate(As[feat], Pis[feat]) for feat in feature_bins}

    # Stationarity check
    rows = []
    for feat in feature_bins:
        err_raw = stationary_error(Pis[feat], As[feat], ord=1)
        err_mh = stationary_error(Pis[feat], A_mh[feat], ord=1)
        rows.append((feat, err_raw, err_mh))
    stats = pd.DataFrame(rows, columns=["feature", "||piA - pi||_1 (raw A)", "||piA - pi||_1 (MH-calibrated)"])
    stats.to_csv(f"{output_dir}/stationarity_check.csv", index=False)

    # Save matrices
    for feat in feature_bins:
        mat_to_df(Ms[feat], "v", "y").to_csv(f"{output_dir}/{feat}__M[v,y].csv", index=True)
        mat_to_df(Rs[feat], "y", "v").to_csv(f"{output_dir}/{feat}__R[y,v].csv", index=True)
        mat_to_df(As[feat], "v", "vprime").to_csv(f"{output_dir}/{feat}__A[v,v'].csv", index=True)
        mat_to_df(A_mh[feat], "v", "vprime").to_csv(f"{output_dir}/{feat}__A_MH[v,v'].csv", index=True)

    print("Saved outputs to", output_dir)
    print(stats)

if __name__ == "__main__":
    run_demo(output_dir="output")
