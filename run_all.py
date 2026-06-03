"""
Run representative sensitivity sweeps and export CSV/PNG outputs.

Usage:
    python run_all.py --out outputs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from effect_lab_core import (
    diagnose_raw_signal,
    make_raw_signal,
    simulate_graybox,
    simulate_jit,
    simulate_multicollinearity,
    simulate_regression,
    simulate_transfer,
)


def savefig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def sweep_regression(out: Path) -> pd.DataFrame:
    rows = []
    for y_noise in np.linspace(0.05, 2.0, 12):
        r = simulate_regression(y_noise=float(y_noise), x_noise=0.0, seed=2)
        rows.append({"theme": "regression", "parameter": "y_noise", "value": y_noise, **r["metrics"]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["value"], df["corr"], marker="o", label="corr / standardized slope")
    ax.plot(df["value"], df["r2"], marker="o", label="R2")
    ax.set_xlabel("y_noise")
    ax.set_ylabel("metric")
    ax.set_title("Effect of y_noise on correlation and R2")
    ax.legend()
    savefig(fig, out / "sweep_regression_noise.png")
    return df


def sweep_multicollinearity(out: Path) -> pd.DataFrame:
    rows = []
    for c in np.linspace(0.1, 0.99, 12):
        r = simulate_multicollinearity(collinearity=float(c), pls_components=2, seed=3, perturb_repeats=20)
        for _, m in r["metrics"].iterrows():
            stab = r["stability"].set_index("model").loc[m["model"]].to_dict()
            rows.append({
                "theme": "multicollinearity",
                "parameter": "collinearity",
                "value": c,
                "model": m["model"],
                "RMSE_test": m["RMSE_test"],
                "R2_test": m["R2_test"],
                "condition_number": r["summary"]["condition_number"],
                "max_vif": r["summary"]["max_vif"],
                **stab,
            })
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    sub = df[df["model"] == "OLS"]
    ax.plot(sub["value"], sub["condition_number"], marker="o", label="condition number")
    ax.set_xlabel("collinearity")
    ax.set_ylabel("condition number")
    ax.set_title("Effect of collinearity on condition number")
    ax.legend()
    savefig(fig, out / "sweep_multicollinearity_condition.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for model, g in df.groupby("model"):
        ax.plot(g["value"], g["coef_std_mean"], marker="o", label=model)
    ax.set_xlabel("collinearity")
    ax.set_ylabel("mean coefficient std under tiny perturbation")
    ax.set_title("Coefficient instability under tiny perturbations")
    ax.legend()
    savefig(fig, out / "sweep_multicollinearity_stability.png")
    return df


def sweep_jit(out: Path) -> pd.DataFrame:
    rows = []
    for k in [3, 5, 10, 20, 40, 80, 120]:
        for bw in [0.25, 0.5, 0.8, 1.2, 2.0]:
            r = simulate_jit(k=k, bandwidth=bw, wear_current=0.85, drift_strength=1.2, seed=4)
            for _, m in r["metrics"].iterrows():
                rows.append({"theme": "JIT", "parameter": "k_bandwidth", "k": k, "bandwidth": bw, **m.to_dict()})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    sub = df[df["method"] == "LW-PLS"]
    for bw, g in sub.groupby("bandwidth"):
        ax.plot(g["k"], g["abs_error_vs_truth"], marker="o", label=f"bandwidth={bw}")
    ax.set_xlabel("k")
    ax.set_ylabel("absolute error vs truth")
    ax.set_title("JIT: effect of k and bandwidth")
    ax.legend(fontsize=8)
    savefig(fig, out / "sweep_jit_k_bandwidth.png")
    return df


def sweep_graybox(out: Path) -> pd.DataFrame:
    rows = []
    for obs in np.linspace(0, 1, 11):
        r = simulate_graybox(residual_observability=float(obs), residual_strength=0.9, parameter_drift=0.7, physics_bias=0.4, seed=5)
        for _, m in r["metrics"].iterrows():
            rows.append({"theme": "graybox", "parameter": "residual_observability", "value": obs, **m.to_dict()})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    for model, g in df.groupby("model"):
        ax.plot(g["value"], g["RMSE_test"], marker="o", label=model)
    ax.set_xlabel("residual_observability")
    ax.set_ylabel("RMSE_test")
    ax.set_title("Gray-box correction improves when residual is observable")
    ax.legend(fontsize=7)
    savefig(fig, out / "sweep_graybox_observability.png")
    return df


def sweep_transfer(out: Path) -> pd.DataFrame:
    rows = []
    for n_target in [3, 5, 10, 20, 50, 100]:
        for gap in [0.1, 0.35, 0.7, 1.2]:
            r = simulate_transfer(n_target=n_target, domain_gap=gap, transfer_weight=0.6, seed=6)
            for _, m in r["metrics"].iterrows():
                rows.append({"theme": "transfer", "parameter": "n_target_domain_gap", "n_target": n_target, "domain_gap": gap, **m.to_dict()})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    sub = df[df["domain_gap"] == 0.35]
    for model, g in sub.groupby("model"):
        ax.plot(g["n_target"], g["RMSE_target_test"], marker="o", label=model)
    ax.set_xlabel("n_target")
    ax.set_ylabel("RMSE_target_test")
    ax.set_title("Effect of target sample size")
    ax.legend(fontsize=7)
    savefig(fig, out / "sweep_transfer_ntarget.png")
    return df


def raw_example(out: Path) -> pd.DataFrame:
    df = make_raw_signal(outlier_strength=5, range_shift=1.2, trend_strength=0.8, cycle_strength=1.0, missing_rate=0.06, lower_clip=0.12, two_modes=1.0, seed=7)
    diag = diagnose_raw_signal(df)
    df.to_csv(out / "raw_signal_example.csv", index=False)
    diag.to_csv(out / "raw_signal_diagnostics.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["time"], df["value"], marker="o", markersize=2, linewidth=1)
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.set_title("Raw signal example")
    savefig(fig, out / "raw_signal_example.png")
    return diag


def parameter_effects_table(out: Path) -> pd.DataFrame:
    rows = [
        ["regression", "y_noise", "increase", "相関・R²・標準化傾きが下がる"],
        ["regression", "x_noise", "increase", "OLS の x は正確という前提が崩れ、傾きがずれる"],
        ["multicollinearity", "collinearity", "increase", "条件数・VIF・係数の不安定性が上がる"],
        ["PLS", "n_components", "too many", "OLS に近づき、多重共線性の影響が戻る"],
        ["JIT", "k", "increase", "ノイズには強くなるが、局所性は落ちる"],
        ["JIT", "bandwidth", "increase", "遠いデータも効くようになり、全体モデルに近づく"],
        ["graybox", "residual_observability", "increase", "Parallel / Combined の残差補正が効きやすくなる"],
        ["transfer", "domain_gap", "increase", "素朴な転移は悪化しやすく、負の転移が起きやすい"],
        ["raw", "range_shift", "increase", "単一モデルではなく分割・モード確認が必要になる"],
    ]
    df = pd.DataFrame(rows, columns=["theme", "parameter", "change", "expected_effect"])
    df.to_csv(out / "parameter_effects_summary.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="outputs")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tables = {
        "sweep_regression.csv": sweep_regression(out),
        "sweep_multicollinearity.csv": sweep_multicollinearity(out),
        "sweep_jit.csv": sweep_jit(out),
        "sweep_graybox.csv": sweep_graybox(out),
        "sweep_transfer.csv": sweep_transfer(out),
        "raw_signal_diagnostics_from_run.csv": raw_example(out),
        "parameter_effects_summary_from_run.csv": parameter_effects_table(out),
    }
    for name, df in tables.items():
        df.to_csv(out / name, index=False)
    print(f"Saved outputs to {out.resolve()}")


if __name__ == "__main__":
    main()
