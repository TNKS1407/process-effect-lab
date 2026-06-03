"""
Streamlit UI for Process Effect Lab.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from effect_lab_core import (
    diagnose_raw_signal,
    make_raw_signal,
    simulate_graybox,
    simulate_jit,
    simulate_multicollinearity,
    simulate_regression,
    simulate_transfer,
)

st.set_page_config(page_title="Process Effect Lab", layout="wide")

st.title("Process Effect Lab")
st.caption("パラメータを動かして、係数・精度・診断結果がどう変わるかを確認する実験ツール")

KNOB_TABLE = pd.DataFrame([
    ["単回帰", "y_noise", "y 側のばらつきが増え、相関・R²・標準化傾きが下がる"],
    ["単回帰", "x_noise", "x の測定誤差が増え、OLS の傾きが真値からずれやすい"],
    ["多重共線性", "collinearity", "条件数・VIF・係数のばらつきが大きくなる"],
    ["PLS", "pls_components", "少なすぎると不足、多すぎると OLS に近づく"],
    ["JIT", "k", "大きいほど平均化されるが、局所性が落ちる"],
    ["JIT", "bandwidth", "大きいほど遠いデータも効き、全体モデルに近づく"],
    ["グレーボックス", "physics_bias", "物理モデル単独の系統誤差が増える"],
    ["グレーボックス", "parameter_drift", "Serial 型の補正が効きやすくなる"],
    ["転移", "n_target", "増えるほど Target only が強くなり、転移の必要性が下がる"],
    ["転移", "domain_gap", "素朴な転移が害になりやすい"],
    ["生データ", "range_shift / trend / cycle", "モデル以前に、分割・前処理・運転モード確認が必要になる"],
], columns=["テーマ", "変えるもの", "増やすと起きやすいこと"])

with st.expander("まず見る表：何を変えると何が起きるか", expanded=True):
    st.dataframe(KNOB_TABLE, use_container_width=True, hide_index=True)


def show_metrics(df: pd.DataFrame, title: str = "結果") -> None:
    st.subheader(title)
    st.dataframe(df, use_container_width=True, hide_index=True)


def fig_regression(result):
    df = result["data"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(df["x_observed"], df["y"], s=28, alpha=0.8, label="data")
    order = np.argsort(df["x_observed"].to_numpy())
    ax.plot(df["x_observed"].to_numpy()[order], df["y_pred_ols"].to_numpy()[order], label="OLS")
    ax.set_xlabel("x observed")
    ax.set_ylabel("y")
    ax.set_title("OLS は縦方向の誤差を最小化する")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_coefficients(coef_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(coef_df))
    width = 0.25
    ax.bar(x - width, coef_df["OLS"], width, label="OLS")
    ax.bar(x, coef_df["Ridge"], width, label="Ridge")
    ax.bar(x + width, coef_df["PLS"], width, label="PLS")
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(coef_df["feature"], rotation=45, ha="right")
    ax.set_ylabel("coefficient")
    ax.set_title("係数の比較")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_jit_time(result):
    db = result["database"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(db["time"], db["y"], marker="o", markersize=2, linewidth=1)
    ax.set_xlabel("time")
    ax.set_ylabel("quality y")
    ax.set_title("データベース内の過去品質データ")
    fig.tight_layout()
    return fig


def fig_jit_distance(result):
    details = result["details"]
    if "LW-PLS" in details:
        dist = details["LW-PLS"]["distances"]
    else:
        dist = next(iter(details.values()))["distances"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(dist, bins=30)
    ax.set_xlabel("standardized distance from current point")
    ax.set_ylabel("count")
    ax.set_title("現在点にどれくらい近いデータがあるか")
    fig.tight_layout()
    return fig


def fig_graybox(result):
    pred_df = result["test_predictions"]
    metrics = result["metrics"]
    best = metrics.iloc[0]["model"]
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(pred_df["y"], pred_df[best], s=18, alpha=0.75)
    lo = min(pred_df["y"].min(), pred_df[best].min())
    hi = max(pred_df["y"].max(), pred_df[best].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("measured y")
    ax.set_ylabel(f"predicted y: {best}")
    ax.set_title("最良モデルの予測 vs 実測")
    fig.tight_layout()
    return fig


def fig_bar_metrics(metrics: pd.DataFrame, value_col: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(metrics["model"], metrics[value_col])
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig


def fig_raw(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(df["time"], df["value"], marker="o", markersize=2, linewidth=1)
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.set_title("生データの時系列")
    fig.tight_layout()
    return fig


def fig_raw_hist(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.hist(df["value"].dropna(), bins=30)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    ax.set_title("ヒストグラム")
    fig.tight_layout()
    return fig


tabs = st.tabs(["単回帰", "多重共線性・PLS", "JIT", "グレーボックス", "転移", "生データ診断"])

with tabs[0]:
    st.header("単回帰：ノイズ・測定誤差・外れ値")
    st.write("OLS は x を正確とみなし、y の縦方向の誤差を小さくします。標準化した単回帰の傾きは相関係数になります。")
    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.slider("n", 20, 300, 80, 10)
        true_slope = st.slider("true_slope", -3.0, 3.0, 1.0, 0.1)
    with c2:
        y_noise = st.slider("y_noise", 0.0, 3.0, 0.6, 0.05)
        x_noise = st.slider("x_noise", 0.0, 2.0, 0.0, 0.05)
    with c3:
        outlier_strength = st.slider("outlier_strength", 0.0, 10.0, 0.0, 0.5)
        seed = st.number_input("seed", 0, 9999, 0, 1)
    result = simulate_regression(n=n, true_slope=true_slope, y_noise=y_noise, x_noise=x_noise, outlier_strength=outlier_strength, seed=seed)
    st.pyplot(fig_regression(result))
    st.json(result["metrics"])
    st.info("見るポイント：y_noise を上げると相関と標準化傾きが下がります。x_noise を上げると、OLS の前提が崩れて傾きがずれます。")

with tabs[1]:
    st.header("多重共線性・PLS：係数はどれだけ信用できるか")
    st.write("入力変数同士が似た動きをすると、予測精度がそこそこでも係数解釈は壊れます。PLS の潜在変数数も動かしてください。")
    c1, c2, c3 = st.columns(3)
    with c1:
        n_train = st.slider("n_train", 20, 300, 80, 10)
        n_features = st.slider("n_features", 3, 30, 8, 1)
    with c2:
        collinearity = st.slider("collinearity", 0.0, 0.995, 0.95, 0.005)
        y_noise = st.slider("noise", 0.0, 2.0, 0.25, 0.05, key="mc_noise")
    with c3:
        pls_components = st.slider("pls_components", 1, min(n_features, n_train - 1), min(2, n_features), 1)
        ridge_alpha = st.slider("ridge_alpha", 0.0, 20.0, 1.0, 0.5)
    result = simulate_multicollinearity(n_train=n_train, n_features=n_features, collinearity=collinearity, y_noise=y_noise, pls_components=pls_components, ridge_alpha=ridge_alpha)
    st.write(result["summary"])
    show_metrics(result["metrics"], "予測性能")
    show_metrics(result["stability"], "微小ノイズに対する係数の不安定性")
    st.pyplot(fig_coefficients(result["coefficients"]))
    st.dataframe(result["cv"], use_container_width=True, hide_index=True)
    st.warning("注意：CV_RMSE が最小の成分数が、解釈しやすく安全な成分数とは限りません。潜在変数を増やすほど OLS に近づきます。")

with tabs[2]:
    st.header("JIT 型モデル：近いデータだけをどれくらい使うか")
    st.write("固定モデルと、1-NN、kNN、局所線形、局所重み付き PLS を比較します。")
    c1, c2, c3 = st.columns(3)
    with c1:
        x1_current = st.slider("x1_current", -3.0, 3.0, 0.5, 0.1)
        x2_current = st.slider("x2_current", -2.0, 2.0, 0.0, 0.1)
        wear_current = st.slider("wear_current", 0.0, 1.0, 0.85, 0.01)
    with c2:
        k = st.slider("k", 1, 150, 35, 1)
        bandwidth = st.slider("bandwidth", 0.1, 3.0, 0.8, 0.05)
        pls_components = st.slider("pls_components", 1, 3, 2, 1, key="jit_pls")
    with c3:
        nonlinearity = st.slider("nonlinearity", 0.2, 3.0, 1.3, 0.1)
        drift_strength = st.slider("drift_strength", 0.0, 3.0, 1.0, 0.1)
        noise = st.slider("noise", 0.0, 1.0, 0.12, 0.01, key="jit_noise")
    result = simulate_jit(x1_current=x1_current, x2_current=x2_current, wear_current=wear_current, k=k, bandwidth=bandwidth, pls_components=pls_components, nonlinearity=nonlinearity, drift_strength=drift_strength, noise=noise)
    st.write(f"真の値（シミュレーション上の正解）: {result['y_true']:.4f}")
    show_metrics(result["metrics"], "モデル比較")
    col_a, col_b = st.columns(2)
    with col_a:
        st.pyplot(fig_jit_time(result))
    with col_b:
        st.pyplot(fig_jit_distance(result))
    st.info("見るポイント：k や bandwidth が小さいと局所性は高いがノイズに弱い。大きいと安定するが、違う状態のデータまで混ざります。")

with tabs[3]:
    st.header("グレーボックス：物理モデルに統計補正を足す")
    st.write("物理モデルだけ、統計モデルだけ、Parallel、Serial、Combined を比較します。")
    c1, c2, c3 = st.columns(3)
    with c1:
        physics_bias = st.slider("physics_bias", -2.0, 2.0, 0.3, 0.1)
        parameter_drift = st.slider("parameter_drift", 0.0, 2.0, 0.7, 0.05)
    with c2:
        residual_strength = st.slider("residual_strength", 0.0, 2.0, 0.8, 0.05)
        residual_observability = st.slider("residual_observability", 0.0, 1.0, 0.8, 0.05)
    with c3:
        noise = st.slider("noise", 0.0, 1.0, 0.10, 0.01, key="gray_noise")
        ridge_alpha = st.slider("ridge_alpha", 0.0, 20.0, 1.0, 0.5, key="gray_ridge")
    result = simulate_graybox(physics_bias=physics_bias, parameter_drift=parameter_drift, residual_strength=residual_strength, residual_observability=residual_observability, noise=noise, ridge_alpha=ridge_alpha)
    show_metrics(result["metrics"], "モデル比較")
    st.pyplot(fig_graybox(result))
    st.info("見るポイント：残差が観測変数から説明できるほど Parallel / Combined が効きます。パラメータが運転状態で変わるほど Serial が効きます。")

with tabs[4]:
    st.header("転移学習：過去の似たデータをどれだけ使うか")
    st.write("ターゲットデータが少ない状況で、ターゲットのみ、素朴な転移、ドメイン拡張型の転移を比較します。")
    c1, c2, c3 = st.columns(3)
    with c1:
        n_source = st.slider("n_source", 20, 1000, 300, 20)
        n_target = st.slider("n_target", 3, 100, 10, 1)
    with c2:
        domain_gap = st.slider("domain_gap", 0.0, 2.0, 0.35, 0.05)
        transfer_weight = st.slider("transfer_weight", 0.0, 3.0, 0.6, 0.05)
    with c3:
        n_common = st.slider("n_common", 1, 20, 8, 1)
        n_source_unique = st.slider("n_source_unique", 0, 10, 3, 1)
        n_target_unique = st.slider("n_target_unique", 0, 10, 3, 1)
    result = simulate_transfer(n_source=n_source, n_target=n_target, n_common=n_common, n_source_unique=n_source_unique, n_target_unique=n_target_unique, domain_gap=domain_gap, transfer_weight=transfer_weight)
    show_metrics(result["metrics"], "ターゲットテストでの比較")
    st.pyplot(fig_bar_metrics(result["metrics"], "RMSE_target_test", "転移学習の比較"))
    st.warning("注意：domain_gap が大きいと、ソースを強く混ぜるほど悪くなることがあります。これは負の転移です。")

with tabs[5]:
    st.header("生データ診断：モデル前に見るべきもの")
    st.write("外れ値、レンジ変更、トレンド、周期変動、欠損、下限張り付き、複数モードを入れて、診断結果を確認します。")
    c1, c2, c3 = st.columns(3)
    with c1:
        outlier_strength = st.slider("outlier_strength", 0.0, 10.0, 4.0, 0.5, key="raw_outlier")
        range_shift = st.slider("range_shift", 0.0, 3.0, 1.0, 0.1)
        trend_strength = st.slider("trend_strength", -2.0, 2.0, 0.5, 0.1)
    with c2:
        cycle_strength = st.slider("cycle_strength", 0.0, 3.0, 0.8, 0.1)
        missing_rate = st.slider("missing_rate", 0.0, 0.4, 0.05, 0.01)
    with c3:
        lower_clip = st.slider("lower_clip", 0.0, 0.4, 0.0, 0.01)
        two_modes = st.slider("two_modes", 0.0, 4.0, 0.0, 0.1)
        n = st.slider("n", 50, 1000, 300, 10, key="raw_n")
    df = make_raw_signal(n=n, outlier_strength=outlier_strength, range_shift=range_shift, trend_strength=trend_strength, cycle_strength=cycle_strength, missing_rate=missing_rate, lower_clip=lower_clip, two_modes=two_modes)
    diag = diagnose_raw_signal(df)
    col_a, col_b = st.columns(2)
    with col_a:
        st.pyplot(fig_raw(df))
    with col_b:
        st.pyplot(fig_raw_hist(df))
    show_metrics(diag, "診断結果")
    st.info("flag=True が出たら、モデル選択より先に理由を調べる対象です。")
