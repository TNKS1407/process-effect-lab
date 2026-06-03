"""Process Effect Lab — lightweight web server (no Streamlit)."""
from __future__ import annotations
import base64, http.server, io, json, os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = ["Meiryo", "MS Gothic", "DejaVu Sans"]
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent))

from effect_lab_core import (
    diagnose_raw_signal, make_raw_signal,
    simulate_graybox, simulate_jit,
    simulate_multicollinearity, simulate_regression,
    simulate_transfer,
)

PORT = int(os.environ.get("PORT", 8504))
CORS = {"Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type"}


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def df_to_rows(df: pd.DataFrame) -> dict:
    return {"columns": list(df.columns),
            "rows": [[str(round(v, 5)) if isinstance(v, float) else str(v) for v in row]
                     for row in df.itertuples(index=False)]}


# ── experiment runners ────────────────────────────────────────────────────────

def run_regression(p: dict) -> dict:
    r = simulate_regression(**p)
    df = r["data"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df["x_observed"], df["y"], s=22, alpha=0.75, label="data")
    order = np.argsort(df["x_observed"].to_numpy())
    ax.plot(df["x_observed"].to_numpy()[order], df["y_pred_ols"].to_numpy()[order], lw=2, label="OLS")
    ax.set_xlabel("x observed"); ax.set_ylabel("y")
    ax.set_title("OLS は縦方向の誤差を最小化する")
    ax.legend(); fig.tight_layout()
    metrics_df = pd.DataFrame([r["metrics"]]) if isinstance(r["metrics"], dict) else r["metrics"]
    return {"images": [fig_to_b64(fig)], "tables": [df_to_rows(metrics_df)],
            "table_titles": ["メトリクス"], "note": r.get("note", "")}


def run_multicollinearity(p: dict) -> dict:
    r = simulate_multicollinearity(**p)
    coef = r["coefficients"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(coef)); w = 0.25
    ax.bar(x - w, coef["OLS"], w, label="OLS")
    ax.bar(x,     coef["Ridge"], w, label="Ridge")
    ax.bar(x + w, coef["PLS"], w, label="PLS")
    ax.axhline(0, lw=1); ax.set_xticks(x)
    ax.set_xticklabels(coef["feature"], rotation=40, ha="right")
    ax.set_ylabel("coefficient"); ax.set_title("係数の比較"); ax.legend(); fig.tight_layout()
    return {"images": [fig_to_b64(fig)],
            "tables": [df_to_rows(r["metrics"]), df_to_rows(r["stability"]), df_to_rows(r["cv"])],
            "table_titles": ["予測性能", "係数の不安定性", "交差検証"]}


def run_jit(p: dict) -> dict:
    r = simulate_jit(**p)
    db = r["database"]
    fig1, ax1 = plt.subplots(figsize=(7, 3.8))
    ax1.plot(db["time"], db["y"], marker="o", ms=2, lw=1)
    ax1.set_xlabel("time"); ax1.set_ylabel("quality y")
    ax1.set_title(f"データベース ({len(db)} samples)  真の値: {r['y_true']:.3f}")
    fig1.tight_layout()
    details = r.get("details", {})
    dist = None
    for v in details.values():
        if isinstance(v, dict) and "distances" in v:
            dist = v["distances"]; break
    figs = [fig_to_b64(fig1)]
    if dist is not None:
        fig2, ax2 = plt.subplots(figsize=(6, 3.8))
        ax2.hist(dist, bins=30)
        ax2.set_xlabel("standardized distance"); ax2.set_ylabel("count")
        ax2.set_title("現在点からの距離分布"); fig2.tight_layout()
        figs.append(fig_to_b64(fig2))
    return {"images": figs, "tables": [df_to_rows(r["metrics"])],
            "table_titles": ["モデル比較"]}


def run_graybox(p: dict) -> dict:
    r = simulate_graybox(**p)
    pred = r["test_predictions"]
    best = r["metrics"].iloc[0]["model"]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(pred["y"], pred[best], s=18, alpha=0.75)
    lo = min(pred["y"].min(), pred[best].min()); hi = max(pred["y"].max(), pred[best].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("measured y"); ax.set_ylabel(f"predicted: {best}")
    ax.set_title("最良モデルの予測 vs 実測"); fig.tight_layout()
    return {"images": [fig_to_b64(fig)], "tables": [df_to_rows(r["metrics"])],
            "table_titles": ["モデル比較"]}


def run_transfer(p: dict) -> dict:
    r = simulate_transfer(**p)
    metrics = r["metrics"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(metrics["model"], metrics["RMSE_target_test"])
    ax.set_ylabel("RMSE (target test)"); ax.set_title("転移学習の比較")
    ax.tick_params(axis="x", rotation=25); fig.tight_layout()
    return {"images": [fig_to_b64(fig)], "tables": [df_to_rows(metrics)],
            "table_titles": ["ターゲットテストでの比較"]}


def run_rawdata(p: dict) -> dict:
    df = make_raw_signal(**p)
    diag = diagnose_raw_signal(df)
    fig1, ax1 = plt.subplots(figsize=(8, 3.8))
    ax1.plot(df["time"], df["value"], ms=2, lw=1)
    ax1.set_xlabel("time"); ax1.set_ylabel("value"); ax1.set_title("生データの時系列"); fig1.tight_layout()
    fig2, ax2 = plt.subplots(figsize=(6, 3.8))
    ax2.hist(df["value"].dropna(), bins=30)
    ax2.set_xlabel("value"); ax2.set_ylabel("count"); ax2.set_title("ヒストグラム"); fig2.tight_layout()
    return {"images": [fig_to_b64(fig1), fig_to_b64(fig2)],
            "tables": [df_to_rows(diag)], "table_titles": ["診断結果"]}


RUNNERS = {
    "regression": run_regression,
    "multicollinearity": run_multicollinearity,
    "jit": run_jit,
    "graybox": run_graybox,
    "transfer": run_transfer,
    "rawdata": run_rawdata,
}


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

    def send_json(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for k, v in CORS.items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        for k, v in CORS.items(): self.send_header(k, v)
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            data = (ROOT / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data); return
        self.send_response(404); self.end_headers()

    def do_POST(self):
        if self.path != "/api/run":
            self.send_response(404); self.end_headers(); return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        exp = body.get("exp", "")
        params = body.get("params", {})
        runner = RUNNERS.get(exp)
        if runner is None:
            self.send_json(400, {"error": f"unknown experiment: {exp}"}); return
        try:
            result = runner(params)
            self.send_json(200, result)
        except Exception as e:
            self.send_json(500, {"error": str(e)})


if __name__ == "__main__":
    server = http.server.HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Process Effect Lab at http://127.0.0.1:{PORT}")
    server.serve_forever()
