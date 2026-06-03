
"""
Core numerical functions for Process Effect Lab.

This module intentionally uses only NumPy/Pandas so the calculations are easy
to inspect.  The goal is not to provide a production-grade AutoML system; it is
to expose the mechanism behind regression, collinearity, local modeling,
gray-box modeling, transfer learning, and raw-data diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass
class LinearModel:
    coef: np.ndarray
    intercept: float
    x_mean: Optional[np.ndarray] = None
    x_std: Optional[np.ndarray] = None
    y_mean: Optional[float] = None
    y_std: Optional[float] = None
    name: str = "linear"

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.coef + self.intercept


@dataclass
class PLSModel:
    coef: np.ndarray
    intercept: float
    n_components_used: int
    W: np.ndarray
    P: np.ndarray
    q: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.coef + self.intercept


def rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def safe_std(x: np.ndarray, axis: int = 0, weights: Optional[np.ndarray] = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if weights is None:
        s = np.std(x, axis=axis, ddof=0)
    else:
        w = np.asarray(weights, dtype=float)
        w = w / np.sum(w)
        mu = np.sum(w[:, None] * x, axis=0)
        s = np.sqrt(np.sum(w[:, None] * (x - mu) ** 2, axis=0))
    return np.where(s < EPS, 1.0, s)


def standardize(X: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = as_2d(X)
    if weights is None:
        mu = X.mean(axis=0)
        sd = safe_std(X, axis=0)
    else:
        w = np.asarray(weights, dtype=float)
        w = w / np.sum(w)
        mu = np.sum(w[:, None] * X, axis=0)
        sd = safe_std(X, weights=w)
    return (X - mu) / sd, mu, sd


def standardize_y(y: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, float]:
    y = np.asarray(y, dtype=float).ravel()
    if weights is None:
        mu = float(y.mean())
        sd = float(np.std(y, ddof=0))
    else:
        w = np.asarray(weights, dtype=float)
        w = w / np.sum(w)
        mu = float(np.sum(w * y))
        sd = float(np.sqrt(np.sum(w * (y - mu) ** 2)))
    if sd < EPS:
        sd = 1.0
    return (y - mu) / sd, mu, sd


def add_intercept(X: np.ndarray) -> np.ndarray:
    X = as_2d(X)
    return np.column_stack([np.ones(X.shape[0]), X])


def ridge_fit(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.0,
    standardize_x: bool = False,
    sample_weight: Optional[np.ndarray] = None,
    penalize_intercept: bool = False,
    name: str = "ridge",
) -> LinearModel:
    X = as_2d(X)
    y = np.asarray(y, dtype=float).ravel()

    if standardize_x:
        Xs, xm, xs = standardize(X, weights=sample_weight)
    else:
        Xs = X
        xm = np.zeros(X.shape[1])
        xs = np.ones(X.shape[1])

    Xa = add_intercept(Xs)
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float).ravel()
        w = np.where(w < 0, 0, w)
        if np.sum(w) <= EPS:
            w = np.ones_like(w)
        sw = np.sqrt(w / np.mean(w))
        Xa_w = Xa * sw[:, None]
        y_w = y * sw
    else:
        Xa_w = Xa
        y_w = y

    reg = alpha * np.eye(Xa.shape[1])
    if not penalize_intercept:
        reg[0, 0] = 0.0

    try:
        beta_aug = np.linalg.solve(Xa_w.T @ Xa_w + reg, Xa_w.T @ y_w)
    except np.linalg.LinAlgError:
        beta_aug = np.linalg.pinv(Xa_w.T @ Xa_w + reg) @ Xa_w.T @ y_w

    intercept_s = float(beta_aug[0])
    coef_s = beta_aug[1:]
    coef = coef_s / xs
    intercept = intercept_s - float(xm @ coef)
    return LinearModel(coef=coef, intercept=intercept, x_mean=xm, x_std=xs, name=name)


def ols_fit(X: np.ndarray, y: np.ndarray, standardize_x: bool = False, name: str = "ols") -> LinearModel:
    return ridge_fit(X, y, alpha=0.0, standardize_x=standardize_x, name=name)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < EPS:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def condition_number(X: np.ndarray) -> float:
    X = as_2d(X)
    Xs, _, _ = standardize(X)
    try:
        s = np.linalg.svd(Xs, compute_uv=False)
        if np.min(s) < EPS:
            return float("inf")
        return float(np.max(s) / np.min(s))
    except np.linalg.LinAlgError:
        return float("inf")


def vif_values(X: np.ndarray) -> np.ndarray:
    X = as_2d(X)
    Xs, _, _ = standardize(X)
    c = np.corrcoef(Xs, rowvar=False)
    if c.ndim == 0:
        return np.array([1.0])
    try:
        inv_c = np.linalg.inv(c)
    except np.linalg.LinAlgError:
        inv_c = np.linalg.pinv(c)
    vif = np.diag(inv_c)
    return np.where(np.isfinite(vif), vif, np.inf)


def pls1_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 2,
    scale: bool = True,
    sample_weight: Optional[np.ndarray] = None,
) -> PLSModel:
    """
    PLS1 by a compact NIPALS-style algorithm.

    If sample_weight is supplied, weighted means/covariances are used.  This is
    enough for the local weighted PLS demonstrations in this project.
    """
    X = as_2d(X)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape

    if sample_weight is None:
        weights = np.ones(n) / n
    else:
        weights = np.asarray(sample_weight, dtype=float).ravel()
        weights = np.where(weights < 0, 0.0, weights)
        if np.sum(weights) <= EPS:
            weights = np.ones(n)
        weights = weights / np.sum(weights)

    if scale:
        Xs, x_mean, x_std = standardize(X, weights=weights)
        ys, y_mean, y_std = standardize_y(y, weights=weights)
    else:
        x_mean = np.sum(weights[:, None] * X, axis=0)
        x_std = np.ones(p)
        y_mean = float(np.sum(weights * y))
        y_std = 1.0
        Xs = X - x_mean
        ys = y - y_mean

    Xh = Xs.copy()
    yh = ys.copy()
    max_components = int(max(1, min(n_components, p, max(1, n - 1))))
    W_list: List[np.ndarray] = []
    P_list: List[np.ndarray] = []
    q_list: List[float] = []

    for _ in range(max_components):
        cov = Xh.T @ (weights * yh)
        norm = float(np.linalg.norm(cov))
        if norm < EPS:
            break
        w_vec = cov / norm
        t = Xh @ w_vec
        denom = float(np.sum(weights * t ** 2))
        if denom < EPS:
            break
        p_vec = Xh.T @ (weights * t) / denom
        q_val = float(np.sum(weights * t * yh) / denom)
        Xh = Xh - np.outer(t, p_vec)
        yh = yh - q_val * t
        W_list.append(w_vec)
        P_list.append(p_vec)
        q_list.append(q_val)

    if not W_list:
        coef = np.zeros(p)
        intercept = float(np.sum(weights * y))
        return PLSModel(
            coef=coef,
            intercept=intercept,
            n_components_used=0,
            W=np.zeros((p, 0)),
            P=np.zeros((p, 0)),
            q=np.zeros(0),
            x_mean=x_mean,
            x_std=x_std,
            y_mean=float(y_mean),
            y_std=float(y_std),
        )

    W = np.column_stack(W_list)
    P = np.column_stack(P_list)
    q = np.asarray(q_list)

    try:
        coef_scaled = W @ np.linalg.solve(P.T @ W, q)
    except np.linalg.LinAlgError:
        coef_scaled = W @ (np.linalg.pinv(P.T @ W) @ q)

    coef = coef_scaled * (y_std / x_std)
    intercept = float(y_mean - x_mean @ coef)
    return PLSModel(
        coef=coef,
        intercept=intercept,
        n_components_used=len(q_list),
        W=W,
        P=P,
        q=q,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=float(y_mean),
        y_std=float(y_std),
    )


def kfold_indices(n: int, k: int, seed: int = 0) -> List[np.ndarray]:
    k = int(max(2, min(k, n)))
    idx = rng(seed).permutation(n)
    return [arr for arr in np.array_split(idx, k) if len(arr) > 0]


def kfold_cv_pls(
    X: np.ndarray,
    y: np.ndarray,
    max_components: int,
    k: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    X = as_2d(X)
    y = np.asarray(y, dtype=float).ravel()
    folds = kfold_indices(len(y), k, seed)
    rows = []
    for r in range(1, int(max_components) + 1):
        sqerr = 0.0
        n_eval = 0
        for val_idx in folds:
            train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
            model = pls1_fit(X[train_idx], y[train_idx], n_components=r)
            pred = model.predict(X[val_idx])
            sqerr += float(np.sum((y[val_idx] - pred) ** 2))
            n_eval += len(val_idx)
        rows.append({"components": r, "PRESS": sqerr, "CV_RMSE": math.sqrt(sqerr / max(n_eval, 1))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Regression direction / measurement-error simulator
# ---------------------------------------------------------------------------

def simulate_regression(
    n: int = 80,
    true_slope: float = 1.0,
    y_noise: float = 0.6,
    x_noise: float = 0.0,
    outlier_strength: float = 0.0,
    seed: int = 0,
) -> Dict[str, object]:
    g = rng(seed)
    x_true = g.normal(0, 1, n)
    y = true_slope * x_true + g.normal(0, y_noise, n)
    x = x_true + g.normal(0, x_noise, n)

    if outlier_strength > 0:
        m = max(1, int(0.04 * n))
        out_idx = g.choice(n, size=m, replace=False)
        y[out_idx] += outlier_strength * np.sign(g.normal(size=m)) * (1 + np.abs(x[out_idx]))

    X = x.reshape(-1, 1)
    ols = ols_fit(X, y)
    pred = ols.predict(X)

    # Standardized regression slope equals correlation.
    xz = (x - x.mean()) / (np.std(x) if np.std(x) > EPS else 1)
    yz = (y - y.mean()) / (np.std(y) if np.std(y) > EPS else 1)
    corr = float(np.corrcoef(xz, yz)[0, 1])

    # Total least squares / first principal component line.
    Z = np.column_stack([xz, yz])
    cov = np.cov(Z, rowvar=False, ddof=0)
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, np.argmax(vals)]
    tls_slope_std = float(v[1] / v[0]) if abs(v[0]) > EPS else float("inf")

    metrics = {
        "corr": corr,
        "standardized_ols_slope": corr,
        "ols_slope": float(ols.coef[0]),
        "ols_intercept": float(ols.intercept),
        "tls_slope_standardized": tls_slope_std,
        "rmse": rmse(y, pred),
        "r2": r2_score(y, pred),
        "x_variance_ratio_observed_to_true": float(np.var(x) / max(np.var(x_true), EPS)),
    }
    df = pd.DataFrame({"x_true": x_true, "x_observed": x, "y": y, "y_pred_ols": pred})
    return {"data": df, "metrics": metrics, "ols": ols}


# ---------------------------------------------------------------------------
# 2. Multicollinearity / PLS simulator
# ---------------------------------------------------------------------------

def _make_collinear_data(
    n: int,
    collinearity: float,
    y_noise: float,
    n_features: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = rng(seed)
    collinearity = float(np.clip(collinearity, 0.0, 0.9999))
    z = g.normal(size=(n, 2))
    X = np.zeros((n, n_features))
    residual_scale = math.sqrt(max(1.0 - collinearity ** 2, EPS))
    for j in range(n_features):
        latent = z[:, j % 2]
        X[:, j] = collinearity * latent + residual_scale * g.normal(size=n)
        X[:, j] += 0.05 * g.normal(size=n) * (j + 1) / n_features
    y = 1.2 * z[:, 0] - 0.9 * z[:, 1] + g.normal(scale=y_noise, size=n)
    latent_coef = np.array([1.2, -0.9])
    return X, y, latent_coef


def simulate_multicollinearity(
    n_train: int = 80,
    n_test: int = 400,
    collinearity: float = 0.95,
    y_noise: float = 0.25,
    n_features: int = 8,
    pls_components: int = 2,
    ridge_alpha: float = 1.0,
    perturb_repeats: int = 40,
    seed: int = 0,
) -> Dict[str, object]:
    Xtr, ytr, _ = _make_collinear_data(n_train, collinearity, y_noise, n_features, seed)
    Xte, yte, _ = _make_collinear_data(n_test, collinearity, y_noise, n_features, seed + 10_000)

    ols = ols_fit(Xtr, ytr, standardize_x=True, name="OLS")
    ridge = ridge_fit(Xtr, ytr, alpha=ridge_alpha, standardize_x=True, name="Ridge")
    pls = pls1_fit(Xtr, ytr, n_components=pls_components, scale=True)

    pred = {
        "OLS": ols.predict(Xte),
        "Ridge": ridge.predict(Xte),
        "PLS": pls.predict(Xte),
    }

    metrics_rows = []
    for name, p in pred.items():
        metrics_rows.append({"model": name, "RMSE_test": rmse(yte, p), "R2_test": r2_score(yte, p)})
    metrics = pd.DataFrame(metrics_rows)

    # Coefficient instability under tiny measurement perturbation.
    g = rng(seed + 1234)
    coef_records = {"OLS": [], "Ridge": [], "PLS": []}
    for _ in range(max(1, perturb_repeats)):
        Xp = Xtr + g.normal(scale=0.015, size=Xtr.shape)
        coef_records["OLS"].append(ols_fit(Xp, ytr, standardize_x=True).coef)
        coef_records["Ridge"].append(ridge_fit(Xp, ytr, alpha=ridge_alpha, standardize_x=True).coef)
        coef_records["PLS"].append(pls1_fit(Xp, ytr, n_components=pls_components, scale=True).coef)
    stability_rows = []
    for name, coefs in coef_records.items():
        C = np.vstack(coefs)
        stability_rows.append({
            "model": name,
            "coef_std_mean": float(np.mean(np.std(C, axis=0))),
            "coef_std_max": float(np.max(np.std(C, axis=0))),
            "coef_l2_mean": float(np.mean(np.linalg.norm(C, axis=1))),
        })
    stability = pd.DataFrame(stability_rows)

    summary = {
        "condition_number": condition_number(Xtr),
        "max_vif": float(np.nanmax(vif_values(Xtr))),
        "mean_abs_corr": float(np.mean(np.abs(np.corrcoef(standardize(Xtr)[0], rowvar=False)[np.triu_indices(n_features, 1)]))),
        "pls_components_used": pls.n_components_used,
    }

    coef_df = pd.DataFrame({
        "feature": [f"x{j+1}" for j in range(n_features)],
        "OLS": ols.coef,
        "Ridge": ridge.coef,
        "PLS": pls.coef,
    })

    cv = kfold_cv_pls(Xtr, ytr, max_components=min(n_features, max(1, n_train - 1)), k=min(5, n_train), seed=seed)

    return {
        "X_train": Xtr, "y_train": ytr, "X_test": Xte, "y_test": yte,
        "metrics": metrics, "stability": stability, "summary": summary,
        "coefficients": coef_df, "cv": cv,
    }


# ---------------------------------------------------------------------------
# 3. Just-In-Time / local modeling simulator
# ---------------------------------------------------------------------------

def _jit_true_function(X: np.ndarray, nonlinearity: float, drift_strength: float) -> np.ndarray:
    X = as_2d(X)
    x1, x2, wear = X[:, 0], X[:, 1], X[:, 2]
    return (
        np.sin(nonlinearity * x1)
        + 0.45 * x2
        + 0.25 * x1 * x2
        + drift_strength * (wear - 0.5)
        + 0.25 * np.sin(2 * np.pi * wear)
    )


def make_jit_database(
    n_database: int = 420,
    nonlinearity: float = 1.3,
    drift_strength: float = 1.0,
    noise: float = 0.12,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    g = rng(seed)
    t = np.arange(n_database)
    cycle = max(30, n_database // 5)
    wear = (t % cycle) / cycle
    x1 = g.uniform(-3, 3, n_database) + 0.4 * np.sin(2 * np.pi * wear)
    x2 = g.uniform(-2, 2, n_database) + 0.2 * np.cos(2 * np.pi * wear)
    X = np.column_stack([x1, x2, wear])
    y = _jit_true_function(X, nonlinearity, drift_strength) + g.normal(scale=noise, size=n_database)
    df = pd.DataFrame({"time": t, "x1": x1, "x2": x2, "wear": wear, "y": y})
    return X, y, df


def _distance_standardized(X: np.ndarray, xq: np.ndarray, variable_weights: Optional[np.ndarray] = None) -> np.ndarray:
    X = as_2d(X)
    xq = np.asarray(xq, dtype=float).ravel()
    Xs, mu, sd = standardize(X)
    xqs = (xq - mu) / sd
    if variable_weights is None:
        wv = np.ones(X.shape[1])
    else:
        wv = np.asarray(variable_weights, dtype=float)
    return np.sqrt(np.sum(wv[None, :] * (Xs - xqs[None, :]) ** 2, axis=1))


def jit_predict(
    X: np.ndarray,
    y: np.ndarray,
    xq: np.ndarray,
    method: str = "LW-PLS",
    k: int = 35,
    bandwidth: float = 0.8,
    pls_components: int = 2,
    ridge_alpha: float = 1e-4,
) -> Dict[str, object]:
    X = as_2d(X)
    y = np.asarray(y, dtype=float).ravel()
    xq = np.asarray(xq, dtype=float).ravel()
    dist = _distance_standardized(X, xq)
    order = np.argsort(dist)
    k = int(max(1, min(k, len(y))))
    idx = order[:k]
    dscale = max(float(bandwidth), 1e-3)
    weights_all = np.exp(-(dist / dscale) ** 2)

    if method == "1-NN":
        pred = float(y[order[0]])
        eff_n = 1.0
        used_idx = order[:1]
        weights_used = np.array([1.0])
    elif method == "kNN mean":
        pred = float(np.mean(y[idx]))
        eff_n = float(k)
        used_idx = idx
        weights_used = np.ones(k) / k
    elif method == "weighted kNN":
        w = weights_all[idx]
        if np.sum(w) <= EPS:
            w = np.ones_like(w)
        w = w / np.sum(w)
        pred = float(np.sum(w * y[idx]))
        eff_n = float(1 / np.sum(w ** 2))
        used_idx = idx
        weights_used = w
    elif method == "local linear":
        w = weights_all[idx]
        if np.sum(w) <= EPS:
            w = np.ones_like(w)
        model = ridge_fit(X[idx], y[idx], alpha=ridge_alpha, standardize_x=True, sample_weight=w, name="local linear")
        pred = float(model.predict(xq.reshape(1, -1))[0])
        ww = w / np.sum(w)
        eff_n = float(1 / np.sum(ww ** 2))
        used_idx = idx
        weights_used = ww
    else:
        # Use all data but with exponentially decaying weights.  Very small
        # weights are harmless because ridge/PLS uses weighted covariance.
        w = weights_all.copy()
        if np.sum(w) <= EPS:
            w = np.ones_like(w)
        model = pls1_fit(X, y, n_components=pls_components, sample_weight=w, scale=True)
        pred = float(model.predict(xq.reshape(1, -1))[0])
        ww = w / np.sum(w)
        eff_n = float(1 / np.sum(ww ** 2))
        used_idx = np.argsort(w)[-min(len(w), max(k, 30)):]
        weights_used = ww[used_idx]

    return {
        "prediction": pred,
        "effective_n": eff_n,
        "nearest_distance": float(dist[order[0]]),
        "used_indices": used_idx,
        "used_weights": weights_used,
        "distances": dist,
    }


def simulate_jit(
    n_database: int = 420,
    x1_current: float = 0.5,
    x2_current: float = 0.0,
    wear_current: float = 0.85,
    nonlinearity: float = 1.3,
    drift_strength: float = 1.0,
    noise: float = 0.12,
    k: int = 35,
    bandwidth: float = 0.8,
    pls_components: int = 2,
    seed: int = 0,
) -> Dict[str, object]:
    X, y, db = make_jit_database(n_database, nonlinearity, drift_strength, noise, seed)
    xq = np.array([x1_current, x2_current, wear_current], dtype=float)
    y_true = float(_jit_true_function(xq.reshape(1, -1), nonlinearity, drift_strength)[0])

    global_model = pls1_fit(X, y, n_components=pls_components, scale=True)
    global_pred = float(global_model.predict(xq.reshape(1, -1))[0])

    methods = ["1-NN", "kNN mean", "weighted kNN", "local linear", "LW-PLS"]
    rows = [{"method": "global PLS", "prediction": global_pred, "abs_error_vs_truth": abs(global_pred - y_true), "effective_n": len(y), "nearest_distance": np.nan}]
    details = {}
    for m in methods:
        d = jit_predict(X, y, xq, method=m, k=k, bandwidth=bandwidth, pls_components=pls_components)
        details[m] = d
        rows.append({
            "method": m,
            "prediction": d["prediction"],
            "abs_error_vs_truth": abs(d["prediction"] - y_true),
            "effective_n": d["effective_n"],
            "nearest_distance": d["nearest_distance"],
        })
    metrics = pd.DataFrame(rows).sort_values("abs_error_vs_truth").reset_index(drop=True)
    return {"database": db, "X": X, "y": y, "xq": xq, "y_true": y_true, "metrics": metrics, "details": details}


# ---------------------------------------------------------------------------
# 4. Gray-box modeling simulator
# ---------------------------------------------------------------------------

def _gray_features(u: np.ndarray, temp: np.ndarray) -> np.ndarray:
    u = np.asarray(u)
    temp = np.asarray(temp)
    return np.column_stack([
        u,
        temp,
        u * temp,
        temp ** 2,
        np.sin(1.5 * u + 0.7 * temp),
        np.cos(u),
    ])


def make_graybox_data(
    n: int,
    physics_bias: float,
    parameter_drift: float,
    residual_strength: float,
    residual_observability: float,
    noise: float,
    seed: int,
) -> pd.DataFrame:
    g = rng(seed)
    u = g.uniform(0.2, 3.0, n)
    temp = g.uniform(-2.0, 2.0, n)
    theta_base = 2.0
    theta_true = theta_base + parameter_drift * np.tanh(temp)
    deterministic_residual = np.sin(1.5 * u + 0.7 * temp)
    unobserved_residual = g.normal(size=n)
    residual = residual_strength * (
        residual_observability * deterministic_residual
        + (1.0 - residual_observability) * unobserved_residual
    )
    y = theta_true * u + residual + g.normal(scale=noise, size=n)
    y_physics = (theta_base + physics_bias) * u
    return pd.DataFrame({
        "u": u,
        "temp": temp,
        "theta_true": theta_true,
        "residual_true": residual,
        "y": y,
        "y_physics": y_physics,
    })


def simulate_graybox(
    n_train: int = 120,
    n_test: int = 500,
    physics_bias: float = 0.3,
    parameter_drift: float = 0.7,
    residual_strength: float = 0.8,
    residual_observability: float = 0.8,
    noise: float = 0.10,
    ridge_alpha: float = 1.0,
    seed: int = 0,
) -> Dict[str, object]:
    train = make_graybox_data(n_train, physics_bias, parameter_drift, residual_strength, residual_observability, noise, seed)
    test = make_graybox_data(n_test, physics_bias, parameter_drift, residual_strength, residual_observability, noise, seed + 999)

    Xtr = _gray_features(train["u"].to_numpy(), train["temp"].to_numpy())
    Xte = _gray_features(test["u"].to_numpy(), test["temp"].to_numpy())
    ytr = train["y"].to_numpy()
    yte = test["y"].to_numpy()

    # Statistical only.
    stat = ridge_fit(Xtr, ytr, alpha=ridge_alpha, standardize_x=True, name="statistical")
    pred_stat = stat.predict(Xte)

    # Physical only.
    pred_phys = test["y_physics"].to_numpy()

    # Parallel gray-box: predict physical residual.
    res_tr = ytr - train["y_physics"].to_numpy()
    res_model = ridge_fit(Xtr, res_tr, alpha=ridge_alpha, standardize_x=True, name="parallel residual")
    pred_parallel = pred_phys + res_model.predict(Xte)

    # Serial gray-box: predict a physical parameter from operating conditions.
    theta_obs = ytr / np.maximum(train["u"].to_numpy(), 1e-6)
    theta_model = ridge_fit(_gray_features(train["u"].to_numpy(), train["temp"].to_numpy())[:, [0, 1, 2, 3]], theta_obs, alpha=ridge_alpha, standardize_x=True, name="serial theta")
    Xtheta_te = _gray_features(test["u"].to_numpy(), test["temp"].to_numpy())[:, [0, 1, 2, 3]]
    theta_hat_te = theta_model.predict(Xtheta_te)
    pred_serial = theta_hat_te * test["u"].to_numpy()

    # Combined: serial parameter adjustment + residual correction.
    pred_serial_tr = theta_model.predict(_gray_features(train["u"].to_numpy(), train["temp"].to_numpy())[:, [0, 1, 2, 3]]) * train["u"].to_numpy()
    res2 = ytr - pred_serial_tr
    res2_model = ridge_fit(Xtr, res2, alpha=ridge_alpha, standardize_x=True, name="combined residual")
    pred_combined = pred_serial + res2_model.predict(Xte)

    preds = {
        "Physical only": pred_phys,
        "Statistical only": pred_stat,
        "Parallel gray-box": pred_parallel,
        "Serial gray-box": pred_serial,
        "Combined gray-box": pred_combined,
    }

    metrics = pd.DataFrame([
        {"model": k, "RMSE_test": rmse(yte, v), "MAE_test": mae(yte, v), "R2_test": r2_score(yte, v)}
        for k, v in preds.items()
    ]).sort_values("RMSE_test").reset_index(drop=True)

    pred_df = test.copy()
    for k, v in preds.items():
        pred_df[k] = v

    return {"train": train, "test_predictions": pred_df, "metrics": metrics}


# ---------------------------------------------------------------------------
# 5. Transfer learning simulator
# ---------------------------------------------------------------------------

def _transfer_domain_data(
    n_source: int,
    n_target: int,
    n_test: int,
    n_common: int,
    n_source_unique: int,
    n_target_unique: int,
    domain_gap: float,
    noise: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    g = rng(seed)
    n_common = int(max(1, n_common))
    n_source_unique = int(max(0, n_source_unique))
    n_target_unique = int(max(0, n_target_unique))

    beta_common = g.normal(size=n_common)
    beta_common = beta_common / (np.linalg.norm(beta_common) + EPS)
    beta_target_shift = g.normal(size=n_common)
    beta_target_shift = beta_target_shift / (np.linalg.norm(beta_target_shift) + EPS)
    beta_common_s = beta_common
    beta_common_t = beta_common + domain_gap * beta_target_shift

    beta_s_u = g.normal(scale=0.5, size=n_source_unique)
    beta_t_u = g.normal(scale=0.5, size=n_target_unique)

    Xc_s = g.normal(size=(n_source, n_common))
    Xu_s = g.normal(size=(n_source, n_source_unique)) if n_source_unique else np.empty((n_source, 0))
    Xc_t = g.normal(loc=0.15 * domain_gap, size=(n_target, n_common))
    Xu_t = g.normal(size=(n_target, n_target_unique)) if n_target_unique else np.empty((n_target, 0))
    Xc_te = g.normal(loc=0.15 * domain_gap, size=(n_test, n_common))
    Xu_te = g.normal(size=(n_test, n_target_unique)) if n_target_unique else np.empty((n_test, 0))

    y_s = Xc_s @ beta_common_s + (Xu_s @ beta_s_u if n_source_unique else 0) + g.normal(scale=noise, size=n_source)
    y_t = Xc_t @ beta_common_t + (Xu_t @ beta_t_u if n_target_unique else 0) + g.normal(scale=noise, size=n_target)
    y_te = Xc_te @ beta_common_t + (Xu_te @ beta_t_u if n_target_unique else 0) + g.normal(scale=noise, size=n_test)

    return {
        "Xc_s": Xc_s, "Xu_s": Xu_s, "y_s": y_s,
        "Xc_t": Xc_t, "Xu_t": Xu_t, "y_t": y_t,
        "Xc_te": Xc_te, "Xu_te": Xu_te, "y_te": y_te,
    }


def _fehda_features(
    Xc: np.ndarray,
    Xu: np.ndarray,
    domain: str,
    n_source_unique: int,
    n_target_unique: int,
) -> np.ndarray:
    Xc = as_2d(Xc)
    Xu = as_2d(Xu) if Xu.size else np.empty((Xc.shape[0], 0))
    n, k = Xc.shape
    z_common_source = np.zeros((n, k))
    z_common_target = np.zeros((n, k))
    z_su = np.zeros((n, n_source_unique))
    z_tu = np.zeros((n, n_target_unique))
    if domain == "source":
        z_common_source = Xc.copy()
        if n_source_unique:
            z_su = Xu.copy()
    else:
        z_common_target = Xc.copy()
        if n_target_unique:
            z_tu = Xu.copy()
    return np.column_stack([Xc, z_common_source, z_common_target, z_su, z_tu])


def simulate_transfer(
    n_source: int = 300,
    n_target: int = 10,
    n_test: int = 300,
    n_common: int = 8,
    n_source_unique: int = 3,
    n_target_unique: int = 3,
    domain_gap: float = 0.35,
    transfer_weight: float = 0.6,
    ridge_alpha: float = 3.0,
    noise: float = 0.25,
    seed: int = 0,
) -> Dict[str, object]:
    d = _transfer_domain_data(
        n_source=n_source,
        n_target=n_target,
        n_test=n_test,
        n_common=n_common,
        n_source_unique=n_source_unique,
        n_target_unique=n_target_unique,
        domain_gap=domain_gap,
        noise=noise,
        seed=seed,
    )

    X_t = np.column_stack([d["Xc_t"], d["Xu_t"]])
    X_te = np.column_stack([d["Xc_te"], d["Xu_te"]])
    target_only = ridge_fit(X_t, d["y_t"], alpha=ridge_alpha, standardize_x=True, name="target only")
    pred_target_only = target_only.predict(X_te)

    # Common-only naive transfer: source and target common variables are assumed identical.
    X_common_train = np.vstack([d["Xc_s"], d["Xc_t"]])
    y_common_train = np.concatenate([d["y_s"], d["y_t"]])
    weights_common = np.concatenate([
        np.full(n_source, transfer_weight),
        np.ones(n_target),
    ])
    common_model = ridge_fit(X_common_train, y_common_train, alpha=ridge_alpha, standardize_x=True, sample_weight=weights_common, name="common naive")
    pred_common = common_model.predict(d["Xc_te"])

    # Target with source data handled by domain-augmented feature space.
    Fs = _fehda_features(d["Xc_s"], d["Xu_s"], "source", n_source_unique, n_target_unique)
    Ft = _fehda_features(d["Xc_t"], d["Xu_t"], "target", n_source_unique, n_target_unique)
    Fte = _fehda_features(d["Xc_te"], d["Xu_te"], "target", n_source_unique, n_target_unique)
    F_train = np.vstack([Fs, Ft])
    y_train = np.concatenate([d["y_s"], d["y_t"]])
    weights = np.concatenate([np.full(n_source, transfer_weight), np.ones(n_target)])
    fehda_model = ridge_fit(F_train, y_train, alpha=ridge_alpha, standardize_x=True, sample_weight=weights, name="FEHDA-like")
    pred_fehda = fehda_model.predict(Fte)

    # An optimistic reference: target-only with many target samples.
    d_ref = _transfer_domain_data(
        n_source=0,
        n_target=max(200, n_target * 10),
        n_test=n_test,
        n_common=n_common,
        n_source_unique=n_source_unique,
        n_target_unique=n_target_unique,
        domain_gap=domain_gap,
        noise=noise,
        seed=seed,
    )
    X_ref = np.column_stack([d_ref["Xc_t"], d_ref["Xu_t"]])
    X_ref_te = np.column_stack([d["Xc_te"], d["Xu_te"]])
    ref_model = ridge_fit(X_ref, d_ref["y_t"], alpha=ridge_alpha, standardize_x=True, name="target enough data")
    pred_ref = ref_model.predict(X_ref_te)

    preds = {
        "Target only": pred_target_only,
        "Naive common transfer": pred_common,
        "Domain-augmented transfer": pred_fehda,
        "Reference: many target samples": pred_ref,
    }
    metrics = pd.DataFrame([
        {"model": k, "RMSE_target_test": rmse(d["y_te"], v), "R2_target_test": r2_score(d["y_te"], v)}
        for k, v in preds.items()
    ]).sort_values("RMSE_target_test").reset_index(drop=True)

    return {"data": d, "metrics": metrics, "predictions": preds}


# ---------------------------------------------------------------------------
# 6. Raw data diagnostics simulator
# ---------------------------------------------------------------------------

def make_raw_signal(
    n: int = 300,
    outlier_strength: float = 4.0,
    range_shift: float = 1.0,
    trend_strength: float = 0.5,
    cycle_strength: float = 0.8,
    missing_rate: float = 0.05,
    lower_clip: float = 0.0,
    two_modes: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    g = rng(seed)
    t = np.arange(n)
    base = g.normal(scale=0.25, size=n)
    y = base.copy()
    y += trend_strength * (t / max(n - 1, 1))
    y += cycle_strength * np.sin(2 * np.pi * t / max(20, n // 8))
    if range_shift != 0:
        y[t >= n // 2] += range_shift
    if two_modes > 0:
        group = g.binomial(1, 0.5, size=n)
        y += two_modes * group
    if outlier_strength > 0:
        n_out = max(1, n // 40)
        idx = g.choice(n, size=n_out, replace=False)
        y[idx] += outlier_strength * g.choice([-1, 1], size=n_out)
    if lower_clip > 0:
        clip_level = np.quantile(y, lower_clip)
        y = np.maximum(y, clip_level)
    if missing_rate > 0:
        m = g.random(n) < missing_rate
        y[m] = np.nan
    return pd.DataFrame({"time": t, "value": y})


def diagnose_raw_signal(df: pd.DataFrame) -> pd.DataFrame:
    y = df["value"].to_numpy(dtype=float)
    t = df["time"].to_numpy(dtype=float)
    finite = np.isfinite(y)
    yf = y[finite]
    tf = t[finite]
    rows = []
    missing_rate = 1 - len(yf) / max(len(y), 1)
    rows.append({"check": "missing_rate", "value": missing_rate, "flag": missing_rate > 0.02, "meaning": "欠損が多いほど、有効サンプル数と時系列の連続性が落ちる"})

    if len(yf) < 5:
        rows.append({"check": "enough_data", "value": len(yf), "flag": True, "meaning": "有効データが少なすぎる"})
        return pd.DataFrame(rows)

    med = np.median(yf)
    mad = np.median(np.abs(yf - med)) + EPS
    robust_z = 0.6745 * (yf - med) / mad
    outlier_rate = float(np.mean(np.abs(robust_z) > 3.5))
    rows.append({"check": "outlier_rate", "value": outlier_rate, "flag": outlier_rate > 0.01, "meaning": "外れ値があると、OLSやPCAの軸が引っ張られやすい"})

    # Range shift: compare first/second half medians.
    first = yf[tf < np.median(tf)]
    second = yf[tf >= np.median(tf)]
    pooled_sd = np.std(yf) + EPS
    shift_score = abs(np.median(second) - np.median(first)) / pooled_sd if len(first) and len(second) else 0.0
    rows.append({"check": "range_shift_score", "value": float(shift_score), "flag": shift_score > 0.8, "meaning": "レンジ変更や運転モード変更がある可能性。全期間を一つのモデルで扱うと危険"})

    # Trend: normalized slope.
    A = np.column_stack([np.ones_like(tf), (tf - tf.mean()) / (tf.std() + EPS)])
    beta = np.linalg.pinv(A) @ yf
    yhat = A @ beta
    slope_score = abs(beta[1]) / (np.std(yf) + EPS)
    rows.append({"check": "trend_score", "value": float(slope_score), "flag": slope_score > 0.25, "meaning": "ドリフトや劣化の可能性。固定モデルの精度劣化につながる"})

    # Periodicity: FFT peak ratio after detrending.
    resid = yf - yhat
    if len(resid) > 16:
        spec = np.abs(np.fft.rfft(resid - resid.mean())) ** 2
        if len(spec) > 2:
            peak_ratio = float(np.max(spec[1:]) / (np.sum(spec[1:]) + EPS))
        else:
            peak_ratio = 0.0
    else:
        peak_ratio = 0.0
    rows.append({"check": "periodicity_score", "value": peak_ratio, "flag": peak_ratio > 0.25, "meaning": "周期変動があるなら、周期をまたぐデータ分割・モデル検証が必要"})

    # Lower clipping: repeated minimum.
    ymin = np.nanmin(yf)
    clip_rate = float(np.mean(np.isclose(yf, ymin, atol=max(1e-6, 0.005 * (np.nanmax(yf) - ymin + EPS)))))
    rows.append({"check": "lower_clip_rate", "value": clip_rate, "flag": clip_rate > 0.05, "meaning": "下限張り付きは真値ではなく測定限界かもしれない"})

    # Multimodality: simple histogram peak count.
    hist, edges = np.histogram(yf, bins="auto")
    peaks = 0
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > max(3, 0.08 * len(yf)):
            peaks += 1
    rows.append({"check": "histogram_peak_count", "value": float(peaks), "flag": peaks >= 2, "meaning": "山が複数あるなら、複数の運転モード・製品グレードが混ざっている可能性"})

    return pd.DataFrame(rows)
