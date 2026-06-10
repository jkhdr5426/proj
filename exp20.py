
#!/usr/bin/env python3
"""
Experiment 20: Fixed-α Discrimination Test for BP/SP on LWE
============================================================

Goal
----
Test whether the message-passing observables you defined in Exp 15/19 are
*only* functions of alpha = q / (sigma * sqrt(n)), or whether they still
separate instances when alpha is held approximately fixed.

This script has three main parts:

A. Alpha-collapse check
   - Hold alpha fixed (up to integer q rounding)
   - Vary decomposition across (n, sigma, q)
   - Measure whether SP/BP observables collapse onto alpha

B. Fixed-alpha secret-family scan
   - Hold (alpha, n, sigma, q) fixed
   - Vary secret distribution
   - Ask whether SP/BP observables distinguish secret geometry beyond alpha

C. Optional matrix-structure scan
   - Hold (alpha, secret family) fixed
   - Vary A-matrix family
   - Ask whether observables respond to matrix structure beyond alpha

Outputs
-------
  exp20_results.png
  exp20_summary.txt
  exp20_data.json

Run
---
  sage -python exp20.py
  sage -python exp20.py --quick
  sage -python exp20.py --sp-only

Notes
-----
This script reuses the Exp 15 machinery:
  - lattice-estimator if available
  - otherwise BKZ proxy
  - LWEsurveyPropagation from exp15.py

The intent is to see whether warning_density / survey_entropy / overlap_proxy
contain information beyond alpha.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

try:
    import exp15 as base
except Exception as exc:
    raise SystemExit(
        "exp15.py not found or not importable. Place exp20.py next to exp15.py."
    ) from exc

ESTIMATOR_AVAILABLE = getattr(base, "ESTIMATOR_AVAILABLE", False)
DEFAULT_JOBS = getattr(base, "DEFAULT_JOBS", 4)

ALPHA_TARGETS = [6.0, 9.0, 12.0, 18.0]
N_GRID = [24, 32, 48, 64, 96]
SIGMA_GRID = [0.5, 1.0, 1.5, 2.0]

SECRET_FAMILIES = ["uniform", "binary", "ternary", "sparse", "small_gaussian"]
MATRIX_FAMILIES = ["random_dense", "repeated_rows", "low_rank", "sparse_binary"]

# canonical point for "same alpha, change the structure" scans
CANON_N = 48
CANON_SIGMA = 1.0

# quick mode reductions
QUICK_ALPHA_TARGETS = [6.0, 12.0]
QUICK_N_GRID = [24, 48, 64]
QUICK_SIGMA_GRID = [0.5, 1.5]
QUICK_SECRET_FAMILIES = ["uniform", "binary", "ternary"]
QUICK_MATRIX_FAMILIES = ["random_dense", "repeated_rows"]

BG = "#0f1117"
PBG = "#1a1d27"

def fmt(x: Optional[float], d: int = 4) -> str:
    return "n/a" if x is None or not np.isfinite(float(x)) else f"{float(x):.{d}f}"

def ensure_odd(q: int) -> int:
    q = int(q)
    return q if q % 2 == 1 else q + 1

def corrcoef_nan(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    m = np.isfinite(xa) & np.isfinite(ya)
    if m.sum() < 3 or np.std(xa[m]) == 0 or np.std(ya[m]) == 0:
        return None
    return float(np.corrcoef(xa[m], ya[m])[0, 1])

def spearman_nan(x: Sequence[float], y: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    m = np.isfinite(xa) & np.isfinite(ya)
    if m.sum() < 3:
        return None, None
    r, p = scipy_stats.spearmanr(xa[m], ya[m])
    return float(r), float(p)

def bootstrap_ci_mean(vals: Sequence[float], n_boot: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(123)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), len(arr))
        boots.append(float(np.mean(arr[idx])))
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return lo, hi

def get_estimator(n: int, q: int, sigma: float, use_proxy: bool):
    if use_proxy or not ESTIMATOR_AVAILABLE:
        return base.bkz_proxy_bits(n=n, q=q, sigma=sigma)
    return base.run_estimator(n=n, q=q, sigma=sigma, jobs=DEFAULT_JOBS)

def alpha_from_params(q: int, sigma: float, n: int) -> float:
    return float(q) / (float(sigma) * math.sqrt(float(n)))

def robust_mode(items: Sequence[str]) -> str:
    vals, counts = np.unique(np.asarray(items, dtype=object), return_counts=True)
    return str(vals[int(np.argmax(counts))])

def one_hot(s: np.ndarray, q: int) -> np.ndarray:
    out = np.zeros(q, dtype=np.float64)
    for x in s:
        out[int(x) % q] += 1.0
    return out / max(1.0, out.sum())

def sample_secret(q: int, n: int, family: str, rng: np.random.Generator) -> np.ndarray:
    family = family.lower()
    if family == "uniform":
        return rng.integers(0, q, size=n, dtype=np.int64)

    if family == "binary":
        return rng.integers(0, 2, size=n, dtype=np.int64) % q

    if family == "ternary":
        vals = rng.choice([-1, 0, 1], size=n, p=[0.25, 0.50, 0.25]).astype(np.int64)
        return np.remainder(vals, q).astype(np.int64)

    if family == "sparse":
        s = np.zeros(n, dtype=np.int64)
        k = max(1, n // 8)
        idx = rng.choice(n, size=k, replace=False)
        s[idx] = rng.choice([-1, 1], size=k).astype(np.int64)
        return np.remainder(s, q).astype(np.int64)

    if family == "small_gaussian":
        # Small centered secret, then reduced mod q.
        vals = np.round(rng.normal(0.0, 1.0, size=n)).astype(np.int64)
        return np.remainder(vals, q).astype(np.int64)

    raise ValueError(f"unknown secret family: {family}")

def make_matrix(n: int, m: int, q: int, family: str, rng: np.random.Generator) -> np.ndarray:
    family = family.lower()

    if family == "random_dense":
        return rng.integers(0, q, size=(m, n), dtype=np.int64)

    if family == "repeated_rows":
        # Build a smaller random template and repeat rows.
        template_rows = max(1, m // 4)
        A0 = rng.integers(0, q, size=(template_rows, n), dtype=np.int64)
        reps = int(math.ceil(m / template_rows))
        A = np.vstack([A0 for _ in range(reps)])[:m]
        perm = rng.permutation(m)
        return A[perm]

    if family == "low_rank":
        # A = U V mod q with small rank.
        r = max(2, min(n // 4, 8))
        U = rng.integers(0, q, size=(m, r), dtype=np.int64)
        V = rng.integers(0, q, size=(r, n), dtype=np.int64)
        A = (U @ V) % q
        return A.astype(np.int64)

    if family == "sparse_binary":
        p = min(0.20, max(0.03, 2.0 / math.sqrt(max(1, n))))
        A = (rng.random((m, n)) < p).astype(np.int64)
        return A % q

    raise ValueError(f"unknown matrix family: {family}")

def generate_custom_lwe(
    n: int,
    m: int,
    q: int,
    sigma: float,
    seed: int,
    secret_family: str = "uniform",
    matrix_family: str = "random_dense",
) -> base.LWEInstance:
    rng = np.random.default_rng(seed)
    A = make_matrix(n=n, m=m, q=q, family=matrix_family, rng=rng)
    s = sample_secret(q=q, n=n, family=secret_family, rng=rng)
    e = np.round(rng.normal(0.0, sigma, size=m)).astype(np.int64)
    b = (A @ s + e) % q
    idx = np.arange(m, dtype=np.int64)
    return base.LWEInstance(n=n, m=m, q=q, sigma=sigma, A=A, b=b, s_true=s, row_idx=idx)

def run_message_passing(
    lwe: base.LWEInstance,
    bp_iter: int = 60,
    sp_iter: int = 20,
    n_restarts: int = 8,
    light_sp: bool = False,
) -> Dict[str, Any]:
    sp = base.LWEsurveyPropagation(lwe)
    return sp.run(max_iter=sp_iter, n_restarts=n_restarts, bp_iter=bp_iter, light_sp=light_sp)

def run_one_point(
    n: int,
    sigma: float,
    alpha_target: float,
    seed: int,
    secret_family: str,
    matrix_family: str,
    use_proxy: bool,
    quick: bool,
) -> Dict[str, Any]:
    m = 3 * n
    q = ensure_odd(max(n + 2, int(round(alpha_target * sigma * math.sqrt(n)))))
    lwe = generate_custom_lwe(
        n=n,
        m=m,
        q=q,
        sigma=sigma,
        seed=seed,
        secret_family=secret_family,
        matrix_family=matrix_family,
    )
    actual_alpha = alpha_from_params(q=q, sigma=sigma, n=n)

    if quick:
        sp_metrics = run_message_passing(lwe, bp_iter=40, sp_iter=10, n_restarts=4, light_sp=False)
    else:
        sp_metrics = run_message_passing(lwe, bp_iter=60, sp_iter=20, n_restarts=8, light_sp=False)

    sec = get_estimator(n, q, sigma, use_proxy)
    return {
        "n": n,
        "m": m,
        "q": q,
        "sigma": sigma,
        "alpha_target": alpha_target,
        "alpha_actual": actual_alpha,
        "seed": seed,
        "secret_family": secret_family,
        "matrix_family": matrix_family,
        "hardness": sec.get("best_bits"),
        "family": sec.get("best_family", "other"),
        "dominance_gap": sec.get("dominance_gap"),
        "sp": sp_metrics,
    }

def aggregate_runs(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(rows[0])
    keys = ["warning_density", "survey_entropy_norm", "overlap_proxy",
            "complexity_proxy", "map_accuracy", "decimation_accuracy",
            "bp_delta", "sp_delta", "xi_ratio"]
    sp_agg = {}
    for k in keys:
        vals = [r["sp"].get(k) for r in rows if r["sp"].get(k) is not None and np.isfinite(r["sp"].get(k, np.nan))]
        sp_agg[k] = float(np.mean(vals)) if vals else None
    out["sp"] = sp_agg
    out["hardness"] = float(np.mean([r["hardness"] for r in rows if r["hardness"] is not None]))
    out["dominance_gap"] = float(np.mean([r["dominance_gap"] for r in rows if r["dominance_gap"] is not None])) if any(r["dominance_gap"] is not None for r in rows) else None
    out["family"] = robust_mode([r["family"] for r in rows if r["family"]])
    return out

def summarize_group(rows: List[Dict[str, Any]], metric: str) -> Tuple[float, float, float]:
    vals = [r["sp"].get(metric) for r in rows if r["sp"].get(metric) is not None and np.isfinite(r["sp"].get(metric, np.nan))]
    if not vals:
        return float("nan"), float("nan"), float("nan")
    vals = np.asarray(vals, dtype=np.float64)
    mn = float(np.mean(vals))
    sd = float(np.std(vals))
    cv = sd / (abs(mn) + 1e-9)
    return mn, sd, cv

def kruskal_by_family(rows: List[Dict[str, Any]], metric: str, key: str) -> Tuple[Optional[float], Optional[float]]:
    groups: Dict[str, List[float]] = {}
    for r in rows:
        fam = str(r[key])
        val = r["sp"].get(metric)
        if val is None or not np.isfinite(val):
            continue
        groups.setdefault(fam, []).append(float(val))
    if len(groups) < 2:
        return None, None
    data = [groups[k] for k in groups if len(groups[k]) >= 2]
    if len(data) < 2:
        return None, None
    h, p = scipy_stats.kruskal(*data)
    return float(h), float(p)

def nearest_centroid_accuracy(rows: List[Dict[str, Any]], label_key: str, metric_keys: List[str]) -> Tuple[float, int]:
    # Simple leave-one-out nearest centroid classifier on z-scored features.
    X = []
    y = []
    for r in rows:
        feat = []
        ok = True
        for k in metric_keys:
            v = r["sp"].get(k)
            if v is None or not np.isfinite(v):
                ok = False
                break
            feat.append(float(v))
        if ok:
            X.append(feat)
            y.append(str(r[label_key]))
    if len(X) < 4:
        return float("nan"), 0
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=object)
    # z-score
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    classes = sorted(set(y.tolist()))
    if len(classes) < 2:
        return float("nan"), len(classes)

    preds = []
    for i in range(len(X)):
        mask = np.ones(len(X), dtype=bool)
        mask[i] = False
        centroids = {}
        for c in classes:
            xc = X[mask & (y == c)]
            if len(xc) == 0:
                continue
            centroids[c] = xc.mean(axis=0)
        if not centroids:
            preds.append(classes[0])
            continue
        dists = {c: float(np.linalg.norm(X[i] - mu)) for c, mu in centroids.items()}
        preds.append(min(dists.items(), key=lambda kv: kv[1])[0])
    acc = float(np.mean(np.asarray(preds, dtype=object) == y))
    return acc, len(classes)

def part_a_alpha_collapse(use_proxy: bool, quick: bool) -> List[Dict[str, Any]]:
    print("\n" + "=" * 72)
    print("PART A: ALPHA COLLAPSE CHECK")
    print("  Hold alpha fixed while varying (n, sigma, q).")
    print("=" * 72)

    alphas = QUICK_ALPHA_TARGETS if quick else ALPHA_TARGETS
    ns = QUICK_N_GRID if quick else N_GRID
    sigmas = QUICK_SIGMA_GRID if quick else SIGMA_GRID
    families = ["uniform"]  # isolate alpha first
    seeds = 2 if quick else 4
    rows: List[Dict[str, Any]] = []

    for alpha in alphas:
        for n in ns:
            for sigma in sigmas:
                q = ensure_odd(max(n + 2, int(round(alpha * sigma * math.sqrt(n)))))
                for s in range(seeds):
                    rows.append(
                        run_one_point(
                            n=n,
                            sigma=sigma,
                            alpha_target=alpha,
                            seed=1000 + 31 * s + 17 * n + int(10 * sigma),
                            secret_family="uniform",
                            matrix_family="random_dense",
                            use_proxy=use_proxy,
                            quick=quick,
                        )
                    )

    print("\n  Alpha-collapse summary (uniform secrets only):")
    for alpha in alphas:
        sub = [r for r in rows if abs(r["alpha_target"] - alpha) < 1e-9]
        warn_mean, warn_sd, warn_cv = summarize_group(sub, "warning_density")
        hs_mean, hs_sd, hs_cv = summarize_group(sub, "survey_entropy_norm")
        cp_mean, cp_sd, cp_cv = summarize_group(sub, "complexity_proxy")
        ra, pa = spearman_nan([r["alpha_actual"] for r in sub], [r["sp"]["warning_density"] for r in sub])
        print(
            f"  alpha={alpha:>5.1f}  warn_d={warn_mean:.4f}±{warn_sd:.4f} CV={warn_cv:.4f}  "
            f"H_surv={hs_mean:.4f} CV={hs_cv:.4f}  comp={cp_mean:.4f} CV={cp_cv:.4f}  "
            f"r(alpha, warn_d)={fmt(ra,3)} p={fmt(pa,3)}"
        )
    return rows

def part_b_secret_family_scan(use_proxy: bool, quick: bool) -> List[Dict[str, Any]]:
    print("\n" + "=" * 72)
    print("PART B: FIXED-ALPHA SECRET-FAMILY SCAN")
    print("  Hold (alpha, n, sigma, q) fixed; vary the secret distribution.")
    print("=" * 72)

    families = QUICK_SECRET_FAMILIES if quick else SECRET_FAMILIES
    alphas = QUICK_ALPHA_TARGETS if quick else ALPHA_TARGETS
    seeds = 3 if quick else 5
    rows: List[Dict[str, Any]] = []

    # Canonical decomposition per alpha.
    for alpha in alphas:
        n = CANON_N if not quick else 32
        sigma = CANON_SIGMA if not quick else 1.0
        q = ensure_odd(max(n + 2, int(round(alpha * sigma * math.sqrt(n)))))
        for fam in families:
            for s in range(seeds):
                rows.append(
                    run_one_point(
                        n=n,
                        sigma=sigma,
                        alpha_target=alpha,
                        seed=2000 + 41 * s + 13 * n + int(100 * sigma),
                        secret_family=fam,
                        matrix_family="random_dense",
                        use_proxy=use_proxy,
                        quick=quick,
                    )
                )

    print("\n  Family-scan summary:")
    metric_keys = ["warning_density", "survey_entropy_norm", "overlap_proxy", "complexity_proxy"]
    for alpha in alphas:
        sub = [r for r in rows if abs(r["alpha_target"] - alpha) < 1e-9]
        print(f"\n  alpha={alpha:.1f}")
        for fam in families:
            fam_rows = [r for r in sub if r["secret_family"] == fam]
            warn_mean, warn_sd, warn_cv = summarize_group(fam_rows, "warning_density")
            hs_mean, hs_sd, hs_cv = summarize_group(fam_rows, "survey_entropy_norm")
            print(
                f"    {fam:>14}: warn_d={warn_mean:.4f}±{warn_sd:.4f} CV={warn_cv:.4f}  "
                f"H_surv={hs_mean:.4f} CV={hs_cv:.4f}"
            )
        for metric in metric_keys:
            h, p = kruskal_by_family(sub, metric, "secret_family")
            print(f"    Kruskal {metric:>18}: H={fmt(h,3)}  p={fmt(p,4)}")
    return rows

def part_c_matrix_family_scan(use_proxy: bool, quick: bool) -> List[Dict[str, Any]]:
    print("\n" + "=" * 72)
    print("PART C: FIXED-ALPHA MATRIX-FAMILY SCAN")
    print("  Hold alpha and secret family fixed; vary A-matrix structure.")
    print("=" * 72)

    matrix_families = QUICK_MATRIX_FAMILIES if quick else MATRIX_FAMILIES
    alpha = 12.0 if not quick else 6.0
    n = CANON_N if not quick else 32
    sigma = CANON_SIGMA if not quick else 1.0
    seeds = 3 if quick else 5
    rows: List[Dict[str, Any]] = []

    for fam in matrix_families:
        for s in range(seeds):
            rows.append(
                run_one_point(
                    n=n,
                    sigma=sigma,
                    alpha_target=alpha,
                    seed=3000 + 53 * s + int(7 * n),
                    secret_family="uniform",
                    matrix_family=fam,
                    use_proxy=use_proxy,
                    quick=quick,
                )
            )

    print("\n  Matrix-family summary:")
    for fam in matrix_families:
        fam_rows = [r for r in rows if r["matrix_family"] == fam]
        warn_mean, warn_sd, warn_cv = summarize_group(fam_rows, "warning_density")
        hs_mean, hs_sd, hs_cv = summarize_group(fam_rows, "survey_entropy_norm")
        cp_mean, cp_sd, cp_cv = summarize_group(fam_rows, "complexity_proxy")
        print(
            f"    {fam:>14}: warn_d={warn_mean:.4f}±{warn_sd:.4f} CV={warn_cv:.4f}  "
            f"H_surv={hs_mean:.4f} CV={hs_cv:.4f}  comp={cp_mean:.4f} CV={cp_cv:.4f}"
        )

    for metric in ["warning_density", "survey_entropy_norm", "overlap_proxy", "complexity_proxy"]:
        h, p = kruskal_by_family(rows, metric, "matrix_family")
        print(f"    Kruskal {metric:>18}: H={fmt(h,3)}  p={fmt(p,4)}")
    return rows

def part_d_classification(rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 72)
    print("PART D: CAN SP/BP OBSERVABLES PREDICT FAMILY AT FIXED ALPHA?")
    print("=" * 72)

    metric_keys = ["warning_density", "survey_entropy_norm", "overlap_proxy", "complexity_proxy"]
    acc_secret, n_cls_secret = nearest_centroid_accuracy(rows, "secret_family", metric_keys)
    acc_matrix, n_cls_matrix = nearest_centroid_accuracy(rows, "matrix_family", metric_keys)

    chance_secret = 1.0 / max(1, n_cls_secret)
    chance_matrix = 1.0 / max(1, n_cls_matrix)

    print(f"  Secret-family LOOCV nearest-centroid acc = {fmt(acc_secret,3)} (chance ~ {fmt(chance_secret,3)})")
    print(f"  Matrix-family LOOCV nearest-centroid acc = {fmt(acc_matrix,3)} (chance ~ {fmt(chance_matrix,3)})")

def part_e_stability(use_proxy: bool, quick: bool) -> List[Dict[str, Any]]:
    print("\n" + "=" * 72)
    print("PART E: SEED STABILITY")
    print("  Which SP metrics remain stable across seeds?")
    print("=" * 72)

    configs = [
        (32, 0.5, 6.0),
        (48, 1.0, 9.0),
        (64, 1.0, 12.0),
        (64, 2.0, 18.0),
    ]
    if quick:
        configs = configs[:2]
    seeds = 5 if quick else 20
    rows: List[Dict[str, Any]] = []

    for n, sigma, alpha in configs:
        sub = []
        for s in range(seeds):
            sub.append(
                run_one_point(
                    n=n,
                    sigma=sigma,
                    alpha_target=alpha,
                    seed=4000 + 71 * s + n,
                    secret_family="uniform",
                    matrix_family="random_dense",
                    use_proxy=use_proxy,
                    quick=quick,
                )
            )
        row = {"n": n, "sigma": sigma, "alpha": alpha}
        print(f"\n  config n={n}, sigma={sigma}, alpha={alpha}")
        for metric in ["warning_density", "survey_entropy_norm", "overlap_proxy", "complexity_proxy", "map_accuracy", "bp_delta"]:
            vals = [r["sp"].get(metric) for r in sub if r["sp"].get(metric) is not None and np.isfinite(r["sp"].get(metric, np.nan))]
            if not vals:
                continue
            mn = float(np.mean(vals))
            sd = float(np.std(vals))
            cv = sd / (abs(mn) + 1e-9)
            row[f"{metric}_mean"] = mn
            row[f"{metric}_cv"] = cv
            verdict = "STABLE" if cv < 0.10 else ("usable" if cv < 0.20 else "NOISY")
            print(f"    {metric:>22}: mean={mn:.4f}  sd={sd:.4f}  CV={cv:.4f}  {verdict}")
        rows.append(row)

    return rows

def plot_results(a: List[Dict[str, Any]],
                 b: List[Dict[str, Any]],
                 c: List[Dict[str, Any]],
                 e: List[Dict[str, Any]],
                 save: str = "exp20_results.png") -> None:
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    def sty(ax, title, xl="", yl=""):
        ax.set_facecolor(PBG)
        ax.set_title(title, color="white", fontsize=9)
        ax.set_xlabel(xl, color="white", fontsize=8)
        ax.set_ylabel(yl, color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    # A: warning density vs alpha
    ax0 = fig.add_subplot(gs[0, 0])
    if a:
        xs = [r["alpha_actual"] for r in a]
        ys = [r["sp"]["warning_density"] for r in a]
        cs = [r["n"] for r in a]
        sc = ax0.scatter(xs, ys, c=cs, cmap="plasma", s=35, edgecolors="white", lw=0.2)
        plt.colorbar(sc, ax=ax0, fraction=0.046)
    sty(ax0, "A: warning_density vs actual alpha", "alpha", "warning_density")

    # B: family scan scatter
    ax1 = fig.add_subplot(gs[0, 1])
    if b:
        fams = sorted(set(r["secret_family"] for r in b))
        fam_colors = {f: plt.cm.tab10(i / max(1, len(fams) - 1)) for i, f in enumerate(fams)}
        for fam in fams:
            rr = [r for r in b if r["secret_family"] == fam]
            ax1.scatter(
                [r["sp"]["warning_density"] for r in rr],
                [r["sp"]["survey_entropy_norm"] for r in rr],
                label=fam,
                s=40,
                alpha=0.9,
                edgecolors="white",
                lw=0.2,
                color=fam_colors[fam],
            )
        ax1.legend(fontsize=7, facecolor=PBG, labelcolor="white")
    sty(ax1, "B: family scan in SP-space", "warning_density", "survey_entropy")

    # C: matrix scan scatter
    ax2 = fig.add_subplot(gs[0, 2])
    if c:
        fams = sorted(set(r["matrix_family"] for r in c))
        fam_colors = {f: plt.cm.tab10(i / max(1, len(fams) - 1)) for i, f in enumerate(fams)}
        for fam in fams:
            rr = [r for r in c if r["matrix_family"] == fam]
            ax2.scatter(
                [r["sp"]["warning_density"] for r in rr],
                [r["sp"]["complexity_proxy"] for r in rr],
                label=fam,
                s=40,
                alpha=0.9,
                edgecolors="white",
                lw=0.2,
                color=fam_colors[fam],
            )
        ax2.legend(fontsize=7, facecolor=PBG, labelcolor="white")
    sty(ax2, "C: matrix scan in SP-space", "warning_density", "complexity_proxy")

    # E: stability
    ax3 = fig.add_subplot(gs[1, :])
    if e:
        metrics = ["warning_density", "survey_entropy_norm", "overlap_proxy", "complexity_proxy"]
        labels = [f"n={r['n']},σ={r['sigma']},α={r['alpha']}" for r in e]
        mat = np.full((len(metrics), len(labels)), np.nan)
        for j, row in enumerate(e):
            for i, k in enumerate(metrics):
                mat[i, j] = row.get(f"{k}_cv", np.nan)
        im = ax3.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.4)
        ax3.set_xticks(range(len(labels)))
        ax3.set_yticks(range(len(metrics)))
        ax3.set_xticklabels(labels, rotation=0, fontsize=7, color="white")
        ax3.set_yticklabels(metrics, fontsize=8, color="white")
        for i in range(len(metrics)):
            for j in range(len(labels)):
                if np.isfinite(mat[i, j]):
                    ax3.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", fontsize=6, color="black")
        plt.colorbar(im, ax=ax3, fraction=0.015)
    sty(ax3, "E: metric stability (CV)", "config", "metric")

    fig.suptitle(
        "Experiment 20 — Fixed-α Discrimination Test\n"
        "Does the BP/SP observable collapse onto α, or does it see structure beyond α?",
        color="white",
        fontsize=12,
        y=1.01,
    )
    plt.savefig(save, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Plot → {save}")

def write_summary(a: List[Dict[str, Any]],
                  b: List[Dict[str, Any]],
                  c: List[Dict[str, Any]],
                  e: List[Dict[str, Any]],
                  path: str = "exp20_summary.txt") -> None:
    sep = "=" * 72
    with open(path, "w", encoding="utf-8") as f:
        f.write(sep + "\nEXPERIMENT 20 — FIXED-α DISCRIMINATION TEST\n" + sep + "\n\n")

        f.write("PART A: ALPHA COLLAPSE\n" + "-" * 40 + "\n")
        for alpha in sorted(set(r["alpha_target"] for r in a)):
            sub = [r for r in a if abs(r["alpha_target"] - alpha) < 1e-9]
            warn = [r["sp"]["warning_density"] for r in sub if r["sp"].get("warning_density") is not None]
            hs = [r["sp"]["survey_entropy_norm"] for r in sub if r["sp"].get("survey_entropy_norm") is not None]
            comp = [r["sp"]["complexity_proxy"] for r in sub if r["sp"].get("complexity_proxy") is not None]
            ra, pa = spearman_nan([r["alpha_actual"] for r in sub], [r["sp"]["warning_density"] for r in sub])
            f.write(
                f"  alpha={alpha:>5.1f}: warn_d range=[{min(warn):.4f},{max(warn):.4f}] "
                f"H_surv_mean={np.mean(hs):.4f}  comp_mean={np.mean(comp):.4f}  "
                f"r(alpha, warn_d)={fmt(ra,3)}  p={fmt(pa,3)}\n"
            )
        f.write("\n")

        f.write("PART B: SECRET-FAMILY SCAN\n" + "-" * 40 + "\n")
        for alpha in sorted(set(r["alpha_target"] for r in b)):
            sub = [r for r in b if abs(r["alpha_target"] - alpha) < 1e-9]
            f.write(f"  alpha={alpha:.1f}\n")
            fams = sorted(set(r["secret_family"] for r in sub))
            for fam in fams:
                rr = [r for r in sub if r["secret_family"] == fam]
                mn, sd, cv = summarize_group(rr, "warning_density")
                mh, sh, cvh = summarize_group(rr, "survey_entropy_norm")
                f.write(
                    f"    {fam:>14}: warn_d={mn:.4f}±{sd:.4f} CV={cv:.4f}  "
                    f"H_surv={mh:.4f}±{sh:.4f} CV={cvh:.4f}\n"
                )
            for metric in ["warning_density", "survey_entropy_norm", "overlap_proxy", "complexity_proxy"]:
                h, p = kruskal_by_family(sub, metric, "secret_family")
                f.write(f"    Kruskal {metric:>18}: H={fmt(h,3)}  p={fmt(p,4)}\n")
        f.write("\n")

        f.write("PART C: MATRIX-FAMILY SCAN\n" + "-" * 40 + "\n")
        if c:
            for fam in sorted(set(r["matrix_family"] for r in c)):
                rr = [r for r in c if r["matrix_family"] == fam]
                mn, sd, cv = summarize_group(rr, "warning_density")
                mh, sh, cvh = summarize_group(rr, "survey_entropy_norm")
                mc, sc, cvc = summarize_group(rr, "complexity_proxy")
                f.write(
                    f"  {fam:>14}: warn_d={mn:.4f}±{sd:.4f} CV={cv:.4f}  "
                    f"H_surv={mh:.4f} CV={cvh:.4f}  comp={mc:.4f} CV={cvc:.4f}\n"
                )
            for metric in ["warning_density", "survey_entropy_norm", "overlap_proxy", "complexity_proxy"]:
                h, p = kruskal_by_family(c, metric, "matrix_family")
                f.write(f"    Kruskal {metric:>18}: H={fmt(h,3)}  p={fmt(p,4)}\n")
        f.write("\n")

        f.write("PART E: STABILITY\n" + "-" * 40 + "\n")
        for row in e:
            f.write(f"  config n={row['n']} sigma={row['sigma']} alpha={row['alpha']}\n")
            for metric in ["warning_density", "survey_entropy_norm", "overlap_proxy", "complexity_proxy", "map_accuracy", "bp_delta"]:
                cv = row.get(f"{metric}_cv")
                if cv is None:
                    continue
                f.write(f"    {metric:>22}: CV={cv:.4f}\n")
        f.write("\n" + sep + "\n")
        f.write("INTERPRETATION GUIDE\n" + sep + "\n")
        f.write(
            "1. If Part A collapses tightly, the observable is mostly an alpha/SNR meter.\n"
            "2. If Part B separates secret families at fixed alpha, it sees structure beyond alpha.\n"
            "3. If Part C separates matrix families at fixed alpha, it is sensitive to A-geometry.\n"
            "4. Use only low-CV observables as paper claims.\n"
        )
        f.write(sep + "\n")
    print(f"Summary → {path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 20: Fixed-alpha discrimination test")
    parser.add_argument("--quick", action="store_true", help="Fast pilot")
    parser.add_argument("--sp-only", action="store_true", help="Use BKZ proxy instead of lattice-estimator")
    parser.add_argument("--skip-a", action="store_true", help="Skip part A")
    parser.add_argument("--skip-b", action="store_true", help="Skip part B")
    parser.add_argument("--skip-c", action="store_true", help="Skip part C")
    parser.add_argument("--skip-e", action="store_true", help="Skip part E")
    args = parser.parse_args()

    use_proxy = bool(args.sp_only) or not ESTIMATOR_AVAILABLE

    print("=" * 72)
    print("Experiment 20: Fixed-α Discrimination Test")
    print(f"quick={args.quick}  use_proxy={use_proxy}  estimator_available={ESTIMATOR_AVAILABLE}")
    print("=" * 72)

    t0 = time.time()
    a = b = c = e = []

    if not args.skip_a:
        a = part_a_alpha_collapse(use_proxy, args.quick)
    if not args.skip_b:
        b = part_b_secret_family_scan(use_proxy, args.quick)
        part_d_classification(b)
    if not args.skip_c:
        c = part_c_matrix_family_scan(use_proxy, args.quick)
    if not args.skip_e:
        e = part_e_stability(use_proxy, args.quick)

    print(f"\nDone in {(time.time() - t0) / 60.0:.1f} min. Writing outputs...")
    plot_results(a, b, c, e)
    write_summary(a, b, c, e)

    result = {"A": a, "B": b, "C": c, "E": e}
    with open("exp20_data.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print("Data → exp20_data.json")
    print("Outputs: exp20_results.png  exp20_summary.txt  exp20_data.json")

if __name__ == "__main__":
    main()
