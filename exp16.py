#!/usr/bin/env python3
"""
Experiment 16: Controlled follow-up to Experiment 15
======

Goal
----
Disentangle whether survey-propagation (SP) observables are tracking:
  - hardness,
  - dimension n,
  - modulus q,
  - or noise sigma.

This script runs three controlled sweeps with one parameter fixed at a time:
  1) sigma sweep at fixed n and q
  2) n sweep at fixed sigma and q
  3) q sweep at fixed n and sigma

It reuses the Experiment 15 implementation for instance generation, SP, and
lattice-estimator / BKZ proxy evaluation.

Outputs
-------
- exp16_controlled_results.json
- exp16_controlled_summary.txt
- exp16_controlled_results.png
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# tqdm is optional.
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

# Import Experiment 15 helpers

try:
    import exp15 as base
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import exp15.py. Place this script next to exp15.py "
        "and run it from that directory."
    ) from exc

# Defaults

FIXED_N = 48
FIXED_SIGMA = 1.5
FIXED_Q = 101

SIGMA_SWEEP = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
N_SWEEP = [24, 32, 40, 48, 56, 64]
Q_SWEEP = [59, 73, 89, 101, 127, 149]

DEFAULT_QUICK_SEEDS = 2
DEFAULT_FULL_SEEDS = 5

# Faster quick-mode settings.
QUICK_SP_MAX_ITER = 8
QUICK_BP_ITER = 15
QUICK_N_RESTARTS = 2
FULL_SP_MAX_ITER = 18
FULL_BP_ITER = 35
FULL_N_RESTARTS = 4

# Helpers

def ensure_odd(q: int) -> int:
    return q if (q % 2 == 1) else q + 1


def mean_of(items: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = []
    for row in items:
        v = row.get(key)
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isfinite(fv):
            vals.append(fv)
    return float(np.mean(vals)) if vals else None


def corr_report(x: List[float], y: List[float]) -> Dict[str, Any]:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    m = np.isfinite(xa) & np.isfinite(ya)
    out: Dict[str, Any] = {"n": int(m.sum())}
    if m.sum() < 3:
        out.update({"pearson": None, "spearman": None, "pvalue": None})
        return out
    try:
        out["pearson"] = float(np.corrcoef(xa[m], ya[m])[0, 1])
    except Exception:
        out["pearson"] = None
    try:
        sr, sp = scipy_stats.spearmanr(xa[m], ya[m])
        out["spearman"] = float(sr)
        out["pvalue"] = float(sp)
    except Exception:
        out["spearman"] = None
        out["pvalue"] = None
    return out


def fmt_bits(x: Optional[float]) -> str:
    return "n/a" if x is None else f"2^{x:.1f}"


def fmt_f(x: Optional[float], d: int = 3) -> str:
    return "n/a" if x is None or not np.isfinite(float(x)) else f"{float(x):.{d}f}"


def pick_estimator(use_proxy: bool) -> bool:
    return bool(use_proxy or not getattr(base, "ESTIMATOR_AVAILABLE", False))


def make_bar(iterable, *, total=None, desc="", leave=False):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=leave, dynamic_ncols=True)


def sweep_point_params(label: str, value: float, fixed_n: int, fixed_q: int, fixed_sigma: float) -> tuple[int, int, float]:
    if label == "sigma":
        return fixed_n, fixed_q, float(value)
    if label == "n":
        return int(value), fixed_q, fixed_sigma
    if label == "q":
        return fixed_n, int(value), fixed_sigma
    raise ValueError(f"Unknown sweep label: {label}")


# Core run function

def run_point_custom(
    n: int,
    q: int,
    sigma: float,
    seed: int,
    use_proxy: bool,
    quick: bool,
) -> Dict[str, Any]:
    q = int(ensure_odd(q))
    m_full = 3 * int(n)
    max_rows = min(m_full, n + 8) if quick else min(m_full, max(2 * n, 48))

    lwe = base.generate_lwe(n=n, m=m_full, q=q, sigma=sigma, seed=seed, max_rows=max_rows)
    sp = base.LWEsurveyPropagation(lwe)

    if quick:
        sp_metrics = sp.run(max_iter=QUICK_SP_MAX_ITER, n_restarts=QUICK_N_RESTARTS, bp_iter=QUICK_BP_ITER, light_sp=True)
    else:
        sp_metrics = sp.run(max_iter=FULL_SP_MAX_ITER, n_restarts=FULL_N_RESTARTS, bp_iter=FULL_BP_ITER, light_sp=False)

    if use_proxy:
        sec = base.bkz_proxy_bits(n=n, q=q, sigma=sigma)
    else:
        sec = base.run_estimator(n=n, q=q, sigma=sigma, jobs=getattr(base, "DEFAULT_JOBS", 1))

    return {
        "n": n,
        "q": q,
        "sigma": sigma,
        "seed": seed,
        "alpha_like": q / (sigma * math.sqrt(n)),
        "sec": sec,
        "sp": sp_metrics,
    }


def run_sweep(
    label: str,
    values: List[float],
    fixed_n: int,
    fixed_q: int,
    fixed_sigma: float,
    seeds: int,
    quick: bool,
    use_proxy: bool,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    t0 = time.time()

    value_bar = make_bar(values, total=len(values), desc=f"{label} sweep", leave=True)
    for idx, value in enumerate(value_bar):
        n, q, sigma = sweep_point_params(label, value, fixed_n, fixed_q, fixed_sigma)
        point_rows = []

        seed_bar = make_bar(range(seeds), total=seeds, desc=f"  {label}={value}", leave=False)
        for s in seed_bar:
            seed = 10_000 + 97 * idx + s
            if hasattr(seed_bar, "set_postfix_str"):
                seed_bar.set_postfix_str(f"n={n}, q={q}, sigma={sigma}")
            point_rows.append(
                run_point_custom(
                    n=n,
                    q=q,
                    sigma=sigma,
                    seed=seed,
                    use_proxy=use_proxy,
                    quick=quick,
                )
            )

        best_fams = [r["sec"].get("best_family", "other") for r in point_rows]
        best_bits_list = [r["sec"]["best_bits"] for r in point_rows if r["sec"]["best_bits"] is not None]

        agg = {
            "label": label,
            "value": float(value),
            "rows": point_rows,
            "n": point_rows[0]["n"],
            "q": point_rows[0]["q"],
            "sigma": point_rows[0]["sigma"],
            "sec": {
                "best_family": max(set(best_fams), key=best_fams.count) if best_fams else "other",
                "best_bits": float(np.mean(best_bits_list)) if best_bits_list else None,
                "dominance_gap": mean_of([r["sec"] for r in point_rows], "dominance_gap"),
            },
            "sp": {
                "warning_density": mean_of([r["sp"] for r in point_rows], "warning_density"),
                "survey_entropy_norm": mean_of([r["sp"] for r in point_rows], "survey_entropy_norm"),
                "overlap_proxy": mean_of([r["sp"] for r in point_rows], "overlap_proxy"),
                "complexity_proxy": mean_of([r["sp"] for r in point_rows], "complexity_proxy"),
                "map_accuracy": mean_of([r["sp"] for r in point_rows], "map_accuracy"),
                "xi_ratio": mean_of([r["sp"] for r in point_rows], "xi_ratio"),
            },
        }
        rows.append(agg)

    elapsed_min = (time.time() - t0) / 60.0

    hardness = [r["sec"]["best_bits"] for r in rows]
    warn = [r["sp"]["warning_density"] for r in rows]
    hs = [r["sp"]["survey_entropy_norm"] for r in rows]
    ov = [r["sp"]["overlap_proxy"] for r in rows]
    cp = [r["sp"]["complexity_proxy"] for r in rows]

    return {
        "label": label,
        "values": values,
        "fixed_n": fixed_n,
        "fixed_q": fixed_q,
        "fixed_sigma": fixed_sigma,
        "seeds": seeds,
        "quick": quick,
        "use_proxy": use_proxy,
        "elapsed_min": elapsed_min,
        "rows": rows,
        "correlations": {
            "warning_density": corr_report(warn, hardness),
            "survey_entropy": corr_report(hs, hardness),
            "overlap_proxy": corr_report(ov, hardness),
            "complexity_proxy": corr_report(cp, hardness),
        },
    }


def make_plots(results: Dict[str, Any], outpath: str) -> None:
    sweeps = results["sweeps"]
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    fig.patch.set_facecolor("white")

    metric_keys = [
        ("warning_density", "warning density"),
        ("survey_entropy_norm", "survey entropy"),
        ("overlap_proxy", "overlap proxy"),
        ("complexity_proxy", "complexity proxy"),
    ]

    for row_idx, sweep_name in enumerate(["sigma", "n", "q"]):
        sweep = sweeps[sweep_name]
        xs = [r["value"] for r in sweep["rows"]]
        hardness = [r["sec"]["best_bits"] for r in sweep["rows"]]
        for col_idx, (key, title) in enumerate(metric_keys):
            ax = axes[row_idx, col_idx]
            ys = [r["sp"][key] for r in sweep["rows"]]

            ax.scatter(xs, ys)
            ax.set_title(f"{sweep_name} sweep: {title}", fontsize=10)
            ax.set_xlabel(sweep_name)
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.25)

            if col_idx == 3:
                ax2 = ax.twinx()
                ax2.plot(xs, hardness, marker="o", linestyle="--")
                ax2.set_ylabel("hardness bits")

    plt.tight_layout()
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_summary(results: Dict[str, Any], path: str) -> None:
    sweeps = results["sweeps"]
    lines = []
    lines.append("=" * 80)
    lines.append("EXPERIMENT 16 — CONTROLLED FOLLOW-UP TO EXPERIMENT 15")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Purpose: one-parameter-at-a-time sweeps to separate hardness from")
    lines.append("dimension/modulus/noise effects.")
    lines.append("")

    for sweep_name in ["sigma", "n", "q"]:
        sweep = sweeps[sweep_name]
        lines.append(f"[{sweep_name.upper()} SWEEP]")
        lines.append(f"fixed_n={sweep['fixed_n']} fixed_q={sweep['fixed_q']} fixed_sigma={sweep['fixed_sigma']}")
        lines.append(f"seeds={sweep['seeds']} quick={sweep['quick']} proxy={sweep['use_proxy']}")
        lines.append(f"elapsed={sweep['elapsed_min']:.1f} min")
        lines.append("correlations vs hardness (bits):")
        for k, rep in sweep["correlations"].items():
            lines.append(
                f"  {k:18s} n={rep['n']:>3}  "
                f"Pearson={fmt_f(rep['pearson'], 4)}  "
                f"Spearman={fmt_f(rep['spearman'], 4)}  "
                f"p={fmt_f(rep['pvalue'], 4)}"
            )
        lines.append("")

        lines.append("per-point means:")
        lines.append(f"{'val':>8}  {'n':>4}  {'q':>4}  {'sigma':>5}  {'family':>12}  {'bits':>8}  {'H_surv':>8}  {'overlap':>8}  {'Sigma*':>8}")
        for r in sweep["rows"]:
            lines.append(
                f"{r['value']:>8.3f}  {r['n']:>4}  {r['q']:>4}  {r['sigma']:>5.2f}  "
                f"{(r['sec']['best_family'] or '?'):>12}  "
                f"{fmt_bits(r['sec']['best_bits']):>8}  "
                f"{fmt_f(r['sp']['survey_entropy_norm'], 4):>8}  "
                f"{fmt_f(r['sp']['overlap_proxy'], 4):>8}  "
                f"{fmt_f(r['sp']['complexity_proxy'], 3):>8}"
            )
        lines.append("")

    lines.append("Interpretation guide:")
    lines.append("- if correlations stay strong in the sigma sweep but weaken in the n/q sweeps,")
    lines.append("  SP is probably tracking noise-driven clustering rather than hardness itself.")
    lines.append("- if complexity_proxy remains strong in all three sweeps, that is the best")
    lines.append("  evidence that SP is detecting a hardness-linked geometric signal.")
    lines.append("- if only one axis drives the signal, then the pilot result was confounded.")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))



def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 16: controlled SP sweeps for LWE")
    parser.add_argument("--quick", action="store_true", help="Faster pilot with fewer points / restarts")
    parser.add_argument("--sp-only", action="store_true", help="Skip lattice-estimator and use BKZ proxy")
    parser.add_argument("--seeds", type=int, default=None, help="Seeds per point (overrides default)")
    parser.add_argument("--fixed-n", type=int, default=FIXED_N)
    parser.add_argument("--fixed-sigma", type=float, default=FIXED_SIGMA)
    parser.add_argument("--fixed-q", type=int, default=FIXED_Q)
    parser.add_argument("--out-prefix", type=str, default="exp16_controlled")
    args = parser.parse_args()

    use_proxy = pick_estimator(args.sp_only)
    seeds = args.seeds if args.seeds is not None else (DEFAULT_QUICK_SEEDS if args.quick else DEFAULT_FULL_SEEDS)

    results: Dict[str, Any] = {
        "meta": {
            "fixed_n": args.fixed_n,
            "fixed_sigma": args.fixed_sigma,
            "fixed_q": ensure_odd(args.fixed_q),
            "seeds": seeds,
            "quick": args.quick,
            "use_proxy": use_proxy,
            "generated_at_unix": time.time(),
        },
        "sweeps": {},
    }

    print("=" * 80, flush=True)
    print("Experiment 16: Controlled Follow-up to Experiment 15", flush=True)
    print(f"quick={args.quick}  seeds={seeds}  proxy={use_proxy}", flush=True)
    print("=" * 80, flush=True)

    results["sweeps"]["sigma"] = run_sweep(
        label="sigma",
        values=SIGMA_SWEEP,
        fixed_n=args.fixed_n,
        fixed_q=ensure_odd(args.fixed_q),
        fixed_sigma=args.fixed_sigma,
        seeds=seeds,
        quick=args.quick,
        use_proxy=use_proxy,
    )
    results["sweeps"]["n"] = run_sweep(
        label="n",
        values=N_SWEEP,
        fixed_n=args.fixed_n,
        fixed_q=ensure_odd(args.fixed_q),
        fixed_sigma=args.fixed_sigma,
        seeds=seeds,
        quick=args.quick,
        use_proxy=use_proxy,
    )
    results["sweeps"]["q"] = run_sweep(
        label="q",
        values=Q_SWEEP,
        fixed_n=args.fixed_n,
        fixed_q=ensure_odd(args.fixed_q),
        fixed_sigma=args.fixed_sigma,
        seeds=seeds,
        quick=args.quick,
        use_proxy=use_proxy,
    )

    json_path = f"{args.out_prefix}_results.json"
    txt_path = f"{args.out_prefix}_summary.txt"
    png_path = f"{args.out_prefix}_results.png"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    write_summary(results, txt_path)
    make_plots(results, png_path)

    print(f"Saved: {json_path}", flush=True)
    print(f"Saved: {txt_path}", flush=True)
    print(f"Saved: {png_path}", flush=True)


if __name__ == "__main__":
    main()
