#!/usr/bin/env python3
"""
Experiment 17: Hardness-Constant Manifold Test
=

Goal
----
Hold lattice-estimator hardness approximately constant, then vary representation
parameters (n, q, sigma) to test whether SP observables still move.

Outputs
-------
- exp17_shell_results.json
- exp17_shell_summary.txt
- exp17_shell_results.png

Recommended usage
------------------
# Quick pilot
sage -python exp17.py --quick --target-bits 37.5 --tol 0.5 --num-points 12

# Stronger run
sage -python exp17.py --target-bits 37.5 --tol 0.4 --num-points 24 --seeds 5

# BKZ proxy only
sage -python exp17.py --sp-only
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    import exp15 as base
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import exp15.py. Place this script next to exp15.py and run it from that directory."
    ) from exc


# Defaults

DEFAULT_TARGET_BITS = 37.5
DEFAULT_TOL = 0.5
DEFAULT_NUM_POINTS = 18
DEFAULT_SEEDS_QUICK = 2
DEFAULT_SEEDS_FULL = 5

TARGET_ALPHA_GRID = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
N_GRID = [24, 32, 40, 48, 56, 64, 80, 96]
SIGMA_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]

QUICK_SP_MAX_ITER = 10
QUICK_BP_ITER = 20
QUICK_N_RESTARTS = 2

FULL_SP_MAX_ITER = 22
FULL_BP_ITER = 45
FULL_N_RESTARTS = 6


# Helpers

def ensure_odd(q: int) -> int:
    return q if (q % 2 == 1) else q + 1


def make_bar(iterable, *, total=None, desc="", leave=False):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=leave, dynamic_ncols=True)


def fmt_bits(x: Optional[float]) -> str:
    return "n/a" if x is None or not np.isfinite(float(x)) else f"2^{float(x):.1f}"


def fmt_f(x: Optional[float], d: int = 3) -> str:
    return "n/a" if x is None or not np.isfinite(float(x)) else f"{float(x):.{d}f}"


def mean_of(items: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    vals: List[float] = []
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


def corr_report(x: Sequence[float], y: Sequence[float]) -> Dict[str, Any]:
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


def pick_estimator(use_proxy: bool) -> bool:
    return bool(use_proxy or not getattr(base, "ESTIMATOR_AVAILABLE", False))


def sweep_q(n: int, sigma: float, alpha: float) -> int:
    q = int(round(alpha * sigma * math.sqrt(n)))
    q = max(q, n + 2)
    return ensure_odd(q)


def sp_settings(quick: bool) -> Tuple[int, int, int, bool]:
    if quick:
        return QUICK_SP_MAX_ITER, QUICK_N_RESTARTS, QUICK_BP_ITER, True
    return FULL_SP_MAX_ITER, FULL_N_RESTARTS, FULL_BP_ITER, False


# Candidate generation and selection

@dataclass(frozen=True)
class Candidate:
    n: int
    sigma: float
    alpha: float
    q: int
    hardness: Optional[float]
    family: Optional[str]
    dominance_gap: Optional[float]


def estimate_hardness(n: int, q: int, sigma: float, use_proxy: bool) -> Dict[str, Any]:
    if use_proxy:
        return base.bkz_proxy_bits(n=n, q=q, sigma=sigma)
    return base.run_estimator(n=n, q=q, sigma=sigma, jobs=getattr(base, "DEFAULT_JOBS", 1))


def build_candidate_pool(use_proxy: bool) -> List[Candidate]:
    pool: List[Candidate] = []
    for n in N_GRID:
        for sigma in SIGMA_GRID:
            for alpha in TARGET_ALPHA_GRID:
                q = sweep_q(n, sigma, alpha)
                sec = estimate_hardness(n=n, q=q, sigma=sigma, use_proxy=use_proxy)
                pool.append(
                    Candidate(
                        n=n,
                        sigma=sigma,
                        alpha=alpha,
                        q=q,
                        hardness=sec.get("best_bits"),
                        family=sec.get("best_family"),
                        dominance_gap=sec.get("dominance_gap"),
                    )
                )
    return pool


def select_shell(
    pool: Sequence[Candidate],
    target_bits: float,
    tol: float,
    num_points: int,
) -> List[Candidate]:
    ranked = []
    for c in pool:
        if c.hardness is None or not np.isfinite(float(c.hardness)):
            continue
        diff = abs(float(c.hardness) - target_bits)
        ranked.append((diff, c))
    ranked.sort(key=lambda t: (t[0], t[1].n, t[1].sigma, t[1].q))

    within = [c for d, c in ranked if d <= tol]
    if len(within) >= num_points:
        return within[:num_points]

    selected = within[:]
    seen = {(c.n, c.sigma, c.q) for c in selected}
    for _, c in ranked:
        key = (c.n, c.sigma, c.q)
        if key in seen:
            continue
        selected.append(c)
        seen.add(key)
        if len(selected) >= num_points:
            break
    return selected[:num_points]



def run_point(
    n: int,
    q: int,
    sigma: float,
    seed: int,
    use_proxy: bool,
    quick: bool,
) -> Dict[str, Any]:
    q = ensure_odd(int(q))
    m_full = 3 * int(n)
    max_rows = min(m_full, n + 8) if quick else min(m_full, max(2 * n, 48))

    lwe = base.generate_lwe(n=n, m=m_full, q=q, sigma=sigma, seed=seed, max_rows=max_rows)
    sp = base.LWEsurveyPropagation(lwe)

    sp_max_iter, n_restarts, bp_iter, light_sp = sp_settings(quick)
    sp_metrics = sp.run(
        max_iter=sp_max_iter,
        n_restarts=n_restarts,
        bp_iter=bp_iter,
        light_sp=light_sp,
    )

    sec = estimate_hardness(n=n, q=q, sigma=sigma, use_proxy=use_proxy)

    return {
        "n": n,
        "q": q,
        "sigma": sigma,
        "seed": seed,
        "alpha_like": q / (sigma * math.sqrt(n)),
        "sec": sec,
        "sp": sp_metrics,
    }


def aggregate_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    best_fams = [r["sec"].get("best_family", "other") for r in rows if r["sec"].get("best_family")]
    best_bits = [r["sec"].get("best_bits") for r in rows if r["sec"].get("best_bits") is not None]

    agg = {
        "n": rows[0]["n"],
        "q": rows[0]["q"],
        "sigma": rows[0]["sigma"],
        "seed_rows": list(rows),
        "sec": {
            "best_family": max(set(best_fams), key=best_fams.count) if best_fams else "other",
            "best_bits": float(np.mean(best_bits)) if best_bits else None,
            "dominance_gap": mean_of([r["sec"] for r in rows], "dominance_gap"),
        },
        "sp": {
            "warning_density": mean_of([r["sp"] for r in rows], "warning_density"),
            "survey_entropy_norm": mean_of([r["sp"] for r in rows], "survey_entropy_norm"),
            "overlap_proxy": mean_of([r["sp"] for r in rows], "overlap_proxy"),
            "complexity_proxy": mean_of([r["sp"] for r in rows], "complexity_proxy"),
            "map_accuracy": mean_of([r["sp"] for r in rows], "map_accuracy"),
            "xi_ratio": mean_of([r["sp"] for r in rows], "xi_ratio"),
        },
    }
    return agg


# Plotting / summary

def make_plots(results: Dict[str, Any], outpath: str) -> None:
    rows = results["rows"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor("white")

    xs_bits = [r["sec"]["best_bits"] for r in rows]
    xs_n = [r["n"] for r in rows]
    xs_sigma = [r["sigma"] for r in rows]
    xs_q = [r["q"] for r in rows]
    y_warn = [r["sp"]["warning_density"] for r in rows]
    y_h = [r["sp"]["survey_entropy_norm"] for r in rows]
    y_ov = [r["sp"]["overlap_proxy"] for r in rows]
    y_cp = [r["sp"]["complexity_proxy"] for r in rows]

    panels = [
        (axes[0, 0], xs_bits, y_warn, "warning_density vs hardness", "log2(ROP)", "warning_density"),
        (axes[0, 1], xs_bits, y_h, "survey_entropy vs hardness", "log2(ROP)", "H_surv/log(q)"),
        (axes[0, 2], xs_bits, y_cp, "complexity_proxy vs hardness", "log2(ROP)", "Sigma*"),
        (axes[1, 0], xs_n, y_cp, "complexity_proxy vs n", "n", "Sigma*"),
        (axes[1, 1], xs_sigma, y_cp, "complexity_proxy vs sigma", "sigma", "Sigma*"),
        (axes[1, 2], xs_q, y_cp, "complexity_proxy vs q", "q", "Sigma*"),
    ]

    for ax, xs, ys, title, xl, yl in panels:
        ax.scatter(xs, ys)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_summary(results: Dict[str, Any], path: str) -> None:
    rows = results["rows"]
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("EXPERIMENT 17 — HARDNESS-CONSTANT MANIFOLD TEST")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"target_bits={results['meta']['target_bits']}")
    lines.append(f"tol={results['meta']['tol']}")
    lines.append(f"selected_points={len(rows)}")
    lines.append(f"seeds={results['meta']['seeds']}")
    lines.append(f"quick={results['meta']['quick']}")
    lines.append(f"proxy={results['meta']['use_proxy']}")
    lines.append(f"elapsed={results['elapsed_min']:.1f} min")
    lines.append("")

    hardness = [r["sec"]["best_bits"] for r in rows]
    warn = [r["sp"]["warning_density"] for r in rows]
    hs = [r["sp"]["survey_entropy_norm"] for r in rows]
    ov = [r["sp"]["overlap_proxy"] for r in rows]
    cp = [r["sp"]["complexity_proxy"] for r in rows]
    nvals = [r["n"] for r in rows]
    sigmas = [r["sigma"] for r in rows]
    qvals = [r["q"] for r in rows]
    hdiff = [
        (hb - results["meta"]["target_bits"]) if hb is not None else np.nan
        for hb in hardness
    ]

    lines.append("correlations with hardness (bits):")
    for label, arr in [
        ("warning_density", warn),
        ("survey_entropy", hs),
        ("overlap_proxy", ov),
        ("complexity_proxy", cp),
    ]:
        rep = corr_report(arr, hardness)
        lines.append(
            f"  {label:18s} n={rep['n']:>3}  "
            f"Pearson={fmt_f(rep['pearson'], 4)}  "
            f"Spearman={fmt_f(rep['spearman'], 4)}  "
            f"p={fmt_f(rep['pvalue'], 4)}"
        )
    lines.append("")

    lines.append("correlations with hardness deviation from target:")
    for label, arr in [
        ("warning_density", warn),
        ("survey_entropy", hs),
        ("overlap_proxy", ov),
        ("complexity_proxy", cp),
    ]:
        rep = corr_report(arr, hdiff)
        lines.append(
            f"  {label:18s} n={rep['n']:>3}  "
            f"Pearson={fmt_f(rep['pearson'], 4)}  "
            f"Spearman={fmt_f(rep['spearman'], 4)}  "
            f"p={fmt_f(rep['pvalue'], 4)}"
        )
    lines.append("")

    lines.append("per-point means:")
    lines.append(
        f"{'bits':>8}  {'Δbits':>8}  {'n':>4}  {'q':>4}  {'sigma':>5}  "
        f"{'family':>12}  {'H_surv':>8}  {'overlap':>8}  {'Sigma*':>8}"
    )
    for r in rows:
        b = r["sec"]["best_bits"]
        db = (b - results["meta"]["target_bits"]) if b is not None else None
        lines.append(
            f"{fmt_bits(b):>8}  {fmt_f(db, 3):>8}  {r['n']:>4}  {r['q']:>4}  {r['sigma']:>5.2f}  "
            f"{(r['sec']['best_family'] or '?'):>12}  "
            f"{fmt_f(r['sp']['survey_entropy_norm'], 4):>8}  "
            f"{fmt_f(r['sp']['overlap_proxy'], 4):>8}  "
            f"{fmt_f(r['sp']['complexity_proxy'], 3):>8}"
        )
    lines.append("")

    lines.append("Interpretation guide:")
    lines.append("- if SP remains structured inside the hardness shell, it is not merely a hardness proxy.")
    lines.append("- if SP collapses when hardness is held fixed, it is mostly tracking attack cost.")
    lines.append("- complexity_proxy is the main candidate to watch in this test.")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))



def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 17: hardness-constant manifold test")
    parser.add_argument("--quick", action="store_true", help="Faster pilot mode")
    parser.add_argument("--sp-only", action="store_true", help="Use BKZ proxy instead of lattice-estimator")
    parser.add_argument("--target-bits", type=float, default=DEFAULT_TARGET_BITS)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--num-points", type=int, default=DEFAULT_NUM_POINTS)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--out-prefix", type=str, default="exp17_shell")
    args = parser.parse_args()

    use_proxy = pick_estimator(args.sp_only)
    seeds = args.seeds if args.seeds is not None else (DEFAULT_SEEDS_QUICK if args.quick else DEFAULT_SEEDS_FULL)

    t0 = time.time()
    print("=" * 80, flush=True)
    print("Experiment 17: Hardness-Constant Manifold Test", flush=True)
    print(
        f"target_bits={args.target_bits} tol={args.tol} num_points={args.num_points} "
        f"seeds={seeds} quick={args.quick} proxy={use_proxy}",
        flush=True,
    )
    print("=" * 80, flush=True)

    print("Calibrating candidate pool...", flush=True)
    pool = build_candidate_pool(use_proxy=use_proxy)

    selected = select_shell(
        pool=pool,
        target_bits=args.target_bits,
        tol=args.tol,
        num_points=args.num_points,
    )

    if not selected:
        raise SystemExit("No candidates selected. Expand the grid or tolerance.")

    print(f"Selected {len(selected)} points from candidate pool.", flush=True)

    rows: List[Dict[str, Any]] = []
    candidate_bar = make_bar(selected, total=len(selected), desc="shell points", leave=True)

    for idx, c in enumerate(candidate_bar):
        point_rows = []
        seed_bar = make_bar(range(seeds), total=seeds, desc=f"  n={c.n}, q={c.q}, sigma={c.sigma}", leave=False)
        for s in seed_bar:
            if hasattr(seed_bar, "set_postfix_str"):
                seed_bar.set_postfix_str(f"target={args.target_bits:.2f}±{args.tol:.2f}")
            point_rows.append(
                run_point(
                    n=c.n,
                    q=c.q,
                    sigma=c.sigma,
                    seed=10_000 + idx * 97 + s,
                    use_proxy=use_proxy,
                    quick=args.quick,
                )
            )

        agg = aggregate_rows(point_rows)
        agg["target_bits"] = args.target_bits
        agg["tol"] = args.tol
        agg["candidate"] = {
            "alpha": c.alpha,
            "hardness_calibrated": c.hardness,
            "family": c.family,
            "dominance_gap": c.dominance_gap,
        }
        agg["hardness_diff"] = (
            float(agg["sec"]["best_bits"] - args.target_bits)
            if agg["sec"]["best_bits"] is not None
            else None
        )
        rows.append(agg)

        print(
            f"n={c.n:>3} q={c.q:>4} sigma={c.sigma:>4.2f} "
            f"bits={fmt_bits(agg['sec']['best_bits']):>8} "
            f"H_surv={fmt_f(agg['sp']['survey_entropy_norm'], 4):>8} "
            f"Sigma*={fmt_f(agg['sp']['complexity_proxy'], 3):>8}",
            flush=True,
        )

    elapsed_min = (time.time() - t0) / 60.0

    results: Dict[str, Any] = {
        "meta": {
            "target_bits": args.target_bits,
            "tol": args.tol,
            "num_points": args.num_points,
            "seeds": seeds,
            "quick": args.quick,
            "use_proxy": use_proxy,
            "generated_at_unix": time.time(),
        },
        "elapsed_min": elapsed_min,
        "rows": rows,
    }

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