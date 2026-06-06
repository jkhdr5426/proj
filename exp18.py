#!/usr/bin/env python3
"""
Experiment 18: Umbrella Sampling for LWE + SP


Goal
----
Use umbrella sampling over lattice-estimator hardness to focus compute near a
chosen hardness band, then test whether SP observables still vary inside that
biased shell.

This script is designed to be careful experimentally:
- lattice-estimator output is silenced during calibration and evaluation
- candidate pools are cached
- umbrella windows are sampled with Metropolis-Hastings on the discrete pool
- reweighting recovers an approximately uniform-over-pool estimate
- console output stays compact and only shows the current / most recent run

Outputs
-------
- exp18_umbrella_results.json
- exp18_umbrella_summary.txt
- exp18_umbrella_results.png
- exp18_candidate_pool.json (cache)

Recommended usage
------------------
# Quick pilot
sage -python exp18_umbrella_sampling.py --quick --target-bits 37.5 --centers 36.8,37.2,37.5,37.8,38.2

# Stronger run
sage -python exp18_umbrella_sampling.py --target-bits 37.5 --centers 36.5,37.0,37.5,38.0,38.5 --samples-per-window 16 --repeats 2

# Proxy only
sage -python exp18_umbrella_sampling.py --sp-only
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import random
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
DEFAULT_CENTERS = [36.8, 37.1, 37.4, 37.7, 38.0]
DEFAULT_UMBRELLA_WIDTH = 0.35
DEFAULT_SAMPLES_PER_WINDOW = 12
DEFAULT_REPEATS = 1
DEFAULT_BURN_IN = 120
DEFAULT_THIN = 10
DEFAULT_CHAIN_STEPS = 400
DEFAULT_N_CHAINS = 2
DEFAULT_POOL_CACHE = "exp18_candidate_pool.json"
DEFAULT_OUT_PREFIX = "exp18_umbrella"

TARGET_ALPHA_GRID = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
N_GRID = [24, 32, 40, 48, 56, 64, 80, 96]
SIGMA_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]

QUICK_SP_MAX_ITER = 10
QUICK_BP_ITER = 20
QUICK_N_RESTARTS = 2

FULL_SP_MAX_ITER = 22
FULL_BP_ITER = 45
FULL_N_RESTARTS = 6

WEIGHT_SCALE = 1e6


# Helpers

def ensure_odd(q: int) -> int:
    return q if (q % 2 == 1) else q + 1


def fmt_bits(x: Optional[float]) -> str:
    return "n/a" if x is None or not np.isfinite(float(x)) else f"2^{float(x):.1f}"


def fmt_f(x: Optional[float], d: int = 3) -> str:
    return "n/a" if x is None or not np.isfinite(float(x)) else f"{float(x):.{d}f}"


def human_seconds(seconds: Optional[float]) -> str:
    if seconds is None or not np.isfinite(float(seconds)):
        return "?:??"
    s = max(0, int(round(float(seconds))))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


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


def scaled_weight(raw: float) -> float:
    return max(0.05, float(raw) / WEIGHT_SCALE)


@contextlib.contextmanager
def suppress_output() -> Iterable[None]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield


class WeightedProgress:
    """Weighted progress bar with a dynamically refined ETA."""

    def __init__(self, total_weight: float, desc: str, leave: bool = False):
        self.total_weight = float(max(total_weight, 1e-9))
        self.done_weight = 0.0
        self.rate_sec_per_weight: Optional[float] = None
        self.alpha = 0.25
        self.bar = (
            tqdm(total=self.total_weight, desc=desc, leave=leave, dynamic_ncols=True)
            if tqdm is not None
            else None
        )

    def update(self, weight: float, elapsed: Optional[float] = None, **postfix: Any) -> None:
        w = float(max(weight, 0.0))
        if elapsed is not None and w > 0:
            inst = float(elapsed) / w
            if self.rate_sec_per_weight is None or not np.isfinite(self.rate_sec_per_weight):
                self.rate_sec_per_weight = inst
            else:
                self.rate_sec_per_weight = self.alpha * inst + (1.0 - self.alpha) * self.rate_sec_per_weight
        self.done_weight += w
        if self.bar is not None:
            self.bar.update(w)
            rem = max(0.0, self.total_weight - self.done_weight)
            eta = None if self.rate_sec_per_weight is None else rem * self.rate_sec_per_weight
            pf = dict(postfix)
            pf["eta"] = human_seconds(eta)
            self.bar.set_postfix(pf, refresh=False)

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()


# Data classes

@dataclass(frozen=True)
class Candidate:
    n: int
    sigma: float
    alpha: float
    q: int
    hardness: Optional[float]
    family: Optional[str]
    dominance_gap: Optional[float]


@dataclass
class SampleRecord:
    window_center: float
    umbrella_width: float
    candidate_index: int
    n: int
    q: int
    sigma: float
    hardness: Optional[float]
    umbrella_energy: float
    bias_weight: float
    chain_id: int
    step_id: int
    seed: int
    sec: Dict[str, Any]
    sp: Dict[str, Any]



def run_estimator_silent(n: int, q: int, sigma: float, use_proxy: bool) -> Dict[str, Any]:
    if use_proxy:
        return base.bkz_proxy_bits(n=n, q=q, sigma=sigma)
    with suppress_output():
        return base.run_estimator(n=n, q=q, sigma=sigma, jobs=getattr(base, "DEFAULT_JOBS", 1))


def build_candidate_pool(use_proxy: bool, cache_path: str, refresh: bool) -> List[Candidate]:
    if os.path.isfile(cache_path) and not refresh:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pool = [Candidate(**row) for row in data]
        return pool

    pool: List[Candidate] = []
    combos: List[Tuple[int, float, float, int]] = []
    for n in N_GRID:
        for sigma in SIGMA_GRID:
            for alpha in TARGET_ALPHA_GRID:
                q = sweep_q(n, sigma, alpha)
                combos.append((n, sigma, alpha, q))

    # Use a rough weight model to build a useful ETA for the calibration phase.
    total_weight = 0.0
    weights: List[float] = []
    for n, sigma, _alpha, q in combos:
        raw = (n**2) * math.log2(q + 2.0) * (1.0 + 0.15 * sigma)
        raw *= 0.8 if use_proxy else 1.15
        w = scaled_weight(raw)
        weights.append(w)
        total_weight += w

    meter = WeightedProgress(total_weight=total_weight, desc="calibrating candidate pool", leave=True)

    for (n, sigma, alpha, q), w in zip(combos, weights):
        t0 = time.time()
        sec = run_estimator_silent(n=n, q=q, sigma=sigma, use_proxy=use_proxy)
        elapsed = time.time() - t0
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
        meter.update(w, elapsed=elapsed, n=n, q=q, sigma=sigma, bits=fmt_bits(sec.get("best_bits")))

    meter.close()

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump([candidate.__dict__ for candidate in pool], f, indent=2, default=str)

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



def run_sp_point(
    candidate: Candidate,
    seed: int,
    use_proxy: bool,
    quick: bool,
) -> Dict[str, Any]:
    n, q, sigma = candidate.n, ensure_odd(candidate.q), candidate.sigma
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

    sec = run_estimator_silent(n=n, q=q, sigma=sigma, use_proxy=use_proxy)

    return {
        "n": n,
        "q": q,
        "sigma": sigma,
        "seed": seed,
        "alpha_like": q / (sigma * math.sqrt(n)),
        "sec": sec,
        "sp": sp_metrics,
    }


def umbrella_energy(hardness: float, center: float, width: float) -> float:
    return 0.5 * ((hardness - center) / max(width, 1e-9)) ** 2


def umbrella_bias_weight(hardness: float, center: float, width: float) -> float:
    return math.exp(-umbrella_energy(hardness, center, width))


def metropolis_window_samples(
    pool: Sequence[Candidate],
    center: float,
    width: float,
    num_samples: int,
    burn_in: int,
    thin: int,
    chain_steps: int,
    n_chains: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int, float]]:
    """Return tuples (candidate_index, chain_id, step_id) drawn from the biased distribution."""
    hardness = np.array([float(c.hardness) if c.hardness is not None else np.nan for c in pool], dtype=float)
    valid = np.isfinite(hardness)
    if not np.any(valid):
        raise ValueError("Candidate pool has no valid hardness values.")

    logw = np.full(len(pool), -np.inf, dtype=float)
    logw[valid] = -0.5 * ((hardness[valid] - center) / max(width, 1e-9)) ** 2

    start_idx = int(np.nanargmin(np.abs(hardness - center)))

    per_chain = max(1, int(math.ceil(num_samples / max(1, n_chains))))
    collected: List[Tuple[int, int, int]] = []

    for chain_id in range(n_chains):
        current = start_idx if chain_id == 0 else int(rng.integers(0, len(pool)))
        # burn-in
        for _ in range(burn_in):
            prop = int(rng.integers(0, len(pool)))
            if not valid[prop]:
                continue
            if math.log(rng.random() + 1e-300) < (logw[prop] - logw[current]):
                current = prop

        # sampling
        produced = 0
        step_id = 0
        while produced < per_chain and len(collected) < num_samples:
            for _ in range(thin):
                prop = int(rng.integers(0, len(pool)))
                if not valid[prop]:
                    continue
                if math.log(rng.random() + 1e-300) < (logw[prop] - logw[current]):
                    current = prop
                step_id += 1
                if step_id >= chain_steps:
                    break
            collected.append((current, chain_id, step_id))
            produced += 1
            if step_id >= chain_steps:
                break

    return collected[:num_samples]



def aggregate_records(records: Sequence[SampleRecord]) -> Dict[str, Any]:
    sec_bits = [r.sec.get("best_bits") for r in records if r.sec.get("best_bits") is not None]
    hardness = [r.hardness for r in records if r.hardness is not None]
    warn = [r.sp.get("warning_density") for r in records if r.sp.get("warning_density") is not None]
    hs = [r.sp.get("survey_entropy_norm") for r in records if r.sp.get("survey_entropy_norm") is not None]
    ov = [r.sp.get("overlap_proxy") for r in records if r.sp.get("overlap_proxy") is not None]
    cp = [r.sp.get("complexity_proxy") for r in records if r.sp.get("complexity_proxy") is not None]

    # Reweight to approximately uniform-over-pool by inverting the umbrella bias.
    weights = np.array([math.exp(+r.umbrella_energy) for r in records], dtype=float)
    weights = weights / max(weights.mean(), 1e-300)

    def wmean(vals: Sequence[Optional[float]]) -> Optional[float]:
        xa = np.array([np.nan if v is None else float(v) for v in vals], dtype=float)
        m = np.isfinite(xa) & np.isfinite(weights)
        if not np.any(m):
            return None
        ww = weights[m]
        xx = xa[m]
        return float(np.sum(ww * xx) / np.sum(ww))

    def ess(w: np.ndarray) -> float:
        s1 = float(np.sum(w))
        s2 = float(np.sum(w * w))
        return (s1 * s1 / s2) if s2 > 0 else 0.0

    return {
        "n_records": len(records),
        "hardness_mean": float(np.mean(hardness)) if hardness else None,
        "hardness_std": float(np.std(hardness)) if hardness else None,
        "best_bits_mean": float(np.mean(sec_bits)) if sec_bits else None,
        "best_bits_std": float(np.std(sec_bits)) if sec_bits else None,
        "warning_density_mean": float(np.mean(warn)) if warn else None,
        "survey_entropy_norm_mean": float(np.mean(hs)) if hs else None,
        "overlap_proxy_mean": float(np.mean(ov)) if ov else None,
        "complexity_proxy_mean": float(np.mean(cp)) if cp else None,
        "warning_density_reweighted": wmean([r.sp.get("warning_density") for r in records]),
        "survey_entropy_norm_reweighted": wmean([r.sp.get("survey_entropy_norm") for r in records]),
        "overlap_proxy_reweighted": wmean([r.sp.get("overlap_proxy") for r in records]),
        "complexity_proxy_reweighted": wmean([r.sp.get("complexity_proxy") for r in records]),
        "effective_sample_size": ess(weights),
        "corr_hardness_warning": corr_report(hardness, warn),
        "corr_hardness_entropy": corr_report(hardness, hs),
        "corr_hardness_overlap": corr_report(hardness, ov),
        "corr_hardness_complexity": corr_report(hardness, cp),
    }


def write_summary(results: Dict[str, Any], path: str) -> None:
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("EXPERIMENT 18 — UMBRELLA SAMPLING FOR HARDNESS + SP")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"target_bits={results['meta']['target_bits']}")
    lines.append(f"centers={results['meta']['centers']}")
    lines.append(f"umbrella_width={results['meta']['umbrella_width']}")
    lines.append(f"samples_per_window={results['meta']['samples_per_window']}")
    lines.append(f"repeats={results['meta']['repeats']}")
    lines.append(f"quick={results['meta']['quick']}")
    lines.append(f"proxy={results['meta']['use_proxy']}")
    lines.append(f"elapsed={results['elapsed_min']:.1f} min")
    lines.append("")

    lines.append("overall reweighted estimates (approximately uniform over pool):")
    overall = results["overall"]
    for label, key in [
        ("warning_density", "warning_density_reweighted"),
        ("survey_entropy", "survey_entropy_norm_reweighted"),
        ("overlap_proxy", "overlap_proxy_reweighted"),
        ("complexity_proxy", "complexity_proxy_reweighted"),
    ]:
        lines.append(f"  {label:18s}: {fmt_f(overall.get(key), 4)}")
    lines.append(f"  effective_sample_size: {fmt_f(overall.get('effective_sample_size'), 2)}")
    lines.append("")

    lines.append("per-window summaries:")
    for win in results["windows"]:
        lines.append(
            f"window center={win['center']:.2f}  accepted={win['n_records']}  "
            f"acc_rate={fmt_f(win['acceptance_rate'], 3)}  "
            f"ESS={fmt_f(win['effective_sample_size'], 1)}"
        )
        lines.append(
            f"  hardness mean±std: {fmt_f(win['hardness_mean'], 4)} ± {fmt_f(win['hardness_std'], 4)}"
        )
        lines.append(
            f"  complexity mean±std: {fmt_f(win['complexity_proxy_mean'], 4)} ± {fmt_f(win['complexity_proxy_std'], 4)}"
        )
        lines.append(
            f"  reweighted complexity: {fmt_f(win['complexity_proxy_reweighted'], 4)}"
        )
    lines.append("")

    lines.append("interpretation:")
    lines.append("- if SP observables vary strongly even after umbrella reweighting, SP is capturing structure beyond hardness.")
    lines.append("- if the biased windows separate sharply in SP observables, the hardness landscape and SP landscape are coupled.")
    lines.append("- if the reweighted estimates remain stable while biased-window values move, then the umbrella bias is doing useful variance reduction rather than changing the underlying signal.")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def make_plots(results: Dict[str, Any], outpath: str) -> None:
    windows = results["windows"]
    all_records = []
    for win in windows:
        all_records.extend(win["records"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor("white")

    hardness = [r["hardness"] for r in all_records]
    warn = [r["sp"]["warning_density"] for r in all_records]
    hs = [r["sp"]["survey_entropy_norm"] for r in all_records]
    cp = [r["sp"]["complexity_proxy"] for r in all_records]
    centers = [r["window_center"] for r in all_records]

    sc0 = axes[0, 0].scatter(hardness, warn, c=centers)
    axes[0, 0].set_title("warning_density vs hardness")
    axes[0, 0].set_xlabel("log2(ROP)")
    axes[0, 0].set_ylabel("warning_density")
    axes[0, 0].grid(True, alpha=0.25)
    fig.colorbar(sc0, ax=axes[0, 0], label="window center")

    sc1 = axes[0, 1].scatter(hardness, hs, c=centers)
    axes[0, 1].set_title("survey_entropy vs hardness")
    axes[0, 1].set_xlabel("log2(ROP)")
    axes[0, 1].set_ylabel("H_surv/log(q)")
    axes[0, 1].grid(True, alpha=0.25)
    fig.colorbar(sc1, ax=axes[0, 1], label="window center")

    sc2 = axes[1, 0].scatter(hardness, cp, c=centers)
    axes[1, 0].set_title("complexity_proxy vs hardness")
    axes[1, 0].set_xlabel("log2(ROP)")
    axes[1, 0].set_ylabel("Sigma*")
    axes[1, 0].grid(True, alpha=0.25)
    fig.colorbar(sc2, ax=axes[1, 0], label="window center")

    # Histogram of hardness with window centers shown.
    axes[1, 1].hist(hardness, bins=18, alpha=0.75)
    for c in results["meta"]["centers"]:
        axes[1, 1].axvline(c, linestyle="--")
    axes[1, 1].set_title("Umbrella-sampled hardness distribution")
    axes[1, 1].set_xlabel("log2(ROP)")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)



def parse_centers(text: Optional[str], target_bits: float) -> List[float]:
    if not text:
        return list(DEFAULT_CENTERS)
    out: List[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out if out else [target_bits]


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 18: umbrella sampling for SP hardness shells")
    parser.add_argument("--quick", action="store_true", help="Faster pilot mode")
    parser.add_argument("--sp-only", action="store_true", help="Use BKZ proxy instead of lattice-estimator")
    parser.add_argument("--target-bits", type=float, default=DEFAULT_TARGET_BITS)
    parser.add_argument("--centers", type=str, default=None, help="Comma-separated umbrella window centers")
    parser.add_argument("--umbrella-width", type=float, default=DEFAULT_UMBRELLA_WIDTH)
    parser.add_argument("--samples-per-window", type=int, default=DEFAULT_SAMPLES_PER_WINDOW)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="SP+estimator repeats per accepted sample")
    parser.add_argument("--burn-in", type=int, default=DEFAULT_BURN_IN)
    parser.add_argument("--thin", type=int, default=DEFAULT_THIN)
    parser.add_argument("--chain-steps", type=int, default=DEFAULT_CHAIN_STEPS)
    parser.add_argument("--chains", type=int, default=DEFAULT_N_CHAINS)
    parser.add_argument("--pool-cache", type=str, default=DEFAULT_POOL_CACHE)
    parser.add_argument("--refresh-pool", action="store_true", help="Recompute candidate pool even if cache exists")
    parser.add_argument("--out-prefix", type=str, default=DEFAULT_OUT_PREFIX)
    args = parser.parse_args()

    use_proxy = pick_estimator(args.sp_only)
    centers = parse_centers(args.centers, args.target_bits)
    repeats = max(1, args.repeats)

    print("=" * 80, flush=True)
    print("Experiment 18: Umbrella Sampling for Hardness + SP", flush=True)
    print(
        f"target_bits={args.target_bits} centers={centers} width={args.umbrella_width} "
        f"samples/window={args.samples_per_window} repeats={repeats} quick={args.quick} proxy={use_proxy}",
        flush=True,
    )
    print("=" * 80, flush=True)

    t0 = time.time()
    pool = build_candidate_pool(use_proxy=use_proxy, cache_path=args.pool_cache, refresh=args.refresh_pool)
    selected = select_shell(pool=pool, target_bits=args.target_bits, tol=max(0.0, args.umbrella_width), num_points=max(1, len(pool)))

    # We intentionally keep the full candidate pool but only sample from windows.
    # A shell summary based on the closest points is still helpful for diagnostics.
    shell_preview = select_shell(pool=pool, target_bits=args.target_bits, tol=args.umbrella_width, num_points=min(12, len(pool)))

    print(f"Candidate pool: {len(pool)} points", flush=True)
    print(f"Preview shell: {len(shell_preview)} nearest points", flush=True)

    rng = np.random.default_rng(202406)
    windows_output: List[Dict[str, Any]] = []
    all_records: List[SampleRecord] = []

    # Weighted estimate of total work for the umbrella stage.
    avg_candidate = sum(
        estimate := ((c.n**2) * math.log2(c.q + 2.0) * (1.0 + 0.15 * c.sigma))
        for c in selected
    ) / max(1, len(selected))
    avg_point_weight = scaled_weight(avg_candidate * (5.0 if args.quick else 8.0) * repeats)
    total_weight = avg_point_weight * len(centers) * args.samples_per_window
    umbrella_meter = WeightedProgress(total_weight=total_weight, desc="umbrella sampling", leave=True)

    for w_idx, center in enumerate(centers):
        t_window = time.time()
        window_records: List[SampleRecord] = []
        accepted = 0
        total_chain_accepts = 0
        total_chain_steps = 0

        # Collect candidate indices from the biased distribution.
        selected_samples = metropolis_window_samples(
            pool=pool,
            center=center,
            width=args.umbrella_width,
            num_samples=args.samples_per_window,
            burn_in=args.burn_in,
            thin=args.thin,
            chain_steps=args.chain_steps,
            n_chains=args.chains,
            rng=rng,
        )

        sample_bar = tqdm(selected_samples, desc=f"window {w_idx+1}/{len(centers)} center={center:.2f}", leave=False, dynamic_ncols=True) if tqdm is not None else selected_samples

        for sample_idx, (cand_idx, chain_id, step_id) in enumerate(sample_bar):
            cand = pool[cand_idx]
            if cand.hardness is None:
                continue
            energy = umbrella_energy(float(cand.hardness), center=center, width=args.umbrella_width)
            bias_w = umbrella_bias_weight(float(cand.hardness), center=center, width=args.umbrella_width)

            # Each accepted umbrella sample is evaluated repeats times with distinct seeds.
            point_rows: List[Dict[str, Any]] = []
            t_point = time.time()
            for rep in range(repeats):
                seed = 100000 + 1000 * w_idx + 37 * sample_idx + rep
                point_rows.append(
                    run_sp_point(
                        candidate=cand,
                        seed=seed,
                        use_proxy=use_proxy,
                        quick=args.quick,
                    )
                )
            point_elapsed = time.time() - t_point

            # Aggregate repeats for this umbrella draw.
            agg_point = {
                "n": cand.n,
                "q": cand.q,
                "sigma": cand.sigma,
                "hardness": cand.hardness,
                "sec": {
                    "best_family": max(
                        [r["sec"].get("best_family", "other") for r in point_rows],
                        key=lambda fam: [r["sec"].get("best_family", "other") for r in point_rows].count(fam),
                    ),
                    "best_bits": float(np.mean([r["sec"].get("best_bits") for r in point_rows if r["sec"].get("best_bits") is not None])) if any(r["sec"].get("best_bits") is not None for r in point_rows) else None,
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
                "seed_rows": point_rows,
            }

            rec = SampleRecord(
                window_center=center,
                umbrella_width=args.umbrella_width,
                candidate_index=cand_idx,
                n=cand.n,
                q=cand.q,
                sigma=cand.sigma,
                hardness=float(cand.hardness),
                umbrella_energy=energy,
                bias_weight=bias_w,
                chain_id=chain_id,
                step_id=step_id,
                seed=seed,
                sec=agg_point["sec"],
                sp=agg_point["sp"],
            )
            window_records.append(rec)
            all_records.append(rec)
            accepted += 1
            total_chain_accepts += 1
            total_chain_steps += step_id

            umbrella_meter.update(
                avg_point_weight,
                elapsed=point_elapsed,
                center=f"{center:.2f}",
                n=cand.n,
                q=cand.q,
                sigma=f"{cand.sigma:.2f}",
                bits=fmt_bits(cand.hardness),
                eta=human_seconds(
                    umbrella_meter.rate_sec_per_weight * max(0.0, umbrella_meter.total_weight - umbrella_meter.done_weight)
                    if umbrella_meter.rate_sec_per_weight is not None else None
                ),
            )

        window_summary = aggregate_records(window_records)
        window_summary.update(
            {
                "center": center,
                "width": args.umbrella_width,
                "acceptance_rate": accepted / max(1, len(selected_samples)),
                "mean_step_id": float(total_chain_steps / max(1, accepted)),
                "records": [r.__dict__ for r in window_records],
                "wall_clock_min": (time.time() - t_window) / 60.0,
                "hardness_diff_mean": (
                    float(np.mean([r.hardness - args.target_bits for r in window_records])) if window_records else None
                ),
            }
        )
        windows_output.append(window_summary)

        print(
            f"window center={center:.2f}  accepted={accepted}/{len(selected_samples)}  "
            f"H={fmt_f(window_summary['hardness_mean'], 4)}  "
            f"Sigma*={fmt_f(window_summary['complexity_proxy_mean'], 4)}  "
            f"wall={window_summary['wall_clock_min']:.1f} min",
            flush=True,
        )

    umbrella_meter.close()

    overall = aggregate_records(all_records)
    elapsed_min = (time.time() - t0) / 60.0

    results: Dict[str, Any] = {
        "meta": {
            "target_bits": args.target_bits,
            "centers": centers,
            "umbrella_width": args.umbrella_width,
            "samples_per_window": args.samples_per_window,
            "repeats": repeats,
            "burn_in": args.burn_in,
            "thin": args.thin,
            "chain_steps": args.chain_steps,
            "chains": args.chains,
            "quick": args.quick,
            "use_proxy": use_proxy,
            "pool_cache": args.pool_cache,
            "generated_at_unix": time.time(),
        },
        "elapsed_min": elapsed_min,
        "pool_size": len(pool),
        "preview_shell": [c.__dict__ for c in shell_preview],
        "windows": windows_output,
        "overall": overall,
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
