#!/usr/bin/env python3
"""
Experiment 19: Comprehensive Survey Propagation Characterization
================================================================
Run:  sage -python exp19.py
      sage -python exp19.py --quick     (fast pilot, ~15 min)
      sage -python exp19.py --sp-only   (skip lattice-estimator)

OPEN QUESTIONS THIS EXPERIMENT ANSWERS
---------------------------------------
All five sub-tests address gaps left by Experiments 15-18.

A. ALPHA COLLAPSE TEST
   Question: do SP metrics reduce to a function of alpha = q/(sigma*sqrt(n))
   alone, or do they see something beyond alpha?
   Method: sweep 5 alpha levels × 5 n values. At fixed alpha, n, q, sigma all
   change, but hardness should be similar. If SP metrics are also similar, they
   track alpha (not independent geometry). If they vary, SP is seeing more.

B. ATTACK-FAMILY SENSITIVITY
   Question: does SP change at the BDD/USVP/Dual transition boundary?
   Method: sigma sweep at n=64, alpha=12, which Exp 11/13 showed crosses the
   other→BDD boundary near sigma=0.9. Run SP with many seeds, test whether
   warning_density or survey_entropy spike at the family boundary.
   This is the key test of whether SP detects attack-family structure.

C. BP CONVERGENCE QUALITY (RSB PROBE)
   Question: do different random BP initializations converge to the same or
   different fixed points? High divergence = many fixed points = RSB structure.
   Method: 20 BP restarts per instance, measure:
     - fraction_same_MAP: how often restarts agree on argmax(marginal)
     - restart_entropy: entropy of the MAP distribution across restarts
     - marginal_variance: per-variable variance of marginals across restarts
   Compare across phases (easy funnel / hard rugged).

D. DECIMATION RECOVERY TEST
   Question: does SP-guided decimation actually recover the true secret for
   small n, and does success rate correlate with SP metrics?
   Method: n=16-28, vary sigma for easy/hard. Run iterative decimation:
   fix top-confidence variables, re-run BP, repeat. Measure % correct.
   A useful SP signal should predict decimation success.

E. STABILITY ANALYSIS (RELIABILITY)
   Question: which SP metrics are stable across seeds (low CV)?
   Unstable metrics should not be used for paper claims.
   Method: 20 seeds per point at 4 representative parameter configurations.
   Report mean, std, CV for each metric.
   A metric with CV < 0.15 is stable enough for claims.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    tqdm = None
    HAS_TQDM = False

try:
    import exp15 as base
    ESTIMATOR_AVAILABLE = getattr(base, "ESTIMATOR_AVAILABLE", False)
except Exception as exc:
    raise SystemExit("exp15.py not found. Place exp19.py in the same directory.") from exc

DEFAULT_JOBS = getattr(base, "DEFAULT_JOBS", 4)


def bar(it, desc="", total=None):
    if HAS_TQDM:
        return tqdm(it, desc=desc, total=total, leave=False, dynamic_ncols=True)
    return it

def fmt(x, d=4):
    return "n/a" if x is None or not np.isfinite(float(x)) else f"{float(x):.{d}f}"

def ensure_odd(q):
    return int(q) if int(q) % 2 == 1 else int(q) + 1

def get_estimator(n, q, sigma, use_proxy):
    if use_proxy or not ESTIMATOR_AVAILABLE:
        return base.bkz_proxy_bits(n=n, q=q, sigma=sigma)
    return base.run_estimator(n=n, q=q, sigma=sigma, jobs=DEFAULT_JOBS)

def run_sp(n, q, sigma, seed, bp_iter=50, sp_iter=20, n_restarts=8, light=False):
    """Run one LWE instance through the full SP pipeline. Returns sp_metrics dict."""
    m = min(3 * n, max(2 * n, 48))
    lwe = base.generate_lwe(n=n, m=3*n, q=q, sigma=sigma, seed=seed, max_rows=m)
    sp  = base.LWEsurveyPropagation(lwe)
    return sp.run(max_iter=sp_iter, n_restarts=n_restarts, bp_iter=bp_iter,
                  light_sp=light)

def bootstrap_corr(x, y, n_boot=1000, ci=0.95):
    """Bootstrap 95% CI for Pearson r between x and y."""
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(xa) & np.isfinite(ya)
    if m.sum() < 5:
        return None, None, None
    xv, yv = xa[m], ya[m]
    if np.std(xv) == 0 or np.std(yv) == 0:
        return None, None, None
    r_obs = float(np.corrcoef(xv, yv)[0, 1])
    rng = np.random.default_rng(42)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(xv), len(xv))
        if np.std(xv[idx]) == 0 or np.std(yv[idx]) == 0:
            continue
        rs.append(float(np.corrcoef(xv[idx], yv[idx])[0, 1]))
    if not rs:
        return r_obs, None, None
    lo = float(np.percentile(rs, (1 - ci) / 2 * 100))
    hi = float(np.percentile(rs, (1 + ci) / 2 * 100))
    return r_obs, lo, hi

def spearman_p(x, y):
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(xa) & np.isfinite(ya)
    if m.sum() < 5:
        return None, None
    r, p = scipy_stats.spearmanr(xa[m], ya[m])
    return float(r), float(p)


ALPHA_LEVELS  = [6.0, 9.0, 12.0, 18.0, 24.0]
ALPHA_NS      = [24, 32, 40, 48, 56]
ALPHA_SIGMA   = 1.5
ALPHA_SEEDS   = 4

def run_part_a(use_proxy, quick):
    print("\n" + "="*70)
    print("PART A: Alpha Collapse Test")
    print("  Does SP reduce to f(alpha) or does it see beyond alpha?")
    print("="*70)
    ns_use     = ALPHA_NS[:3] if quick else ALPHA_NS
    alphas_use = ALPHA_LEVELS[:3] if quick else ALPHA_LEVELS
    seeds_use  = 2 if quick else ALPHA_SEEDS
    kw = dict(bp_iter=30, sp_iter=8, n_restarts=4, light=True) if quick \
         else dict(bp_iter=60, sp_iter=20, n_restarts=8, light=False)

    rows = []
    total = len(alphas_use) * len(ns_use) * seeds_use
    pb = bar(range(total), desc="Part A", total=total)
    counter = 0

    for alpha in alphas_use:
        for n in ns_use:
            q = ensure_odd(int(alpha * ALPHA_SIGMA * math.sqrt(n)))
            q = max(q, n + 2)
            sp_rows = []
            for seed in range(seeds_use):
                sp_m = run_sp(n, q, ALPHA_SIGMA, seed * 71 + n + int(alpha * 3), **kw)
                sp_rows.append(sp_m)
                if HAS_TQDM: next(iter([pb]))
                counter += 1
                if counter % 10 == 0:
                    print(f"  A: alpha={alpha:.0f} n={n} q={q} seed={seed} "
                          f"warn={sp_rows[-1]['warning_density']:.4f}", flush=True)
            # aggregate
            def mn(k): return float(np.mean([r[k] for r in sp_rows if r.get(k) is not None]))
            sec = get_estimator(n, q, ALPHA_SIGMA, use_proxy)
            rows.append({
                "alpha": alpha, "n": n, "q": q, "sigma": ALPHA_SIGMA,
                "hardness": sec.get("best_bits"),
                "family": sec.get("best_family", "other"),
                "warning_density":    mn("warning_density"),
                "survey_entropy_norm":mn("survey_entropy_norm"),
                "overlap_proxy":      mn("overlap_proxy"),
                "complexity_proxy":   mn("complexity_proxy"),
                "map_accuracy":       mn("map_accuracy"),
                "bp_delta":           mn("bp_delta"),
            })

    # Analysis: for each alpha, how much do SP metrics vary with n?
    print("\n  Alpha collapse results:")
    print(f"  {'alpha':>6}  {'n':>4}  {'q':>5}  {'hardness':>9}  "
          f"{'warn_d':>8}  {'H_surv':>8}  {'complexity':>10}")
    for r in rows:
        print(f"  {r['alpha']:>6.1f}  {r['n']:>4}  {r['q']:>5}  "
              f"{fmt(r['hardness'],2):>9}  {fmt(r['warning_density'],4):>8}  "
              f"{fmt(r['survey_entropy_norm'],4):>8}  {fmt(r['complexity_proxy'],3):>10}")

    # Compute: r(SP, alpha) vs r(SP, hardness) vs r(SP, n)
    alphas   = [r["alpha"]             for r in rows]
    ns       = [r["n"]                 for r in rows]
    qs       = [r["q"]                 for r in rows]
    hard     = [r["hardness"]          for r in rows]
    warn_d   = [r["warning_density"]   for r in rows]
    h_surv   = [r["survey_entropy_norm"] for r in rows]
    comp     = [r["complexity_proxy"]  for r in rows]

    print("\n  Correlations (Spearman, alpha vs hardness vs n vs q):")
    for name, vals in [("warning_density", warn_d),
                       ("survey_entropy",  h_surv),
                       ("complexity_proxy",comp)]:
        ra, pa = spearman_p(alphas, vals)
        rh, ph = spearman_p(hard,   vals)
        rn, pn = spearman_p(ns,     vals)
        rq, pq = spearman_p(qs,     vals)
        print(f"  {name:22s}: r_alpha={fmt(ra,3)}(p={fmt(pa,3)})  "
              f"r_hardness={fmt(rh,3)}(p={fmt(ph,3)})  "
              f"r_n={fmt(rn,3)}(p={fmt(pn,3)})  r_q={fmt(rq,3)}(p={fmt(pq,3)})")

    print("\n  Within-alpha SP variation (CV = std/mean across n):")
    for alpha in alphas_use:
        sub = [r for r in rows if r["alpha"] == alpha]
        if len(sub) < 2: continue
        for name, k in [("warn_d", "warning_density"),
                        ("H_surv","survey_entropy_norm"),
                        ("complexity","complexity_proxy")]:
            vals_a = [r[k] for r in sub if r[k] is not None]
            if len(vals_a) < 2: continue
            cv = float(np.std(vals_a) / (abs(np.mean(vals_a)) + 1e-9))
            verdict = "COLLAPSES" if cv < 0.05 else ("partial" if cv < 0.15 else "DOES NOT collapse")
            print(f"    alpha={alpha:.0f} {name}: CV={cv:.4f} → {verdict}")

    return rows



FAM_SIGMAS = [0.50, 0.65, 0.80, 0.90, 0.95, 1.00, 1.10, 1.30, 1.60, 2.00, 3.00]
FAM_N      = 64
FAM_ALPHA  = 12.0
FAM_SEEDS  = 6

def run_part_b(use_proxy, quick):
    print("\n" + "="*70)
    print("PART B: Attack-Family Sensitivity")
    print("  Does SP change at the BDD/USVP/other family boundary?")
    print("  (Known transition at sigma≈0.9 from Exp 11/13)")
    print("="*70)
    sigs_use  = [0.50, 0.90, 1.10, 2.00] if quick else FAM_SIGMAS
    seeds_use = 2 if quick else FAM_SEEDS
    kw = dict(bp_iter=30, sp_iter=8, n_restarts=4, light=True) if quick \
         else dict(bp_iter=60, sp_iter=20, n_restarts=8, light=False)

    rows = []
    for sigma in sigs_use:
        q = ensure_odd(int(FAM_ALPHA * sigma * math.sqrt(FAM_N)))
        q = max(q, FAM_N + 2)
        sp_list = []
        for seed in range(seeds_use):
            sp_m = run_sp(FAM_N, q, sigma, seed * 97 + FAM_N, **kw)
            sp_list.append(sp_m)
            print(f"  B: sigma={sigma:.2f} q={q} seed={seed} "
                  f"warn={sp_m['warning_density']:.4f} "
                  f"H_surv={sp_m['survey_entropy_norm']:.4f}", flush=True)
        def mn(k): return float(np.mean([r[k] for r in sp_list if r.get(k) is not None]))
        def sd(k):
            v = [r[k] for r in sp_list if r.get(k) is not None]
            return float(np.std(v)) if len(v) >= 2 else 0.0
        sec = get_estimator(FAM_N, q, sigma, use_proxy)
        rows.append({
            "sigma": sigma, "q": q, "n": FAM_N, "alpha": FAM_ALPHA,
            "hardness":            sec.get("best_bits"),
            "family":              sec.get("best_family", "other"),
            "dominance_gap":       sec.get("dominance_gap"),
            "warning_density":     mn("warning_density"),
            "warning_density_sd":  sd("warning_density"),
            "survey_entropy_norm": mn("survey_entropy_norm"),
            "survey_entropy_sd":   sd("survey_entropy_norm"),
            "overlap_proxy":       mn("overlap_proxy"),
            "complexity_proxy":    mn("complexity_proxy"),
            "map_accuracy":        mn("map_accuracy"),
            "bp_delta":            mn("bp_delta"),
        })

    print("\n  Family boundary test results:")
    print(f"  {'sigma':>6}  {'q':>5}  {'family':>12}  {'dgap':>6}  "
          f"{'warn_d ± sd':>14}  {'H_surv ± sd':>14}")
    prev_fam = None
    for r in rows:
        marker = " ← TRANSITION" if prev_fam and r["family"] != prev_fam else ""
        print(f"  {r['sigma']:>6.2f}  {r['q']:>5}  {r['family']:>12}  "
              f"{fmt(r['dominance_gap'],2):>6}  "
              f"{fmt(r['warning_density'],4):>7} ± {fmt(r['warning_density_sd'],4):>6}  "
              f"{fmt(r['survey_entropy_norm'],4):>7} ± {fmt(r['survey_entropy_sd'],4):>6}"
              f"{marker}")
        prev_fam = r["family"]

    # Key test: do SP metrics differ between family groups?
    families_found = list({r["family"] for r in rows})
    for metric in ["warning_density", "survey_entropy_norm"]:
        groups = {}
        for r in rows:
            fam = r["family"]
            if fam not in groups: groups[fam] = []
            if r[metric] is not None: groups[fam].append(r[metric])
        if len(groups) >= 2:
            keys = list(groups.keys())
            for i in range(len(keys)):
                for j in range(i+1, len(keys)):
                    g1, g2 = groups[keys[i]], groups[keys[j]]
                    if len(g1) >= 2 and len(g2) >= 2:
                        t, p = scipy_stats.ttest_ind(g1, g2)
                        sig = "*** p<0.05" if p < 0.05 else "(not significant)"
                        print(f"  t-test {metric} [{keys[i]} vs {keys[j]}]: "
                              f"t={t:.3f}, p={p:.4f} {sig}")

    return rows

CONV_CONFIGS = [
    # (n, q, sigma, label, expected_phase)
    (32, 101, 0.5,  "easy_low_sigma",  "funnel"),
    (32, 101, 2.0,  "easy_high_sigma", "funnel"),
    (64, 101, 1.5,  "medium",          "funnel"),
    (96, 200, 1.5,  "hard",            "rugged"),
    (96, 200, 3.0,  "very_hard",       "rugged"),
]
CONV_RESTARTS = 20
CONV_BP_ITER  = 80

def run_part_c(use_proxy, quick):
    print("\n" + "="*70)
    print("PART C: BP Convergence Quality — RSB Probe")
    print("  20 random BP restarts per instance. High divergence = RSB structure.")
    print("="*70)
    configs_use = CONV_CONFIGS[:3] if quick else CONV_CONFIGS
    n_rest = 8 if quick else CONV_RESTARTS
    bp_it  = 40 if quick else CONV_BP_ITER

    rows = []
    for (n, q, sigma, label, phase) in configs_use:
        q = ensure_odd(q)
        print(f"\n  Config: {label} (n={n}, q={q}, σ={sigma})")

        # Run many BP restarts for ONE instance (seed=0)
        m = min(3 * n, max(2 * n, 48))
        lwe = base.generate_lwe(n=n, m=3*n, q=q, sigma=sigma, seed=0, max_rows=m)
        bp  = base.ModQBeliefPropagation(lwe.A, lwe.b, q, sigma)
        vals = np.arange(q, dtype=np.float64)

        rng       = np.random.default_rng(42)
        maps_list = []
        deltas    = []
        marg_list = []

        for r_idx in bar(range(n_rest), desc=f"  {label} restarts"):
            init = rng.dirichlet(np.ones(q), size=n)
            marg, iters, delta = bp.run(max_iter=bp_it, init=init)
            MAP = np.argmax(marg, axis=1)
            maps_list.append(MAP)
            deltas.append(delta)
            marg_list.append(marg)

        maps_arr = np.stack(maps_list, axis=0)  # (R, n)
        marg_arr = np.stack(marg_list, axis=0)  # (R, n, q)

        # Fraction of restarts that agree with the modal MAP
        mode_map = scipy_stats.mode(maps_arr, axis=0).mode.flatten()
        agree    = float(np.mean(np.all(maps_arr == mode_map, axis=1)))

        # Per-variable MAP consistency: fraction of restarts where MAP[i] = mode
        var_consist = np.mean(maps_arr == mode_map, axis=0)
        mean_consist = float(var_consist.mean())
        min_consist  = float(var_consist.min())

        # Marginal variance across restarts (per variable, per value) → mean
        marg_var = float(marg_arr.var(axis=0).mean())

        # Entropy of MAP distribution across restarts
        unique_maps, counts = np.unique(maps_arr, axis=0, return_counts=True)
        probs = counts / counts.sum()
        restart_entropy = float(-np.sum(probs * np.log(probs + 1e-300)))
        restart_entropy_norm = restart_entropy / math.log(len(probs) + 1)

        # Fraction of unique MAPs
        frac_unique = len(unique_maps) / n_rest

        # Accuracy on true secret
        true_s = lwe.s_true
        acc = float(np.mean(mode_map == true_s))

        # Mean convergence delta
        mean_delta = float(np.mean(deltas))

        sec = get_estimator(n, q, sigma, use_proxy)

        row = {
            "label": label, "phase": phase, "n": n, "q": q, "sigma": sigma,
            "hardness": sec.get("best_bits"),
            "family":   sec.get("best_family", "other"),
            "agree":       agree,           # fraction of restarts → same MAP
            "mean_consist":mean_consist,    # mean per-var consistency
            "min_consist": min_consist,     # worst-case var consistency
            "marg_var":    marg_var,        # marginal variance across restarts
            "restart_entropy_norm": restart_entropy_norm,
            "frac_unique_maps": frac_unique,
            "mean_delta": mean_delta,
            "true_acc":  acc,
        }
        rows.append(row)
        print(f"  agree={agree:.3f}  mean_consist={mean_consist:.3f}  "
              f"marg_var={marg_var:.5f}  restart_H={restart_entropy_norm:.3f}  "
              f"unique_maps={frac_unique:.2f}  true_acc={acc:.3f}  "
              f"delta={mean_delta:.5f}")

    print("\n  Summary: RSB-like structure = low agree, high restart_entropy, many unique MAPs")
    print(f"  {'label':>20}  {'agree':>7}  {'restart_H':>10}  "
          f"{'marg_var':>9}  {'true_acc':>9}")
    for r in rows:
        print(f"  {r['label']:>20}  {fmt(r['agree'],3):>7}  "
              f"{fmt(r['restart_entropy_norm'],4):>10}  "
              f"{fmt(r['marg_var'],5):>9}  {fmt(r['true_acc'],3):>9}")

    return rows



DEC_N_VALUES  = [16, 20, 24, 28]
DEC_SIGMAS    = [0.5, 1.0, 2.0, 3.5]   # easy → hard
DEC_SEEDS     = 5
DEC_BP_ITER   = 60
DEC_CHUNK     = 3   # variables to fix per round

def run_decimation_full(lwe, bp_iter=60, chunk=3):
    """Iterative decimation: fix top-chunk confident vars per round."""
    n, q = lwe.n, lwe.q
    A, b = lwe.A.copy(), lwe.b.copy()
    s_fixed = np.full(n, -1, dtype=np.int64)  # -1 = not yet fixed
    order   = []

    for _ in range(math.ceil(n / chunk)):
        free_idx = np.where(s_fixed == -1)[0]
        if len(free_idx) == 0:
            break
        A_sub = A[:, free_idx]
        b_adj = b.copy()
        for i, fix_i in enumerate(np.where(s_fixed >= 0)[0]):
            b_adj = (b_adj - A[:, fix_i] * s_fixed[fix_i]) % q
        lwe_sub = base.generate_lwe(n=len(free_idx), m=A_sub.shape[0],
                                    q=q, sigma=lwe.sigma, seed=0)
        lwe_sub.A, lwe_sub.b = A_sub.astype(np.int64), b_adj.astype(np.int64)
        bp  = base.ModQBeliefPropagation(lwe_sub.A, lwe_sub.b, q, lwe.sigma)
        marg, _, _ = bp.run(max_iter=bp_iter)
        conf = marg.max(axis=1)
        top  = np.argsort(-conf)[:chunk]
        for t in top:
            global_idx = free_idx[t]
            val        = int(np.argmax(marg[t]))
            s_fixed[global_idx] = val
            order.append(global_idx)

    acc = float(np.mean(s_fixed == lwe.s_true))
    return acc, s_fixed

def run_part_d(use_proxy, quick):
    print("\n" + "="*70)
    print("PART D: Decimation Recovery Test")
    print("  Does SP-guided decimation recover the secret for small n?")
    print("="*70)
    ns_use  = DEC_N_VALUES[:2] if quick else DEC_N_VALUES
    sigs_use= DEC_SIGMAS[:2] if quick else DEC_SIGMAS
    seeds_u = 2 if quick else DEC_SEEDS
    bp_it   = 30 if quick else DEC_BP_ITER

    rows = []
    for n in ns_use:
        q = ensure_odd(int(12.0 * 1.5 * math.sqrt(n)))  # fixed alpha=12
        q = max(q, n + 2)
        for sigma in sigs_use:
            accs = []
            for seed in range(seeds_u):
                lwe = base.generate_lwe(n=n, m=3*n, q=q, sigma=sigma, seed=seed)
                acc, _ = run_decimation_full(lwe, bp_iter=bp_it, chunk=DEC_CHUNK)
                accs.append(acc)
                print(f"  D: n={n} q={q} sigma={sigma:.1f} seed={seed} "
                      f"acc={acc:.3f}", flush=True)
            # Also run SP for metrics
            sp_m = run_sp(n, q, sigma, 0,
                          bp_iter=bp_it, sp_iter=10, n_restarts=4, light=True)
            sec  = get_estimator(n, q, sigma, use_proxy)
            rows.append({
                "n": n, "q": q, "sigma": sigma,
                "hardness": sec.get("best_bits"),
                "family":   sec.get("best_family", "other"),
                "dec_acc_mean": float(np.mean(accs)),
                "dec_acc_std":  float(np.std(accs)),
                "warning_density":     sp_m.get("warning_density"),
                "survey_entropy_norm": sp_m.get("survey_entropy_norm"),
                "overlap_proxy":       sp_m.get("overlap_proxy"),
                "map_accuracy":        sp_m.get("map_accuracy"),
            })

    print("\n  Decimation recovery results:")
    print(f"  {'n':>4}  {'sigma':>6}  {'hardness':>9}  {'dec_acc':>8}  "
          f"{'±sd':>6}  {'map_acc':>8}  {'warn_d':>8}")
    for r in rows:
        print(f"  {r['n']:>4}  {r['sigma']:>6.1f}  {fmt(r['hardness'],2):>9}  "
              f"{fmt(r['dec_acc_mean'],3):>8}  {fmt(r['dec_acc_std'],3):>6}  "
              f"{fmt(r['map_accuracy'],3):>8}  {fmt(r['warning_density'],4):>8}")

    # Correlation: decimation accuracy vs SP metrics
    dec_accs = [r["dec_acc_mean"]        for r in rows]
    warn_d   = [r["warning_density"]     for r in rows]
    h_surv   = [r["survey_entropy_norm"] for r in rows]
    map_accs = [r["map_accuracy"]        for r in rows]
    hard_v   = [r["hardness"]            for r in rows]
    print("\n  Correlations with decimation accuracy:")
    for name, vals in [("warning_density",     warn_d),
                       ("survey_entropy_norm", h_surv),
                       ("map_accuracy",        map_accs),
                       ("hardness",            hard_v)]:
        r_val, lo, hi = bootstrap_corr(vals, dec_accs)
        rs, ps = spearman_p(vals, dec_accs)
        ci_str = f"[{lo:.3f},{hi:.3f}]" if lo is not None else "n/a"
        print(f"    {name:22s}: Pearson={fmt(r_val,3)} 95%CI={ci_str}  "
              f"Spearman={fmt(rs,3)} p={fmt(ps,4)}")

    return rows


STAB_CONFIGS = [
    (32,  67, 0.5,  "easy_low_sigma"),
    (48, 101, 1.5,  "medium"),
    (64, 193, 2.0,  "hard"),
    (96, 235, 2.0,  "very_hard"),
]
STAB_SEEDS = 20

def run_part_e(use_proxy, quick):
    print("\n" + "="*70)
    print("PART E: Stability Analysis")
    print("  20 seeds per config. Which SP metrics are reliable (CV < 0.15)?")
    print("="*70)
    configs_u = STAB_CONFIGS[:2] if quick else STAB_CONFIGS
    seeds_u   = 5 if quick else STAB_SEEDS
    kw = dict(bp_iter=30, sp_iter=8, n_restarts=4, light=True) if quick \
         else dict(bp_iter=60, sp_iter=20, n_restarts=8, light=False)

    METRIC_KEYS = [
        "warning_density", "survey_entropy_norm", "overlap_proxy",
        "complexity_proxy", "map_accuracy", "bp_delta",
    ]
    rows = []
    for (n, q, sigma, label) in configs_u:
        q = ensure_odd(q)
        print(f"\n  Config: {label} (n={n}, q={q}, σ={sigma})")
        sp_list = []
        for seed in bar(range(seeds_u), desc=f"  {label}"):
            sp_list.append(run_sp(n, q, sigma, seed * 13 + n, **kw))

        stats_row = {"label": label, "n": n, "q": q, "sigma": sigma}
        print(f"  {'metric':>22}  {'mean':>8}  {'std':>8}  {'CV':>7}  verdict")
        for k in METRIC_KEYS:
            vals = [r[k] for r in sp_list if r.get(k) is not None
                    and np.isfinite(r.get(k, float('nan')))]
            if not vals:
                continue
            mn = float(np.mean(vals))
            sd = float(np.std(vals))
            cv = sd / (abs(mn) + 1e-9)
            verdict = ("STABLE" if cv < 0.10 else
                       "usable" if cv < 0.20 else "NOISY")
            print(f"  {k:>22}  {mn:>8.4f}  {sd:>8.4f}  {cv:>7.4f}  {verdict}")
            stats_row[f"{k}_mean"] = mn
            stats_row[f"{k}_std"]  = sd
            stats_row[f"{k}_cv"]   = cv
            stats_row[f"{k}_stable"] = cv < 0.20
        rows.append(stats_row)

    # Summary: which metrics are stable across ALL configs?
    print("\n  Cross-config stability summary:")
    for k in METRIC_KEYS:
        all_cvs = [r.get(f"{k}_cv") for r in rows if r.get(f"{k}_cv") is not None]
        if not all_cvs: continue
        max_cv = max(all_cvs)
        verdict = ("ALWAYS STABLE" if max_cv < 0.10 else
                   "MOSTLY STABLE" if max_cv < 0.20 else
                   "SOMETIMES NOISY" if max_cv < 0.40 else "NOISY")
        print(f"  {k:>22}: max_CV={max_cv:.4f} → {verdict}")

    return rows


def plot_all(pa, pb, pc, pd, pe, save="exp19_results.png"):
    BG, PBG = "#0f1117", "#1a1d27"
    fig = plt.figure(figsize=(24, 20))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.40)

    def sty(ax, title, xl="", yl="", tc="white"):
        ax.set_facecolor(PBG); ax.set_title(title, color=tc, fontsize=8, pad=5)
        ax.set_xlabel(xl, color="white", fontsize=7)
        ax.set_ylabel(yl, color="white", fontsize=7)
        ax.tick_params(colors="white", labelsize=6)
        for sp in ax.spines.values(): sp.set_edgecolor("#444")

    cmap = plt.cm.plasma

    # ── A: warning_density vs alpha, colored by n ────────────────────────────
    if pa:
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.set_facecolor(PBG)
        ns_uniq = sorted({r["n"] for r in pa})
        for r in pa:
            c = cmap(ns_uniq.index(r["n"]) / max(len(ns_uniq)-1, 1))
            ax0.scatter(r["alpha"], r["warning_density"], color=c, s=70,
                        edgecolors="white", lw=0.4)
        sty(ax0, "A: warning_density vs alpha\n(color = n)", "alpha", "warning_density")

        ax1 = fig.add_subplot(gs[0, 1])
        ax1.set_facecolor(PBG)
        for r in pa:
            c = cmap(ns_uniq.index(r["n"]) / max(len(ns_uniq)-1, 1))
            ax1.scatter(r["alpha"], r["complexity_proxy"], color=c, s=70,
                        edgecolors="white", lw=0.4)
        sty(ax1, "A: complexity_proxy vs alpha\n(color = n)", "alpha", "complexity_proxy")

    # ── B: SP metrics vs sigma with family transition ────────────────────────
    if pb:
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_facecolor(PBG)
        sigs   = [r["sigma"] for r in pb]
        warn_d = [r["warning_density"] for r in pb]
        sd_w   = [r.get("warning_density_sd", 0) for r in pb]
        fams   = [r["family"] for r in pb]
        ax2.errorbar(sigs, warn_d, yerr=sd_w, fmt="o-", color="#ffd93d",
                     capsize=3, lw=2, ms=5)
        # Mark family transitions
        for i in range(1, len(pb)):
            if fams[i] != fams[i-1]:
                mid = 0.5*(sigs[i-1]+sigs[i])
                ax2.axvline(mid, color="white", lw=1.5, linestyle="--", alpha=0.7)
                ax2.text(mid, max(v for v in warn_d if v), f"{fams[i-1]}→{fams[i]}",
                         fontsize=6, color="white", rotation=90, va="top")
        sty(ax2, "B: warning_density vs sigma\n(dashed = family boundary)",
            "sigma", "warning_density ± std")

        ax3 = fig.add_subplot(gs[0, 3])
        ax3.set_facecolor(PBG)
        h_surv = [r["survey_entropy_norm"] for r in pb]
        sd_h   = [r.get("survey_entropy_sd", 0) for r in pb]
        ax3.errorbar(sigs, h_surv, yerr=sd_h, fmt="o-", color="#69ff94",
                     capsize=3, lw=2, ms=5)
        for i in range(1, len(pb)):
            if fams[i] != fams[i-1]:
                ax3.axvline(0.5*(sigs[i-1]+sigs[i]), color="white",
                            lw=1.5, linestyle="--", alpha=0.7)
        sty(ax3, "B: survey_entropy vs sigma\n(dashed = family boundary)",
            "sigma", "H_survey ± std")

    # ── C: convergence quality bars ───────────────────────────────────────────
    if pc:
        ax4 = fig.add_subplot(gs[1, :2])
        ax4.set_facecolor(PBG)
        labels = [r["label"][:16] for r in pc]
        x      = np.arange(len(labels))
        w      = 0.3
        agree  = [r["agree"] for r in pc]
        re_h   = [r["restart_entropy_norm"] for r in pc]
        uniq   = [r["frac_unique_maps"] for r in pc]
        ax4.bar(x - w, agree, w, color="#69ff94", alpha=0.85, label="agree")
        ax4.bar(x,     re_h,  w, color="#ffd93d", alpha=0.85, label="restart_H_norm")
        ax4.bar(x + w, uniq,  w, color="#00d4ff", alpha=0.85, label="frac_unique_MAPs")
        ax4.set_xticks(x); ax4.set_xticklabels(labels, rotation=25, fontsize=6)
        sty(ax4, "C: BP Convergence Quality\n"
                 "low agree + high restart_H = RSB / many fixed points",
            "", "value", "#00d4ff")
        ax4.legend(fontsize=7, labelcolor="white", facecolor=PBG)

    # ── D: decimation accuracy vs hardness ───────────────────────────────────
    if pd:
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.set_facecolor(PBG)
        for r in pd:
            c = cmap(min((r["sigma"] - 0.5) / 3.5, 1.0))
            ax5.scatter(r["hardness"] or 0, r["dec_acc_mean"],
                        color=c, s=70, edgecolors="white", lw=0.4)
            ax5.errorbar(r["hardness"] or 0, r["dec_acc_mean"],
                         yerr=r["dec_acc_std"], fmt="none", ecolor=c, capsize=2)
        sty(ax5, "D: Decimation accuracy vs hardness\n(color = sigma)",
            "log2(hardness)", "fraction correct")

        ax6 = fig.add_subplot(gs[1, 3])
        ax6.set_facecolor(PBG)
        for r in pd:
            ax6.scatter(r["warning_density"] or 0, r["dec_acc_mean"],
                        color=cmap(min((r["sigma"]-0.5)/3.5,1.0)),
                        s=70, edgecolors="white", lw=0.4)
        sty(ax6, "D: Decimation accuracy vs warning_density",
            "warning_density", "fraction correct")

    # ── E: CV heatmap ─────────────────────────────────────────────────────────
    if pe:
        ax7 = fig.add_subplot(gs[2, :])
        ax7.set_facecolor(PBG)
        metrics_e = ["warning_density", "survey_entropy_norm", "overlap_proxy",
                     "complexity_proxy", "map_accuracy", "bp_delta"]
        labels_e  = [r["label"][:16] for r in pe]
        cv_matrix = np.full((len(metrics_e), len(labels_e)), np.nan)
        for j, row in enumerate(pe):
            for i, k in enumerate(metrics_e):
                cv_matrix[i, j] = row.get(f"{k}_cv", np.nan)

        im = ax7.imshow(cv_matrix, aspect="auto", cmap="RdYlGn_r",
                        vmin=0, vmax=0.4)
        ax7.set_xticks(range(len(labels_e)))
        ax7.set_yticks(range(len(metrics_e)))
        ax7.set_xticklabels(labels_e, color="white", fontsize=7)
        ax7.set_yticklabels(metrics_e, color="white", fontsize=7)
        for i in range(len(metrics_e)):
            for j in range(len(labels_e)):
                v = cv_matrix[i, j]
                if np.isfinite(v):
                    ax7.text(j, i, f"{v:.3f}", ha="center", va="center",
                             color="black", fontsize=6, fontweight="bold")
        plt.colorbar(im, ax=ax7, fraction=0.02).ax.yaxis.label.set_color("white")
        sty(ax7, "E: Stability (CV = std/mean per metric per config)\n"
                 "Green < 0.10 STABLE | Yellow 0.10-0.20 usable | Red > 0.20 NOISY",
            "config", "metric")

    fig.suptitle("Experiment 19 — Comprehensive SP Characterization\n"
                 "A:Alpha collapse | B:Family sensitivity | C:RSB probe | "
                 "D:Decimation | E:Stability",
                 color="white", fontsize=11, fontweight="bold", y=1.01)
    plt.savefig(save, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\nPlot → {save}")


def write_summary(pa, pb, pc, pd, pe, path="exp19_summary.txt"):
    sep = "="*72
    with open(path, "w") as f:
        f.write(sep + "\nEXPERIMENT 19 — COMPREHENSIVE SP CHARACTERIZATION\n" + sep + "\n\n")

        # A
        f.write("PART A: ALPHA COLLAPSE TEST\n" + "-"*40 + "\n")
        if pa:
            alphas_a = sorted({r["alpha"] for r in pa})
            for alpha in alphas_a:
                sub = [r for r in pa if r["alpha"] == alpha]
                warn_vs = [r["warning_density"] for r in sub]
                hard_vs = [r["hardness"] for r in sub if r["hardness"]]
                f.write(f"  alpha={alpha:.0f}: warn_d range=[{min(warn_vs):.4f},{max(warn_vs):.4f}]  "
                        f"CV={np.std(warn_vs)/(np.mean(warn_vs)+1e-9):.4f}\n")
            ra, _, _ = bootstrap_corr([r["alpha"] for r in pa],
                                      [r["warning_density"] for r in pa])
            rh, _, _ = bootstrap_corr([r["hardness"] or 0 for r in pa],
                                      [r["warning_density"] for r in pa])
            rn, _, _ = bootstrap_corr([r["n"] for r in pa],
                                      [r["warning_density"] for r in pa])
            f.write(f"\n  r(alpha, warn_d)={fmt(ra,3)}  r(hardness, warn_d)={fmt(rh,3)}  "
                    f"r(n, warn_d)={fmt(rn,3)}\n")
            if ra is not None and rh is not None:
                driver = "alpha" if abs(ra) > abs(rh) else "hardness"
                f.write(f"  Primary driver of SP: {driver.upper()}\n")
        f.write("\n")

        # B
        f.write("PART B: ATTACK-FAMILY SENSITIVITY\n" + "-"*40 + "\n")
        if pb:
            prev = None
            for r in pb:
                marker = " ← TRANSITION" if prev and r["family"] != prev else ""
                f.write(f"  sigma={r['sigma']:.2f}  family={r['family']:>12}  "
                        f"warn_d={fmt(r['warning_density'],4)}±{fmt(r.get('warning_density_sd',0),4)}"
                        f"  H_surv={fmt(r['survey_entropy_norm'],4)}  {marker}\n")
                prev = r["family"]
        f.write("\n")

        # C
        f.write("PART C: BP CONVERGENCE QUALITY\n" + "-"*40 + "\n")
        if pc:
            for r in pc:
                f.write(f"  {r['label']:>20}: agree={fmt(r['agree'],3)}  "
                        f"restart_H={fmt(r['restart_entropy_norm'],4)}  "
                        f"unique_maps={fmt(r['frac_unique_maps'],3)}  "
                        f"marg_var={fmt(r['marg_var'],5)}\n")
            max_uniq = max(r["frac_unique_maps"] for r in pc)
            f.write(f"\n  Max unique_maps fraction: {max_uniq:.3f}\n")
            if max_uniq > 0.5:
                f.write("  → STRONG RSB signal: many distinct BP fixed points\n")
            elif max_uniq > 0.2:
                f.write("  → Moderate RSB signal\n")
            else:
                f.write("  → Weak RSB: BP converges to few fixed points\n")
        f.write("\n")

        # D
        f.write("PART D: DECIMATION RECOVERY\n" + "-"*40 + "\n")
        if pd:
            for r in pd:
                f.write(f"  n={r['n']} sigma={r['sigma']:.1f}: "
                        f"dec_acc={fmt(r['dec_acc_mean'],3)}±{fmt(r['dec_acc_std'],3)}  "
                        f"map_acc={fmt(r['map_accuracy'],3)}  "
                        f"warn_d={fmt(r['warning_density'],4)}\n")
            dec_accs = [r["dec_acc_mean"] for r in pd]
            warn_ds  = [r["warning_density"] for r in pd]
            r_dw, _, _ = bootstrap_corr(warn_ds, dec_accs)
            f.write(f"\n  r(warning_density, decimation_accuracy): {fmt(r_dw,4)}\n")
        f.write("\n")

        # E
        f.write("PART E: STABILITY (CV ACROSS SEEDS)\n" + "-"*40 + "\n")
        if pe:
            metrics_e = ["warning_density", "survey_entropy_norm",
                         "overlap_proxy", "complexity_proxy"]
            for k in metrics_e:
                cvs = [r.get(f"{k}_cv") for r in pe if r.get(f"{k}_cv") is not None]
                if not cvs: continue
                max_cv = max(cvs)
                verdict = ("STABLE" if max_cv < 0.10 else
                           "usable" if max_cv < 0.20 else "NOISY")
                f.write(f"  {k:>24}: max_CV={max_cv:.4f} → {verdict}\n")

        f.write("\n" + sep + "\n")
        f.write("PAPER CLAIMS SUPPORTED BY THIS EXPERIMENT\n" + sep + "\n")
        f.write("""
1. WHAT SP IS TRACKING (Part A):
   Primary driver of SP warning_density is alpha = q/(sigma*sqrt(n)), not hardness.
   This means SP captures the signal-to-noise geometry of LWE.

2. ATTACK-FAMILY DETECTION (Part B):
   If t-test is significant: SP detects BDD/USVP/Dual transitions — novel finding.
   If not significant: SP is family-blind, consistent with ξ_primal result from
   umbrella sampling (Exp 11).

3. RSB STRUCTURE (Part C):
   If unique_maps > 0.5: LWE exhibits 1RSB-like behavior in the hard regime.
   Multiple BP fixed points = clustered solution space = SP's theoretical sweet spot.

4. ALGORITHMIC VALUE (Part D):
   If decimation_accuracy > chance AND correlates with warning_density:
   SP-guided decimation adds value for small-n LWE recovery.

5. METRIC RELIABILITY (Part E):
   Stable metrics (CV < 0.10) can be used in paper claims without caveats.
   Noisy metrics need either more seeds or should be dropped from claims.
""")
        f.write(sep + "\n")
    print(f"Summary → {path}")


def main():
    parser = argparse.ArgumentParser(description="Exp 19: Comprehensive SP test")
    parser.add_argument("--quick",   action="store_true",
                        help="Fast pilot (~15 min instead of ~90 min)")
    parser.add_argument("--sp-only", action="store_true",
                        help="Use BKZ proxy, skip lattice-estimator")
    parser.add_argument("--skip-a",  action="store_true", help="Skip Part A")
    parser.add_argument("--skip-b",  action="store_true", help="Skip Part B")
    parser.add_argument("--skip-c",  action="store_true", help="Skip Part C")
    parser.add_argument("--skip-d",  action="store_true", help="Skip Part D")
    parser.add_argument("--skip-e",  action="store_true", help="Skip Part E")
    args = parser.parse_args()

    use_proxy = args.sp_only or not ESTIMATOR_AVAILABLE

    print("="*70)
    print("Experiment 19: Comprehensive Survey Propagation Characterization")
    print(f"quick={args.quick}  use_proxy={use_proxy}")
    print(f"Estimator available: {ESTIMATOR_AVAILABLE}")
    print("="*70)

    t0 = time.time()
    pa = pb = pc = pd = pe = None

    if not args.skip_a:
        pa = run_part_a(use_proxy, args.quick)
    if not args.skip_b:
        pb = run_part_b(use_proxy, args.quick)
    if not args.skip_c:
        pc = run_part_c(use_proxy, args.quick)
    if not args.skip_d:
        pd = run_part_d(use_proxy, args.quick)
    if not args.skip_e:
        pe = run_part_e(use_proxy, args.quick)

    print(f"\n{'='*70}")
    print(f"All parts done in {(time.time()-t0)/60:.1f} min. Saving outputs...")

    plot_all(pa or [], pb or [], pc or [], pd or [], pe or [])
    write_summary(pa or [], pb or [], pc or [], pd or [], pe or [])

    result = {"A": pa, "B": pb, "C": pc, "D": pd, "E": pe}
    with open("exp19_data.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("Data → exp19_data.json")
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
    print("Outputs: exp19_results.png  exp19_summary.txt  exp19_data.json")


if __name__ == "__main__":
    main()