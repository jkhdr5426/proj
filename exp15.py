#!/usr/bin/env python3
"""
Experiment 15: Survey Propagation on LWE vs Lattice-Estimator Hardness
=============
Recommended (real lattice-estimator ground truth) — run inside WSL with Sage:

    # One-time setup in WSL:
    git clone https://github.com/malb/lattice-estimator ~/lattice-estimator

    # From WSL (project lives on the Windows drive):
    cd "/mnt/c/Users/Edward Kim/Desktop/proj"
    sage -python exp15.py
    sage -python exp15.py --quick

    # From Windows PowerShell (delegates to WSL):
    wsl bash -lc "cd '/mnt/c/Users/Edward Kim/Desktop/proj' && sage -python exp15.py --quick"

Windows-only ``python exp15.py`` works for SP development but falls back to a BKZ
proxy unless Sage + lattice-estimator are available on that Python.

Background
----------
Survey propagation (SP) is a message-passing algorithm from the cavity method /
1RSB formalism (Mezard, Parisi, Zecchina). It succeeded on random k-SAT near
the satisfiability threshold where single-chain MCMC and umbrella sampling fail,
by operating on distributions over clusters of solutions rather than individual
configurations.  The same framework applies to lattice problems: Kim et al.
(2022, arXiv:2207.00504) derive cavity / SP equations for CVP; we adapt the
zero-temperature warning-propagation layer to mod-q LWE recovery.

Core question
-------------
Do SP-derived clustering signals (survey entropy, warning density, 1RSB overlap
proxy) track concrete lattice-estimator attack cost?

Method (this script)
--------------------
1. Encode LWE as a factor graph: n variables s_i in Z_q, m soft mod-q linear
   constraints (one per row of A, subsampled for tractability).
2. Run mod-q belief propagation (mean-field on dense rows) to obtain marginals.
3. Run survey propagation: iterate warning surveys eta_{a->i}(v) on whether
   value v is incompatible with small residual |r_a| <= T.
4. Multi-restart BP gives a 1RSB overlap proxy (marginal dispersion).
5. Optional decimation: fix high-confidence variables and measure secret recovery.
6. Sweep (n, sigma) at fixed alpha; compare SP metrics to lattice-estimator.

Outputs: exp15_results.png, exp15_summary.txt, exp15_data.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

BG = "#0f1117"
PBG = "#1a1d27"

FAMILY_COLORS = {
    "bdd": "#69ff94",
    "usvp": "#00d4ff",
    "dual": "#ffd93d",
    "dual_hybrid": "#ff6b6b",
    "other": "#b0b0b0",
}

# ── Lattice estimator ────────────────────────────────────────────────────────
def _estimator_paths() -> List[str]:
    home = os.path.expanduser("~")
    return [
        os.path.join(home, "lattice-estimator"),
        os.path.join(home, "Desktop", "lattice-estimator"),
        os.path.join(home, "Documents", "lattice-estimator"),
    ]


for _p in _estimator_paths():
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

ESTIMATOR_AVAILABLE = False
LWE = ND = RC = None
RED_COST_MODEL = None
try:
    from estimator import LWE, ND, RC  # type: ignore
    ESTIMATOR_AVAILABLE = True
    RED_COST_MODEL = getattr(RC, "BDGL16", None)
except Exception:
    pass

DEFAULT_JOBS = max(1, min(16, os.cpu_count() or 1))
SWEEP_ALPHA = 12.0

WSL_PROJECT_DIR = "/mnt/c/Users/Edward Kim/Desktop/proj"
WSL_RUN_QUICK = (
    f'wsl bash -lc "cd \'{WSL_PROJECT_DIR}\' && sage -python exp15.py --quick"'
)
WSL_RUN_FULL = (
    f'wsl bash -lc "cd \'{WSL_PROJECT_DIR}\' && sage -python exp15.py"'
)


def _running_in_wsl() -> bool:
    try:
        with open("/proc/version", "r", encoding="utf-8", errors="ignore") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def _print_estimator_help() -> None:
    print("lattice-estimator / Sage not found on this Python.")
    print()
    print("For real attack-cost ground truth, use WSL + Sage:")
    print("  git clone https://github.com/malb/lattice-estimator ~/lattice-estimator")
    print(f"  cd \"{WSL_PROJECT_DIR}\"")
    print("  sage -python exp15.py --quick")
    print()
    print("From Windows PowerShell:")
    print(f"  {WSL_RUN_QUICK}")
    print()
    print("Dev fallback (BKZ proxy, no Sage):")
    print("  python exp15.py --quick --sp-only")


# Helpers

def fmt_bits(x: Optional[float]) -> str:
    return "n/a" if x is None else f"2^{x:.1f}"


def fmt_f(x: Optional[float], d: int = 3) -> str:
    return "n/a" if x is None or not np.isfinite(float(x)) else f"{float(x):.{d}f}"


def corrcoef_nan(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    m = np.isfinite(xa) & np.isfinite(ya)
    if m.sum() < 3 or np.std(xa[m]) == 0 or np.std(ya[m]) == 0:
        return None
    return float(np.corrcoef(xa[m], ya[m])[0, 1])


def center_mod(x: np.ndarray, q: int) -> np.ndarray:
    r = np.remainder(x, q).astype(np.float64)
    half = q // 2
    return np.where(r > half, r - q, r)


def parse_rop_bits(text: str) -> Optional[float]:
    m = re.search(r"rop:\s*≈?\s*2\^([0-9]+(?:\.[0-9]+)?)", text)
    return float(m.group(1)) if m else None


def parse_beta(text: str) -> Optional[float]:
    m = re.search(r"\bβ:\s*([0-9]+(?:\.[0-9]+)?)", text)
    return float(m.group(1)) if m else None


def family_from_name(name: str) -> str:
    n = name.lower().replace("-", "_").replace(" ", "_")
    if "bdd" in n:
        return "bdd"
    if "dual_hybrid" in n or "dualhybrid" in n:
        return "dual_hybrid"
    if "dual" in n:
        return "dual"
    if "usvp" in n or "primal" in n:
        return "usvp"
    return "other"


def parse_result(result: Any) -> Dict[str, Any]:
    table: Dict[str, Dict[str, Optional[float]]] = {}
    if hasattr(result, "items"):
        for k, v in result.items():
            t = str(v)
            table[str(k)] = {"rop": parse_rop_bits(t), "beta": parse_beta(t)}
    fam_best: Dict[str, Optional[float]] = {f: None for f in ("bdd", "usvp", "dual", "dual_hybrid")}
    fam_beta: Dict[str, Optional[float]] = {f: None for f in fam_best}
    for k, rec in table.items():
        rop = rec["rop"]
        if rop is None:
            continue
        fam = family_from_name(k)
        if fam not in fam_best:
            continue
        if fam_best[fam] is None or rop < fam_best[fam]:
            fam_best[fam] = rop
            fam_beta[fam] = rec["beta"]
    valid = [(f, v) for f, v in fam_best.items() if v is not None]
    if not valid:
        return {
            "fam_best": fam_best,
            "fam_beta": fam_beta,
            "best_family": None,
            "best_bits": None,
            "best_beta": None,
            "dominance_gap": None,
        }
    valid.sort(key=lambda x: x[1])
    best_family, best_bits = valid[0]
    dominance_gap = float(valid[1][1] - valid[0][1]) if len(valid) >= 2 else None
    return {
        "fam_best": fam_best,
        "fam_beta": fam_beta,
        "best_family": best_family,
        "best_bits": best_bits,
        "best_beta": fam_beta[best_family],
        "dominance_gap": dominance_gap,
    }


def run_estimator(n: int, q: int, sigma: float, jobs: int = DEFAULT_JOBS) -> Dict[str, Any]:
    if not ESTIMATOR_AVAILABLE:
        raise RuntimeError("lattice-estimator unavailable")
    m = 3 * n
    try:
        Xs = ND.UniformMod(q)
    except Exception:
        Xs = ND.Uniform(0, q)
    try:
        Xe = ND.DiscreteGaussian(stddev=sigma)
    except Exception:
        Xe = ND.DiscreteGaussian(sigma)
    params = LWE.Parameters(n=n, q=q, Xs=Xs, Xe=Xe, m=m)
    kw: Dict[str, Any] = {"jobs": jobs}
    if RED_COST_MODEL:
        kw["red_cost_model"] = RED_COST_MODEL
    try:
        result = LWE.estimate(params, **kw)
    except TypeError:
        try:
            result = LWE.estimate(params, jobs=jobs)
        except TypeError:
            result = LWE.estimate(params)
    return parse_result(result)


# LWE instance

@dataclass
class LWEInstance:
    n: int
    m: int
    q: int
    sigma: float
    A: np.ndarray
    b: np.ndarray
    s_true: np.ndarray
    row_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))


def generate_lwe(n: int, m: int, q: int, sigma: float, seed: int = 0,
                 max_rows: Optional[int] = None) -> LWEInstance:
    rng = np.random.default_rng(seed)
    A = rng.integers(0, q, size=(m, n), dtype=np.int64)
    s = rng.integers(0, q, size=(n,), dtype=np.int64)
    e = np.round(rng.normal(0, sigma, m)).astype(np.int64)
    b = (A @ s + e) % q
    if max_rows is not None and max_rows < m:
        idx = rng.choice(m, size=max_rows, replace=False)
        idx.sort()
        A = A[idx]
        b = b[idx]
        m = max_rows
    else:
        idx = np.arange(m, dtype=np.int64)
    return LWEInstance(n=n, m=m, q=q, sigma=sigma, A=A, b=b, s_true=s, row_idx=idx)


def residual_norm(A: np.ndarray, b: np.ndarray, s: np.ndarray, q: int) -> float:
    r = center_mod(A @ s - b, q)
    return float(np.linalg.norm(r))


# Mod-q Belief Propagation (mean-field on dense linear rows)

class ModQBeliefPropagation:
    """
    Loopy BP with mean-field factor updates for soft mod-q LWE rows.

    Factor a: psi_a(s) ~ exp(-r_a(s)^2 / (2 sigma^2)),  r_a centered mod q.
    """

    def __init__(self, A: np.ndarray, b: np.ndarray, q: int, sigma: float,
                 tol: float = 1e-6, damping: float = 0.5):
        self.A = A.astype(np.int64)
        self.b = b.astype(np.int64)
        self.q = int(q)
        self.sigma = float(max(sigma, 1e-6))
        self.m, self.n = A.shape
        self.tol = tol
        self.damping = damping
        self.inv_2s2 = 1.0 / (2.0 * self.sigma * self.sigma)

    def _factor_to_var(self, a: int, i: int, marginals: np.ndarray) -> np.ndarray:
        """Message factor a -> variable i, shape (q,)."""
        q = self.q
        vals = np.arange(q, dtype=np.float64)
        exp_contrib = np.zeros(self.m, dtype=np.float64)
        for j in range(self.n):
            if j == i:
                continue
            exp_contrib += self.A[:, j] * (marginals[j] @ vals)
        ai = self.A[a, i]
        base = exp_contrib[a] - self.b[a]
        r = center_mod(ai * vals + base, q)
        out = np.exp(-r * r * self.inv_2s2)
        s = out.sum()
        return np.full(q, 1.0 / q) if s <= 0 else out / s

    def run(self, max_iter: int = 80, init: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int, float]:
        q, n, m = self.q, self.n, self.m
        vals = np.arange(q, dtype=np.float64)
        if init is None:
            marginals = np.full((n, q), 1.0 / q, dtype=np.float64)
        else:
            marginals = init.copy()
        means = np.array([marginals[j] @ vals for j in range(n)])
        delta = 1.0
        it = 0
        for it in range(1, max_iter + 1):
            exp_all = self.A.astype(np.float64) @ means
            new = np.zeros_like(marginals)
            for i in range(n):
                if not np.any(self.A[:, i]):
                    new[i] = marginals[i]
                    continue
                exp_ex = exp_all - self.A[:, i].astype(np.float64) * means[i]
                log_m = np.zeros(q, dtype=np.float64)
                cols = np.nonzero(self.A[:, i])[0]
                for a in cols:
                    ai = float(self.A[a, i])
                    base = float(exp_ex[a] - self.b[a])
                    r = center_mod(ai * vals + base, q)
                    msg = np.exp(-r * r * self.inv_2s2)
                    log_m += np.log(msg + 1e-300)
                log_m -= log_m.max()
                new[i] = np.exp(log_m)
                s = new[i].sum()
                new[i] /= s if s > 0 else 1.0
            new = self.damping * new + (1.0 - self.damping) * marginals
            delta = float(np.max(np.abs(new - marginals)))
            marginals = new
            means = np.array([marginals[j] @ vals for j in range(n)])
            if delta < self.tol:
                break
        return marginals, it, delta


# Survey Propagation for mod-q LWE (warning surveys on value compatibility)

class LWEsurveyPropagation:
    """
    Survey propagation layer for mod-q LWE, adapted from Braunstein-Zecchina
    warning propagation / SP (k-SAT) to soft linear constraints.

    eta[a, i, v] in [0,1]: survey fraction of clusters where value v is warned
    (incompatible with small residual) at factor a.

    Uses mean-field marginals from BP to evaluate residual compatibility.
    """

    def __init__(self, lwe: LWEInstance, threshold_sigma: float = 3.0,
                 bp_damping: float = 0.55):
        self.lwe = lwe
        self.q = lwe.q
        self.n = lwe.n
        self.m = lwe.m
        self.A = lwe.A.astype(np.int64)
        self.b = lwe.b.astype(np.int64)
        self.sigma = float(max(lwe.sigma, 1e-6))
        self.T = float(threshold_sigma * self.sigma)
        self.bp = ModQBeliefPropagation(self.A, self.b, self.q, self.sigma, damping=bp_damping)

    def _expected_contribution(self, marginals: np.ndarray, exclude_i: int) -> np.ndarray:
        """Expected sum_j A_{aj} s_j excluding variable i, per row a."""
        exp = np.zeros(self.m, dtype=np.float64)
        for j in range(self.n):
            if j == exclude_i:
                continue
            vals = np.arange(self.q, dtype=np.float64)
            exp += self.A[:, j] * (marginals[j] @ vals)
        return exp

    def _warning_prob(self, marginals: np.ndarray, a: int, i: int, v: int) -> float:
        """Probability that value v is warned at edge (a, i)."""
        if self.A[a, i] == 0:
            return 0.0
        base = self._expected_contribution(marginals, i)[a] - self.b[a]
        r = center_mod(np.array([self.A[a, i] * v + base], dtype=np.float64), self.q)[0]
        # Soft warning: logistic in excess beyond threshold T
        excess = max(0.0, abs(r) - self.T)
        return 1.0 / (1.0 + math.exp(-excess / max(self.sigma, 1e-6)))

    def _update_eta(self, eta: np.ndarray, marginals: np.ndarray) -> np.ndarray:
        """
        SP fixed-point update (independence approximation).

        eta[a,i,v] <- product_{j in row a, j!=i} rho_j->a(v)
        where rho is derived from incoming warnings on other rows touching j.
        """
        q, n, m = self.q, self.n, self.m
        new_eta = np.zeros_like(eta)

        # Precompute per-variable warning aggregates from other factors
        for a in range(m):
            cols = np.nonzero(self.A[a])[0]
            for i in cols:
                for v in range(q):
                    prod = 1.0
                    for j in cols:
                        if j == i:
                            continue
                        # rho_j->a: prob j warns against value v via factor a
                        # Use surveys from other rows attached to j
                        rows_j = np.nonzero(self.A[:, j])[0]
                        warn = 0.0
                        cnt = 0
                        for b in rows_j:
                            if b == a:
                                continue
                            warn += eta[b, j, v]
                            cnt += 1
                        rho = warn / cnt if cnt else self._warning_prob(marginals, a, j, v)
                        prod *= (1.0 - rho)
                    direct = self._warning_prob(marginals, a, i, v)
                    new_eta[a, i, v] = 1.0 - (1.0 - direct) * prod
        return new_eta

    def run(self, max_iter: int = 40, n_restarts: int = 6,
            bp_iter: int = 60, light_sp: bool = False) -> Dict[str, Any]:
        q, n, m = self.q, self.n, self.m
        marginals, bp_iters, bp_delta = self.bp.run(max_iter=bp_iter)

        sp_delta = 0.0
        sp_iters = 0
        if light_sp:
            # One-shot warning surveys from BP marginals (fast quick-mode path)
            eta = np.zeros((m, n, q), dtype=np.float64)
            for a in range(m):
                cols = np.nonzero(self.A[a])[0]
                for i in cols:
                    for v in range(q):
                        eta[a, i, v] = self._warning_prob(marginals, a, i, v)
        else:
            eta = np.full((m, n, q), 0.05, dtype=np.float64)
            for sp_iters in range(1, max_iter + 1):
                new_eta = self._update_eta(eta, marginals)
                sp_delta = float(np.max(np.abs(new_eta - eta)))
                eta = 0.6 * new_eta + 0.4 * eta
                if sp_delta < 1e-5:
                    break

        # Multi-restart BP for 1RSB overlap proxy
        restarts: List[np.ndarray] = []
        rng = np.random.default_rng(42)
        for _r in range(n_restarts):
            init = rng.dirichlet(np.ones(q), size=n)
            m_r, _, _ = self.bp.run(max_iter=bp_iter, init=init)
            restarts.append(m_r)

        stack = np.stack(restarts, axis=0)  # (R, n, q)
        mean_m = stack.mean(axis=0)
        # Overlap proxy: average pairwise cosine similarity of marginals
        overlaps = []
        for i in range(n_restarts):
            for j in range(i + 1, n_restarts):
                vi = stack[i].reshape(-1)
                vj = stack[j].reshape(-1)
                denom = np.linalg.norm(vi) * np.linalg.norm(vj)
                overlaps.append(float(vi @ vj / denom) if denom > 0 else 0.0)
        overlap_proxy = float(np.mean(overlaps)) if overlaps else 0.0

        # Survey entropy (mean per-variable entropy of mean marginal)
        ent = 0.0
        for i in range(n):
            p = mean_m[i]
            p = p / (p.sum() + 1e-300)
            ent += -float(np.sum(p * np.log(p + 1e-300)))
        survey_entropy = ent / n
        survey_entropy_norm = survey_entropy / math.log(q)

        warning_density = float(eta.mean())
        warning_max = float(eta.max())

        # Complexity proxy: log effective number of clusters ~ n * H_survey / (1 - overlap)
        complexity_proxy = float(
            n * survey_entropy_norm / max(1e-6, 1.0 - overlap_proxy + 1e-6)
        )

        # Decimation: fix top-k confident variables, re-run BP
        confidence = mean_m.max(axis=1)
        dec_fixed = int(max(1, n // 8))
        order = np.argsort(-confidence)
        fixed_idx = order[:dec_fixed]
        fixed_vals = np.argmax(mean_m[fixed_idx], axis=1)

        dec_acc = (
            self._decimation_accuracy(fixed_idx, fixed_vals, bp_iter)
            if not light_sp and n <= 40
            else float("nan")
        )
        map_acc = float(np.mean(np.argmax(mean_m, axis=1) == self.lwe.s_true))

        # Marginal bias toward true secret (KL proxy)
        kl_true = 0.0
        for i in range(n):
            p = mean_m[i]
            p = p / (p.sum() + 1e-300)
            kl_true += -math.log(p[self.lwe.s_true[i]] + 1e-300)
        kl_true /= n

        xi_true = residual_norm(self.A, self.b, self.lwe.s_true, q)
        xi_map = residual_norm(self.A, self.b, np.argmax(mean_m, axis=1), q)

        return {
            "bp_iters": bp_iters,
            "bp_delta": float(bp_delta),
            "sp_iters": sp_iters,
            "sp_delta": float(sp_delta),
            "survey_entropy": survey_entropy,
            "survey_entropy_norm": survey_entropy_norm,
            "overlap_proxy": overlap_proxy,
            "warning_density": warning_density,
            "warning_max": warning_max,
            "complexity_proxy": complexity_proxy,
            "map_accuracy": map_acc,
            "decimation_accuracy": dec_acc,
            "kl_to_true": kl_true,
            "xi_true": xi_true,
            "xi_map": xi_map,
            "xi_ratio": xi_map / (xi_true + 1e-9),
            "n_restarts": n_restarts,
        }

    def _decimation_accuracy(self, fixed_idx: np.ndarray, fixed_vals: np.ndarray,
                             bp_iter: int) -> float:
        q, n = self.q, self.n
        init = np.full((n, q), 1.0 / q, dtype=np.float64)
        for idx, val in zip(fixed_idx, fixed_vals):
            init[idx, :] = 0.0
            init[idx, val] = 1.0
        marginals, _, _ = self.bp.run(max_iter=bp_iter, init=init)
        guess = np.argmax(marginals, axis=1)
        return float(np.mean(guess == self.lwe.s_true))


# Single parameter point

def bkz_proxy_bits(n: int, q: int, sigma: float) -> Dict[str, Any]:
    """Rough BKZ proxy when lattice-estimator is unavailable (dev only)."""
    m = 3 * n
    log_q = math.log2(q)
    log_m = math.log2(m)
    beta = max(40.0, 0.35 * n - 0.5 * log_q + 2.0 * math.log2(sigma + 1.0))
    rop = beta * log_m + 0.25 * n
    return {
        "fam_best": {"bdd": rop, "usvp": rop + 5, "dual": rop + 8, "dual_hybrid": rop + 12},
        "fam_beta": {"bdd": beta, "usvp": beta, "dual": beta, "dual_hybrid": beta},
        "best_family": "bdd",
        "best_bits": rop,
        "best_beta": beta,
        "dominance_gap": 5.0,
        "proxy": True,
    }


def run_point(n: int, q: int, sigma: float, seed: int = 0,
              max_rows: Optional[int] = None,
              est_jobs: int = DEFAULT_JOBS,
              use_proxy: bool = False,
              quick: bool = False) -> Dict[str, Any]:
    m_full = 3 * n
    if max_rows is None:
        max_rows = min(m_full, n + 8) if quick else min(m_full, max(2 * n, 48))
    lwe = generate_lwe(n, m_full, q, sigma, seed=seed, max_rows=max_rows)
    alpha = q / (sigma * math.sqrt(n))

    sp = LWEsurveyPropagation(lwe)
    if quick:
        # Medium SP: full warning iteration, fewer restarts than full sweep
        sp_metrics = sp.run(max_iter=25, n_restarts=4, bp_iter=45, light_sp=False)
    else:
        sp_metrics = sp.run(max_iter=40, n_restarts=8, bp_iter=60, light_sp=False)

    if use_proxy:
        sec = bkz_proxy_bits(n, q, sigma)
    else:
        sec = run_estimator(n, q, sigma, jobs=est_jobs)

    return {
        "n": n,
        "q": q,
        "sigma": sigma,
        "alpha": alpha,
        "seed": seed,
        "m_sp": lwe.m,
        "sec": sec,
        "sp": sp_metrics,
    }


# Sweep + analysis

SWEEP_NS = [24, 32, 48, 64, 96]
SWEEP_SIGMAS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
QUICK_NS = [24, 32, 48, 64]
QUICK_SIGMAS = [0.5, 1.0, 1.5, 2.0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 15: Survey Propagation vs lattice-estimator")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--seeds", type=int, default=5,
                        help="Independent LWE instances per (n,sigma) point (default 5)")
    parser.add_argument("--quick", action="store_true",
                        help="Medium sweep: 4 n x 4 sigma, 3 seeds (~48 SP runs)")
    parser.add_argument("--sp-only", action="store_true",
                        help="Skip lattice-estimator; use BKZ proxy (dev/test only)")
    args = parser.parse_args()

    use_proxy = bool(args.sp_only)
    if not ESTIMATOR_AVAILABLE and not args.sp_only:
        print("WARNING: lattice-estimator unavailable on this Python - using BKZ proxy.")
        print(f"  WSL detected: {_running_in_wsl()}")
        _print_estimator_help()
        use_proxy = True
    elif ESTIMATOR_AVAILABLE:
        where = "WSL" if _running_in_wsl() else "local"
        print(f"Using lattice-estimator ({where}).\n")

    ns = QUICK_NS if args.quick else SWEEP_NS
    sigmas = QUICK_SIGMAS if args.quick else SWEEP_SIGMAS
    n_seeds = 3 if args.quick else args.seeds

    print("=" * 80)
    print("Experiment 15: Survey Propagation on LWE vs Lattice-Estimator")
    n_configs = len(ns) * len(sigmas)
    print(f"Sweep: {len(ns)} n x {len(sigmas)} sigma = {n_configs} configs")
    print(f"Fixed alpha = {SWEEP_ALPHA}, seeds per point = {n_seeds}")
    print(f"Total SP runs: {n_configs * n_seeds}")
    print("=" * 80)
    print(
        f"\n{'n':>4}  {'sigma':>6}  {'q':>6}  {'family':>12}  "
        f"{'bits':>8}  {'warn_d':>8}  {'H_surv':>8}  {'overlap':>8}  {'Sigma*':>8}"
    )
    print("-" * 82)

    all_results: List[Dict[str, Any]] = []
    t0 = time.time()

    for n in ns:
        for sigma in sigmas:
            q = int(SWEEP_ALPHA * sigma * math.sqrt(n))
            q = max(q, n + 2)
            if q % 2 == 0:
                q += 1

            seed_rows = []
            for seed in range(n_seeds):
                try:
                    seed_rows.append(
                        run_point(n, q, sigma, seed=seed * 91 + n,
                                  est_jobs=args.jobs, use_proxy=use_proxy,
                                  quick=args.quick)
                    )
                except Exception as exc:
                    print(f"  n={n} sigma={sigma} seed={seed} ERROR: {exc}")

            if not seed_rows:
                continue

            def mean_sp(key: str) -> Optional[float]:
                vals = [r["sp"][key] for r in seed_rows if r["sp"].get(key) is not None]
                vals = [float(v) for v in vals if np.isfinite(v)]
                return float(np.mean(vals)) if vals else None

            agg = dict(seed_rows[0])
            agg["seed_rows"] = seed_rows
            agg["sp"] = {
                "warning_density": mean_sp("warning_density"),
                "survey_entropy_norm": mean_sp("survey_entropy_norm"),
                "overlap_proxy": mean_sp("overlap_proxy"),
                "complexity_proxy": mean_sp("complexity_proxy"),
                "map_accuracy": mean_sp("map_accuracy"),
                "decimation_accuracy": mean_sp("decimation_accuracy"),
                "sp_iters": mean_sp("sp_iters"),
                "xi_ratio": mean_sp("xi_ratio"),
            }
            fams = [r["sec"]["best_family"] for r in seed_rows if r["sec"]["best_family"]]
            agg["sec"]["best_family"] = max(set(fams), key=fams.count) if fams else "other"
            agg["sec"]["best_bits"] = float(np.mean(
                [r["sec"]["best_bits"] for r in seed_rows if r["sec"]["best_bits"] is not None]
            ))
            agg["sec"]["dominance_gap"] = float(np.mean(
                [r["sec"]["dominance_gap"] for r in seed_rows
                 if r["sec"]["dominance_gap"] is not None]
            )) if any(r["sec"]["dominance_gap"] is not None for r in seed_rows) else None
            all_results.append(agg)

            fam = agg["sec"]["best_family"] or "other"
            print(
                f"{n:>4}  {sigma:>6.1f}  {q:>6}  {fam:>12}  "
                f"{fmt_bits(agg['sec']['best_bits']):>8}  "
                f"{fmt_f(agg['sp']['warning_density'], 4):>8}  "
                f"{fmt_f(agg['sp']['survey_entropy_norm'], 4):>8}  "
                f"{fmt_f(agg['sp']['overlap_proxy'], 4):>8}  "
                f"{fmt_f(agg['sp']['complexity_proxy'], 3):>8}"
            )

    elapsed = (time.time() - t0) / 60.0
    print(f"\n{len(all_results)} configs in {elapsed:.1f} min")

    hardness = np.array([r["sec"]["best_bits"] or np.nan for r in all_results])
    warn_d = np.array([r["sp"]["warning_density"] or np.nan for r in all_results])
    h_surv = np.array([r["sp"]["survey_entropy_norm"] or np.nan for r in all_results])
    overlap = np.array([r["sp"]["overlap_proxy"] or np.nan for r in all_results])
    sigma_c = np.array([r["sp"]["complexity_proxy"] or np.nan for r in all_results])
    dom_gaps = np.array([r["sec"]["dominance_gap"] or np.nan for r in all_results])
    families = [r["sec"]["best_family"] or "other" for r in all_results]

    def corr_report(x, y, label: str) -> None:
        m = np.isfinite(x) & np.isfinite(y)
        n = int(m.sum())
        if n < 5:
            print(f"  {label}: n={n} (too few for reliable correlation)")
            return
        pr = corrcoef_nan(x, y)
        sr, sp = scipy_stats.spearmanr(x[m], y[m])
        print(f"  {label}: n={n}  Pearson={fmt_f(pr, 4)}  Spearman={fmt_f(sr, 4)}  p={fmt_f(sp, 4)}")

    print(f"\nCORRELATIONS WITH LATTICE-ESTIMATOR HARDNESS (log2 ROP), {len(all_results)} points:")
    corr_report(warn_d, hardness, "warning_density vs hardness")
    corr_report(h_surv, hardness, "survey_entropy vs hardness ")
    corr_report(overlap, hardness, "overlap_proxy vs hardness  ")
    corr_report(sigma_c, hardness, "complexity_proxy vs hardness")
    corr_report(warn_d, dom_gaps, "warning_density vs dom_gap")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    def styled(ax, title, xl="", yl="", tc="white"):
        ax.set_facecolor(PBG)
        ax.set_title(title, color=tc, fontsize=9, pad=6)
        ax.set_xlabel(xl, color="white", fontsize=8)
        ax.set_ylabel(yl, color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    # 1. Attack family phase map
    ax0 = fig.add_subplot(gs[0, :2])
    for r in all_results:
        fam = r["sec"]["best_family"] or "other"
        col = FAMILY_COLORS.get(fam, "#888")
        dgap = r["sec"]["dominance_gap"]
        size = 260 if (dgap is not None and dgap < 10) else 120
        ax0.scatter(r["n"], r["sigma"], color=col, s=size,
                    edgecolors="white", lw=(2 if size > 200 else 0.5))
        ax0.text(r["n"], r["sigma"], fam[:3].upper(),
                 ha="center", va="center", color="black", fontsize=7, fontweight="bold")
    styled(ax0, f"Attack-Family Phase Map (alpha={SWEEP_ALPHA})", "n", "sigma")
    ax0.legend(handles=[Patch(color=FAMILY_COLORS[f], label=f)
                        for f in ["bdd", "usvp", "dual", "dual_hybrid"]],
               fontsize=7, labelcolor="white", facecolor=PBG)

    # 2. Warning density map
    ax1 = fig.add_subplot(gs[0, 2])
    for r in all_results:
        wd = r["sp"]["warning_density"]
        if wd is None:
            continue
        col = plt.cm.inferno(min(wd / 0.5, 1.0))
        ax1.scatter(r["n"], r["sigma"], color=col, s=180, edgecolors="white", lw=0.5)
        ax1.text(r["n"], r["sigma"], f"{wd:.3f}", ha="center", va="center",
                 color="white", fontsize=6)
    styled(ax1, "SP Warning Density", "n", "sigma", "#ffd93d")

    # 3. KEY: complexity proxy vs hardness
    ax2 = fig.add_subplot(gs[1, 0])
    for r, fam in zip(all_results, families):
        bits = r["sec"]["best_bits"]
        cp = r["sp"]["complexity_proxy"]
        if bits is None or cp is None:
            continue
        ax2.scatter(bits, cp, color=FAMILY_COLORS.get(fam, "#888"),
                    s=90, edgecolors="white", lw=0.5)
    styled(ax2, "KEY: SP Complexity Proxy vs Estimator Hardness",
           "log2(ROP) [lattice-estimator]", "complexity proxy")

    # 4. Survey entropy vs hardness
    ax3 = fig.add_subplot(gs[1, 1])
    for r, fam in zip(all_results, families):
        bits = r["sec"]["best_bits"]
        hs = r["sp"]["survey_entropy_norm"]
        if bits is None or hs is None:
            continue
        ax3.scatter(bits, hs, color=FAMILY_COLORS.get(fam, "#888"),
                    s=90, edgecolors="white", lw=0.5)
    styled(ax3, "Survey Entropy vs Hardness", "log2(ROP)", "H_survey / log(q)")

    # 5. Overlap vs warning density
    ax4 = fig.add_subplot(gs[1, 2])
    for r, fam in zip(all_results, families):
        ov = r["sp"]["overlap_proxy"]
        wd = r["sp"]["warning_density"]
        if ov is None or wd is None:
            continue
        ax4.scatter(wd, ov, color=FAMILY_COLORS.get(fam, "#888"),
                    s=90, edgecolors="white", lw=0.5)
    styled(ax4, "1RSB Overlap Proxy vs Warning Density",
           "warning density", "overlap proxy", "#69ff94")

    fig.suptitle(
        "Experiment 15 — Survey Propagation on LWE vs Lattice-Estimator\n"
        "Cavity-method clustering signals vs concrete attack cost",
        color="white", fontsize=13, fontweight="bold", y=1.01,
    )
    plt.savefig("exp15_results.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("\nPlot saved -> exp15_results.png")

    # ── JSON ──────────────────────────────────────────────────────────────────
    with open("exp15_data.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ── Summary ───────────────────────────────────────────────────────────────
    with open("exp15_summary.txt", "w") as f:
        f.write("=" * 72 + "\n")
        f.write("EXPERIMENT 15 — SURVEY PROPAGATION ON LWE VS LATTICE-ESTIMATOR\n")
        f.write("=" * 72 + "\n\n")
        f.write("Method: mod-q BP + survey propagation warning surveys on LWE rows.\n")
        gt = "BKZ proxy (dev)" if use_proxy else "lattice-estimator min log2(ROP)"
        f.write(f"Ground truth: {gt} and attack family.\n")
        f.write("\nWSL + Sage (recommended):\n")
        f.write(f"  cd \"{WSL_PROJECT_DIR}\"\n")
        f.write("  sage -python exp15.py --quick\n")
        f.write(f"  # or from PowerShell: {WSL_RUN_QUICK}\n\n")
        f.write(f"Fixed alpha = {SWEEP_ALPHA}\n")
        f.write(f"Configs: {len(all_results)}, elapsed {elapsed:.1f} min\n\n")
        f.write(f"CORRELATIONS WITH HARDNESS (n={len(all_results)} points)\n" + "-" * 40 + "\n")
        for xv, yv, name in [
            (warn_d, hardness, "warning_density"),
            (h_surv, hardness, "survey_entropy"),
            (overlap, hardness, "overlap_proxy"),
            (sigma_c, hardness, "complexity_proxy"),
        ]:
            m = np.isfinite(xv) & np.isfinite(yv)
            if m.sum() >= 5:
                sr, sp = scipy_stats.spearmanr(xv[m], yv[m])
                f.write(f"{name}: Pearson={fmt_f(corrcoef_nan(xv, yv), 4)}  "
                        f"Spearman={fmt_f(sr, 4)}  p={fmt_f(sp, 4)}\n")
            else:
                f.write(f"{name}: n={m.sum()} (insufficient)\n")
        f.write("\n")
        f.write("PER-CONFIG\n" + "-" * 40 + "\n")
        f.write(
            f"{'n':>4}  {'sigma':>6}  {'family':>12}  {'bits':>8}  "
            f"{'warn_d':>8}  {'H_surv':>8}  {'overlap':>8}\n"
        )
        for r in all_results:
            f.write(
                f"{r['n']:>4}  {r['sigma']:>6.1f}  "
                f"{(r['sec']['best_family'] or '?'):>12}  "
                f"{fmt_bits(r['sec']['best_bits']):>8}  "
                f"{fmt_f(r['sp']['warning_density'], 4):>8}  "
                f"{fmt_f(r['sp']['survey_entropy_norm'], 4):>8}  "
                f"{fmt_f(r['sp']['overlap_proxy'], 4):>8}\n"
            )
        f.write("\nINTERPRETATION\n" + "-" * 40 + "\n")
        f.write(
            "High survey entropy + low overlap -> 1RSB/clustered solution space.\n"
            "if complexity_proxy tracks estimator hardness, SP sees the same\n"
            "phase structure as BKZ attacks without running sieving.\n"
        )
        f.write("=" * 72 + "\n")
    print("Summary saved -> exp15_summary.txt")
    print("Data saved -> exp15_data.json")


if __name__ == "__main__":
    main()
