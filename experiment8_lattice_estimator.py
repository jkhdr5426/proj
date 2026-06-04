#!/usr/bin/env python3
"""
Experiment 8: Umbrella texture vs real lattice-estimator
=========================================================

Run:
    sage -python experiment8_lattice_estimator.py

Outputs:
    - lwe_exp8_results.png
    - lwe_exp8_summary.txt
    - lwe_exp8_data.json
"""


from __future__ import annotations


import sys
sys.path.insert(0, "/home/ekim/lattice-estimator")
import argparse
import json
import math
import os
import re
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")


# =============================================================================
# Optional torch setup
# =============================================================================
try:
    import torch

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
except Exception as exc:
    raise RuntimeError(
        "PyTorch is required for this script. Install torch and rerun."
    ) from exc


# =============================================================================
# Optional lattice-estimator setup
# =============================================================================
ESTIMATOR_AVAILABLE = False
LWE = None
ND = None
RC = None
schemes = None

try:
    from estimator import LWE, ND, RC, schemes  # type: ignore

    ESTIMATOR_AVAILABLE = True
except Exception:
    ESTIMATOR_AVAILABLE = False


# =============================================================================
# Styling
# =============================================================================
BG = "#0f1117"
PBG = "#1a1d27"

PHASE_COLORS = {
    "funnel": "#69ff94",
    "rugged": "#ffd93d",
    "cliff": "#ff6b6b",
    "unknown": "#888888",
}

FAMILY_COLORS = {
    "bdd": "#69ff94",
    "usvp": "#00d4ff",
    "dual": "#ffd93d",
    "dual_hybrid": "#ff6b6b",
    "other": "#b0b0b0",
}

DEFAULT_JOBS = max(1, min(16, os.cpu_count() or 1))
USE_FULL_ESTIMATOR = True
RED_COST_MODEL = getattr(RC, "BDGL16", None)


# =============================================================================
# Small utilities
# =============================================================================
def fmt_bits(x: Optional[float]) -> str:
    return "n/a" if x is None else f"2^{x:.1f}"


def fmt_beta(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.1f}"


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        s = str(x)
        m = re.search(r"([-+]?\d+(?:\.\d+)?)", s)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


def parse_rop_bits(text: str) -> Optional[float]:
    patterns = [
        r"rop:\s*[^\d-]*2\^([0-9]+(?:\.[0-9]+)?)",
        r"rop\s*=\s*[^\d-]*2\^([0-9]+(?:\.[0-9]+)?)",
        r"2\^([0-9]+(?:\.[0-9]+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return float(m.group(1))
    return None


def extract_cost_field(cost_obj: Any, field: str) -> Optional[float]:
    try:
        if hasattr(cost_obj, "get"):
            v = cost_obj.get(field)
            out = safe_float(v)
            if out is not None:
                return out
    except Exception:
        pass

    try:
        if hasattr(cost_obj, field):
            out = safe_float(getattr(cost_obj, field))
            if out is not None:
                return out
    except Exception:
        pass

    try:
        txt = str(cost_obj)
        if field == "rop":
            return parse_rop_bits(txt)
        pat = rf"{re.escape(field)}\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)"
        m = re.search(pat, txt)
        if m:
            return float(m.group(1))
    except Exception:
        pass

    return None


def cost_to_record(cost_obj: Any) -> Dict[str, Optional[float]]:
    return {
        "rop": extract_cost_field(cost_obj, "rop"),
        "beta": extract_cost_field(cost_obj, "beta"),
        "d": extract_cost_field(cost_obj, "d"),
        "m": extract_cost_field(cost_obj, "m"),
        "delta": extract_cost_field(cost_obj, "delta"),
    }


def normalize_attack_name(name: str) -> str:
    n = name.lower().strip()
    n = n.replace(" ", "_").replace("-", "_")
    n = re.sub(r"__+", "_", n)
    return n


def family_from_attack_name(name: str) -> str:
    n = normalize_attack_name(name)
    if "bdd" in n:
        return "bdd"
    if "usvp" in n or "primal" in n:
        return "usvp"
    if "dual_hybrid" in n or "dualhybrid" in n:
        return "dual_hybrid"
    if n == "dual" or n.startswith("dual_") or "_dual_" in n:
        return "dual"
    return "other"


def avg_ignore_none(values: Sequence[Optional[float]]) -> Optional[float]:
    xs = [v for v in values if v is not None]
    if not xs:
        return None
    return float(np.mean(xs))


def std_ignore_none(values: Sequence[Optional[float]]) -> Optional[float]:
    xs = [v for v in values if v is not None]
    if len(xs) < 2:
        return 0.0 if xs else None
    return float(np.std(xs))


def clean_for_json(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): clean_for_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean_for_json(v) for v in x]
    if isinstance(x, tuple):
        return [clean_for_json(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


# =============================================================================
# Estimator wrappers
# =============================================================================
def make_secret_dist(q: int):
    if not ESTIMATOR_AVAILABLE:
        raise RuntimeError("lattice-estimator unavailable")

    if hasattr(ND, "UniformMod"):
        return ND.UniformMod(q)
    if hasattr(ND, "UniformModQ"):
        return ND.UniformModQ(q)
    if hasattr(ND, "Uniform"):
        try:
            return ND.Uniform(0, q)
        except Exception:
            pass

    raise RuntimeError("Could not construct a uniform-mod-q secret distribution")


def make_error_dist(sigma: float):
    if not ESTIMATOR_AVAILABLE:
        raise RuntimeError("lattice-estimator unavailable")

    if hasattr(ND, "DiscreteGaussian"):
        try:
            return ND.DiscreteGaussian(stddev=sigma)
        except TypeError:
            try:
                return ND.DiscreteGaussian(sigma)
            except Exception:
                pass

    if hasattr(ND, "Gaussian"):
        try:
            return ND.Gaussian(stddev=sigma)
        except TypeError:
            try:
                return ND.Gaussian(sigma)
            except Exception:
                pass

    if hasattr(ND, "CenteredBinomial"):
        eta = max(1, int(round(2.0 * sigma * sigma)))
        return ND.CenteredBinomial(eta)

    raise RuntimeError("Could not construct an error distribution")


def call_estimator_func(func, params, jobs: Optional[int] = None, red_cost_model=None):
    kwargs: Dict[str, Any] = {}
    if jobs is not None:
        kwargs["jobs"] = jobs
    if red_cost_model is not None:
        kwargs["red_cost_model"] = red_cost_model

    try:
        return func(params, **kwargs)
    except TypeError:
        pass

    if "jobs" in kwargs:
        kwargs.pop("jobs", None)
        try:
            return func(params, **kwargs)
        except TypeError:
            pass

    if "red_cost_model" in kwargs:
        kwargs.pop("red_cost_model", None)
        try:
            return func(params, **kwargs)
        except TypeError:
            pass

    return func(params)


def build_lwe_params(n: int, q: int, sigma: float, m_factor: int = 3):
    if not ESTIMATOR_AVAILABLE:
        raise RuntimeError("lattice-estimator is unavailable")
    m = m_factor * n
    Xs = make_secret_dist(q)
    Xe = make_error_dist(sigma)
    return LWE.Parameters(n=n, q=q, Xs=Xs, Xe=Xe, m=m, tag=f"synthetic-n{n}-q{q}")


def estimate_from_params(params, full: bool = True, jobs: Optional[int] = None) -> Dict[str, Any]:
    if not ESTIMATOR_AVAILABLE:
        raise RuntimeError(
            "The lattice-estimator package is unavailable. Run inside Sage with the estimator module installed."
        )

    jobs = DEFAULT_JOBS if jobs is None else jobs

    if full:
        try:
            result = call_estimator_func(LWE.estimate, params, jobs=jobs, red_cost_model=RED_COST_MODEL)
        except Exception:
            result = call_estimator_func(LWE.estimate, params, jobs=jobs)
    else:
        rough = getattr(LWE.estimate, "rough", None)
        if rough is None:
            raise RuntimeError("LWE.estimate.rough is unavailable in this estimator version")
        try:
            result = call_estimator_func(rough, params, jobs=jobs, red_cost_model=RED_COST_MODEL)
        except Exception:
            result = call_estimator_func(rough, params, jobs=jobs)

    attack_table: Dict[str, Dict[str, Optional[float]]] = {}

    if hasattr(result, "items"):
        try:
            items = list(result.items())
        except Exception:
            items = []
        for attack_name, cost_obj in items:
            key = normalize_attack_name(str(attack_name))
            attack_table[key] = cost_to_record(cost_obj)

    if not attack_table:
        attack_table["single"] = cost_to_record(result)

    family_best: Dict[str, Dict[str, Optional[float]]] = {}
    for fam in ("bdd", "usvp", "dual", "dual_hybrid"):
        best_key = None
        best_bits = None
        best_rec = None

        for key, rec in attack_table.items():
            if fam == "bdd":
                ok = "bdd" in key
            elif fam == "usvp":
                ok = ("usvp" in key) or ("primal" in key)
            elif fam == "dual":
                ok = ("dual" in key) and ("dual_hybrid" not in key) and ("dualhybrid" not in key)
            elif fam == "dual_hybrid":
                ok = ("dual_hybrid" in key) or ("dualhybrid" in key)
            else:
                ok = False

            if not ok:
                continue

            bits = rec.get("rop")
            if bits is None:
                continue

            if best_bits is None or bits < best_bits:
                best_bits = bits
                best_key = key
                best_rec = rec

        if best_rec is not None:
            family_best[fam] = {
                "attack": best_key,
                "rop": best_rec.get("rop"),
                "beta": best_rec.get("beta"),
                "d": best_rec.get("d"),
                "m": best_rec.get("m"),
                "delta": best_rec.get("delta"),
            }

    overall_attack = None
    overall_bits = None
    overall_beta = None
    overall_family = None

    for attack_name, rec in attack_table.items():
        bits = rec.get("rop")
        if bits is None:
            continue
        if overall_bits is None or bits < overall_bits:
            overall_bits = bits
            overall_attack = attack_name
            overall_beta = rec.get("beta")
            overall_family = family_from_attack_name(attack_name)

    family_bits = {fam: rec["rop"] for fam, rec in family_best.items()}
    family_beta = {fam: rec.get("beta") for fam, rec in family_best.items()}

    best_family = None
    best_family_bits = None
    best_family_beta = None
    best_family_attack = None

    for fam, bits in family_bits.items():
        if bits is None:
            continue
        if best_family_bits is None or bits < best_family_bits:
            best_family_bits = bits
            best_family = fam
            best_family_beta = family_beta.get(fam)
            best_family_attack = family_best[fam].get("attack")

    sorted_families = sorted(
        [(fam, bits) for fam, bits in family_bits.items() if bits is not None],
        key=lambda x: x[1],
    )
    dominance_gap = None
    if len(sorted_families) >= 2:
        dominance_gap = float(sorted_families[1][1] - sorted_families[0][1])

    return {
        "params": params,
        "raw_result": result,
        "attack_table": attack_table,
        "family_best": family_best,
        "overall_attack": overall_attack,
        "overall_bits": overall_bits,
        "overall_beta": overall_beta,
        "overall_family": overall_family,
        "best_family": best_family,
        "best_family_bits": best_family_bits,
        "best_family_beta": best_family_beta,
        "best_family_attack": best_family_attack,
        "dominance_gap": dominance_gap,
        "family_bits": family_bits,
        "family_beta": family_beta,
    }


def estimate_synthetic_lwe(
    n: int,
    q: int,
    sigma: float,
    m_factor: int = 3,
    full: bool = True,
    jobs: Optional[int] = None,
) -> Dict[str, Any]:
    params = build_lwe_params(n=n, q=q, sigma=sigma, m_factor=m_factor)
    return estimate_from_params(params, full=full, jobs=jobs)


def estimate_kyber1024_baseline(full: bool = True, jobs: Optional[int] = None) -> Dict[str, Any]:
    if not ESTIMATOR_AVAILABLE or schemes is None:
        return {}
    try:
        params = schemes.Kyber1024
        return estimate_from_params(params, full=full, jobs=jobs)
    except Exception:
        return {}


# =============================================================================
# LWE instance generation and umbrella sampler
# =============================================================================
@dataclass
class LWEInstance:
    n: int
    m: int
    q: int
    sigma: float
    A: np.ndarray
    b: np.ndarray
    s_true: np.ndarray
    A_gpu: Any = None
    b_gpu: Any = None

    def to_gpu(self):
        self.A_gpu = torch.tensor(self.A, dtype=torch.float32, device=DEVICE)
        self.b_gpu = torch.tensor(self.b, dtype=torch.float32, device=DEVICE)
        return self


def generate_lwe(n: int, m: int, q: int, sigma: float, seed: Optional[int] = None) -> LWEInstance:
    rng = np.random.default_rng(seed)
    A = rng.integers(0, q, size=(m, n), dtype=np.int64)
    s = rng.integers(0, q, size=(n,), dtype=np.int64)
    e = np.round(rng.normal(0.0, sigma, size=(m,))).astype(np.int64)
    b = (A @ s + e) % q
    return LWEInstance(n=n, m=m, q=q, sigma=sigma, A=A, b=b, s_true=s)


def residual_norm_cpu(lwe: LWEInstance, s: np.ndarray) -> float:
    r = (lwe.A @ s - lwe.b) % lwe.q
    r = np.where(r > lwe.q // 2, r - lwe.q, r)
    return float(np.linalg.norm(r))


def residual_norm_gpu(lwe: LWEInstance, s_batch: torch.Tensor) -> torch.Tensor:
    q = lwe.q
    r = (lwe.A_gpu @ s_batch.T - lwe.b_gpu.unsqueeze(1)) % q
    r = torch.where(r > q // 2, r - q, r)
    return torch.norm(r.T, dim=1)


def mh_all_windows_gpu(
    lwe: LWEInstance,
    xi_centers: Sequence[float],
    k: float,
    T: float,
    n_steps: int,
    n_chains: int = 8,
    s_init_batch: Optional[torch.Tensor] = None,
    step_size: Optional[int] = None,
    block_size: Optional[int] = None,
):
    n_win = len(xi_centers)
    n, q = lwe.n, lwe.q
    B = n_win * n_chains

    if step_size is None:
        step_size = max(1, q // 64)
    if block_size is None:
        block_size = max(1, n // 16)

    xc_gpu = torch.tensor(
        np.repeat(np.asarray(xi_centers, dtype=np.float32), n_chains),
        device=DEVICE,
    )

    if s_init_batch is None:
        s_batch = torch.randint(0, q, (B, n), dtype=torch.int64, device=DEVICE).to(torch.float32)
    else:
        noise = torch.randint(
            -step_size, step_size + 1, (B, n), dtype=torch.int64, device=DEVICE
        ).to(torch.float32)
        s_batch = (s_init_batch + noise) % q

    xi_cur = residual_norm_gpu(lwe, s_batch)
    bias_cur = 0.5 * k * (xi_cur - xc_gpu) ** 2

    xi_records = torch.zeros((B, n_steps), dtype=torch.float32, device=DEVICE)
    acc_count = torch.zeros(B, dtype=torch.float32, device=DEVICE)

    for step in range(n_steps):
        col_idx = torch.stack([torch.randperm(n, device=DEVICE)[:block_size] for _ in range(B)])
        deltas = torch.randint(
            -step_size, step_size + 1, (B, block_size), dtype=torch.int64, device=DEVICE
        ).to(torch.float32)

        s_prop = s_batch.clone()
        s_prop.scatter_add_(1, col_idx, deltas)
        s_prop = s_prop % q

        xi_prop = residual_norm_gpu(lwe, s_prop)
        bias_prop = 0.5 * k * (xi_prop - xc_gpu) ** 2
        db = bias_prop - bias_cur

        u = torch.rand(B, device=DEVICE).clamp_min(1e-30)
        accept = (db <= 0) | (torch.log(u) < (-db / T))

        s_batch = torch.where(accept.unsqueeze(1), s_prop, s_batch)
        xi_cur = torch.where(accept, xi_prop, xi_cur)
        bias_cur = torch.where(accept, bias_prop, bias_cur)
        xi_records[:, step] = xi_cur
        acc_count += accept.float()

    xi_cpu = xi_records.cpu().numpy()
    acc_rates = (acc_count / n_steps).cpu().numpy()
    xi_trajs = xi_cpu.reshape(n_win, n_chains, n_steps).reshape(n_win, n_chains * n_steps)
    acc_by_win = acc_rates.reshape(n_win, n_chains).mean(axis=1)

    return xi_trajs, s_batch, acc_by_win


def wham_stable(
    window_xis: Sequence[np.ndarray],
    xi_centers: Sequence[float],
    k: float,
    T: float,
    xi_bins: np.ndarray,
    n_iter: int = 500,
    tol: float = 1e-8,
):
    n_win = len(window_xis)
    n_bins = len(xi_bins) - 1
    bc = 0.5 * (xi_bins[:-1] + xi_bins[1:])

    H = np.zeros((n_win, n_bins), dtype=np.float64)
    N = np.zeros(n_win, dtype=np.float64)
    for i, xis in enumerate(window_xis):
        H[i], _ = np.histogram(xis, bins=xi_bins)
        N[i] = len(xis)

    bias = np.exp(
        -np.array(
            [[0.5 * k * (b - xi_centers[i]) ** 2 / T for b in bc] for i in range(n_win)],
            dtype=np.float64,
        )
    )

    f = np.zeros(n_win, dtype=np.float64)
    rho = np.ones(n_bins, dtype=np.float64)

    for _ in range(n_iter):
        rho_old = rho.copy()
        denom = (N[:, None] * bias * np.exp(f[:, None])).sum(axis=0)
        rho = np.where(denom > 0, H.sum(axis=0) / denom, 0.0)
        Z_i = (rho[None, :] * bias).sum(axis=1)
        f = -np.log(np.where(Z_i > 0, Z_i, 1e-300))
        if np.max(np.abs(rho - rho_old)) < tol:
            break

    F = -np.log(np.where(rho > 0, rho, 1e-300))
    F -= np.nanmin(F)
    finite = np.isfinite(F)
    if np.any(finite):
        clip_hi = np.nanpercentile(F[finite], 95)
        F = np.clip(F, 0, clip_hi)
    return bc, F


def classify_phase(rugosity: float, mobility: float, rug_thresh: float = 0.16, mob_thresh: float = 0.015) -> str:
    if rugosity > rug_thresh:
        return "rugged"
    if mobility < mob_thresh:
        return "cliff"
    return "funnel"


def run_config_stable(
    n: int,
    q: int,
    sigma: float,
    seed: Optional[int] = None,
    n_windows: int = 14,
    n_steps_burn: int = 350,
    n_steps_prod: int = 1200,
    k_base: float = 0.006,
    T: float = 1.0,
    n_chains: int = 6,
    estimator_full: bool = True,
    estimator_jobs: Optional[int] = None,
):
    lwe = generate_lwe(n, 3 * n, q, sigma, seed=seed).to_gpu()
    alpha = q / (sigma * np.sqrt(n))

    xi_true = residual_norm_cpu(lwe, lwe.s_true)
    rng_est = np.random.default_rng(0 if seed is None else seed)
    xi_random = float(
        np.mean(
            [
                residual_norm_cpu(lwe, rng_est.integers(0, q, size=(n,), dtype=np.int64))
                for _ in range(30)
            ]
        )
    )

    xi_max = xi_random * 1.03
    xi_min = max(xi_true * 0.75, 1.0)
    xi_range = xi_max - xi_min + 1e-9

    k_scaled = k_base * (500.0 / xi_range) ** 2
    k_scaled = max(k_scaled, 1e-10)

    xi_centers = np.linspace(xi_max, xi_min, n_windows)

    _, s_batch, _ = mh_all_windows_gpu(
        lwe, xi_centers, k_scaled, T, n_steps_burn, n_chains=n_chains
    )
    xi_trajs, s_batch, acc_by_win = mh_all_windows_gpu(
        lwe,
        xi_centers,
        k_scaled,
        T,
        n_steps_prod,
        n_chains=n_chains,
        s_init_batch=s_batch,
    )

    window_means = xi_trajs.mean(axis=1)
    window_gap = float(np.mean(np.abs(window_means - xi_centers) / xi_range))
    mobility = float(acc_by_win.mean())

    xi_bins = np.linspace(xi_min * 0.85, xi_max * 1.08, 80)
    bc, F = wham_stable(list(xi_trajs), xi_centers, k_scaled, T, xi_bins)
    valid = np.isfinite(F)
    rugosity = float(np.std(np.diff(F[valid])) / (float(F[valid].max()) + 1e-9)) if valid.sum() > 2 else 0.0

    try:
        del lwe.A_gpu, lwe.b_gpu, s_batch, xi_trajs
    except Exception:
        pass
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    security = estimate_synthetic_lwe(
        n=n,
        q=q,
        sigma=sigma,
        m_factor=3,
        full=estimator_full,
        jobs=estimator_jobs,
    )

    phase = classify_phase(rugosity, mobility)

    return {
        "n": n,
        "q": q,
        "sigma": sigma,
        "alpha": alpha,
        "rugosity": rugosity,
        "rugosity_std": 0.0,
        "window_gap": window_gap,
        "mobility": mobility,
        "mobility_std": 0.0,
        "phase": phase,
        "bin_centers": bc.tolist(),
        "F": F.tolist(),
        "xi_range": xi_range,
        "k_used": k_scaled,
        "security": security,
        "seed": seed,
        "xi_true": xi_true,
        "xi_random": xi_random,
    }


# =============================================================================
# Plotting and summary
# =============================================================================
def style_axis(ax, title: str, xlabel: str = "", ylabel: str = "", title_color: str = "white"):
    ax.set_facecolor(PBG)
    ax.set_title(title, color=title_color, fontsize=11, pad=8)
    ax.set_xlabel(xlabel, color="white", fontsize=9)
    ax.set_ylabel(ylabel, color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")


def plot_experiment_8(results: List[Dict[str, Any]], baseline: Dict[str, Any], save: str = "lwe_exp8_results.png"):
    fig = plt.figure(figsize=(24, 17))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.24)

    ax0 = fig.add_subplot(gs[0, 0])
    alphas = np.array([r["alpha"] for r in results], dtype=float)
    rugs = np.array([r["rugosity"] for r in results], dtype=float)
    rug_err = np.array([r.get("rugosity_std", 0.0) for r in results], dtype=float)
    phases = [r["phase"] for r in results]

    ax0.errorbar(alphas, rugs, yerr=rug_err, fmt="o-", lw=2, ms=6, color="#00d4ff", capsize=3)
    for a, rg, ph in zip(alphas, rugs, phases):
        ax0.scatter(
            [a],
            [rg],
            s=120,
            marker={"funnel": "o", "rugged": "^", "cliff": "s"}.get(ph, "o"),
            color=PHASE_COLORS.get(ph, "#888888"),
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
        )
    ax0.axhline(0.16, color="#ffd93d", linestyle="--", lw=1.5)
    style_axis(ax0, "Experiment 8 — rugosity + phase", "α = q/(σ√n)", "rugosity")

    ax1 = fig.add_subplot(gs[0, 1])
    min_bits = np.array(
        [r["security"]["overall_bits"] if r["security"]["overall_bits"] is not None else np.nan for r in results],
        dtype=float,
    )
    ax1.plot(alphas, min_bits, "o-", lw=2, ms=6, color="#69ff94", label="min attack cost")

    if baseline and baseline.get("overall_bits") is not None:
        ax1.axhline(
            baseline["overall_bits"],
            color="#ff6b6b",
            linestyle="--",
            lw=1.5,
            label=f"baseline ≈ {fmt_bits(baseline['overall_bits'])}",
        )

    for r in results:
        best_fam = r["security"]["best_family"] or "other"
        bits = r["security"]["overall_bits"]
        if bits is None:
            continue
        ax1.scatter(
            [r["alpha"]],
            [bits],
            s=100,
            color=FAMILY_COLORS.get(best_fam, FAMILY_COLORS["other"]),
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )

    style_axis(ax1, "Real lattice-estimator security", "α = q/(σ√n)", "min log2(rop)")
    ax1.legend(fontsize=7, facecolor=PBG, labelcolor="white")

    ax2 = fig.add_subplot(gs[1, 0])
    order = np.argsort(alphas)

    fam_lines = {fam: [] for fam in ("bdd", "usvp", "dual", "dual_hybrid")}
    for r in results:
        sec = r["security"]["family_bits"]
        for fam in fam_lines:
            fam_lines[fam].append(sec.get(fam, np.nan) if sec else np.nan)

    for fam in ("bdd", "usvp", "dual", "dual_hybrid"):
        y = np.array(fam_lines[fam], dtype=float)
        if np.all(np.isnan(y)):
            continue
        ax2.plot(
            alphas[order],
            y[order],
            "o-",
            lw=2,
            ms=5,
            label=fam,
            color=FAMILY_COLORS.get(fam, FAMILY_COLORS["other"]),
        )

    style_axis(ax2, "Attack-family comparison", "α = q/(σ√n)", "log2(rop)")
    ax2.legend(fontsize=7, facecolor=PBG, labelcolor="white")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(PBG)
    ax3.axis("off")

    lines = [
        "EXPERIMENT 8 — KEY NUMBERS",
        "",
        f"Device: {DEVICE_NAME}",
        f"Estimator mode: {'full' if USE_FULL_ESTIMATOR else 'rough'}",
        f"Jobs: {DEFAULT_JOBS}",
        f"n: {results[0]['n'] if results else 'n/a'}",
        f"sigma: {results[0]['sigma'] if results else 'n/a'}",
        "",
        "Per-point summary:",
    ]

    for r in results:
        sec = r["security"]
        overall_bits = sec["overall_bits"]
        best_attack = sec["overall_attack"]
        best_family = sec["best_family"]
        family_bits = sec["family_bits"]

        lines.append(
            f"  α={r['alpha']:.0f}: phase={r['phase']}, rug={r['rugosity']:.3f}, "
            f"min={fmt_bits(overall_bits)}, best={best_family or 'n/a'}:{best_attack or 'n/a'}"
        )
        lines.append(
            "     "
            f"bdd={fmt_bits(family_bits.get('bdd'))}, "
            f"usvp={fmt_bits(family_bits.get('usvp'))}, "
            f"dual={fmt_bits(family_bits.get('dual'))}, "
            f"dual_hybrid={fmt_bits(family_bits.get('dual_hybrid'))}"
        )

    if baseline and baseline.get("overall_bits") is not None:
        lines += [
            "",
            "Kyber1024 baseline:",
            f"  min security ≈ {fmt_bits(baseline['overall_bits'])}",
            f"  best family: {baseline.get('best_family') or 'n/a'}",
            f"  best attack: {baseline.get('overall_attack') or 'n/a'}",
        ]

    ax3.text(
        0.03,
        0.98,
        "\n".join(lines),
        transform=ax3.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="white",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#0f1117", alpha=0.92),
    )
    style_axis(ax3, "Summary")

    fig.suptitle(
        "Experiment 8 — umbrella texture vs real lattice-estimator\n"
        "Min attack cost, attack-family dominance, beta, and instability",
        color="white",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.savefig(save, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Plot saved → {save}")


def write_summary(
    results: List[Dict[str, Any]],
    baseline: Dict[str, Any],
    path: str = "lwe_exp8_summary.txt",
):
    sep = "=" * 80
    with open(path, "w", encoding="utf-8") as f:
        f.write(sep + "\n")
        f.write("EXPERIMENT 8 — UMBRELLA TEXTURE VS REAL LATTICE-ESTIMATOR\n")
        f.write(sep + "\n\n")
        f.write("GOALS\n")
        f.write("- Replace old BKZ proxy bits with real lattice-estimator outputs.\n")
        f.write("- Track bdd, usvp, dual, and dual_hybrid when available.\n")
        f.write("- Compare rugosity against min attack cost, attack-family dominance, beta, and instability.\n\n")

        f.write("PER-POINT RESULTS\n")
        f.write("-" * 80 + "\n")
        for r in results:
            sec = r["security"]
            overall_bits = sec["overall_bits"]
            best_family = sec["best_family"]
            best_attack = sec["overall_attack"]
            dom_gap = sec["dominance_gap"]

            f.write(
                f"alpha={r['alpha']:.1f}  phase={r['phase']}  rugosity={r['rugosity']:.4f}  "
                f"mobility={r['mobility']:.4f}  security={fmt_bits(overall_bits)}  "
                f"best_family={best_family or 'n/a'}  best_attack={best_attack or 'n/a'}  "
                f"dominance_gap={('n/a' if dom_gap is None else f'{dom_gap:.2f}')}\n"
            )

            fb = sec["family_bits"]
            bb = sec["family_beta"]

            line_bits = (
                f"  bdd={fmt_bits(fb.get('bdd'))}"
                + (f",β={fmt_beta(bb.get('bdd'))}" if bb.get("bdd") is not None else "")
                + f", usvp={fmt_bits(fb.get('usvp'))}"
                + (f",β={fmt_beta(bb.get('usvp'))}" if bb.get("usvp") is not None else "")
                + f", dual={fmt_bits(fb.get('dual'))}"
                + (f",β={fmt_beta(bb.get('dual'))}" if bb.get("dual") is not None else "")
                + f", dual_hybrid={fmt_bits(fb.get('dual_hybrid'))}"
                + (f",β={fmt_beta(bb.get('dual_hybrid'))}" if bb.get("dual_hybrid") is not None else "")
                + "\n"
            )
            f.write(line_bits)

        f.write("\n")
        if baseline and baseline.get("overall_bits") is not None:
            f.write("KYBER1024 BASELINE\n")
            f.write("-" * 80 + "\n")
            f.write(f"min security ≈ {fmt_bits(baseline['overall_bits'])}\n")
            f.write(f"best family: {baseline.get('best_family') or 'n/a'}\n")
            f.write(f"best attack: {baseline.get('overall_attack') or 'n/a'}\n\n")

        rugs = np.array([r["rugosity"] for r in results], dtype=float)
        mins = np.array(
            [r["security"]["overall_bits"] if r["security"]["overall_bits"] is not None else np.nan for r in results],
            dtype=float,
        )
        mob = np.array([r["mobility"] for r in results], dtype=float)

        f.write("AGGREGATES\n")
        f.write("-" * 80 + "\n")
        f.write(f"rugosity mean={np.nanmean(rugs):.4f}, std={np.nanstd(rugs):.4f}\n")
        f.write(f"mobility mean={np.nanmean(mob):.4f}, std={np.nanstd(mob):.4f}\n")
        f.write(f"min_bits mean={np.nanmean(mins):.2f}, std={np.nanstd(mins):.2f}\n\n")

        fam_counts: Dict[str, int] = {}
        for r in results:
            fam = r["security"]["best_family"] or "n/a"
            fam_counts[fam] = fam_counts.get(fam, 0) + 1

        f.write("FAMILY DOMINANCE COUNTS\n")
        f.write("-" * 80 + "\n")
        for fam, cnt in sorted(fam_counts.items(), key=lambda x: (-x[1], x[0])):
            f.write(f"{fam}: {cnt}\n")

    print(f"Summary saved → {path}")


def save_json(results: List[Dict[str, Any]], baseline: Dict[str, Any], path: str = "lwe_exp8_data.json"):
    payload = {
        "results": clean_for_json(results),
        "baseline": clean_for_json(baseline),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"JSON saved → {path}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--alphas", type=float, nargs="*", default=[36, 52, 72, 96, 128, 160, 208])
    parser.add_argument("--seeds-per-point", type=int, default=2)
    parser.add_argument("--n-windows", type=int, default=12)
    parser.add_argument("--burn-steps", type=int, default=250)
    parser.add_argument("--prod-steps", type=int, default=900)
    parser.add_argument("--n-chains", type=int, default=6)
    parser.add_argument("--k-base", type=float, default=0.006)
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--rough", action="store_true", help="Use rough estimator mode instead of full.")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    if not ESTIMATOR_AVAILABLE:
        raise RuntimeError(
            "lattice-estimator unavailable. Run this inside Sage with the estimator module installed."
        )

    estimator_full = not args.rough

    print("=" * 80)
    print("Experiment 8: umbrella texture vs real lattice-estimator")
    print(f"Device: {DEVICE_NAME}")
    print(f"Estimator mode: {'full' if estimator_full else 'rough'}")
    print(f"Jobs: {args.jobs}")
    print("=" * 80)

    results: List[Dict[str, Any]] = []
    t0 = time.time()

    for alpha in args.alphas:
        q = int(alpha * args.sigma * math.sqrt(args.n))
        q = max(q, args.n + 2)
        if q % 2 == 0:
            q += 1

        print(f"\nalpha={alpha:.1f} -> q={q}")
        seed_reports: List[Dict[str, Any]] = []

        for seed in range(args.seeds_per_point):
            try:
                r = run_config_stable(
                    n=args.n,
                    q=q,
                    sigma=args.sigma,
                    seed=seed * 101 + int(alpha * 10),
                    n_windows=args.n_windows,
                    n_steps_burn=args.burn_steps,
                    n_steps_prod=args.prod_steps,
                    n_chains=args.n_chains,
                    k_base=args.k_base,
                    estimator_full=estimator_full,
                    estimator_jobs=args.jobs,
                )
                seed_reports.append(r)
                sec = r["security"]
                print(
                    f"  seed {seed}: rug={r['rugosity']:.4f} mob={r['mobility']:.4f} "
                    f"phase={r['phase']} min={fmt_bits(sec['overall_bits'])} "
                    f"best={sec['best_family'] or 'n/a'}:{sec['overall_attack'] or 'n/a'}"
                )
            except Exception as e:
                print(f"  seed {seed}: ERROR -> {e}")

        if not seed_reports:
            continue

        rugosity_vals = [x["rugosity"] for x in seed_reports]
        mob_vals = [x["mobility"] for x in seed_reports]
        overall_bits_vals = [x["security"]["overall_bits"] for x in seed_reports]
        phases = [x["phase"] for x in seed_reports]
        best_families = [x["security"]["best_family"] for x in seed_reports]

        agg = dict(seed_reports[0])
        agg["rugosity"] = float(np.mean(rugosity_vals))
        agg["rugosity_std"] = float(np.std(rugosity_vals))
        agg["mobility"] = float(np.mean(mob_vals))
        agg["mobility_std"] = float(np.std(mob_vals))
        agg["security"] = dict(seed_reports[0]["security"])
        agg["security"]["overall_bits"] = avg_ignore_none(overall_bits_vals)
        agg["security"]["overall_bits_std"] = std_ignore_none(overall_bits_vals)

        non_none_fams = [x for x in best_families if x is not None]
        agg["security"]["best_family"] = max(set(non_none_fams), key=non_none_fams.count) if non_none_fams else None
        agg["phase"] = max(set(phases), key=phases.count)

        agg["seed_reports"] = seed_reports
        agg["seed_rugs"] = rugosity_vals
        agg["seed_mobility"] = mob_vals
        agg["seed_phases"] = phases
        results.append(agg)

    baseline = {}
    if not args.skip_baseline:
        try:
            baseline = estimate_kyber1024_baseline(full=estimator_full, jobs=args.jobs)
        except Exception as e:
            print(f"Kyber1024 baseline failed: {e}")
            baseline = {}

    print("\nGenerating outputs...")
    plot_experiment_8(results, baseline)
    write_summary(results, baseline)
    save_json(results, baseline)

    elapsed = (time.time() - t0) / 60.0
    print(f"\nTotal runtime: {elapsed:.1f} min")
    print("Outputs: lwe_exp8_results.png  lwe_exp8_summary.txt  lwe_exp8_data.json")


if __name__ == "__main__":
    main()