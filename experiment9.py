#!/usr/bin/env python3
"""
Experiment 9: Umbrella-sampled free-energy landscape vs lattice-estimator hardness

Run:
    sage -python experiment9_free_energy_vs_hardness.py

Outputs:
    - exp9_results.png
    - exp9_summary.txt
    - exp9_data.json

This script answers:
    "Does the free-energy geometry from umbrella sampling track changes in
     concrete lattice-estimator hardness?"

It computes, per parameter point:
    - free-energy profile F(xi)
    - barrier height
    - entropy of the reconstructed xi distribution
    - number of modes / peak separation
    - roughness of F
    - lattice-estimator attack costs (bdd / usvp / dual / dual_hybrid)
    - minimum attack cost
    - best attack family
    - correlation between geometry features and hardness
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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# Config
DEFAULT_REPO_PATH = os.path.expanduser("~/lattice-estimator")
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
RED_COST_MODEL = None  # filled after import if available

# Optional torch
try:
    import torch

    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
except Exception:
    torch = None
    TORCH_AVAILABLE = False
    DEVICE = "cpu"
    DEVICE_NAME = "CPU (torch unavailable)"

# Local lattice-estimator import
if os.path.isdir(DEFAULT_REPO_PATH) and DEFAULT_REPO_PATH not in sys.path:
    sys.path.insert(0, DEFAULT_REPO_PATH)

ESTIMATOR_AVAILABLE = False
LWE = None
ND = None
RC = None
schemes = None

try:
    from estimator import LWE, ND, RC, schemes  # type: ignore
    ESTIMATOR_AVAILABLE = True
    RED_COST_MODEL = getattr(RC, "BDGL16", None)
except Exception:
    ESTIMATOR_AVAILABLE = False

# Helpers
def fmt_bits(x: Optional[float]) -> str:
    return "n/a" if x is None else f"2^{x:.1f}"


def fmt_float(x: Optional[float], nd: int = 3) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def clean_json(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean_json(v) for v in x]
    if isinstance(x, tuple):
        return [clean_json(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def moving_average(x: np.ndarray, w: int = 5) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x.copy()
    kernel = np.ones(w, dtype=np.float64) / float(w)
    y = np.convolve(x, kernel, mode="same")
    return y


def corrcoef_nan(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if mask.sum() < 3:
        return None
    xc = xa[mask]
    yc = ya[mask]
    if np.std(xc) == 0 or np.std(yc) == 0:
        return None
    return float(np.corrcoef(xc, yc)[0, 1])


def pairwise_rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64) + 1.0
    return ranks


def spearman_nan(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if mask.sum() < 3:
        return None
    rx = pairwise_rankdata(xa[mask])
    ry = pairwise_rankdata(ya[mask])
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def parse_number(text: str) -> Optional[float]:
    m = re.search(r"([-+]?\d+(?:\.\d+)?)", text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def parse_rop_bits_from_text(text: str) -> Optional[float]:
    pats = [
        r"rop:\s*≈?\s*2\^([0-9]+(?:\.[0-9]+)?)",
        r"rop:\s*2\^([0-9]+(?:\.[0-9]+)?)",
        r"rop\s*=\s*≈?\s*2\^([0-9]+(?:\.[0-9]+)?)",
        r"rop\s*=\s*2\^([0-9]+(?:\.[0-9]+)?)",
    ]
    for pat in pats:
        m = re.search(pat, text)
        if m:
            return float(m.group(1))
    return None


def parse_beta_from_text(text: str) -> Optional[float]:
    pats = [
        r"\bβ:\s*([0-9]+(?:\.[0-9]+)?)",
        r"\bbeta:\s*([0-9]+(?:\.[0-9]+)?)",
        r"\bb:\s*([0-9]+(?:\.[0-9]+)?)",
        r"\bβ\s*=\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for pat in pats:
        m = re.search(pat, text)
        if m:
            return float(m.group(1))
    return None


def parse_m_from_text(text: str) -> Optional[float]:
    m = re.search(r"\bm:\s*≈?\s*([0-9]+(?:\.[0-9]+)?)", text)
    if m:
        return float(m.group(1))
    return None


def parse_d_from_text(text: str) -> Optional[float]:
    m = re.search(r"\bd:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if m:
        return float(m.group(1))
    return None


def parse_delta_from_text(text: str) -> Optional[float]:
    m = re.search(r"\bδ:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if m:
        return float(m.group(1))
    return None


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
    if "dual" in n:
        return "dual"
    return "other"


def attack_record_from_cost_obj(cost_obj: Any) -> Dict[str, Optional[float]]:
    txt = str(cost_obj)
    return {
        "rop": parse_rop_bits_from_text(txt),
        "beta": parse_beta_from_text(txt),
        "m": parse_m_from_text(txt),
        "d": parse_d_from_text(txt),
        "delta": parse_delta_from_text(txt),
        "text": txt,
    }


def family_minima(attack_table: Dict[str, Dict[str, Optional[float]]]) -> Dict[str, Dict[str, Optional[float]]]:
    families = ("bdd", "usvp", "dual", "dual_hybrid")
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for fam in families:
        best_key = None
        best_rop = None
        best_beta = None
        best_m = None
        best_d = None
        best_delta = None
        for key, rec in attack_table.items():
            kn = normalize_attack_name(key)
            ok = False
            if fam == "bdd":
                ok = "bdd" in kn
            elif fam == "usvp":
                ok = ("usvp" in kn) or ("primal" in kn)
            elif fam == "dual":
                ok = ("dual" in kn) and ("dual_hybrid" not in kn) and ("dualhybrid" not in kn)
            elif fam == "dual_hybrid":
                ok = ("dual_hybrid" in kn) or ("dualhybrid" in kn)
            if not ok:
                continue
            rop = rec.get("rop")
            if rop is None:
                continue
            if best_rop is None or rop < best_rop:
                best_key = key
                best_rop = rop
                best_beta = rec.get("beta")
                best_m = rec.get("m")
                best_d = rec.get("d")
                best_delta = rec.get("delta")
        if best_rop is not None:
            out[fam] = {
                "attack": best_key,
                "rop": best_rop,
                "beta": best_beta,
                "m": best_m,
                "d": best_d,
                "delta": best_delta,
            }
    return out


def overall_best_attack(attack_table: Dict[str, Dict[str, Optional[float]]]) -> Dict[str, Any]:
    best_key = None
    best_rec = None
    for key, rec in attack_table.items():
        rop = rec.get("rop")
        if rop is None:
            continue
        if best_rec is None or rop < best_rec["rop"]:
            best_key = key
            best_rec = rec
    if best_rec is None:
        return {
            "attack": None,
            "rop": None,
            "beta": None,
            "family": None,
        }
    return {
        "attack": best_key,
        "rop": best_rec.get("rop"),
        "beta": best_rec.get("beta"),
        "family": family_from_attack_name(best_key or ""),
    }


# Estimator
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
    raise RuntimeError("Could not construct secret distribution")


def make_error_dist(sigma: float):
    if not ESTIMATOR_AVAILABLE:
        raise RuntimeError("lattice-estimator unavailable")
    if hasattr(ND, "DiscreteGaussian"):
        try:
            return ND.DiscreteGaussian(stddev=sigma)
        except Exception:
            try:
                return ND.DiscreteGaussian(sigma)
            except Exception:
                pass
    if hasattr(ND, "Gaussian"):
        try:
            return ND.Gaussian(stddev=sigma)
        except Exception:
            try:
                return ND.Gaussian(sigma)
            except Exception:
                pass
    if hasattr(ND, "CenteredBinomial"):
        eta = max(1, int(round(2.0 * sigma * sigma)))
        return ND.CenteredBinomial(eta)
    raise RuntimeError("Could not construct error distribution")


def call_estimator(func, params, jobs: Optional[int] = None, red_cost_model=None):
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
        raise RuntimeError("lattice-estimator unavailable")
    m = m_factor * n
    Xs = make_secret_dist(q)
    Xe = make_error_dist(sigma)
    return LWE.Parameters(n=n, q=q, Xs=Xs, Xe=Xe, m=m, tag=f"synthetic-n{n}-q{q}")


def estimate_security(n: int, q: int, sigma: float, m_factor: int = 3, full: bool = True, jobs: Optional[int] = None) -> Dict[str, Any]:
    params = build_lwe_params(n=n, q=q, sigma=sigma, m_factor=m_factor)
    jobs = DEFAULT_JOBS if jobs is None else jobs
    if full:
        try:
            result = call_estimator(LWE.estimate, params, jobs=jobs, red_cost_model=RED_COST_MODEL)
        except Exception:
            result = call_estimator(LWE.estimate, params, jobs=jobs)
    else:
        rough = getattr(LWE.estimate, "rough", None)
        if rough is None:
            raise RuntimeError("LWE.estimate.rough unavailable")
        try:
            result = call_estimator(rough, params, jobs=jobs, red_cost_model=RED_COST_MODEL)
        except Exception:
            result = call_estimator(rough, params, jobs=jobs)

    attack_table: Dict[str, Dict[str, Optional[float]]] = {}
    if hasattr(result, "items"):
        try:
            items = list(result.items())
        except Exception:
            items = []
        for attack_name, cost_obj in items:
            attack_table[str(attack_name)] = attack_record_from_cost_obj(cost_obj)
    else:
        attack_table["single"] = attack_record_from_cost_obj(result)

    fam = family_minima(attack_table)
    best = overall_best_attack(attack_table)
    best_family = None
    best_family_bits = None
    best_family_attack = None
    best_family_beta = None
    for fam_name, rec in fam.items():
        rop = rec.get("rop")
        if rop is None:
            continue
        if best_family_bits is None or rop < best_family_bits:
            best_family = fam_name
            best_family_bits = rop
            best_family_attack = rec.get("attack")
            best_family_beta = rec.get("beta")

    sorted_families = sorted(
        [(k, v["rop"]) for k, v in fam.items() if v.get("rop") is not None],
        key=lambda x: x[1],
    )
    dominance_gap = None
    if len(sorted_families) >= 2:
        dominance_gap = float(sorted_families[1][1] - sorted_families[0][1])

    return {
        "params": params,
        "raw_result": result,
        "attack_table": attack_table,
        "family_best": fam,
        "overall_attack": best["attack"],
        "overall_bits": best["rop"],
        "overall_beta": best["beta"],
        "overall_family": best["family"],
        "best_family": best_family,
        "best_family_bits": best_family_bits,
        "best_family_beta": best_family_beta,
        "best_family_attack": best_family_attack,
        "dominance_gap": dominance_gap,
    }


def estimate_kyber_baseline(full: bool = True, jobs: Optional[int] = None) -> Dict[str, Any]:
    if not ESTIMATOR_AVAILABLE or schemes is None:
        return {}
    try:
        params = schemes.Kyber1024
        return estimate_from_params(params, full=full, jobs=jobs)
    except Exception:
        return {}


def estimate_from_params(params, full: bool = True, jobs: Optional[int] = None) -> Dict[str, Any]:
    jobs = DEFAULT_JOBS if jobs is None else jobs
    if full:
        try:
            result = call_estimator(LWE.estimate, params, jobs=jobs, red_cost_model=RED_COST_MODEL)
        except Exception:
            result = call_estimator(LWE.estimate, params, jobs=jobs)
    else:
        rough = getattr(LWE.estimate, "rough", None)
        if rough is None:
            raise RuntimeError("LWE.estimate.rough unavailable")
        try:
            result = call_estimator(rough, params, jobs=jobs, red_cost_model=RED_COST_MODEL)
        except Exception:
            result = call_estimator(rough, params, jobs=jobs)

    attack_table: Dict[str, Dict[str, Optional[float]]] = {}
    if hasattr(result, "items"):
        try:
            items = list(result.items())
        except Exception:
            items = []
        for attack_name, cost_obj in items:
            attack_table[str(attack_name)] = attack_record_from_cost_obj(cost_obj)
    else:
        attack_table["single"] = attack_record_from_cost_obj(result)

    fam = family_minima(attack_table)
    best = overall_best_attack(attack_table)
    best_family = None
    best_family_bits = None
    best_family_attack = None
    for fam_name, rec in fam.items():
        rop = rec.get("rop")
        if rop is None:
            continue
        if best_family_bits is None or rop < best_family_bits:
            best_family = fam_name
            best_family_bits = rop
            best_family_attack = rec.get("attack")

    sorted_families = sorted(
        [(k, v["rop"]) for k, v in fam.items() if v.get("rop") is not None],
        key=lambda x: x[1],
    )
    dominance_gap = None
    if len(sorted_families) >= 2:
        dominance_gap = float(sorted_families[1][1] - sorted_families[0][1])

    return {
        "params": params,
        "raw_result": result,
        "attack_table": attack_table,
        "family_best": fam,
        "overall_attack": best["attack"],
        "overall_bits": best["rop"],
        "overall_beta": best["beta"],
        "overall_family": best["family"],
        "best_family": best_family,
        "best_family_bits": best_family_bits,
        "best_family_attack": best_family_attack,
        "dominance_gap": dominance_gap,
    }


# LWE instance + umbrella sampling
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
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch unavailable")
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


def residual_norm_gpu(lwe: LWEInstance, s_batch: "torch.Tensor") -> "torch.Tensor":
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
    n_chains: int = 6,
    s_init_batch: Optional["torch.Tensor"] = None,
    step_size: Optional[int] = None,
    block_size: Optional[int] = None,
):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for umbrella sampler")
    n_win = len(xi_centers)
    n, q = lwe.n, lwe.q
    B = n_win * n_chains

    if step_size is None:
        step_size = max(1, q // 64)
    if block_size is None:
        block_size = max(1, n // 16)

    xc_gpu = torch.tensor(np.repeat(np.asarray(xi_centers, dtype=np.float32), n_chains), device=DEVICE)

    if s_init_batch is None:
        s_batch = torch.randint(0, q, (B, n), dtype=torch.int64, device=DEVICE).to(torch.float32)
    else:
        noise = torch.randint(-step_size, step_size + 1, (B, n), dtype=torch.int64, device=DEVICE).to(torch.float32)
        s_batch = (s_init_batch + noise) % q

    xi_cur = residual_norm_gpu(lwe, s_batch)
    bias_cur = 0.5 * k * (xi_cur - xc_gpu) ** 2

    xi_records = torch.zeros((B, n_steps), dtype=torch.float32, device=DEVICE)
    acc_count = torch.zeros(B, dtype=torch.float32, device=DEVICE)

    for step in range(n_steps):
        col_idx = torch.randint(0, n, (B, block_size), dtype=torch.int64, device=DEVICE)
        deltas = torch.randint(-step_size, step_size + 1, (B, block_size), dtype=torch.int64, device=DEVICE).to(torch.float32)

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
    n_iter: int = 400,
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
        hi = np.nanpercentile(F[finite], 95)
        F = np.clip(F, 0, hi)

    return bc, F


def landscape_features(bc: np.ndarray, F: np.ndarray) -> Dict[str, Any]:
    valid = np.isfinite(F)
    out = {
        "barrier_height": None,
        "entropy": None,
        "roughness": None,
        "free_energy_span": None,
        "n_peaks": 0,
        "peak_separation": None,
        "peak_ratio": None,
    }
    if valid.sum() < 5:
        return out

    bcv = bc[valid]
    Fv = F[valid]
    Fv = Fv - np.nanmin(Fv)
    rho = np.exp(-Fv)
    if rho.sum() <= 0:
        return out
    p = rho / rho.sum()

    barrier = float(np.nanmax(Fv) - np.nanmin(Fv))
    span = float(np.nanpercentile(Fv, 90) - np.nanpercentile(Fv, 10))
    diff = np.diff(Fv)
    rough = float(np.std(diff) / (barrier + 1e-9)) if len(diff) > 1 else 0.0

    entropy = -float(np.sum(p * np.log(p + 1e-300))) / math.log(len(p) + 1e-300)

    ps = moving_average(p, 5)
    peaks = []
    for i in range(1, len(ps) - 1):
        if ps[i] >= ps[i - 1] and ps[i] > ps[i + 1]:
            peaks.append(i)
    peaks = sorted(peaks, key=lambda i: ps[i], reverse=True)
    n_peaks = len(peaks)

    peak_sep = None
    peak_ratio = None
    if len(peaks) >= 2:
        p1, p2 = peaks[0], peaks[1]
        peak_sep = float(abs(bcv[p1] - bcv[p2]))
        if ps[p1] > 0:
            peak_ratio = float(ps[p2] / ps[p1])

    out.update(
        {
            "barrier_height": barrier,
            "entropy": entropy,
            "roughness": rough,
            "free_energy_span": span,
            "n_peaks": n_peaks,
            "peak_separation": peak_sep,
            "peak_ratio": peak_ratio,
        }
    )
    return out


def classify_phase(barrier: Optional[float], entropy: Optional[float], n_peaks: int) -> str:
    if barrier is None or entropy is None:
        return "unknown"
    if n_peaks >= 3 or entropy > 0.75:
        return "rugged"
    if barrier < 0.8:
        return "cliff"
    return "funnel"


def run_one_point(
    n: int,
    q: int,
    sigma: float,
    seed: int,
    n_windows: int,
    n_steps_burn: int,
    n_steps_prod: int,
    n_chains: int,
    k_base: float,
    T: float,
    estimator_full: bool,
    estimator_jobs: Optional[int],
):
    lwe = generate_lwe(n=n, m=3 * n, q=q, sigma=sigma, seed=seed).to_gpu()

    alpha = q / (sigma * math.sqrt(n))
    xi_true = residual_norm_cpu(lwe, lwe.s_true)

    rng = np.random.default_rng(seed + 12345)
    xi_random = float(
        np.mean(
            [
                residual_norm_cpu(lwe, rng.integers(0, q, size=(n,), dtype=np.int64))
                for _ in range(24)
            ]
        )
    )

    xi_max = xi_random * 1.03
    xi_min = max(xi_true * 0.75, 1.0)
    xi_range = xi_max - xi_min + 1e-9
    k_scaled = max(k_base * (500.0 / xi_range) ** 2, 1e-10)

    xi_centers = np.linspace(xi_max, xi_min, n_windows)

    _, s_batch, _ = mh_all_windows_gpu(
        lwe=lwe,
        xi_centers=xi_centers,
        k=k_scaled,
        T=T,
        n_steps=n_steps_burn,
        n_chains=n_chains,
    )

    xi_trajs, _, acc_by_win = mh_all_windows_gpu(
        lwe=lwe,
        xi_centers=xi_centers,
        k=k_scaled,
        T=T,
        n_steps=n_steps_prod,
        n_chains=n_chains,
        s_init_batch=s_batch,
    )

    window_means = xi_trajs.mean(axis=1)
    window_gap = float(np.mean(np.abs(window_means - xi_centers) / xi_range))
    mobility = float(acc_by_win.mean())

    xi_bins = np.linspace(xi_min * 0.85, xi_max * 1.08, 90)
    bc, F = wham_stable(list(xi_trajs), xi_centers, k_scaled, T, xi_bins)
    feats = landscape_features(bc, F)
    phase = classify_phase(feats["barrier_height"], feats["entropy"], feats["n_peaks"])

    sec = estimate_security(
        n=n,
        q=q,
        sigma=sigma,
        m_factor=3,
        full=estimator_full,
        jobs=estimator_jobs,
    )

    try:
        del lwe.A_gpu, lwe.b_gpu, s_batch, xi_trajs
    except Exception:
        pass
    if TORCH_AVAILABLE and DEVICE != "cpu" and torch is not None:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return {
        "n": n,
        "q": q,
        "sigma": sigma,
        "alpha": alpha,
        "xi_true": xi_true,
        "xi_random": xi_random,
        "xi_range": xi_range,
        "k_used": k_scaled,
        "window_gap": window_gap,
        "mobility": mobility,
        "phase": phase,
        "bin_centers": bc.tolist(),
        "F": F.tolist(),
        "features": feats,
        "security": sec,
        "seed": seed,
    }


# Plotting / summary
def style_axis(ax, title: str, xlabel: str = "", ylabel: str = "", title_color: str = "white"):
    ax.set_facecolor(PBG)
    ax.set_title(title, color=title_color, fontsize=11, pad=8)
    ax.set_xlabel(xlabel, color="white", fontsize=9)
    ax.set_ylabel(ylabel, color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")


def plot_results(results: List[Dict[str, Any]], baseline: Dict[str, Any], save: str = "exp9_results.png"):
    fig = plt.figure(figsize=(24, 17))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.25)

    alphas = np.array([r["alpha"] for r in results], dtype=np.float64)
    hard = np.array([r["security"]["overall_bits"] if r["security"]["overall_bits"] is not None else np.nan for r in results], dtype=np.float64)
    barrier = np.array([r["features"]["barrier_height"] if r["features"]["barrier_height"] is not None else np.nan for r in results], dtype=np.float64)
    entropy = np.array([r["features"]["entropy"] if r["features"]["entropy"] is not None else np.nan for r in results], dtype=np.float64)
    roughness = np.array([r["features"]["roughness"] if r["features"]["roughness"] is not None else np.nan for r in results], dtype=np.float64)
    peaks = np.array([r["features"]["n_peaks"] for r in results], dtype=np.float64)
    fam = [r["security"]["best_family"] or "other" for r in results]

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(alphas, hard, "o-", lw=2, ms=6, color="#69ff94", label="min attack cost")
    if baseline and baseline.get("overall_bits") is not None:
        ax0.axhline(
            baseline["overall_bits"],
            color="#ff6b6b",
            linestyle="--",
            lw=1.5,
            label=f"Kyber1024 baseline {fmt_bits(baseline['overall_bits'])}",
        )
    for a, h, f in zip(alphas, hard, fam):
        if not np.isfinite(h):
            continue
        ax0.scatter([a], [h], s=110, color=FAMILY_COLORS.get(f, FAMILY_COLORS["other"]), edgecolors="white", linewidths=0.8)
    style_axis(ax0, "Hardness curve", "α = q/(σ√n)", "min log2(rop)")
    ax0.legend(fontsize=7, facecolor=PBG, labelcolor="white")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(alphas, barrier, "o-", lw=2, ms=6, color="#00d4ff", label="barrier height")
    ax1.plot(alphas, entropy, "o-", lw=2, ms=6, color="#ffd93d", label="entropy")
    ax1.plot(alphas, roughness, "o-", lw=2, ms=6, color="#ff6b6b", label="roughness")
    style_axis(ax1, "Free-energy features", "α = q/(σ√n)", "feature value")
    ax1.legend(fontsize=7, facecolor=PBG, labelcolor="white")

    ax2 = fig.add_subplot(gs[1, 0])
    for idx in [0, len(results) // 2, len(results) - 1]:
        r = results[idx]
        bc = np.asarray(r["bin_centers"], dtype=np.float64)
        F = np.asarray(r["F"], dtype=np.float64)
        valid = np.isfinite(F)
        if valid.sum() < 4:
            continue
        ax2.plot(bc[valid], F[valid], lw=2, label=f"α≈{r['alpha']:.0f} [{r['phase']}]")
    style_axis(ax2, "Representative free-energy profiles", "ξ (residual norm)", "F(ξ)")
    ax2.legend(fontsize=7, facecolor=PBG, labelcolor="white")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(PBG)
    ax3.axis("off")

    lines = [
        "EXPERIMENT 9 — summmarryyyy",
        "",
        f"Device: {DEVICE_NAME}",
        f"Estimator mode: {'full' if USE_FULL_ESTIMATOR else 'rough'}",
        f"n: {results[0]['n'] if results else 'n/a'}",
        f"sigma: {results[0]['sigma'] if results else 'n/a'}",
        "",
        "Correlations with hardness:",
        f"  barrier vs hardness: {fmt_float(corrcoef_nan(barrier, hard), 3)}",
        f"  entropy vs hardness: {fmt_float(corrcoef_nan(entropy, hard), 3)}",
        f"  roughness vs hardness: {fmt_float(corrcoef_nan(roughness, hard), 3)}",
        f"  peaks vs hardness: {fmt_float(spearman_nan(peaks, hard), 3)}",
        "",
        "Per-point best-family labels:",
    ]
    for r in results:
        sec = r["security"]
        feats = r["features"]
        lines.append(
            f"  α={r['alpha']:.0f}: {r['phase']}, min={fmt_bits(sec['overall_bits'])}, "
            f"best={sec['best_family'] or 'n/a'}:{sec['overall_attack'] or 'n/a'}, "
            f"bar={fmt_float(feats['barrier_height'])}, ent={fmt_float(feats['entropy'])}"
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
        "Experiment 9 — umbrella free energy vs lattice-estimator hardness",
        color="white",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.savefig(save, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Plot saved >>>> {save}")


def write_summary(results: List[Dict[str, Any]], baseline: Dict[str, Any], path: str = "exp9_summary.txt"):
    hard = np.array([r["security"]["overall_bits"] if r["security"]["overall_bits"] is not None else np.nan for r in results], dtype=np.float64)
    barrier = np.array([r["features"]["barrier_height"] if r["features"]["barrier_height"] is not None else np.nan for r in results], dtype=np.float64)
    entropy = np.array([r["features"]["entropy"] if r["features"]["entropy"] is not None else np.nan for r in results], dtype=np.float64)
    roughness = np.array([r["features"]["roughness"] if r["features"]["roughness"] is not None else np.nan for r in results], dtype=np.float64)
    peaks = np.array([r["features"]["n_peaks"] for r in results], dtype=np.float64)
    sep = np.array([r["features"]["peak_separation"] if r["features"]["peak_separation"] is not None else np.nan for r in results], dtype=np.float64)

    sep_corr = corrcoef_nan(sep, hard)
    peak_corr = spearman_nan(peaks, hard)

    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("EXPERIMENT 9 — UMBRELLA FREE-ENERGY VS HARDNESS\n")
        f.write("=" * 80 + "\n\n")
        f.write("GOALS\n")
        f.write("- Compare free-energy geometry from umbrella sampling against lattice-estimator hardness.\n")
        f.write("- Test whether barrier height, entropy, and mode structure track attack cost.\n\n")

        f.write("CORRELATIONS\n")
        f.write("-" * 80 + "\n")
        f.write(f"barrier vs hardness: {fmt_float(corrcoef_nan(barrier, hard), 4)}\n")
        f.write(f"entropy vs hardness: {fmt_float(corrcoef_nan(entropy, hard), 4)}\n")
        f.write(f"roughness vs hardness: {fmt_float(corrcoef_nan(roughness, hard), 4)}\n")
        f.write(f"peak separation vs hardness: {fmt_float(sep_corr, 4)}\n")
        f.write(f"mode count vs hardness (Spearman): {fmt_float(peak_corr, 4)}\n\n")

        f.write("PER-POINT RESULTS\n")
        f.write("-" * 80 + "\n")
        for r in results:
            sec = r["security"]
            feats = r["features"]
            f.write(
                f"alpha={r['alpha']:.1f}  "
                f"phase={r['phase']}  "
                f"hardness={fmt_bits(sec['overall_bits'])}  "
                f"best_family={sec['best_family'] or 'n/a'}  "
                f"best_attack={sec['overall_attack'] or 'n/a'}  "
                f"dominance_gap={fmt_float(sec['dominance_gap'])}  "
                f"barrier={fmt_float(feats['barrier_height'])}  "
                f"entropy={fmt_float(feats['entropy'])}  "
                f"roughness={fmt_float(feats['roughness'])}  "
                f"peaks={feats['n_peaks']}  "
                f"peak_sep={fmt_float(feats['peak_separation'])}\n"
            )
            fb = sec["family_best"]
            fam_bits = []
            for fam_name in ("bdd", "usvp", "dual", "dual_hybrid"):
                if fam_name in fb:
                    fam_bits.append(f"{fam_name}={fmt_bits(fb[fam_name].get('rop'))}")
            f.write("  " + ", ".join(fam_bits) + "\n")

        f.write("\nKYBER1024 BASELINE\n")
        f.write("-" * 80 + "\n")
        if baseline and baseline.get("overall_bits") is not None:
            f.write(f"min security ≈ {fmt_bits(baseline['overall_bits'])}\n")
            f.write(f"best family: {baseline.get('best_family') or 'n/a'}\n")
            f.write(f"best attack: {baseline.get('overall_attack') or 'n/a'}\n")
        else:
            f.write("baseline unavailable\n")

        f.write("\nAGGREGATES\n")
        f.write("-" * 80 + "\n")
        f.write(f"hardness mean={np.nanmean(hard):.2f}, std={np.nanstd(hard):.2f}\n")
        f.write(f"barrier mean={np.nanmean(barrier):.4f}, std={np.nanstd(barrier):.4f}\n")
        f.write(f"entropy mean={np.nanmean(entropy):.4f}, std={np.nanstd(entropy):.4f}\n")
        f.write(f"roughness mean={np.nanmean(roughness):.4f}, std={np.nanstd(roughness):.4f}\n")

    print(f"Summary saved >>>> {path}")


def save_json(results: List[Dict[str, Any]], baseline: Dict[str, Any], path: str = "exp9_data.json"):
    payload = {
        "results": clean_json(results),
        "baseline": clean_json(baseline),
        "correlations": {
            "barrier_vs_hardness": corrcoef_nan(
                [r["features"]["barrier_height"] if r["features"]["barrier_height"] is not None else np.nan for r in results],
                [r["security"]["overall_bits"] if r["security"]["overall_bits"] is not None else np.nan for r in results],
            ),
            "entropy_vs_hardness": corrcoef_nan(
                [r["features"]["entropy"] if r["features"]["entropy"] is not None else np.nan for r in results],
                [r["security"]["overall_bits"] if r["security"]["overall_bits"] is not None else np.nan for r in results],
            ),
            "roughness_vs_hardness": corrcoef_nan(
                [r["features"]["roughness"] if r["features"]["roughness"] is not None else np.nan for r in results],
                [r["security"]["overall_bits"] if r["security"]["overall_bits"] is not None else np.nan for r in results],
            ),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"JSON saved >>>> {path}")


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
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--rough", action="store_true", help="Use rough estimator mode.")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for umbrella sampling.")
    if not ESTIMATOR_AVAILABLE:
        raise RuntimeError(
            "lattice-estimator unavailable. Add your lattice-estimator repo to ~/lattice-estimator "
            "or edit DEFAULT_REPO_PATH."
        )

    estimator_full = not args.rough

    print("=" * 80)
    print("Experiment 9: umbrella free-energy vs real lattice-estimator")
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
        seed_runs: List[Dict[str, Any]] = []

        for seed in range(args.seeds_per_point):
            try:
                r = run_one_point(
                    n=args.n,
                    q=q,
                    sigma=args.sigma,
                    seed=seed * 101 + int(alpha * 10),
                    n_windows=args.n_windows,
                    n_steps_burn=args.burn_steps,
                    n_steps_prod=args.prod_steps,
                    n_chains=args.n_chains,
                    k_base=args.k_base,
                    T=args.temperature,
                    estimator_full=estimator_full,
                    estimator_jobs=args.jobs,
                )
                seed_runs.append(r)
                sec = r["security"]
                feats = r["features"]
                print(
                    f"  seed {seed}: "
                    f"hard={fmt_bits(sec['overall_bits'])} "
                    f"best={sec['best_family'] or 'n/a'}:{sec['overall_attack'] or 'n/a'} "
                    f"bar={fmt_float(feats['barrier_height'])} "
                    f"ent={fmt_float(feats['entropy'])} "
                    f"rough={fmt_float(feats['roughness'])} "
                    f"peaks={feats['n_peaks']}"
                )
            except Exception as e:
                print(f"  seed {seed}: ERROR -> {e}")

        if not seed_runs:
            continue

        agg = dict(seed_runs[0])
        agg["seed_runs"] = seed_runs

        def mean_key(path: List[str]) -> Optional[float]:
            vals = []
            for r in seed_runs:
                cur: Any = r
                for k in path:
                    cur = cur.get(k) if isinstance(cur, dict) else None
                if cur is not None and np.isfinite(cur):
                    vals.append(float(cur))
            if not vals:
                return None
            return float(np.mean(vals))

        def std_key(path: List[str]) -> Optional[float]:
            vals = []
            for r in seed_runs:
                cur: Any = r
                for k in path:
                    cur = cur.get(k) if isinstance(cur, dict) else None
                if cur is not None and np.isfinite(cur):
                    vals.append(float(cur))
            if len(vals) < 2:
                return 0.0 if vals else None
            return float(np.std(vals))

        agg["security"] = dict(seed_runs[0]["security"])
        agg["security"]["overall_bits"] = mean_key(["security", "overall_bits"])
        agg["security"]["overall_bits_std"] = std_key(["security", "overall_bits"])
        agg["security"]["dominance_gap"] = mean_key(["security", "dominance_gap"])
        agg["features"] = dict(seed_runs[0]["features"])
        for key in ["barrier_height", "entropy", "roughness", "free_energy_span", "peak_separation", "peak_ratio"]:
            agg["features"][key] = mean_key(["features", key])
            agg["features"][f"{key}_std"] = std_key(["features", key])

        peaks_list = [r["features"]["n_peaks"] for r in seed_runs]
        agg["features"]["n_peaks"] = int(round(float(np.mean(peaks_list))))
        agg["features"]["n_peaks_std"] = float(np.std(peaks_list)) if len(peaks_list) > 1 else 0.0

        sec_fams = [r["security"]["best_family"] for r in seed_runs if r["security"]["best_family"] is not None]
        if sec_fams:
            agg["security"]["best_family"] = max(set(sec_fams), key=sec_fams.count)
        sec_attacks = [r["security"]["overall_attack"] for r in seed_runs if r["security"]["overall_attack"] is not None]
        if sec_attacks:
            agg["security"]["overall_attack"] = max(set(sec_attacks), key=sec_attacks.count)

        phase_votes = [r["phase"] for r in seed_runs]
        agg["phase"] = max(set(phase_votes), key=phase_votes.count)

        results.append(agg)

    baseline = {}
    if not args.skip_baseline:
        try:
            baseline = estimate_kyber_baseline(full=estimator_full, jobs=args.jobs)
        except Exception as e:
            print(f"Kyber1024 baseline failed: {e}")
            baseline = {}

    print("\nGenerating outputs...")
    plot_results(results, baseline)
    write_summary(results, baseline)
    save_json(results, baseline)

    hard = np.array([r["security"]["overall_bits"] if r["security"]["overall_bits"] is not None else np.nan for r in results], dtype=np.float64)
    barrier = np.array([r["features"]["barrier_height"] if r["features"]["barrier_height"] is not None else np.nan for r in results], dtype=np.float64)
    entropy = np.array([r["features"]["entropy"] if r["features"]["entropy"] is not None else np.nan for r in results], dtype=np.float64)
    roughness = np.array([r["features"]["roughness"] if r["features"]["roughness"] is not None else np.nan for r in results], dtype=np.float64)

    print("\nCORRELATIONS")
    print("-" * 80)
    print(f"barrier vs hardness: {fmt_float(corrcoef_nan(barrier, hard), 4)}")
    print(f"entropy vs hardness: {fmt_float(corrcoef_nan(entropy, hard), 4)}")
    print(f"roughness vs hardness: {fmt_float(corrcoef_nan(roughness, hard), 4)}")

    elapsed = (time.time() - t0) / 60.0
    print(f"\nTotal runtime: {elapsed:.1f} min")
    print("Outputs: exp9_results.png  exp9_summary.txt  exp9_data.json")


if __name__ == "__main__":
    main()