"""
Experiment 10: Attack-Family Transition Sweep

Run with:  sage -python experiment10.py
           sage -python experiment10.py --quick   (fast test)

Core question:
    Does free-energy roughness change when the optimal attack family
    switches from BDD to Dual or DualHybrid?
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
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

DEFAULT_REPO_PATH = os.path.expanduser("~/lattice-estimator")
BG = "#0f1117"; PBG = "#1a1d27"

FAMILY_COLORS = {
    "bdd":         "#69ff94",
    "usvp":        "#00d4ff",
    "dual":        "#ffd93d",
    "dual_hybrid": "#ff6b6b",
    "other":       "#b0b0b0",
}

#  PyTorch 
try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE_NAME = (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else "CPU")
except Exception:
    torch = None
    TORCH_AVAILABLE = False
    DEVICE = "cpu"; DEVICE_NAME = "CPU"

#  Lattice estimator 
if os.path.isdir(DEFAULT_REPO_PATH) and DEFAULT_REPO_PATH not in sys.path:
    sys.path.insert(0, DEFAULT_REPO_PATH)

ESTIMATOR_AVAILABLE = False
LWE = ND = RC = schemes = None

try:
    from estimator import LWE, ND, RC, schemes
    ESTIMATOR_AVAILABLE = True
    RED_COST_MODEL = getattr(RC, "BDGL16", None)
except Exception:
    RED_COST_MODEL = None

DEFAULT_JOBS = max(1, min(16, os.cpu_count() or 1))

# HELPERS

def parse_rop_bits(text):
    m = re.search(r"rop:\s*≈?\s*2\^([0-9]+(?:\.[0-9]+)?)", text)
    return float(m.group(1)) if m else None

def parse_beta(text):
    m = re.search(r"\bβ:\s*([0-9]+(?:\.[0-9]+)?)", text)
    return float(m.group(1)) if m else None

def attack_record(cost_obj):
    t = str(cost_obj)
    return {"rop": parse_rop_bits(t), "beta": parse_beta(t)}

def family_from_name(name):
    n = name.lower().replace("-","_").replace(" ","_")
    if "bdd" in n: return "bdd"
    if "dual_hybrid" in n or "dualhybrid" in n: return "dual_hybrid"
    if "dual" in n: return "dual"
    if "usvp" in n or "primal" in n: return "usvp"
    return "other"

def parse_result(result):
    table = {}
    if hasattr(result, "items"):
        for k, v in result.items():
            table[str(k)] = attack_record(v)
    families = ("bdd", "usvp", "dual", "dual_hybrid")
    fam_best = {f: None for f in families}
    fam_beta = {f: None for f in families}
    for k, rec in table.items():
        rop = rec["rop"]
        if rop is None: continue
        fam = family_from_name(k)
        if fam not in fam_best: continue
        if fam_best[fam] is None or rop < fam_best[fam]:
            fam_best[fam] = rop
            fam_beta[fam] = rec["beta"]
    valid = [(f, v) for f, v in fam_best.items() if v is not None]
    if not valid:
        return {"fam_best": fam_best, "fam_beta": fam_beta,
                "best_family": None, "best_bits": None,
                "best_beta": None, "dominance_gap": None}
    valid_sorted = sorted(valid, key=lambda x: x[1])
    best_family, best_bits = valid_sorted[0]
    best_beta = fam_beta[best_family]
    dominance_gap = (float(valid_sorted[1][1] - valid_sorted[0][1])
                     if len(valid_sorted) >= 2 else None)
    return {"fam_best": fam_best, "fam_beta": fam_beta,
            "best_family": best_family, "best_bits": best_bits,
            "best_beta": best_beta, "dominance_gap": dominance_gap}

def fmt_bits(x): return "n/a" if x is None else f"2^{x:.1f}"
def fmt_f(x, d=3):
    return "n/a" if (x is None or not np.isfinite(float(x))) else f"{float(x):.{d}f}"

def corrcoef_nan(x, y):
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(xa) & np.isfinite(ya)
    if m.sum() < 3 or np.std(xa[m])==0 or np.std(ya[m])==0: return None
    return float(np.corrcoef(xa[m], ya[m])[0,1])

# ESTIMATOR

def run_estimator(n, q, sigma, jobs=DEFAULT_JOBS):
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
    kw = {"jobs": jobs}
    if RED_COST_MODEL: kw["red_cost_model"] = RED_COST_MODEL
    try:
        result = LWE.estimate(params, **kw)
    except TypeError:
        try:
            result = LWE.estimate(params, jobs=jobs)
        except TypeError:
            result = LWE.estimate(params)
    return parse_result(result)

# LWE + GPU SAMPLER

@dataclass
class LWEInstance:
    n: int; m: int; q: int; sigma: float
    A: np.ndarray; b: np.ndarray; s_true: np.ndarray
    A_gpu: Any = None; b_gpu: Any = None

    def to_gpu(self):
        self.A_gpu = torch.tensor(self.A, dtype=torch.float32, device=DEVICE)
        self.b_gpu = torch.tensor(self.b, dtype=torch.float32, device=DEVICE)
        return self

def generate_lwe(n, m, q, sigma, seed=None):
    rng = np.random.default_rng(seed)
    A = rng.integers(0, q, size=(m, n), dtype=np.int64)
    s = rng.integers(0, q, size=(n,),   dtype=np.int64)
    e = np.round(rng.normal(0, sigma, m)).astype(np.int64)
    b = (A @ s + e) % q
    return LWEInstance(n=n, m=m, q=q, sigma=sigma, A=A, b=b, s_true=s)

def res_norm_cpu(lwe, s):
    r = (lwe.A @ s - lwe.b) % lwe.q
    r = np.where(r > lwe.q//2, r-lwe.q, r)
    return float(np.linalg.norm(r))

def res_norm_gpu(lwe, s_batch):
    q = lwe.q
    r = (lwe.A_gpu @ s_batch.T - lwe.b_gpu.unsqueeze(1)) % q
    r = torch.where(r > q//2, r-q, r)
    return torch.norm(r.T, dim=1)

def mh_gpu(lwe, xi_centers, k, T, n_steps, n_chains=6, s_init=None):
    n_win = len(xi_centers); n, q = lwe.n, lwe.q; B = n_win * n_chains
    step_sz = max(1, q//64); blk = max(1, n//16)
    xc = torch.tensor(np.repeat(np.asarray(xi_centers, float), n_chains),
                       dtype=torch.float32, device=DEVICE)
    if s_init is None:
        s_batch = torch.randint(0, q, (B, n), dtype=torch.float32, device=DEVICE)
    else:
        noise = torch.randint(-step_sz, step_sz+1, (B, n),
                              dtype=torch.float32, device=DEVICE)
        s_batch = (s_init + noise) % q
    xi_c  = res_norm_gpu(lwe, s_batch)
    bc_c  = 0.5 * k * (xi_c - xc) ** 2
    xi_r  = torch.zeros((B, n_steps), dtype=torch.float32, device=DEVICE)
    acc   = torch.zeros(B, dtype=torch.float32, device=DEVICE)
    for step_i in range(n_steps):
        col = torch.randint(0, n, (B, blk), device=DEVICE)
        d   = torch.randint(-step_sz, step_sz+1, (B, blk),
                            dtype=torch.float32, device=DEVICE)
        sp  = s_batch.clone(); sp.scatter_add_(1, col, d); sp = sp % q
        xi_p = res_norm_gpu(lwe, sp)
        bc_p = 0.5*k*(xi_p-xc)**2; db = bc_p-bc_c
        ok   = (db<=0)|(torch.log(torch.rand(B, device=DEVICE).clamp_min(1e-30))<-db/T)
        s_batch = torch.where(ok.unsqueeze(1), sp, s_batch)
        xi_c = torch.where(ok, xi_p, xi_c)
        bc_c = torch.where(ok, bc_p, bc_c)
        xi_r[:, step_i] = xi_c; acc += ok.float()
    xi_np  = xi_r.cpu().numpy()
    acc_np = (acc/n_steps).cpu().numpy()
    trajs  = xi_np.reshape(n_win, n_chains, n_steps).reshape(n_win, n_chains*n_steps)
    acc_w  = acc_np.reshape(n_win, n_chains).mean(1)
    return trajs, s_batch, acc_w

def wham(window_xis, xi_centers, k, T, xi_bins, n_iter=400, tol=1e-8):
    n_win = len(window_xis); nb = len(xi_bins)-1
    bc    = 0.5*(xi_bins[:-1]+xi_bins[1:])
    H = np.zeros((n_win, nb)); N = np.zeros(n_win)
    for i, xis in enumerate(window_xis):
        H[i], _ = np.histogram(xis, bins=xi_bins); N[i] = len(xis)
    bias = np.exp(-np.array([[0.5*k*(b-xi_centers[i])**2/T
                               for b in bc] for i in range(n_win)]))
    f=np.zeros(n_win); rho=np.ones(nb)
    for _ in range(n_iter):
        rold = rho.copy()
        denom = (N[:,None]*bias*np.exp(f[:,None])).sum(0)
        rho   = np.where(denom>0, H.sum(0)/denom, 0.)
        Z     = (rho[None,:]*bias).sum(1)
        f     = -np.log(np.where(Z>0, Z, 1e-300))
        if np.max(np.abs(rho-rold)) < tol: break
    F = -np.log(np.where(rho>0, rho, 1e-300)); F -= np.nanmin(F)
    v = np.isfinite(F)
    if v.any(): F = np.clip(F, 0, np.nanpercentile(F[v], 95))
    return bc, F

def landscape_features(bc, F):
    v = np.isfinite(F)
    out = dict(barrier=None, roughness=None, entropy=None,
               n_peaks=0, dominance_gap_geom=None)
    if v.sum() < 5: return out
    Fv = F[v] - np.nanmin(F[v])
    rho = np.exp(-Fv); p = rho/rho.sum()
    barrier   = float(np.nanmax(Fv))
    diff      = np.diff(Fv)
    roughness = float(np.std(diff)/(barrier+1e-9)) if len(diff)>1 else 0.
    entropy   = float(-np.sum(p*np.log(p+1e-300))/math.log(len(p)+1e-300))
    ps = np.convolve(p, np.ones(5)/5, mode="same")
    peaks = [i for i in range(1,len(ps)-1)
             if ps[i]>=ps[i-1] and ps[i]>ps[i+1]]
    peaks = sorted(peaks, key=lambda i: ps[i], reverse=True)
    geom_gap = (float(abs(Fv[peaks[1]]-Fv[peaks[0]])) if len(peaks)>=2 else None)
    out.update(barrier=barrier, roughness=roughness, entropy=entropy,
               n_peaks=len(peaks), dominance_gap_geom=geom_gap)
    return out

XI_REF = 500.0

def run_point(n, q, sigma, seed=0, n_win=14, n_burn=300, n_prod=1000,
              n_chains=6, k_base=0.006, T=1., est_jobs=DEFAULT_JOBS):
    lwe   = generate_lwe(n, 3*n, q, sigma, seed).to_gpu()
    alpha = q/(sigma*math.sqrt(n))
    xi_t  = res_norm_cpu(lwe, lwe.s_true)
    rng2  = np.random.default_rng(seed+99999)
    xi_r  = float(np.mean([res_norm_cpu(lwe, rng2.integers(0,q,n,dtype=np.int64))
                            for _ in range(24)]))
    xi_max = xi_r*1.03; xi_min = max(xi_t*0.75, 1.)
    xi_rng = xi_max-xi_min+1e-9
    k_sc   = max(k_base*(XI_REF/xi_rng)**2, 1e-10)
    xcs    = np.linspace(xi_max, xi_min, n_win)
    _, s0, _    = mh_gpu(lwe, xcs, k_sc, T, n_burn, n_chains)
    trajs, _, _ = mh_gpu(lwe, xcs, k_sc, T, n_prod, n_chains, s_init=s0)
    window_gap  = float(np.mean(np.abs(trajs.mean(1)-xcs)/xi_rng))
    xi_bins = np.linspace(xi_min*0.85, xi_max*1.08, 80)
    bc, F   = wham(list(trajs), xcs, k_sc, T, xi_bins)
    feats   = landscape_features(bc, F)
    sec     = run_estimator(n, q, sigma, jobs=est_jobs)
    try:
        del lwe.A_gpu, lwe.b_gpu, s0, trajs
        if TORCH_AVAILABLE and torch is not None:
            torch.cuda.empty_cache()
    except Exception: pass
    return dict(n=n, q=q, sigma=sigma, alpha=alpha,
                window_gap=window_gap,
                bc=bc.tolist(), F=F.tolist(),
                feats=feats, sec=sec, seed=seed)

# SWEEP PARAMETERS

SWEEP_NS     = [32, 48, 64, 96, 128, 192, 256]
SWEEP_SIGMAS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
SWEEP_ALPHA  = 12.0   # fixed alpha, q = alpha * sigma * sqrt(n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs",  type=int, default=DEFAULT_JOBS)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--quick", action="store_true",
                        help="Small sweep for testing")
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")
    if not ESTIMATOR_AVAILABLE:
        raise RuntimeError("lattice-estimator unavailable — "
                           "add repo to ~/lattice-estimator")

    ns     = [48, 96, 192]       if args.quick else SWEEP_NS
    sigmas = [0.5, 1.5, 4.0]    if args.quick else SWEEP_SIGMAS
    n_seeds = 1 if args.quick else args.seeds

    print("="*80)
    print("Experiment 10: Attack-Family Transition Sweep")
    print(f"Device: {DEVICE_NAME}")
    print(f"Sweep: {len(ns)} n × {len(sigmas)} sigma = {len(ns)*len(sigmas)} configs")
    print(f"Fixed alpha = {SWEEP_ALPHA}")
    print("="*80)
    print(f"\n{'n':>5}  {'sigma':>6}  {'q':>6}  {'family':>12}  "
          f"{'bits':>8}  {'dom_gap':>8}  {'roughness':>10}  {'barrier':>8}")
    print("-"*75)

    all_results = []
    t0 = time.time()

    for n in ns:
        for sigma in sigmas:
            q = int(SWEEP_ALPHA * sigma * math.sqrt(n))
            q = max(q, n+2)
            if q%2==0: q+=1

            seed_rows = []
            for seed in range(n_seeds):
                try:
                    r = run_point(n, q, sigma, seed=seed*73+n,
                                  n_win=14, n_burn=250, n_prod=800,
                                  n_chains=5, est_jobs=args.jobs)
                    seed_rows.append(r)
                except Exception as e:
                    print(f"  n={n} sigma={sigma} seed={seed} ERROR: {e}")

            if not seed_rows: continue

            # Aggregate seeds
            def mn(key_path):
                vals = []
                for row in seed_rows:
                    cur = row
                    for k in key_path:
                        cur = cur.get(k) if isinstance(cur, dict) else None
                    if cur is not None and np.isfinite(float(cur)):
                        vals.append(float(cur))
                return float(np.mean(vals)) if vals else None

            agg = dict(seed_rows[0])
            agg["seed_rows"] = seed_rows
            agg["feats"] = {
                "barrier":            mn(["feats","barrier"]),
                "roughness":          mn(["feats","roughness"]),
                "entropy":            mn(["feats","entropy"]),
                "n_peaks":            int(round(mn(["feats","n_peaks"]) or 0)),
                "dominance_gap_geom": mn(["feats","dominance_gap_geom"]),
            }
            fams = [row["sec"]["best_family"] for row in seed_rows
                    if row["sec"]["best_family"]]
            agg["sec"]["best_family"] = (max(set(fams), key=fams.count)
                                          if fams else "other")
            agg["sec"]["best_bits"]      = mn(["sec","best_bits"])
            agg["sec"]["dominance_gap"]  = mn(["sec","dominance_gap"])
            all_results.append(agg)

            fam  = agg["sec"]["best_family"] or "other"
            bits = agg["sec"]["best_bits"]
            dgap = agg["sec"]["dominance_gap"]
            rgh  = agg["feats"]["roughness"]
            bar  = agg["feats"]["barrier"]
            print(f"{n:>5}  {sigma:>6.1f}  {q:>6}  {fam:>12}  "
                  f"{fmt_bits(bits):>8}  {fmt_f(dgap,2):>8}  "
                  f"{fmt_f(rgh,4):>10}  {fmt_f(bar,2):>8}")

    print(f"\n{len(all_results)} configs in {(time.time()-t0)/60:.1f} min")

    #  Analysis 
    families  = [r["sec"]["best_family"] or "other" for r in all_results]
    dom_gaps  = np.array([r["sec"]["dominance_gap"] or np.nan
                           for r in all_results])
    roughness = np.array([r["feats"]["roughness"] or np.nan
                           for r in all_results])
    barrier   = np.array([r["feats"]["barrier"]   or np.nan
                           for r in all_results])
    entropy   = np.array([r["feats"]["entropy"]   or np.nan
                           for r in all_results])
    hardness  = np.array([r["sec"]["best_bits"]   or np.nan
                           for r in all_results])
    near_boundary = np.isfinite(dom_gaps) & (dom_gaps < 10)

    print("\nATTACK FAMILY DISTRIBUTION:")
    for f in ["bdd","usvp","dual","dual_hybrid","other"]:
        count = families.count(f)
        if count > 0:
            print(f"  {f:>14}: {count:>3}  ({100*count/max(len(all_results),1):.0f}%)")

    print(f"\nNear-boundary configs (dominance_gap < 10 bits): {near_boundary.sum()}")

    print("\nCORRELATIONS:")
    print(f"  roughness vs hardness:    {fmt_f(corrcoef_nan(roughness,hardness),4)}")
    print(f"  barrier vs hardness:      {fmt_f(corrcoef_nan(barrier,hardness),4)}")
    print(f"  dom_gap vs roughness:     {fmt_f(corrcoef_nan(dom_gaps,roughness),4)}")

    # KEY TEST
    nb_rough = roughness[near_boundary & np.isfinite(roughness)]
    fb_rough = roughness[(~near_boundary) & np.isfinite(roughness)]
    if len(nb_rough) >= 2 and len(fb_rough) >= 2:
        t_stat, p_val = scipy_stats.ttest_ind(nb_rough, fb_rough)
        print(f"\nKEY TEST: roughness near vs far boundary:")
        print(f"  near-boundary mean: {np.mean(nb_rough):.4f} (n={len(nb_rough)})")
        print(f"  far-boundary mean:  {np.mean(fb_rough):.4f} (n={len(fb_rough)})")
        print(f"  t={t_stat:.3f}, p={p_val:.4f}")
        if p_val < 0.05:
            print("  *** SIGNIFICANT: roughness IS higher near attack-family boundaries")
            print("  *** This is the publishable result")
        else:
            print("  Not significant — try alpha=5 to force more family diversity")

    #  Plots ─
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    def styled(ax, title, xl="", yl="", tc="white"):
        ax.set_facecolor(PBG)
        ax.set_title(title, color=tc, fontsize=9, pad=6)
        ax.set_xlabel(xl, color="white", fontsize=8)
        ax.set_ylabel(yl, color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#444")

    # 1. Family phase map
    ax0 = fig.add_subplot(gs[0, :2])
    ax0.set_facecolor(PBG)
    for r in all_results:
        fam  = r["sec"]["best_family"] or "other"
        col  = FAMILY_COLORS.get(fam, "#888")
        dgap = r["sec"]["dominance_gap"]
        size = 280 if (dgap is not None and dgap < 10) else 130
        ax0.scatter(r["n"], r["sigma"], color=col, s=size,
                    edgecolors="white", lw=(2 if size>200 else 0.5))
        ax0.text(r["n"], r["sigma"], fam[:3].upper(),
                 ha="center", va="center", color="black",
                 fontsize=7, fontweight="bold")
    styled(ax0,
           f"Attack-Family Phase Map (n vs sigma, fixed α={SWEEP_ALPHA})\n"
           "Large marker = near boundary (dom_gap < 10 bits)",
           "dimension n", "sigma")
    ax0.legend(handles=[Patch(color=FAMILY_COLORS[f], label=f)
                         for f in ["bdd","usvp","dual","dual_hybrid"]],
               fontsize=7, labelcolor="white", facecolor=PBG)

    # 2. Roughness map
    ax1 = fig.add_subplot(gs[0, 2])
    ax1.set_facecolor(PBG)
    for r in all_results:
        rgh = r["feats"]["roughness"]
        if rgh is None: continue
        norm_rgh = min(rgh/0.25, 1.0)
        col = plt.cm.plasma(norm_rgh)
        ax1.scatter(r["n"], r["sigma"], color=col, s=200,
                    edgecolors="white", lw=0.5)
        ax1.text(r["n"], r["sigma"], f"{rgh:.3f}",
                 ha="center", va="center", color="white", fontsize=6)
    styled(ax1, "Roughness Map", "n", "sigma", "#ffd93d")

    # 3. KEY PLOT: roughness vs dominance_gap
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(PBG)
    for r, f in zip(all_results, families):
        dg = r["sec"]["dominance_gap"]
        rg = r["feats"]["roughness"]
        if dg is None or rg is None: continue
        ax2.scatter(dg, rg, color=FAMILY_COLORS.get(f,"#888"),
                    s=80, edgecolors="white", lw=0.5)
    ax2.axvline(10, color="white", lw=1.5, linestyle="--", alpha=0.6,
                label="near-boundary (10 bits)")
    styled(ax2, "KEY: Roughness vs Dominance Gap\n"
           "(gap>>>>0 = BDD and Dual nearly equal)",
           "dominance gap (bits)", "roughness")
    ax2.legend(fontsize=7, labelcolor="white", facecolor=PBG)

    # 4. Roughness vs hardness by family
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(PBG)
    for r, f in zip(all_results, families):
        bits = r["sec"]["best_bits"]
        rgh  = r["feats"]["roughness"]
        if bits is None or rgh is None: continue
        ax3.scatter(bits, rgh, color=FAMILY_COLORS.get(f,"#888"),
                    s=80, edgecolors="white", lw=0.5)
    styled(ax3, "Roughness vs Hardness\n(colored by attack family)",
           "log2(hardness)", "roughness")
    ax3.legend(handles=[Patch(color=FAMILY_COLORS[f], label=f)
                         for f in ["bdd","usvp","dual","dual_hybrid"]],
               fontsize=7, labelcolor="white", facecolor=PBG)

    # 5. F(xi) profiles (near vs far boundary)
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor(PBG)
    nb_res  = sorted([r for r in all_results
                       if r["sec"]["dominance_gap"] is not None
                       and r["sec"]["dominance_gap"] < 15],
                     key=lambda r: r["sec"]["dominance_gap"])
    far_res = [r for r in all_results
                if r["sec"]["dominance_gap"] is not None
                and r["sec"]["dominance_gap"] >= 15]
    for i, r in enumerate(nb_res[:3]):
        bc = np.asarray(r["bc"]); F = np.asarray(r["F"])
        v  = np.isfinite(F)
        if v.sum() < 4: continue
        fam = r["sec"]["best_family"] or "?"
        ax4.plot(bc[v], F[v], lw=2.5, color=plt.cm.plasma(i/4),
                 label=f"n={r['n']} σ={r['sigma']:.1f} [{fam}] "
                       f"gap={r['sec']['dominance_gap']:.1f}b")
    for i, r in enumerate(far_res[:2]):
        bc = np.asarray(r["bc"]); F = np.asarray(r["F"])
        v  = np.isfinite(F)
        if v.sum() < 4: continue
        fam = r["sec"]["best_family"] or "?"
        ax4.plot(bc[v], F[v], lw=1.5, linestyle="--",
                 color=plt.cm.Blues(0.5+0.2*i),
                 label=f"n={r['n']} σ={r['sigma']:.1f} [{fam}] FAR")
    styled(ax4, "F(ξ) Profiles\nsolid=near-boundary, dashed=far",
           "ξ (residual norm)", "F(ξ)")
    ax4.legend(fontsize=6, labelcolor="white", facecolor=PBG)

    fig.suptitle(
        "Experiment 10 — Attack-Family Transition Sweep\n"
        "Does free-energy roughness peak at BDD↔Dual phase boundaries?",
        color="white", fontsize=13, fontweight="bold", y=1.005)
    plt.savefig("exp10_results.png", dpi=150, bbox_inches="tight", facecolor=BG)
    print("\nPlot saved >>>> exp10_results.png")
    plt.close()

    #  Summary file 
    with open("exp10_summary.txt","w") as f:
        f.write("="*70+"\n")
        f.write("EXPERIMENT 10 — ATTACK-FAMILY TRANSITION SWEEP\n")
        f.write("="*70+"\n\n")
        f.write(f"Fixed alpha = {SWEEP_ALPHA}\n")
        f.write(f"n range: {ns}\nsigma range: {sigmas}\n\n")
        f.write("ATTACK FAMILY DISTRIBUTION\n"+"-"*40+"\n")
        for fam in ["bdd","usvp","dual","dual_hybrid","other"]:
            c = families.count(fam)
            if c>0: f.write(f"  {fam}: {c}\n")
        f.write("\nPER-CONFIG RESULTS\n"+"-"*40+"\n")
        f.write(f"{'n':>4}  {'sigma':>6}  {'q':>6}  {'family':>12}  "
                f"{'bits':>8}  {'dom_gap':>8}  {'roughness':>10}\n")
        f.write("-"*65+"\n")
        for r, fam in zip(all_results, families):
            f.write(f"{r['n']:>4}  {r['sigma']:>6.1f}  {r['q']:>6}  "
                    f"{fam:>12}  {fmt_bits(r['sec']['best_bits']):>8}  "
                    f"{fmt_f(r['sec']['dominance_gap'],2):>8}  "
                    f"{fmt_f(r['feats']['roughness'],4):>10}\n")
        f.write("\nCORRELATIONS\n"+"-"*40+"\n")
        f.write(f"roughness vs hardness:   {fmt_f(corrcoef_nan(roughness,hardness),4)}\n")
        f.write(f"dom_gap vs roughness:    {fmt_f(corrcoef_nan(dom_gaps,roughness),4)}\n")
        f.write("\nNEXT STEPS\n"+"-"*40+"\n")
        if near_boundary.sum() == 0:
            f.write("  No near-boundary configs found.\n")
            f.write("  Action: reduce alpha to 5 or 8 and rerun.\n")
        else:
            f.write(f"  {near_boundary.sum()} near-boundary configs found.\n")
            f.write("  Check roughness vs dominance_gap plot.\n")
        f.write("="*70+"\n")
    print("Summary saved >>>> exp10_summary.txt")

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
    print("\nWHAT TO LOOK FOR:")
    print("  1. Does the attack family change across the sweep?")
    print("     if all BDD: reduce alpha to 5 and rerun.")
    print("  2. Is roughness higher at near-boundary configs?")
    print("     if YES (p<0.05): this is your central publishable finding.")
    print("  3. What is corr(dominance_gap, roughness)?")
    print("     Negative = roughness tracks attack competitiveness.")

if __name__ == "__main__":
    main()