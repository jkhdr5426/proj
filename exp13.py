"""
Experiment 13: Per-Instance Hardness Variance + ξ_dual Dip Validation
=============Run with:  sage -python experiment13.py
           sage -python experiment13.py --quick

first partt :) : Per-Instance Hardness Variance
---------------------------------------
BKZ gives ONE hardness number for a given (n, q, σ).
Every instance with those parameters gets the same security label.

Umbrella sampling is instance-specific — it samples a PARTICULAR
lattice, not a parameter class. The free energy landscape F(ξ)
depends on the specific A matrix, not just its statistics.

Question: Do F(ξ) metrics (roughness, mean_eff, τ_int) vary
significantly across instances with the same (n, q, σ)?

if YES: Umbrella sampling detects per-instance weakness that
BKZ cannot see. This is genuinely novel.

if NO: The landscape is parameter-determined, not instance-specific.
Still a useful negative result — confirms BKZ is complete for
within-class hardness.

Protocol:
  - Fix (n=96, q=235, σ=2.0) — from Exp 10, rugged regime
  - Also fix (n=64, q=193, σ=2.0) — easier regime for comparison
  - Run 20 independent seeds each (20 different A matrices)
  - Measure per-instance: roughness, mean_eff, τ_int, window_gap
  - Test: CV = std/mean. CV > 0.15 = significant per-instance variance.
  - Kolmogorov-Smirnov test: are roughness values from hard vs easy
    params drawn from different distributions?

second part :) : ξ_dual Dip Validation
-------------------------------
Experiment 12 showed ξ_d_ratio = 11.74 at sigma=0.90 (the exact
BDD/other boundary), vs 49 and 42 on either side. Could be:
  (a) A real geometric signal — dual vector loses alignment at boundary
  (b) A one-instance artifact — specific to that A matrix and seed

Protocol:
  - Fix n=64, sweep sigma = 0.70, 0.80, 0.85, 0.90, 0.95, 1.00, 1.10
  - Run 10 seeds per sigma value
  - Report mean ± std of ξ_d_ratio
  - if dip is reproducible (std small, dip consistent across seeds):
    real geometric signal >>>> publishable.
  - if high std: artifact >>>> not reproducible, just noise.
"""

from __future__ import annotations

import json, math, os, re, sys, time, warnings
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

DEFAULT_REPO = os.path.expanduser("~/lattice-estimator")
BG = "#0f1117"; PBG = "#1a1d27"

try:
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DNAME  = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    TORCH  = True
except Exception:
    torch = None; DEVICE = "cpu"; DNAME = "CPU"; TORCH = False

if os.path.isdir(DEFAULT_REPO) and DEFAULT_REPO not in sys.path:
    sys.path.insert(0, DEFAULT_REPO)

ESTIMATOR = False; LWE = ND = RC = None
try:
    from estimator import LWE, ND, RC
    ESTIMATOR = True; RCM = getattr(RC, "BDGL16", None)
except Exception:
    RCM = None

JOBS = max(1, min(16, os.cpu_count() or 1))

# CORE: LWE generation, dual vector, samplers

@dataclass
class LWE_:
    n: int; m: int; q: int; sigma: float
    A: np.ndarray; b: np.ndarray; s: np.ndarray
    Ag: Any = None; bg: Any = None
    def gpu(self):
        self.Ag = torch.tensor(self.A, dtype=torch.float32, device=DEVICE)
        self.bg = torch.tensor(self.b, dtype=torch.float32, device=DEVICE)
        return self

def gen(n, m, q, sigma, seed):
    rng = np.random.default_rng(seed)
    A = rng.integers(0, q, size=(m, n), dtype=np.int64)
    s = rng.integers(0, q, size=(n,),   dtype=np.int64)
    e = np.round(rng.normal(0, sigma, m)).astype(np.int64)
    return LWE_(n=n, m=m, q=q, sigma=sigma, A=A, b=(A@s+e)%q, s=s)

def dual_vector(A: np.ndarray) -> np.ndarray:
    """First left null-space vector via SVD (approximate dual vector)."""
    U, _, _ = np.linalg.svd(A.astype(np.float64), full_matrices=True)
    n = A.shape[1]
    v = U[:, n] if U.shape[1] > n else U[:, -1]
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 0 else v

def xi_p(A, b, s, q):
    r = (A@s - b) % q; r = np.where(r>q//2, r-q, r)
    return float(np.linalg.norm(r))

def xi_d(A, b, s, q, v_star):
    r = (A@s - b) % q; r = np.where(r>q//2, r-q, r)
    return float(abs(np.dot(v_star, r.astype(np.float64))))

def xi_gpu_fn(lwe, sb):
    q = lwe.q
    r = (lwe.Ag @ sb.T - lwe.bg.unsqueeze(1)) % q
    r = torch.where(r > q//2, r-q, r)
    return torch.norm(r.T, dim=1)

def integrated_autocorr(x: np.ndarray, max_lag: int = 200) -> float:
    n = len(x)
    if n < 20: return 0.5
    x = x - x.mean(); var = np.var(x)
    if var < 1e-12: return 0.5
    tau = 0.5
    for lag in range(1, min(max_lag, n//4)):
        rho = np.mean(x[:n-lag]*x[lag:]) / var
        tau += rho
        if rho <= 0: break
    return max(tau, 0.5)

XI_REF = 500.

def run_sampler(lwe, W=14, n_burn=300, n_prod=1000, n_chains=5,
                k=0.006, T=1.):
    """Run umbrella sampler and return landscape metrics."""
    n, q = lwe.n, lwe.q
    st = max(1, q//64); bl = max(1, n//16)

    # Baselines
    xi_true  = xi_p(lwe.A, lwe.b, lwe.s, q)
    rng2 = np.random.default_rng(999)
    xi_rand  = float(np.mean([xi_p(lwe.A, lwe.b,
                                   rng2.integers(0,q,n,dtype=np.int64), q)
                               for _ in range(24)]))
    xi_max = xi_rand*1.03; xi_min = max(xi_true*0.75, 1.)
    xi_range = xi_max - xi_min + 1e-9
    k_sc = max(k*(XI_REF/xi_range)**2, 1e-10)
    cen  = np.linspace(xi_max, xi_min, W)
    B    = W * n_chains

    xc_gpu = torch.tensor(np.repeat(cen, n_chains),
                          dtype=torch.float32, device=DEVICE)

    def step(sb, xi_c, bc_c, steps):
        acc = torch.zeros(B, dtype=torch.float32, device=DEVICE)
        xr  = torch.zeros((B, steps), dtype=torch.float32, device=DEVICE)
        for i in range(steps):
            col = torch.randint(0, n, (B, bl), device=DEVICE)
            d   = torch.randint(-st, st+1, (B, bl),
                                dtype=torch.float32, device=DEVICE)
            sp  = sb.clone(); sp.scatter_add_(1, col, d); sp = sp%q
            xi_p_ = xi_gpu_fn(lwe, sp)
            bc_p  = 0.5*k_sc*(xi_p_-xc_gpu)**2; db = bc_p-bc_c
            ok    = (db<=0)|(torch.log(
                torch.rand(B, device=DEVICE).clamp_min(1e-30)) < -db/T)
            sb    = torch.where(ok.unsqueeze(1), sp, sb)
            xi_c  = torch.where(ok, xi_p_, xi_c)
            bc_c  = torch.where(ok, bc_p,  bc_c)
            xr[:,i] = xi_c; acc += ok.float()
        return sb, xi_c, bc_c, xr, (acc/steps)

    # Init + burn-in
    sb = torch.randint(0, q, (B, n), dtype=torch.float32, device=DEVICE)
    xi_c = xi_gpu_fn(lwe, sb); bc_c = 0.5*k_sc*(xi_c-xc_gpu)**2
    sb, xi_c, bc_c, _, _ = step(sb, xi_c, bc_c, n_burn)

    # Production
    sb, xi_c, bc_c, xr, acc_w = step(sb, xi_c, bc_c, n_prod)

    xi_np = xr.cpu().numpy()                         # (B, T)
    acc_np = acc_w.cpu().numpy()                     # (B,)
    trajs  = xi_np.reshape(W, n_chains, n_prod).reshape(W, n_chains*n_prod)
    acc_by_win = acc_np.reshape(W, n_chains).mean(1)

    # WHAM
    xi_bins = np.linspace(xi_min*0.85, xi_max*1.08, 70)
    bc_w, F = _wham(list(trajs), cen, k_sc, T, xi_bins)

    # Metrics
    v = np.isfinite(F)
    roughness = 0.0; barrier_norm = 0.0
    if v.sum() > 2:
        Fv = F[v] - np.nanmin(F[v])
        bar = float(np.nanmax(Fv))
        roughness    = float(np.std(np.diff(Fv))/(bar+1e-9)) if bar>0 else 0.
        barrier_norm = bar / max(xi_range, 1.)

    taus = []
    for w in range(W):
        taus.append(integrated_autocorr(trajs[w]))
    mean_tau = float(np.mean(taus))
    mean_eff = float(np.mean([1./(2*t) if t>0 else 0. for t in taus]))
    window_gap = float(np.mean(np.abs(trajs.mean(1)-cen)/xi_range))

    try:
        del lwe.Ag, lwe.bg, sb, xi_c, bc_c, xr, acc_w
        if TORCH and torch: torch.cuda.empty_cache()
    except: pass

    return dict(
        roughness=roughness, barrier_norm=barrier_norm,
        mean_tau=mean_tau, mean_eff=mean_eff,
        window_gap=window_gap,
        acc=float(acc_by_win.mean()),
        xi_true=xi_true, xi_rand=xi_rand,
        bc=bc_w.tolist(), F=F.tolist(),
    )

def _wham(wxis, centers, k, T, bins, it=400, tol=1e-8):
    W=len(wxis); nb=len(bins)-1
    bc=0.5*(bins[:-1]+bins[1:])
    H=np.zeros((W,nb)); N=np.zeros(W)
    for i,x in enumerate(wxis): H[i],_=np.histogram(x,bins=bins); N[i]=len(x)
    bias=np.exp(-np.array([[0.5*k*(b-centers[i])**2/T
                             for b in bc] for i in range(W)]))
    f=np.zeros(W); rho=np.ones(nb)
    for _ in range(it):
        ro=rho.copy()
        d=(N[:,None]*bias*np.exp(f[:,None])).sum(0)
        rho=np.where(d>0,H.sum(0)/d,0.)
        Z=(rho[None,:]*bias).sum(1)
        f=-np.log(np.where(Z>0,Z,1e-300))
        if np.max(np.abs(rho-ro))<tol: break
    F=-np.log(np.where(rho>0,rho,1e-300)); F-=np.nanmin(F)
    v=np.isfinite(F)
    if v.any(): F=np.clip(F,0,np.nanpercentile(F[v],95))
    return bc,F

def run_dual_metrics(n, q, sigma, seed):
    """Fast: compute ξ_dual ratio without full sampling."""
    lwe   = gen(n, 3*n, q, sigma, seed)
    v_star = dual_vector(lwe.A)
    xi_pt  = xi_p(lwe.A, lwe.b, lwe.s, q)
    xi_dt  = xi_d(lwe.A, lwe.b, lwe.s, q, v_star)
    rng2   = np.random.default_rng(seed+77777)
    xs     = [rng2.integers(0, q, n, dtype=np.int64) for _ in range(30)]
    xi_pr  = float(np.mean([xi_p(lwe.A, lwe.b, s, q) for s in xs]))
    xi_dr  = float(np.mean([xi_d(lwe.A, lwe.b, s, q, v_star) for s in xs]))
    return {
        "xi_p_ratio": xi_pr/(xi_pt+1e-9),
        "xi_d_ratio": xi_dr/(xi_dt+1e-9),
        "xi_p_true":  xi_pt, "xi_d_true": xi_dt,
        "xi_p_rand":  xi_pr, "xi_d_rand":  xi_dr,
    }

# first partt :) : PER-INSTANCE VARIANCE

def part_a(n_seeds=20, kw=None):
    """
    Two parameter sets, many seeds each.
    Hard: n=96, q=235, sigma=2.0  (rugged from Exp 10)
    Easy: n=64, q=193, sigma=2.0  (funnel/BDD)
    """
    if kw is None:
        kw = dict(W=14, n_burn=300, n_prod=1000, n_chains=5)

    configs = [
        ("hard (n=96, rugged)", 96, 235, 2.0, "#ff6b6b"),
        ("easy (n=64, funnel)", 64, 193, 2.0, "#69ff94"),
    ]

    print("\n" + "="*70)
    print("first partt :) : Per-Instance Variance")
    print("="*70)

    results = {}
    for label, n, q, sigma, color in configs:
        print(f"\n  Config: {label}")
        print(f"  {'seed':>5}  {'roughness':>10}  {'mean_eff':>9}  "
              f"{'tau_int':>8}  {'win_gap':>9}")
        print("  " + "-"*48)
        rows = []
        for seed in range(n_seeds):
            lwe = gen(n, 3*n, q, sigma, seed).gpu()
            r   = run_sampler(lwe, **kw)
            rows.append(r)
            print(f"  {seed:>5}  {r['roughness']:>10.4f}  "
                  f"{r['mean_eff']:>9.4f}  {r['mean_tau']:>8.2f}  "
                  f"{r['window_gap']:>9.5f}")

        rugs  = np.array([r['roughness']    for r in rows])
        effs  = np.array([r['mean_eff']     for r in rows])
        taus  = np.array([r['mean_tau']     for r in rows])
        wgaps = np.array([r['window_gap']   for r in rows])

        cv_rug  = float(np.std(rugs)  / (np.mean(rugs)  + 1e-9))
        cv_eff  = float(np.std(effs)  / (np.mean(effs)  + 1e-9))
        cv_tau  = float(np.std(taus)  / (np.mean(taus)  + 1e-9))
        cv_wgap = float(np.std(wgaps) / (np.mean(wgaps) + 1e-9))

        print(f"\n  summmarryyyy for {label}:")
        print(f"    roughness: mean={np.mean(rugs):.4f}  "
              f"std={np.std(rugs):.4f}  CV={cv_rug:.4f}")
        print(f"    efficiency: mean={np.mean(effs):.4f}  "
              f"std={np.std(effs):.4f}  CV={cv_eff:.4f}")
        print(f"    tau_int:   mean={np.mean(taus):.2f}  "
              f"std={np.std(taus):.2f}   CV={cv_tau:.4f}")
        print(f"    win_gap:   mean={np.mean(wgaps):.5f}  "
              f"std={np.std(wgaps):.5f}  CV={cv_wgap:.4f}")

        if cv_rug > 0.15:
            print(f"    *** HIGH per-instance variance (CV={cv_rug:.3f} > 0.15)")
            print(f"    *** Umbrella sampling distinguishes instances BKZ cannot!")
        else:
            print(f"    ~ Low variance (CV={cv_rug:.3f}): landscape is parameter-determined")

        results[label] = {
            "n": n, "q": q, "sigma": sigma, "color": color,
            "rows": rows, "rugs": rugs.tolist(), "effs": effs.tolist(),
            "taus": taus.tolist(), "wgaps": wgaps.tolist(),
            "cv_rug": cv_rug, "cv_eff": cv_eff,
            "cv_tau": cv_tau, "cv_wgap": cv_wgap,
        }

    # KS test: are hard and easy roughness distributions different?
    labels = list(results.keys())
    if len(labels) >= 2:
        r1 = np.array(results[labels[0]]["rugs"])
        r2 = np.array(results[labels[1]]["rugs"])
        ks, p = scipy_stats.ks_2samp(r1, r2)
        print(f"\n  KS test (hard vs easy roughness distributions):")
        print(f"  KS={ks:.4f}, p={p:.4f}")
        if p < 0.05:
            print("  *** SIGNIFICANT: hard and easy instances have ")
            print("  *** statistically different roughness distributions!")
        else:
            print("  Not significant — distributions overlap.")
        results["ks_stat"]  = float(ks)
        results["ks_pvalue"] = float(p)

    return results

# second part :) : ξ_DUAL DIP VALIDATION

def part_b(n_seeds=10):
    """
    Validate the ξ_dual dip at sigma=0.90 with multiple seeds.
    if the dip is real, it should be consistent (low std) across seeds.
    """
    sigmas_fine = [0.70, 0.80, 0.85, 0.90, 0.95, 1.00, 1.10, 1.30, 1.50]
    n = 64; alpha = 12.0

    print("\n" + "="*70)
    print("second part :) : ξ_dual Dip Validation (n=64, alpha=12)")
    print("  Is the dip at sigma=0.90 reproducible across seeds?")
    print("="*70)
    print(f"\n  {'sigma':>6}  {'q':>5}  {'ξ_d_ratio_mean':>15}  "
          f"{'ξ_d_ratio_std':>14}  {'ξ_p_ratio_mean':>15}  {'CV_d':>6}")
    print("  " + "-"*68)

    all_rows = []
    for sigma in sigmas_fine:
        q = int(alpha * sigma * math.sqrt(n))
        q = max(q, n+2)
        if q%2==0: q+=1

        d_ratios = []; p_ratios = []
        for seed in range(n_seeds):
            try:
                r = run_dual_metrics(n, q, sigma, seed*17+n)
                d_ratios.append(r["xi_d_ratio"])
                p_ratios.append(r["xi_p_ratio"])
            except Exception as e:
                print(f"    ERR sigma={sigma} seed={seed}: {e}")

        if d_ratios:
            d_mean = float(np.mean(d_ratios))
            d_std  = float(np.std(d_ratios))
            p_mean = float(np.mean(p_ratios))
            cv_d   = d_std / (d_mean + 1e-9)
            all_rows.append({
                "sigma": sigma, "q": q,
                "d_mean": d_mean, "d_std": d_std,
                "p_mean": p_mean, "p_std": float(np.std(p_ratios)),
                "cv_d": cv_d, "d_ratios": d_ratios, "p_ratios": p_ratios,
            })
            dip_marker = " ← DIP" if d_mean < 0.5*p_mean else ""
            print(f"  {sigma:>6.2f}  {q:>5}  {d_mean:>15.4f}  "
                  f"{d_std:>14.4f}  {p_mean:>15.4f}  {cv_d:>6.3f}{dip_marker}")

    # Find where dip is
    if all_rows:
        d_means = np.array([r["d_mean"] for r in all_rows])
        sigs    = np.array([r["sigma"]  for r in all_rows])
        dip_idx = int(np.argmin(d_means))
        print(f"\n  Minimum ξ_d_ratio at sigma={sigs[dip_idx]:.2f}: "
              f"{d_means[dip_idx]:.4f}")
        cv_at_dip = all_rows[dip_idx]["cv_d"]
        print(f"  CV at dip point: {cv_at_dip:.4f}")
        if cv_at_dip < 0.20:
            print(f"  *** DIP IS REPRODUCIBLE (CV={cv_at_dip:.3f} < 0.20)")
            print(f"  *** ξ_dual genuinely loses sensitivity near the boundary!")
            print(f"  *** This is a real geometric signal — publishable.")
        elif cv_at_dip < 0.40:
            print(f"  ~ Moderate reproducibility (CV={cv_at_dip:.3f})")
            print(f"  ~ Dip is partially real but noisy.")
        else:
            print(f"  High variance (CV={cv_at_dip:.3f}): dip is not reproducible.")
            print(f"  >>>> The sigma=0.90 dip in Exp 12 was a one-seed artifact.")

    return all_rows


def plot_all(part_a_res, part_b_res, save="exp13_results.png"):
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.36)

    def sty(ax, title, xl="", yl="", tc="white"):
        ax.set_facecolor(PBG)
        ax.set_title(title, color=tc, fontsize=9, pad=6)
        ax.set_xlabel(xl, color="white", fontsize=8)
        ax.set_ylabel(yl, color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#444")

    # ── first partt :) : roughness distributions ──────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor(PBG)
    labels = [k for k in part_a_res if k not in ("ks_stat","ks_pvalue")]
    for label in labels:
        d = part_a_res[label]
        rugs = d["rugs"]
        col  = d["color"]
        ax0.hist(rugs, bins=8, alpha=0.7, color=col, label=label, density=True)
        ax0.axvline(float(np.mean(rugs)), color=col, lw=2.5, linestyle="--")
    sty(ax0, "first partt :)  — Roughness Distribution\n(20 instances per config)",
        "roughness", "density", "#ffd93d")
    ax0.legend(fontsize=7, labelcolor="white", facecolor=PBG)
    if "ks_pvalue" in part_a_res:
        p = part_a_res["ks_pvalue"]
        ax0.text(0.05, 0.96, f"KS p={p:.3f}",
                 transform=ax0.transAxes, color="white",
                 fontsize=8, va="top",
                 fontweight="bold" if p<0.05 else "normal")

    # ── Efficiency distribution ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_facecolor(PBG)
    for label in labels:
        d = part_a_res[label]
        ax1.hist(d["effs"], bins=8, alpha=0.7, color=d["color"],
                 label=label, density=True)
        ax1.axvline(float(np.mean(d["effs"])), color=d["color"],
                    lw=2.5, linestyle="--")
    sty(ax1, "first partt :)  — Efficiency Distribution\n(ESS/n_steps per instance)",
        "efficiency", "density", "#69ff94")
    ax1.legend(fontsize=7, labelcolor="white", facecolor=PBG)

    # ── Per-seed roughness vs seed index ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(PBG)
    for label in labels:
        d = part_a_res[label]
        rugs = d["rugs"]
        xs   = np.arange(len(rugs))
        ax2.plot(xs, rugs, "o-", color=d["color"], lw=1.8, ms=4,
                 alpha=0.85, label=f"{label}\nCV={d['cv_rug']:.3f}")
        ax2.axhline(float(np.mean(rugs)), color=d["color"],
                    lw=1.5, linestyle="--", alpha=0.6)
    sty(ax2, "first partt :)  — Roughness per Instance\n(seed index = different A matrix)",
        "seed (different instance)", "roughness", "#ffd93d")
    ax2.legend(fontsize=7, labelcolor="white", facecolor=PBG)

    # ── second part :) : ξ_dual ratio vs sigma ────────────────────────────────────────
    if part_b_res:
        sigs   = [r["sigma"]  for r in part_b_res]
        d_mean = [r["d_mean"] for r in part_b_res]
        d_std  = [r["d_std"]  for r in part_b_res]
        p_mean = [r["p_mean"] for r in part_b_res]
        p_std  = [r["p_std"]  for r in part_b_res]

        ax3 = fig.add_subplot(gs[1, :2])
        ax3.set_facecolor(PBG)
        ax3.errorbar(sigs, d_mean, yerr=d_std, fmt="o-", color="#ffd93d",
                     lw=2.5, ms=7, capsize=4, elinewidth=1.5,
                     label="ξ_dual ratio (mean ± std)")
        ax3.errorbar(sigs, p_mean, yerr=p_std, fmt="s--", color="#69ff94",
                     lw=2, ms=6, capsize=4, elinewidth=1.2,
                     label="ξ_primal ratio (mean ± std)")
        ax3.fill_between(sigs,
                         [m-s for m,s in zip(d_mean, d_std)],
                         [m+s for m,s in zip(d_mean, d_std)],
                         alpha=0.2, color="#ffd93d")
        # Mark minimum
        d_arr = np.array(d_mean)
        dip_i = int(np.argmin(d_arr))
        ax3.scatter([sigs[dip_i]], [d_arr[dip_i]], color="white",
                    s=200, marker="*", zorder=5,
                    label=f"dip at σ={sigs[dip_i]:.2f}")
        ax3.axvline(sigs[dip_i], color="white", lw=1.5,
                    linestyle=":", alpha=0.6)
        sty(ax3,
            "second part :)  — ξ_dual Ratio vs sigma (n=64, alpha=12)\n"
            "10 seeds per sigma — is the dip at sigma=0.90 reproducible?",
            "sigma", "ratio (xi_random / xi_true)", "#ffd93d")
        ax3.legend(fontsize=7, labelcolor="white", facecolor=PBG)

        # ── CV of dip ─────────────────────────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.set_facecolor(PBG)
        cvs_d = [r["cv_d"] for r in part_b_res]
        bars  = ax4.bar(range(len(sigs)), cvs_d, color="#00d4ff",
                        edgecolor="white", lw=0.8, alpha=0.85)
        ax4.axhline(0.20, color="#ffd93d", lw=2, linestyle="--",
                    label="CV=0.20 (reproducible threshold)")
        ax4.axhline(0.40, color="#ff6b6b", lw=1.5, linestyle=":",
                    label="CV=0.40 (noisy threshold)")
        ax4.set_xticks(range(len(sigs)))
        ax4.set_xticklabels([f"{s:.2f}" for s in sigs],
                            color="white", fontsize=7, rotation=30)
        for bar, cv in zip(bars, cvs_d):
            ax4.text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.005,
                     f"{cv:.2f}", ha="center", color="white", fontsize=7)
        sty(ax4, "second part :)  — CV of ξ_dual at each sigma\n"
                 "(low CV = reproducible = real signal)",
            "sigma", "CV = std/mean", "#00d4ff")
        ax4.legend(fontsize=7, labelcolor="white", facecolor=PBG)

    fig.suptitle("Experiment 13 — Per-Instance Variance + ξ_dual Dip Validation\n"
                 "Does umbrella sampling see differences BKZ cannot?  "
                 "Is the ξ_dual boundary signal real?",
                 color="white", fontsize=12, fontweight="bold", y=1.005)
    plt.savefig(save, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Plot >>>> {save}")
    plt.close()

# summmarryyyy FILE

def write_summary(part_a_res, part_b_res, path="exp13_summary.txt"):
    with open(path, "w") as f:
        sep = "="*70
        f.write(sep+"\n")
        f.write("EXPERIMENT 13 — PER-INSTANCE VARIANCE + ξ_DUAL VALIDATION\n")
        f.write(sep+"\n\n")

        f.write("first partt :) : PER-INSTANCE VARIANCE\n"+"-"*40+"\n")
        labels = [k for k in part_a_res if k not in ("ks_stat","ks_pvalue")]
        for label in labels:
            d = part_a_res[label]
            f.write(f"\n  Config: {label}\n")
            f.write(f"  n={d['n']}, q={d['q']}, sigma={d['sigma']}\n")
            f.write(f"  roughness:  mean={np.mean(d['rugs']):.4f}  "
                    f"std={np.std(d['rugs']):.4f}  CV={d['cv_rug']:.4f}\n")
            f.write(f"  efficiency: mean={np.mean(d['effs']):.4f}  "
                    f"std={np.std(d['effs']):.4f}  CV={d['cv_eff']:.4f}\n")
            f.write(f"  tau_int:    mean={np.mean(d['taus']):.2f}  "
                    f"std={np.std(d['taus']):.2f}  CV={d['cv_tau']:.4f}\n")
            verdict = ("SIGNIFICANT per-instance variance" if d['cv_rug'] > 0.15
                       else "Low variance — parameter-determined landscape")
            f.write(f"  Verdict: {verdict}\n")

        if "ks_pvalue" in part_a_res:
            f.write(f"\n  KS test: stat={part_a_res['ks_stat']:.4f}, "
                    f"p={part_a_res['ks_pvalue']:.4f}\n")
            f.write("  " + ("SIGNIFICANT" if part_a_res['ks_pvalue']<0.05
                            else "Not significant") + "\n")

        f.write("\nsecond part :) : ξ_DUAL DIP VALIDATION\n"+"-"*40+"\n")
        if part_b_res:
            f.write(f"{'sigma':>6}  {'d_ratio_mean':>13}  {'d_ratio_std':>12}  "
                    f"{'CV_d':>6}  {'reproducible':>13}\n")
            f.write("-"*60+"\n")
            for r in part_b_res:
                repro = "YES" if r["cv_d"] < 0.20 else ("partial" if r["cv_d"]<0.40 else "NO")
                f.write(f"{r['sigma']:>6.2f}  {r['d_mean']:>13.4f}  "
                        f"{r['d_std']:>12.4f}  {r['cv_d']:>6.3f}  {repro:>13}\n")

            d_arr = np.array([r["d_mean"] for r in part_b_res])
            sigs  = [r["sigma"] for r in part_b_res]
            dip_i = int(np.argmin(d_arr))
            f.write(f"\n  Dip at sigma={sigs[dip_i]:.2f}, "
                    f"CV={part_b_res[dip_i]['cv_d']:.4f}\n")
            f.write(f"  Verdict: {'REPRODUCIBLE — real geometric signal' if part_b_res[dip_i]['cv_d']<0.20 else 'NOT reproducible — artifact'}\n")

        f.write("\n"+sep+"\n")
        f.write("PAPER IMPLICATIONS\n"+sep+"\n")
        f.write("""
1. if Part A CV > 0.15 (significant per-instance variance):
   >>>> Umbrella sampling detects within-class instance weakness
   >>>> BKZ is blind to this: same gate count for all instances
   >>>> Claim: umbrella rugosity as a per-instance security supplement

2. if Part B ξ_dual dip CV < 0.20 (reproducible):
   >>>> The dual reaction coordinate genuinely loses sensitivity
     at the BDD/USVP family boundary
   >>>> Claim: ξ_dual tracks attack-family geometry

3. if both are negative:
   >>>> Landscape is fully determined by (n,q,σ) parameters
   >>>> Umbrella sampling is a parameter-level tool, not instance-level
   >>>> Still useful: confirms BKZ completeness for within-class hardness
""")
        f.write(sep+"\n")
    print(f"Summary >>>> {path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="5 seeds each, fewer sigma values (~8 min)")
    args = p.parse_args()

    if not TORCH:
        raise RuntimeError("PyTorch required")

    print("="*70)
    print("Experiment 13: Per-Instance Variance + ξ_dual Dip Validation")
    print(f"Device: {DNAME}")
    print("="*70)

    t0 = time.time()

    if args.quick:
        kw = dict(W=10, n_burn=150, n_prod=500, n_chains=4)
        n_seeds_a = 5
        n_seeds_b = 5
    else:
        kw = dict(W=14, n_burn=300, n_prod=1000, n_chains=5)
        n_seeds_a = 20
        n_seeds_b = 10

    part_a_res = part_a(n_seeds=n_seeds_a, kw=kw)
    part_b_res = part_b(n_seeds=n_seeds_b)

    print("\nplots generating.....")
    plot_all(part_a_res, part_b_res)
    write_summary(part_a_res, part_b_res)

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
    print("\nKEY OUTCOMES TO LOOK FOR:")
    print("  Part A CV > 0.15  >>>> per-instance variance is real")
    print("  Part B CV < 0.20 at dip >>>> ξ_dual signal is reproducible")
    print("  Either of these = publishable novel contribution")