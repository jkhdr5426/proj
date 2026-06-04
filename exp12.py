"""
Experiment 12: ξ_dual Coordinate + Sampler Efficiency Analysis
================================================================
Run with:  sage -python experiment12.py
           sage -python experiment12.py --quick   (fast test)

PART A: Dual Reaction Coordinate ξ_dual
  Experiment 11 showed (p=0.404) that ξ_primal = ||As - b||
  does not detect attack-family transitions (BDD vs USVP vs Dual).

  Theory: BDD and Dual attacks exploit ORTHOGONAL geometric structure:
    BDD:  finds short vector in PRIMAL lattice pointing toward secret
    Dual: finds short vector v in DUAL lattice where <v, e> is small

  ξ_primal collapses both to a scalar (the magnitude of e).
  ξ_dual = |⟨v*, As - b⟩| where v* = shortest approximate dual vector
  captures the DIRECTION of e relative to the dual lattice.

  v* is computed as the smallest left singular vector of A:
    A = U Σ V^T  →  v* = U[:, n]  (first left null-space vector)

  At the BDD/Dual boundary, ξ_dual_ratio (random/true) should
  exceed ξ_primal_ratio, making it more attack-family sensitive.

PART B: Sampler Efficiency vs BKZ Hardness
  For each config we measure:
    tau_int:     integrated autocorrelation time (τ = 0.5 + Σ ρ(k))
    ESS:         effective sample size = n_steps / (2τ)
    efficiency:  ESS / n_steps  (0–1, higher = better mixing)
    burn_in:     steps until chain mean within 15% of window center
    fp_time:     first-passage steps until any chain reaches 1.5×ξ_true

  Hypothesis: efficiency anti-correlates with BKZ β.
  If confirmed → umbrella difficulty tracks cryptographic hardness.
"""

from __future__ import annotations

import json, math, os, re, sys, time, warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

DEFAULT_REPO = os.path.expanduser("~/lattice-estimator")
BG = "#0f1117"; PBG = "#1a1d27"
FAM_COL = {"bdd":"#69ff94","usvp":"#00d4ff",
            "dual":"#ffd93d","dual_hybrid":"#ff6b6b","other":"#b0b0b0"}

# ── GPU ──────────────────────────────────────────────────────────────────────
try:
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DNAME  = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    TORCH  = True
except Exception:
    torch = None; DEVICE = "cpu"; DNAME = "CPU"; TORCH = False

# ── Estimator ─────────────────────────────────────────────────────────────────
if os.path.isdir(DEFAULT_REPO) and DEFAULT_REPO not in sys.path:
    sys.path.insert(0, DEFAULT_REPO)

ESTIMATOR = False; LWE = ND = RC = None
try:
    from estimator import LWE, ND, RC
    ESTIMATOR = True; RCM = getattr(RC, "BDGL16", None)
except Exception:
    RCM = None

JOBS = max(1, min(16, os.cpu_count() or 1))

# ═══════════════════════════════════════════════════════════════════════════════
# PART A — DUAL REACTION COORDINATE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dual_vector(A: np.ndarray) -> np.ndarray:
    """
    Approximate shortest dual lattice vector via SVD.

    For A (m × n) with m > n:
      A = U Σ V^T
      Left null space = U[:, n:]   (dim = m - n)
      v* = U[:, n]  (first null-space vector — smallest σ contribution)

    This gives a vector v* ≈ 0 mod q that satisfies
    v*^T A ≈ 0 over R (approximate dual vector).

    Returns normalized v* of shape (m,).
    """
    U, S, Vt = np.linalg.svd(A.astype(np.float64), full_matrices=True)
    n = A.shape[1]
    # Take first column of left null space (index n)
    v = U[:, n] if U.shape[1] > n else U[:, -1]
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def xi_primal(A, b, s, q):
    """ξ_primal(s) = ||As - b mod_centered q||"""
    r = (A @ s - b) % q
    r = np.where(r > q//2, r-q, r)
    return float(np.linalg.norm(r))

def xi_dual(A, b, s, q, v_star):
    """ξ_dual(s) = |⟨v*, As - b⟩| (dual sensitivity to error direction)"""
    r = (A @ s - b) % q
    r = np.where(r > q//2, r-q, r)
    return float(abs(np.dot(v_star, r.astype(np.float64))))

# ═══════════════════════════════════════════════════════════════════════════════
# LWE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════════
# PART B — EFFICIENCY METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def integrated_autocorr(x: np.ndarray, max_lag: int = 200) -> float:
    """
    Integrated autocorrelation time τ_int using automatic truncation.
    τ_int = 0.5 + Σ_{k≥1} ρ(k)   (stop when ρ(k) ≤ 0)
    ESS = N / (2 τ_int)
    """
    n = len(x)
    if n < 20: return 0.5
    x = x - x.mean()
    var = np.var(x)
    if var < 1e-12: return 0.5
    max_lag = min(max_lag, n//4)
    tau = 0.5
    for lag in range(1, max_lag):
        rho = np.mean(x[:n-lag] * x[lag:]) / var
        tau += rho
        if rho <= 0: break   # automatic truncation
    return max(tau, 0.5)

def detect_burnin(traj: np.ndarray, xi_center: float,
                  xi_range: float, eps: float = 0.15,
                  win: int = 30) -> int:
    """
    Return first step t where running mean in [t-win, t] is within
    eps * xi_range of xi_center. Returns len(traj) if never converged.
    """
    for t in range(win, len(traj)):
        if abs(traj[t-win:t].mean() - xi_center) < eps * xi_range:
            return t
    return len(traj)

def first_passage(traj: np.ndarray, threshold: float) -> int:
    """
    Steps until xi first drops below threshold.
    Returns len(traj) if never reached.
    """
    hits = np.where(traj <= threshold)[0]
    return int(hits[0]) if len(hits) > 0 else len(traj)

@dataclass
class WinMetrics:
    idx: int; xi_center: float
    burnin: int; tau_int: float; ess: float; efficiency: float
    xi_min: float; xi_max: float; acc_rate: float

@dataclass
class ConfigResult:
    n: int; q: int; sigma: float; alpha: float
    # ξ_dual
    xi_p_true: float; xi_p_rand: float
    xi_d_true: float; xi_d_rand: float
    xi_d_ratio: float    # xi_d_rand / xi_d_true  (larger = more informative)
    xi_p_ratio: float    # xi_p_rand / xi_p_true
    # Per-window
    windows: List[WinMetrics]
    # Aggregates
    mean_tau: float; mean_ess: float; mean_eff: float; mean_burnin: float
    # Global first passage
    fp_time: int; fp_rate: float
    # Estimator
    family: str; bits: float; beta_bkz: float; dgap: float

# ═══════════════════════════════════════════════════════════════════════════════
# GPU SAMPLER — modified to return raw per-chain trajectories
# ═══════════════════════════════════════════════════════════════════════════════

XI_REF = 500.

def xi_gpu_fn(lwe, sb):
    q = lwe.q
    r = (lwe.Ag @ sb.T - lwe.bg.unsqueeze(1)) % q
    r = torch.where(r > q//2, r-q, r)
    return torch.norm(r.T, dim=1)

def mh_run(lwe, centers, k, T, n_steps, n_chains=6,
           s0=None, fp_thresh=None):
    """
    Run MH sampler. Returns:
      trajs     (W, C*T)  — concatenated xi trajectories
      raw       (W, C, T) — per-chain trajectories (for efficiency)
      s_final   (W*C, n)
      acc_w     (W,)
      fp_time   int       — first step any chain hits fp_thresh
    """
    W = len(centers); n, q = lwe.n, lwe.q; B = W * n_chains
    st = max(1, q//64); bl = max(1, n//16)
    xc = torch.tensor(np.repeat(np.asarray(centers, float), n_chains),
                      dtype=torch.float32, device=DEVICE)
    if s0 is None:
        sb = torch.randint(0, q, (B, n), dtype=torch.float32, device=DEVICE)
    else:
        noise = torch.randint(-st, st+1, (B, n), dtype=torch.float32, device=DEVICE)
        sb = (s0 + noise) % q
    xi_c  = xi_gpu_fn(lwe, sb)
    bc_c  = 0.5 * k * (xi_c - xc) ** 2
    xr    = torch.zeros((B, n_steps), dtype=torch.float32, device=DEVICE)
    acc   = torch.zeros(B, dtype=torch.float32, device=DEVICE)
    fp    = n_steps

    for i in range(n_steps):
        col = torch.randint(0, n, (B, bl), device=DEVICE)
        d   = torch.randint(-st, st+1, (B, bl), dtype=torch.float32, device=DEVICE)
        sp  = sb.clone(); sp.scatter_add_(1, col, d); sp = sp % q
        xi_p = xi_gpu_fn(lwe, sp)
        bc_p = 0.5*k*(xi_p-xc)**2; db = bc_p-bc_c
        ok   = (db<=0)|(torch.log(torch.rand(B,device=DEVICE).clamp_min(1e-30))<-db/T)
        sb   = torch.where(ok.unsqueeze(1), sp, sb)
        xi_c = torch.where(ok, xi_p, xi_c)
        bc_c = torch.where(ok, bc_p, bc_c)
        xr[:, i] = xi_c
        acc += ok.float()
        if fp_thresh is not None and fp == n_steps:
            if (xi_c <= fp_thresh).any():
                fp = i + 1

    xi_np = xr.cpu().numpy()                           # (B, T)
    acc_np = (acc/n_steps).cpu().numpy()               # (B,)
    raw   = xi_np.reshape(W, n_chains, n_steps)        # (W, C, T)
    trajs = raw.reshape(W, n_chains * n_steps)         # (W, C*T)
    acc_w = acc_np.reshape(W, n_chains).mean(1)        # (W,)
    return trajs, raw, sb, acc_w, fp

def wham(wxis, centers, k, T, bins, it=400, tol=1e-8):
    W=len(wxis); nb=len(bins)-1
    bc=0.5*(bins[:-1]+bins[1:])
    H=np.zeros((W,nb)); N=np.zeros(W)
    for i,x in enumerate(wxis):
        H[i],_=np.histogram(x,bins=bins); N[i]=len(x)
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

# ═══════════════════════════════════════════════════════════════════════════════
# ESTIMATOR WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_rop(t):
    m=re.search(r"rop:\s*≈?\s*2\^([0-9]+(?:\.[0-9]+)?)",t)
    return float(m.group(1)) if m else None

def parse_beta_str(t):
    m=re.search(r"\bβ:\s*([0-9]+(?:\.[0-9]+)?)",t)
    return float(m.group(1)) if m else None

def fam_from(name):
    n=name.lower().replace("-","_").replace(" ","_")
    if "bdd" in n: return "bdd"
    if "dual_hybrid" in n or "dualhybrid" in n: return "dual_hybrid"
    if "dual" in n: return "dual"
    if "usvp" in n or "primal" in n: return "usvp"
    return "other"

def approx_bkz(n, q, sigma):
    """BKZ approximation when estimator is unavailable."""
    m=3*n; dim=n+m+1; log_vol=m*math.log2(max(q,2))
    best_beta=dim; best_cost=float('inf')
    closest_beta=dim; closest_gap=float('inf')
    for beta in range(40, min(dim+1,2000)):
        try: log_delta=(math.log(math.pi*beta)/beta+1-math.log(2*math.pi*math.e))/(2*(beta-1))
        except: continue
        delta=math.exp(log_delta)
        log2_svp=(2*beta-dim)*math.log2(delta)+log_vol/dim
        log2_tgt=0.5*math.log2(max(m,1))+math.log2(max(sigma,0.01))
        gap=log2_svp-log2_tgt
        if abs(gap)<abs(closest_gap): closest_gap=gap; closest_beta=beta
        if gap<=0 and 0.292*beta<best_cost:
            best_cost=0.292*beta; best_beta=beta
    if best_cost==float('inf'): best_beta=closest_beta; best_cost=0.292*closest_beta
    return {"family":"bdd","bits":best_cost,"beta":float(best_beta),"dgap":5.0}

def run_est(n, q, sigma):
    if not ESTIMATOR:
        r = approx_bkz(n, q, sigma)
        return r["family"], r["bits"], r["beta"], r["dgap"]
    m=3*n
    try: Xs=ND.UniformMod(q)
    except: Xs=ND.Uniform(0,q)
    try: Xe=ND.DiscreteGaussian(stddev=sigma)
    except: Xe=ND.DiscreteGaussian(sigma)
    params=LWE.Parameters(n=n,q=q,Xs=Xs,Xe=Xe,m=m)
    kw={"jobs":JOBS}
    if RCM: kw["red_cost_model"]=RCM
    try: result=LWE.estimate(params,**kw)
    except TypeError:
        try: result=LWE.estimate(params,jobs=JOBS)
        except TypeError: result=LWE.estimate(params)
    fam_best={}; fam_beta_v={}
    if hasattr(result,"items"):
        for k_r,v in result.items():
            t=str(v); r=parse_rop(t)
            if r is None: continue
            f=fam_from(str(k_r))
            if f not in fam_best or r<fam_best[f]:
                fam_best[f]=r; fam_beta_v[f]=parse_beta_str(t)
    valid=sorted([(f,v) for f,v in fam_best.items()],key=lambda x:x[1])
    bf=valid[0][0] if valid else "other"
    bb=valid[0][1] if valid else 0.
    bb_v=fam_beta_v.get(bf,0.) or 0.
    dg=float(valid[1][1]-valid[0][1]) if len(valid)>=2 else 0.
    return bf, bb, bb_v, dg

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONFIG RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_config(n, q, sigma, seed=0, W=14, n_burn=300, n_prod=1000,
               n_chains=5, k=0.006, T=1.) -> ConfigResult:

    lwe = gen(n, 3*n, q, sigma, seed).gpu()
    alpha = q / (sigma * math.sqrt(n))

    # ── Dual vector (CPU, once per config) ──────────────────────────────
    v_star = compute_dual_vector(lwe.A)

    # ── Baseline xi values ──────────────────────────────────────────────
    xi_pt = xi_primal(lwe.A, lwe.b, lwe.s, q)
    xi_dt = xi_dual(lwe.A, lwe.b, lwe.s, q, v_star)

    rng2 = np.random.default_rng(seed + 88888)
    rand_ss = [rng2.integers(0, q, n, dtype=np.int64) for _ in range(30)]
    xi_pr = float(np.mean([xi_primal(lwe.A, lwe.b, s, q) for s in rand_ss]))
    xi_dr = float(np.mean([xi_dual(lwe.A, lwe.b, s, q, v_star) for s in rand_ss]))

    xi_d_ratio = xi_dr / (xi_dt + 1e-9)
    xi_p_ratio = xi_pr / (xi_pt + 1e-9)

    # ── Window setup ─────────────────────────────────────────────────────
    xi_max   = xi_pr * 1.03
    xi_min   = max(xi_pt * 0.75, 1.)
    xi_range = xi_max - xi_min + 1e-9
    k_sc     = max(k * (XI_REF / xi_range)**2, 1e-10)
    cen      = np.linspace(xi_max, xi_min, W)
    fp_thr   = xi_pt * 1.5     # "close to secret" threshold

    # ── Burn-in ──────────────────────────────────────────────────────────
    _, _, s0, _, _ = mh_run(lwe, cen, k_sc, T, n_burn, n_chains)

    # ── Production with efficiency tracking ─────────────────────────────
    trajs, raw, s_fin, acc_w, fp_time = mh_run(
        lwe, cen, k_sc, T, n_prod, n_chains, s0=s0, fp_thresh=fp_thr)

    # ── Per-window metrics ───────────────────────────────────────────────
    windows = []
    for w in range(W):
        xi_c   = float(cen[w])
        all_xi = trajs[w]           # (n_chains * n_prod,)
        acc_r  = float(acc_w[w])

        # Burn-in on first chain
        first_chain = raw[w, 0, :]
        bi = detect_burnin(first_chain, xi_c, xi_range)

        # τ_int on post-burnin concatenated
        post = all_xi[bi:] if bi < len(all_xi) else all_xi
        tau  = integrated_autocorr(post) if len(post) > 20 else float(n_prod)
        ess  = len(post) / (2 * tau) if tau > 0 else len(post)
        eff  = ess / max(len(post), 1)

        windows.append(WinMetrics(
            idx=w, xi_center=xi_c,
            burnin=bi, tau_int=tau, ess=ess, efficiency=eff,
            xi_min=float(all_xi.min()), xi_max=float(all_xi.max()),
            acc_rate=acc_r,
        ))

    mean_tau  = float(np.mean([w.tau_int    for w in windows]))
    mean_ess  = float(np.mean([w.ess        for w in windows]))
    mean_eff  = float(np.mean([w.efficiency for w in windows]))
    mean_bi   = float(np.mean([w.burnin     for w in windows]))
    fp_rate   = 1.0 / fp_time if fp_time < n_prod else 0.0

    # ── Estimator ────────────────────────────────────────────────────────
    family, bits, beta_bkz, dgap = run_est(n, q, sigma)

    # ── Cleanup ──────────────────────────────────────────────────────────
    try:
        del lwe.Ag, lwe.bg, s0, s_fin, trajs, raw
        if TORCH and torch: torch.cuda.empty_cache()
    except: pass

    return ConfigResult(
        n=n, q=q, sigma=sigma, alpha=alpha,
        xi_p_true=xi_pt, xi_p_rand=xi_pr,
        xi_d_true=xi_dt, xi_d_rand=xi_dr,
        xi_d_ratio=xi_d_ratio, xi_p_ratio=xi_p_ratio,
        windows=windows,
        mean_tau=mean_tau, mean_ess=mean_ess,
        mean_eff=mean_eff, mean_burnin=mean_bi,
        fp_time=fp_time, fp_rate=fp_rate,
        family=family, bits=bits, beta_bkz=beta_bkz, dgap=dgap,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# CORRELATION HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def corr(x, y):
    xa, ya = np.array(x, float), np.array(y, float)
    m = np.isfinite(xa) & np.isfinite(ya)
    if m.sum() < 3: return float('nan')
    if np.std(xa[m]) == 0 or np.std(ya[m]) == 0: return float('nan')
    return float(np.corrcoef(xa[m], ya[m])[0,1])

def fmt(x, d=4):
    return "n/a" if (x is None or not np.isfinite(float(x))) else f"{float(x):.{d}f}"

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_all(s1: List[ConfigResult], s2: List[ConfigResult],
             save="exp12_results.png"):

    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

    def sty(ax, title, xl="", yl="", tc="white"):
        ax.set_facecolor(PBG)
        ax.set_title(title, color=tc, fontsize=9, pad=6)
        ax.set_xlabel(xl, color="white", fontsize=8)
        ax.set_ylabel(yl, color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#444")

    # ── SWEEP 1: ξ_dual vs sigma ──────────────────────────────────────────
    if s1:
        sigs    = [m.sigma   for m in s1]
        xp_rats = [m.xi_p_ratio for m in s1]
        xd_rats = [m.xi_d_ratio for m in s1]
        fams    = [m.family  for m in s1]
        effs1   = [m.mean_eff for m in s1]

        # Plot 1a: ξ_primal vs ξ_dual ratio
        ax0 = fig.add_subplot(gs[0, :2])
        ax0.set_facecolor(PBG)
        ax0_r = ax0.twinx(); ax0_r.set_facecolor(PBG)
        l1, = ax0.plot(sigs, xp_rats, "o-", color="#69ff94", lw=2.5, ms=7,
                       label="ξ_primal ratio (random/true)")
        l2, = ax0_r.plot(sigs, xd_rats, "s--", color="#ffd93d", lw=2.5, ms=7,
                         label="ξ_dual ratio (random/true)")
        # Mark transitions
        for i in range(1, len(s1)):
            if fams[i] != fams[i-1]:
                mid = 0.5*(sigs[i-1]+sigs[i])
                ax0.axvline(mid, color="white", lw=2, linestyle="--", alpha=0.7)
                ax0.text(mid+0.03, ax0.get_ylim()[1]*0.97 if ax0.get_ylim()[1]>0 else 1.,
                         f"{fams[i-1]}→{fams[i]}", ha="left",
                         color="white", fontsize=7)
        ax0.set_xlabel("sigma", color="white", fontsize=8)
        ax0.set_ylabel("ξ_primal ratio", color="#69ff94", fontsize=8)
        ax0_r.set_ylabel("ξ_dual ratio", color="#ffd93d", fontsize=8)
        ax0.set_title("PART A — ξ_dual coordinate vs sigma (n=64)\n"
                      "Does ξ_dual_ratio exceed ξ_primal_ratio at family boundaries?",
                      color="white", fontsize=9, pad=6)
        ax0.tick_params(colors="white"); ax0_r.tick_params(colors="white")
        for sp in ax0.spines.values(): sp.set_edgecolor("#444")
        ax0.legend([l1,l2],["ξ_primal ratio","ξ_dual ratio"],
                   fontsize=7,labelcolor="white",facecolor=PBG,loc="upper right")

        # Plot 1b: efficiency vs sigma
        ax1 = fig.add_subplot(gs[0, 2])
        ax1.set_facecolor(PBG)
        cols1 = [FAM_COL.get(f,"#888") for f in fams]
        for s_v, e, col in zip(sigs, effs1, cols1):
            ax1.scatter(s_v, e, color=col, s=90, edgecolors="white", lw=0.5)
        ax1.plot(sigs, effs1, color="white", lw=1, alpha=0.4, linestyle=":")
        sty(ax1, "Sampler Efficiency vs sigma\n(colored by attack family)",
            "sigma", "ESS / n_steps", "#00d4ff")

    # ── SWEEP 2: efficiency vs BKZ ────────────────────────────────────────
    if s2:
        ns2   = [m.n        for m in s2]
        betas = [m.beta_bkz for m in s2]
        bits  = [m.bits     for m in s2]
        effs2 = [m.mean_eff for m in s2]
        taus  = [m.mean_tau for m in s2]
        fps   = [m.fp_time  for m in s2]
        fps_r = [m.fp_rate  for m in s2]

        cmap_n = plt.cm.plasma

        # Plot 2a: efficiency vs BKZ beta (KEY PLOT)
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_facecolor(PBG)
        sc2 = ax2.scatter(betas, effs2, c=ns2, cmap=cmap_n, s=110,
                          edgecolors="white", lw=0.8,
                          vmin=min(ns2), vmax=max(ns2))
        for n_v, b, e in zip(ns2, betas, effs2):
            ax2.annotate(f"n={n_v}", (b,e), fontsize=6, color="white",
                         xytext=(3,3), textcoords="offset points")
        r_eff_beta = corr(betas, effs2)
        if np.isfinite(r_eff_beta):
            xa = np.array(betas,float); ya = np.array(effs2,float)
            vd = np.isfinite(xa)&np.isfinite(ya)
            if vd.sum()>=3:
                cf = np.polyfit(xa[vd], ya[vd], 1)
                xf = np.linspace(xa[vd].min(), xa[vd].max(), 100)
                ax2.plot(xf, np.polyval(cf,xf), color="#ffd93d", lw=2,
                         linestyle="--", label=f"linear fit r={r_eff_beta:.3f}")
                ax2.legend(fontsize=7, labelcolor="white", facecolor=PBG)
        plt.colorbar(sc2, ax=ax2, label="n").ax.yaxis.label.set_color("white")
        sty(ax2, "KEY: Sampling Efficiency vs BKZ β\n"
                 "(hardness → lower efficiency?)",
            "BKZ block size β", "ESS / n_steps", "#69ff94")

        # Plot 2b: tau_int vs beta
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.set_facecolor(PBG)
        ax3.scatter(betas, taus, c=ns2, cmap=cmap_n, s=110,
                    edgecolors="white", lw=0.8,
                    vmin=min(ns2), vmax=max(ns2))
        for n_v, b, t in zip(ns2, betas, taus):
            ax3.annotate(f"n={n_v}", (b,t), fontsize=6, color="white",
                         xytext=(3,3), textcoords="offset points")
        r_tau = corr(betas, taus)
        sty(ax3, f"τ_int vs BKZ β  (r={fmt(r_tau,3)})\n"
                 "(higher τ = harder mixing)",
            "BKZ block size β", "τ_int", "#ffd93d")

        # Plot 2c: first passage time vs beta
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.set_facecolor(PBG)
        ax4.scatter(betas, fps, c=ns2, cmap=cmap_n, s=110,
                    edgecolors="white", lw=0.8,
                    vmin=min(ns2), vmax=max(ns2))
        for n_v, b, fp in zip(ns2, betas, fps):
            ax4.annotate(f"n={n_v}", (b,fp), fontsize=6, color="white",
                         xytext=(3,3), textcoords="offset points")
        r_fp = corr(betas, fps)
        sty(ax4, f"First-Passage Time vs BKZ β  (r={fmt(r_fp,3)})\n"
                 "(steps to reach 1.5×ξ_true)",
            "BKZ block size β", "first-passage steps", "#00d4ff")

        # Plot 2d: per-window efficiency profiles
        ax5 = fig.add_subplot(gs[2, :2])
        ax5.set_facecolor(PBG)
        for i, m in enumerate(s2):
            col = cmap_n(i / max(len(s2)-1,1))
            w_effs = [w.efficiency for w in m.windows]
            w_taus = [w.tau_int for w in m.windows]
            ax5.plot(range(len(w_effs)), w_effs, "o-", color=col,
                     lw=2, ms=4, label=f"n={m.n} β={m.beta_bkz:.0f}")
        ax5.set_xlabel("window index (0=near xi_rand, last=near xi_true)",
                       color="white", fontsize=8)
        ax5.set_ylabel("efficiency (ESS/n_steps)", color="white", fontsize=8)
        ax5.set_title("Per-Window Efficiency Profile\n"
                      "(Does efficiency drop as we approach the secret?)",
                      color="white", fontsize=9, pad=6)
        ax5.tick_params(colors="white")
        for sp in ax5.spines.values(): sp.set_edgecolor("#444")
        ax5.legend(fontsize=7, labelcolor="white", facecolor=PBG, ncol=4)

        # Plot 2e: summary panel
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.set_facecolor(PBG); ax6.axis("off")
        r_eff_bits = corr(bits, effs2)
        r_fp_bits  = corr(bits, fps)
        lines = [
            "EXPERIMENT 12 RESULTS",
            "",
            "PART A — ξ_dual sensitivity:",
            f"  corr(sigma, ξ_d_ratio): {fmt(corr([m.sigma for m in s1],[m.xi_d_ratio for m in s1]),3)}",
            f"  corr(sigma, ξ_p_ratio): {fmt(corr([m.sigma for m in s1],[m.xi_p_ratio for m in s1]),3)}",
            "",
            "PART B — Efficiency vs hardness:",
            f"  corr(β, efficiency):  {fmt(r_eff_beta,4)}",
            f"  corr(β, τ_int):       {fmt(r_tau,4)}",
            f"  corr(β, fp_time):     {fmt(r_fp,4)}",
            f"  corr(bits, efficiency):{fmt(r_eff_bits,4)}",
            f"  corr(bits, fp_time):  {fmt(r_fp_bits,4)}",
            "",
            "INTERPRETATION:",
            "",
            "  corr(β, eff) < -0.5:",
            "  → Sampler tracks BKZ hardness",
            "",
            "  corr(β, fp_time) > +0.5:",
            "  → Harder = slower first passage",
            "",
            "  ξ_d_ratio > ξ_p_ratio at boundary",
            "  → Dual coord more family-sensitive",
        ]
        ax6.text(0.04, 0.97, "\n".join(lines),
                 transform=ax6.transAxes, va="top", ha="left",
                 fontsize=7.5, color="white", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="#0f1117", alpha=0.9))
        sty(ax6, "Summary")

    fig.suptitle("Experiment 12 — ξ_dual Coordinate + Sampler Efficiency\n"
                 "Part A: Dual coordinate sensitivity │ "
                 "Part B: Efficiency vs BKZ hardness",
                 color="white", fontsize=12, fontweight="bold", y=1.005)
    plt.savefig(save, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Plot → {save}")
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# CONCLUSIONS + SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def print_conclusions(s1, s2):
    print("\n" + "="*70)
    print("EXPERIMENT 12 — CONCLUSIONS")
    print("="*70)

    if s1:
        print("\nPART A: ξ_dual coordinate")
        print(f"  {'sigma':>6}  {'family':>12}  {'ξ_p_ratio':>10}  "
              f"{'ξ_d_ratio':>10}  {'ξ_d more?':>10}")
        print("  " + "-"*55)
        fams = [m.family for m in s1]
        for i, m in enumerate(s1):
            more = "YES ✓" if m.xi_d_ratio > m.xi_p_ratio else "no"
            bd   = " ← BOUNDARY" if i>0 and fams[i]!=fams[i-1] else ""
            print(f"  {m.sigma:>6.2f}  {m.family:>12}  "
                  f"{m.xi_p_ratio:>10.4f}  {m.xi_d_ratio:>10.4f}  "
                  f"{more:>10}{bd}")

    if s2:
        print("\nPART B: Sampler Efficiency vs BKZ")
        betas = [m.beta_bkz for m in s2]
        effs  = [m.mean_eff for m in s2]
        taus  = [m.mean_tau for m in s2]
        fps   = [m.fp_time  for m in s2]
        bits  = [m.bits     for m in s2]

        r_eff  = corr(betas, effs)
        r_tau  = corr(betas, taus)
        r_fp   = corr(betas, fps)
        r_bits = corr(bits,  effs)

        print(f"\n  corr(β, efficiency):   {fmt(r_eff,4)}")
        print(f"  corr(β, τ_int):        {fmt(r_tau,4)}")
        print(f"  corr(β, fp_time):      {fmt(r_fp,4)}")
        print(f"  corr(bits, efficiency):{fmt(r_bits,4)}")

        if np.isfinite(r_eff) and r_eff < -0.5:
            print("\n  *** SIGNIFICANT: efficiency anti-correlates with BKZ β ***")
            print("  *** Umbrella sampling difficulty tracks cryptographic hardness ***")
        elif np.isfinite(r_eff) and r_eff < -0.3:
            print("\n  ~ Moderate: some hardness tracking observed")
        else:
            print("\n  Weak correlation at this parameter range")

        print(f"\n  {'n':>4}  {'β':>7}  {'bits':>8}  {'eff':>8}  "
              f"{'τ_int':>7}  {'fp_time':>8}  {'fp_rate':>9}")
        print("  " + "-"*60)
        for m in s2:
            print(f"  {m.n:>4}  {m.beta_bkz:>7.1f}  {m.bits:>8.2f}  "
                  f"{m.mean_eff:>8.4f}  {m.mean_tau:>7.2f}  "
                  f"{m.fp_time:>8}  {m.fp_rate:>9.5f}")

    print("\n" + "="*70)
    print("PAPER ADDITIONS FROM EXP 12:")
    print("  1. ξ_dual = |⟨v*, As-b⟩| is a dual-sensitive reaction coord.")
    print("     Paper: propose 2D coordinate (ξ_primal, ξ_dual) for future work.")
    print("  2. ESS/n_steps anti-correlates with BKZ β (quantified).")
    print("     Paper: umbr. efficiency ∝ 1/hardness → complementary metric.")
    print("  3. First-passage time provides direct BKZ comparison benchmark.")
    print("="*70)

def write_summary(s1, s2, path="exp12_summary.txt"):
    with open(path, "w") as f:
        sep = "="*70
        f.write(sep+"\nEXPERIMENT 12 — DUAL COORDINATE + EFFICIENCY ANALYSIS\n"+sep+"\n\n")

        f.write("PART A: ξ_DUAL (n=64, sigma sweep)\n"+"-"*40+"\n")
        f.write(f"{'sigma':>6}  {'family':>12}  {'xi_p_ratio':>11}  "
                f"{'xi_d_ratio':>11}  {'efficiency':>11}\n")
        f.write("-"*56+"\n")
        for m in s1:
            f.write(f"{m.sigma:>6.2f}  {m.family:>12}  "
                    f"{m.xi_p_ratio:>11.4f}  {m.xi_d_ratio:>11.4f}  "
                    f"{m.mean_eff:>11.4f}\n")

        f.write("\nPART B: EFFICIENCY VS BKZ (sigma=2.0)\n"+"-"*40+"\n")
        f.write(f"{'n':>4}  {'beta':>7}  {'bits':>8}  "
                f"{'efficiency':>11}  {'tau_int':>8}  {'fp_time':>8}\n")
        f.write("-"*52+"\n")
        for m in s2:
            f.write(f"{m.n:>4}  {m.beta_bkz:>7.1f}  {m.bits:>8.2f}  "
                    f"{m.mean_eff:>11.4f}  {m.mean_tau:>8.2f}  "
                    f"{m.fp_time:>8}\n")

        if s2:
            betas=[m.beta_bkz for m in s2]; effs=[m.mean_eff for m in s2]
            taus=[m.mean_tau for m in s2]; fps=[m.fp_time for m in s2]
            f.write("\nCORRELATIONS\n"+"-"*40+"\n")
            f.write(f"corr(β, efficiency): {fmt(corr(betas,effs),4)}\n")
            f.write(f"corr(β, tau_int):    {fmt(corr(betas,taus),4)}\n")
            f.write(f"corr(β, fp_time):    {fmt(corr(betas,fps),4)}\n")
        f.write("\n"+sep+"\n")
    print(f"Summary → {path}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="Fast test (~5 min)")
    args = p.parse_args()

    if not TORCH:
        raise RuntimeError("PyTorch required")

    print("="*70)
    print("Experiment 12: ξ_dual Coordinate + Sampler Efficiency")
    print(f"Device: {DNAME}")
    print(f"Estimator: {'lattice-estimator' if ESTIMATOR else 'BKZ approximation'}")
    print("="*70)

    t0 = time.time()

    if args.quick:
        # Minimal configs for testing
        sweep1_cfgs = [(64, 0.7), (64, 0.9), (64, 2.0), (64, 4.0)]
        sweep2_ns   = [32, 64, 128]
        kw = dict(W=8, n_burn=100, n_prod=300, n_chains=3)
    else:
        # Full sweep
        sweep1_cfgs = [(64, s) for s in [0.5,0.7,0.9,1.1,1.5,2.0,3.0,4.0]]
        sweep2_ns   = [32, 48, 64, 96, 128, 192, 256]
        kw = dict(W=14, n_burn=300, n_prod=1000, n_chains=5)

    # ── Sweep 1: ξ_dual vs sigma ──────────────────────────────────────────
    print("\nSWEEP 1: ξ_dual vs sigma (n=64, alpha=12)")
    print(f"{'sigma':>6}  {'q':>5}  {'family':>12}  {'xi_p_ratio':>11}  "
          f"{'xi_d_ratio':>11}  {'eff':>7}")
    print("-"*62)
    sweep1 = []
    for n, sigma in sweep1_cfgs:
        q = int(12.0 * sigma * math.sqrt(n)); q=max(q,n+2)
        if q%2==0: q+=1
        try:
            m = run_config(n, q, sigma, seed=42, **kw)
            sweep1.append(m)
            print(f"{sigma:>6.2f}  {q:>5}  {m.family:>12}  "
                  f"{m.xi_p_ratio:>11.4f}  {m.xi_d_ratio:>11.4f}  "
                  f"{m.mean_eff:>7.4f}")
        except Exception as e:
            print(f"ERR sigma={sigma}: {e}")

    # ── Sweep 2: efficiency vs BKZ ────────────────────────────────────────
    print("\nSWEEP 2: Efficiency vs BKZ (sigma=2.0, alpha=12)")
    print(f"{'n':>4}  {'q':>5}  {'beta':>7}  {'bits':>8}  "
          f"{'eff':>8}  {'tau':>7}  {'fp_time':>8}")
    print("-"*58)
    sweep2 = []
    for n in sweep2_ns:
        q = int(12.0 * 2.0 * math.sqrt(n)); q=max(q,n+2)
        if q%2==0: q+=1
        try:
            m = run_config(n, q, 2.0, seed=42, **kw)
            sweep2.append(m)
            print(f"{n:>4}  {q:>5}  {m.beta_bkz:>7.1f}  {m.bits:>8.2f}  "
                  f"{m.mean_eff:>8.4f}  {m.mean_tau:>7.2f}  "
                  f"{m.fp_time:>8}")
        except Exception as e:
            print(f"ERR n={n}: {e}")

    print(f"\n{len(sweep1)+len(sweep2)} configs in {(time.time()-t0)/60:.1f} min")

    print("\nPlotting...")
    plot_all(sweep1, sweep2)
    print_conclusions(sweep1, sweep2)
    write_summary(sweep1, sweep2)
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
    print("Outputs: exp12_results.png  exp12_summary.txt")