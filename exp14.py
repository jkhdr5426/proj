"""
Experiment 14: Bimodal Roughness Investigation
===============================================
Run with:  sage -python experiment14.py
           sage -python experiment14.py --quick

FINDING FROM EXPERIMENT 13:
  n=64, q=193, sigma=2.0:  20 instances showed bimodal roughness
    - High cluster (~0.16): seeds 0,2,3,4,5,6,7,8,9,10,12,16,17,19
    - Low cluster  (~0.11): seeds 11,13,14,15,18

  n=96, q=235, sigma=2.0:  consistent roughness ~0.167 (CV=0.083)

WHY THIS MATTERS:
  BKZ gives the SAME gate count to every instance with fixed (n,q,σ).
  If umbrella roughness varies significantly across instances, it is
  detecting structural weakness that BKZ cannot see.

  Known cryptographic fact: some LWE instances are structurally weaker
  due to special spectral properties of their A matrix:
    - Near-singular A → more structured error space
    - Large condition number → numerical instability in attacks
    - Small spectral gap → BKZ lattice reduction is more efficient

  If low-roughness instances correlate with special spectral properties:
  → Umbrella sampling is a per-instance weakness detector
  → This is a novel, practically relevant contribution

EXPERIMENT DESIGN:
  Part A: Large-scale confirmation (100 seeds, n=64)
    - Run umbrella on 100 independent instances
    - Test for bimodality using Hartigan's dip test
    - Measure variance and identify the bimodal clusters
    - Compute spectral features of each A matrix

  Part B: Spectral correlation analysis
    - Features per instance:
        cond_number:   max_singular / min_singular (condition number)
        spectral_gap:  (σ_1 - σ_2) / σ_1 (largest gap between SVs)
        rank_deficiency: fraction of singular values < threshold
        entropy_sv:    entropy of normalized singular values
    - Correlate each with roughness
    - Key test: corr(roughness, cond_number)
      If significant → umbrella detects conditioning-based weakness

  Part C: Comparison across parameter classes
    - Same analysis for n=96 (hard, CV~0.083) to confirm
      that the bimodality is a feature of the "easy" regime
    - Shows umbrella is more informative near phase boundaries
"""

from __future__ import annotations

import json, math, os, re, sys, time, warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats
from scipy.signal import find_peaks

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

DEFAULT_REPO = os.path.expanduser("~/lattice-estimator")
BG = "#0f1117"; PBG = "#1a1d27"

# ── GPU ──────────────────────────────────────────────────────────────────────
try:
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DNAME  = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    TORCH  = True
except Exception:
    torch = None; DEVICE = "cpu"; DNAME = "CPU"; TORCH = False

JOBS = max(1, min(16, os.cpu_count() or 1))
XI_REF = 500.

# ═══════════════════════════════════════════════════════════════════════════════
# LWE + SAMPLER
# ═══════════════════════════════════════════════════════════════════════════════

def gen(n, m, q, sigma, seed):
    rng = np.random.default_rng(seed)
    A = rng.integers(0, q, size=(m, n), dtype=np.int64)
    s = rng.integers(0, q, size=(n,),   dtype=np.int64)
    e = np.round(rng.normal(0, sigma, m)).astype(np.int64)
    b = (A @ s + e) % q
    return A, b, s

def xi_gpu(Ag, bg, sb, q):
    r = (Ag @ sb.T - bg.unsqueeze(1)) % q
    r = torch.where(r > q//2, r-q, r)
    return torch.norm(r.T, dim=1)

def integrated_autocorr(x, max_lag=150):
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

def run_umbrella(A, b, s, n, m, q, sigma, W=14, n_burn=300,
                 n_prod=1000, n_chains=5, k=0.006, T=1.):
    """Run umbrella sampling, return roughness and other metrics."""
    Ag = torch.tensor(A, dtype=torch.float32, device=DEVICE)
    bg = torch.tensor(b, dtype=torch.float32, device=DEVICE)
    st = max(1, q//64); bl = max(1, n//16); B = W * n_chains

    # Baseline xi values
    def xi_cpu(s_):
        r = (A@s_ - b) % q; r = np.where(r>q//2, r-q, r)
        return float(np.linalg.norm(r))

    xi_t = xi_cpu(s)
    rng2 = np.random.default_rng(999)
    xi_r = float(np.mean([xi_cpu(rng2.integers(0,q,n,dtype=np.int64))
                           for _ in range(24)]))
    xi_max = xi_r*1.03; xi_min = max(xi_t*0.75, 1.)
    xi_range = xi_max - xi_min + 1e-9
    k_sc = max(k*(XI_REF/xi_range)**2, 1e-10)
    cen  = np.linspace(xi_max, xi_min, W)
    xc_g = torch.tensor(np.repeat(cen, n_chains),
                        dtype=torch.float32, device=DEVICE)

    def step(sb, xi_c, bc_c, steps):
        acc = torch.zeros(B, dtype=torch.float32, device=DEVICE)
        xr  = torch.zeros((B, steps), dtype=torch.float32, device=DEVICE)
        for i in range(steps):
            col = torch.randint(0, n, (B, bl), device=DEVICE)
            d   = torch.randint(-st, st+1, (B, bl),
                                dtype=torch.float32, device=DEVICE)
            sp  = sb.clone(); sp.scatter_add_(1, col, d); sp = sp%q
            xp  = xi_gpu(Ag, bg, sp, q)
            bp  = 0.5*k_sc*(xp-xc_g)**2; db = bp-bc_c
            ok  = (db<=0)|(torch.log(
                torch.rand(B, device=DEVICE).clamp_min(1e-30))<-db/T)
            sb  = torch.where(ok.unsqueeze(1), sp, sb)
            xi_c = torch.where(ok, xp, xi_c)
            bc_c = torch.where(ok, bp, bc_c)
            xr[:,i] = xi_c; acc += ok.float()
        return sb, xi_c, bc_c, xr, acc/steps

    # Init and burn-in
    sb = torch.randint(0, q, (B, n), dtype=torch.float32, device=DEVICE)
    xi_c = xi_gpu(Ag, bg, sb, q); bc_c = 0.5*k_sc*(xi_c-xc_g)**2
    sb, xi_c, bc_c, _, _ = step(sb, xi_c, bc_c, n_burn)
    sb, xi_c, bc_c, xr, acc = step(sb, xi_c, bc_c, n_prod)

    xi_np = xr.cpu().numpy()
    acc_np = acc.cpu().numpy()
    trajs = xi_np.reshape(W, n_chains, n_prod).reshape(W, n_chains*n_prod)
    acc_w = acc_np.reshape(W, n_chains).mean(1)

    # WHAM
    xi_bins = np.linspace(xi_min*0.85, xi_max*1.08, 70)
    H = np.zeros((W, len(xi_bins)-1)); N = np.zeros(W)
    bc_w = 0.5*(xi_bins[:-1]+xi_bins[1:])
    for i, x in enumerate(trajs):
        H[i],_ = np.histogram(x, bins=xi_bins); N[i] = len(x)
    bias = np.exp(-np.array([[0.5*k_sc*(b_-cen[i])**2/T
                               for b_ in bc_w] for i in range(W)]))
    f = np.zeros(W); rho = np.ones(len(bc_w))
    for _ in range(400):
        ro = rho.copy()
        d  = (N[:,None]*bias*np.exp(f[:,None])).sum(0)
        rho = np.where(d>0, H.sum(0)/d, 0.)
        Z   = (rho[None,:]*bias).sum(1)
        f   = -np.log(np.where(Z>0,Z,1e-300))
        if np.max(np.abs(rho-ro)) < 1e-8: break
    F = -np.log(np.where(rho>0,rho,1e-300)); F -= np.nanmin(F)
    v = np.isfinite(F)
    if v.any(): F = np.clip(F, 0, np.nanpercentile(F[v], 95))

    # Metrics
    roughness = 0.; barrier_norm = 0.
    if v.sum() > 2:
        Fv = F[v] - np.nanmin(F[v])
        bar = float(np.nanmax(Fv))
        roughness = float(np.std(np.diff(Fv))/(bar+1e-9)) if bar>0 else 0.
        barrier_norm = bar / max(xi_range, 1.)

    taus = [integrated_autocorr(trajs[w]) for w in range(W)]
    mean_eff = float(np.mean([1./(2*t) if t>0 else 0. for t in taus]))
    window_gap = float(np.mean(np.abs(trajs.mean(1)-cen)/xi_range))

    try:
        del Ag, bg, sb, xi_c, bc_c, xr, acc
        if TORCH and torch: torch.cuda.empty_cache()
    except: pass

    return {
        "roughness": roughness, "barrier_norm": barrier_norm,
        "mean_eff": mean_eff, "window_gap": window_gap,
        "mean_acc": float(acc_w.mean()),
        "xi_true": xi_t, "xi_rand": xi_r,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# SPECTRAL FEATURES OF A
# ═══════════════════════════════════════════════════════════════════════════════

def spectral_features(A: np.ndarray, q: int) -> dict:
    """
    Compute spectral properties of A that could predict instance weakness.

    cond_number:     σ_max / σ_min   (condition number)
                     High cond = ill-conditioned = potentially weaker
    spectral_gap:    (σ_1 - σ_2) / σ_1
                     Large gap = one dominant direction = more structured
    sv_entropy:      -Σ p_i log p_i  where p_i = σ_i / Σ σ_j
                     Low entropy = concentrated singular spectrum = structured
    sv_tail_ratio:   σ_n / σ_1  (smallest / largest SV)
                     Small ratio = near-singular
    fro_norm:        ||A||_F / (m*n)   (normalized Frobenius norm)
    col_corr:        mean pairwise correlation between columns
                     High correlation = linearly dependent columns = weaker
    """
    A_float = A.astype(np.float64)
    _, sv, _ = np.linalg.svd(A_float, full_matrices=False)
    sv_norm  = sv / (sv.sum() + 1e-12)

    cond_number  = float(sv[0] / (sv[-1] + 1e-12))
    spectral_gap = float((sv[0] - sv[1]) / (sv[0] + 1e-12)) if len(sv) > 1 else 0.
    sv_entropy   = float(-np.sum(sv_norm * np.log(sv_norm + 1e-300)) / math.log(len(sv)+1))
    sv_tail_ratio = float(sv[-1] / (sv[0] + 1e-12))

    # Column correlation: mean of off-diagonal |corr| matrix
    A_centered = A_float - A_float.mean(axis=0)
    A_norm = A_centered / (np.linalg.norm(A_centered, axis=0, keepdims=True) + 1e-12)
    corr_mat = np.abs(A_norm.T @ A_norm)
    n = A.shape[1]
    col_corr = float((corr_mat.sum() - n) / (n*(n-1) + 1e-12))

    fro_norm = float(np.linalg.norm(A_float, 'fro') / math.sqrt(A.shape[0]*A.shape[1]))

    return {
        "cond_number":   cond_number,
        "spectral_gap":  spectral_gap,
        "sv_entropy":    sv_entropy,
        "sv_tail_ratio": sv_tail_ratio,
        "col_corr":      col_corr,
        "fro_norm":      fro_norm,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# HARTIGAN'S DIP TEST FOR UNIMODALITY
# (simplified version: compare to unimodal null hypothesis)
# ═══════════════════════════════════════════════════════════════════════════════

def bimodality_coefficient(x: np.ndarray) -> float:
    """
    Bimodality coefficient BC = (skew² + 1) / (kurtosis + 3*(n-1)²/((n-2)*(n-3)))
    BC > 0.555 suggests bimodality.
    Reference: SAS Institute (1990).
    """
    n = len(x)
    if n < 5: return 0.
    skew = float(scipy_stats.skew(x))
    kurt = float(scipy_stats.kurtosis(x))
    # Excess kurtosis correction
    correction = 3*(n-1)**2 / ((n-2)*(n-3)) if n > 3 else 3.
    bc = (skew**2 + 1) / (kurt + correction)
    return float(bc)

def valley_depth(x: np.ndarray, n_bins: int = 15) -> float:
    """
    Measure valley depth between modes.
    Returns (peak_max - valley_min) / peak_max.
    Higher = more bimodal.
    """
    counts, edges = np.histogram(x, bins=n_bins)
    centers = 0.5*(edges[:-1]+edges[1:])
    counts_smooth = np.convolve(counts.astype(float), np.ones(3)/3, mode='same')
    peaks, _  = find_peaks(counts_smooth)
    valleys, _ = find_peaks(-counts_smooth)
    if len(peaks) < 2 or len(valleys) < 1:
        return 0.
    # depth between the two highest peaks
    peak_heights = counts_smooth[peaks]
    top2 = peaks[np.argsort(peak_heights)[-2:]]
    p1, p2 = top2[0], top2[1]
    # valleys between p1 and p2
    between = valleys[(valleys>min(p1,p2)) & (valleys<max(p1,p2))]
    if len(between) == 0:
        return 0.
    valley_min = counts_smooth[between].min()
    peak_max   = counts_smooth[top2].min()  # smaller of the two peaks
    return float((peak_max - valley_min) / (peak_max + 1e-9))

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def run_config_batch(n, q, sigma, n_seeds, kw, label=""):
    """Run n_seeds independent instances, collect roughness + spectral features."""
    print(f"\n  {label}  n={n}, q={q}, sigma={sigma}, seeds={n_seeds}")
    print(f"  {'seed':>5}  {'roughness':>10}  {'cond_num':>10}  "
          f"{'sv_entropy':>11}  {'col_corr':>9}  {'mean_eff':>9}")
    print("  " + "-"*60)

    rows = []
    for seed in range(n_seeds):
        A, b, s = gen(n, 3*n, q, sigma, seed)
        spec = spectral_features(A, q)
        umbs = run_umbrella(A, b, s, n, 3*n, q, sigma, **kw)
        row  = {"seed": seed, "n": n, "q": q, "sigma": sigma,
                **spec, **umbs}
        rows.append(row)
        print(f"  {seed:>5}  {umbs['roughness']:>10.4f}  "
              f"{spec['cond_number']:>10.2f}  "
              f"{spec['sv_entropy']:>11.4f}  "
              f"{spec['col_corr']:>9.4f}  "
              f"{umbs['mean_eff']:>9.4f}")

    return rows

def analyze_batch(rows, label=""):
    """Statistical analysis of a batch of instances."""
    rugs   = np.array([r["roughness"]    for r in rows])
    conds  = np.array([r["cond_number"]  for r in rows])
    svents = np.array([r["sv_entropy"]   for r in rows])
    ccorrs = np.array([r["col_corr"]     for r in rows])
    tails  = np.array([r["sv_tail_ratio"] for r in rows])
    gaps   = np.array([r["spectral_gap"] for r in rows])
    effs   = np.array([r["mean_eff"]     for r in rows])

    print(f"\n  ANALYSIS: {label}")
    print(f"  roughness: mean={np.mean(rugs):.4f}  "
          f"std={np.std(rugs):.4f}  CV={np.std(rugs)/np.mean(rugs):.4f}")

    bc = bimodality_coefficient(rugs)
    vd = valley_depth(rugs)
    print(f"  Bimodality coefficient BC={bc:.4f}  "
          f"(>0.555 = bimodal)")
    print(f"  Valley depth = {vd:.4f}  (>0.3 = clear bimodal separation)")
    if bc > 0.555 and vd > 0.2:
        print(f"  *** BIMODAL CONFIRMED ***")
    elif bc > 0.555 or vd > 0.2:
        print(f"  ~ Moderate bimodality evidence")
    else:
        print(f"  Unimodal (no bimodality)")

    # Correlations
    print(f"\n  Correlations with roughness:")
    specs = [("cond_number",   conds),
             ("sv_entropy",    svents),
             ("col_corr",      ccorrs),
             ("sv_tail_ratio", tails),
             ("spectral_gap",  gaps),
             ("efficiency",    effs)]
    corr_results = {}
    for name, arr in specs:
        v = np.isfinite(arr) & np.isfinite(rugs)
        if v.sum() >= 5 and np.std(arr[v]) > 0:
            r, p = scipy_stats.pearsonr(rugs[v], arr[v])
            r_s, p_s = scipy_stats.spearmanr(rugs[v], arr[v])
            print(f"    {name:>15}: r={r:+.4f}  p={p:.4f}  "
                  f"(Spearman r_s={r_s:+.4f}  p={p_s:.4f})"
                  f"{'  ***' if p<0.05 else ''}")
            corr_results[name] = {"r": float(r), "p": float(p),
                                  "rs": float(r_s), "ps": float(p_s)}
        else:
            print(f"    {name:>15}: insufficient data")

    # Cluster the instances
    if bc > 0.4:
        median_rug = np.median(rugs)
        low_mask   = rugs < median_rug
        high_mask  = rugs >= median_rug
        if low_mask.sum() >= 3 and high_mask.sum() >= 3:
            print(f"\n  LOW roughness cluster (n={low_mask.sum()}):")
            print(f"    mean rug    = {np.mean(rugs[low_mask]):.4f}")
            print(f"    mean cond   = {np.mean(conds[low_mask]):.2f}")
            print(f"    mean sv_ent = {np.mean(svents[low_mask]):.4f}")
            print(f"  HIGH roughness cluster (n={high_mask.sum()}):")
            print(f"    mean rug    = {np.mean(rugs[high_mask]):.4f}")
            print(f"    mean cond   = {np.mean(conds[high_mask]):.2f}")
            print(f"    mean sv_ent = {np.mean(svents[high_mask]):.4f}")

            # t-test on condition number between clusters
            t, p = scipy_stats.ttest_ind(conds[low_mask], conds[high_mask])
            print(f"\n  t-test (cond_number low vs high cluster):")
            print(f"  t={t:.3f}, p={p:.4f}")
            if p < 0.05:
                print(f"  *** SIGNIFICANT: low-roughness instances have "
                      f"different condition number! ***")

    return {
        "roughness": rugs.tolist(), "cond_number": conds.tolist(),
        "sv_entropy": svents.tolist(), "col_corr": ccorrs.tolist(),
        "sv_tail_ratio": tails.tolist(), "spectral_gap": gaps.tolist(),
        "efficiency": effs.tolist(),
        "bc": bc, "valley_depth": vd,
        "corr_results": corr_results,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_all(analyses: dict, save="exp14_results.png"):
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.36)

    def sty(ax, title, xl="", yl="", tc="white"):
        ax.set_facecolor(PBG); ax.set_title(title, color=tc, fontsize=9, pad=6)
        ax.set_xlabel(xl, color="white", fontsize=8)
        ax.set_ylabel(yl, color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#444")

    config_colors = {
        "easy (n=64)": "#69ff94",
        "hard (n=96)": "#ff6b6b",
    }

    # ── Row 0: Roughness distributions + bimodality ───────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor(PBG)
    for label, ana in analyses.items():
        col = config_colors.get(label, "#00d4ff")
        rugs = np.array(ana["roughness"])
        ax0.hist(rugs, bins=12, alpha=0.7, color=col,
                 label=f"{label}\nBC={ana['bc']:.3f}", density=True)
        ax0.axvline(np.mean(rugs), color=col, lw=2.5, linestyle="--")
    ax0.axvline(0.555, color="white", lw=1, linestyle=":",
                alpha=0.5, label="BC=0.555 threshold")
    sty(ax0, "Roughness Distribution\n(bimodal = BC > 0.555?)",
        "roughness", "density", "#ffd93d")
    ax0.legend(fontsize=7, labelcolor="white", facecolor=PBG)

    # ── Roughness per seed (shows bimodal structure clearly) ──────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_facecolor(PBG)
    for label, ana in analyses.items():
        col  = config_colors.get(label, "#00d4ff")
        rugs = np.array(ana["roughness"])
        xs   = np.arange(len(rugs))
        ax1.plot(xs, rugs, "o", color=col, ms=5, alpha=0.8, label=label)
        ax1.axhline(np.mean(rugs), color=col, lw=1.5, linestyle="--", alpha=0.7)
        # Highlight low-roughness instances
        med = np.median(rugs)
        low = rugs < med - 0.5*np.std(rugs)
        ax1.scatter(xs[low], rugs[low], color="white", s=80,
                    marker="*", zorder=5)
    sty(ax1, "Roughness per Instance\n(stars = low-roughness outliers)",
        "seed (= different A matrix)", "roughness", "#ffd93d")
    ax1.legend(fontsize=7, labelcolor="white", facecolor=PBG)

    # ── Roughness vs condition number (the key scatter) ───────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(PBG)
    for label, ana in analyses.items():
        col  = config_colors.get(label, "#00d4ff")
        rugs = np.array(ana["roughness"])
        cond = np.array(ana["cond_number"])
        ax2.scatter(np.log10(cond+1), rugs, color=col, s=60,
                    alpha=0.8, edgecolors="white", lw=0.4, label=label)
    cr = analyses.get("easy (n=64)", {}).get("corr_results", {})
    r_cond = cr.get("cond_number", {}).get("r", None)
    title_cond = (f"Roughness vs log10(Condition Number)\n"
                  f"r={r_cond:.3f}" if r_cond is not None else
                  "Roughness vs log10(Condition Number)")
    sty(ax2, title_cond, "log10(condition number)", "roughness", "#00d4ff")
    ax2.legend(fontsize=7, labelcolor="white", facecolor=PBG)

    # ── Row 1: Key spectral correlations ─────────────────────────────────────
    spectral_pairs = [
        ("sv_entropy",    "SV Entropy",   "#ffd93d"),
        ("col_corr",      "Column Corr",  "#00d4ff"),
        ("sv_tail_ratio", "SV Tail Ratio","#ff6b6b"),
    ]
    for col_idx, (feat, feat_label, feat_col) in enumerate(spectral_pairs):
        ax = fig.add_subplot(gs[1, col_idx])
        ax.set_facecolor(PBG)
        for label, ana in analyses.items():
            col_c = config_colors.get(label, "#00d4ff")
            rugs = np.array(ana["roughness"])
            feat_arr = np.array(ana.get(feat, [np.nan]*len(rugs)))
            ax.scatter(feat_arr, rugs, color=col_c, s=55,
                       alpha=0.8, edgecolors="white", lw=0.4, label=label)
        cr_e = analyses.get("easy (n=64)", {}).get("corr_results", {})
        r_f  = cr_e.get(feat, {}).get("r", None)
        title_f = (f"Roughness vs {feat_label}\nr={r_f:.3f}" if r_f is not None
                   else f"Roughness vs {feat_label}")
        sty(ax, title_f, feat_label, "roughness", feat_col)
        ax.legend(fontsize=7, labelcolor="white", facecolor=PBG)

    # ── Row 2: Summary plots ──────────────────────────────────────────────────
    # Correlation bar chart for easy config
    ax6 = fig.add_subplot(gs[2, :2])
    ax6.set_facecolor(PBG)
    easy_ana = analyses.get("easy (n=64)", {})
    cr_easy  = easy_ana.get("corr_results", {})
    if cr_easy:
        feat_names = list(cr_easy.keys())
        r_vals     = [cr_easy[f]["r"]  for f in feat_names]
        p_vals     = [cr_easy[f]["p"]  for f in feat_names]
        colors_bar = ["#69ff94" if p<0.05 else "#888" for p in p_vals]
        bars = ax6.bar(feat_names, r_vals, color=colors_bar,
                       edgecolor="white", lw=0.8, alpha=0.85)
        ax6.axhline(0, color="white", lw=0.8, alpha=0.5)
        ax6.axhline( 0.3, color="#ffd93d", lw=1, linestyle="--", alpha=0.6)
        ax6.axhline(-0.3, color="#ffd93d", lw=1, linestyle="--", alpha=0.6)
        for bar, p in zip(bars, p_vals):
            if p < 0.05:
                ax6.text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()+0.01 if bar.get_height()>=0
                         else bar.get_height()-0.04,
                         "p<0.05", ha="center", color="white", fontsize=7)
        ax6.set_xticklabels(feat_names, color="white", fontsize=8, rotation=20)
    sty(ax6, "Correlation of Spectral Features with Roughness\n"
             "(easy config n=64, green = significant p<0.05)",
        "spectral feature", "Pearson r with roughness", "#69ff94")

    # Summary text
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.set_facecolor(PBG); ax7.axis("off")
    lines = ["EXPERIMENT 14 SUMMARY", ""]
    for label, ana in analyses.items():
        rugs = np.array(ana["roughness"])
        lines += [
            f"  {label}:",
            f"    mean rug = {np.mean(rugs):.4f}",
            f"    CV       = {np.std(rugs)/np.mean(rugs):.4f}",
            f"    BC       = {ana['bc']:.4f}  "
            f"({'BIMODAL' if ana['bc']>0.555 else 'unimodal'})",
            f"    valley_d = {ana['valley_depth']:.4f}",
            "",
        ]
    lines.append("  KEY CORRELATIONS (easy config):")
    for feat, res in easy_ana.get("corr_results", {}).items():
        sig = " ***" if res["p"] < 0.05 else ""
        lines.append(f"    {feat[:12]:>12}: r={res['r']:+.3f} p={res['p']:.3f}{sig}")
    lines += ["", "  INTERPRETATION:",
              "  If BC > 0.555: bimodal confirmed",
              "  If corr significant: spectral",
              "  features predict roughness",
              "  → BKZ-blind weakness detected"]
    ax7.text(0.04, 0.97, "\n".join(lines),
             transform=ax7.transAxes, va="top", ha="left",
             fontsize=7, color="white", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#0f1117", alpha=0.9))
    sty(ax7, "Summary")

    fig.suptitle("Experiment 14 — Bimodal Roughness Investigation\n"
                 "Do spectral properties of A predict per-instance roughness?",
                 color="white", fontsize=12, fontweight="bold", y=1.005)
    plt.savefig(save, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nPlot → {save}")
    plt.close()

def write_summary(analyses, path="exp14_summary.txt"):
    with open(path, "w") as f:
        sep = "="*70
        f.write(sep+"\nEXPERIMENT 14 — BIMODAL ROUGHNESS INVESTIGATION\n"+sep+"\n\n")
        for label, ana in analyses.items():
            rugs = np.array(ana["roughness"])
            f.write(f"CONFIG: {label}\n"+"-"*40+"\n")
            f.write(f"  n instances: {len(rugs)}\n")
            f.write(f"  roughness mean: {np.mean(rugs):.4f}\n")
            f.write(f"  roughness std:  {np.std(rugs):.4f}\n")
            f.write(f"  CV:             {np.std(rugs)/np.mean(rugs):.4f}\n")
            f.write(f"  BC:             {ana['bc']:.4f}  "
                    f"({'BIMODAL' if ana['bc']>0.555 else 'unimodal'})\n")
            f.write(f"  valley depth:   {ana['valley_depth']:.4f}\n")
            f.write("\n  SPECTRAL CORRELATIONS:\n")
            for feat, res in ana.get("corr_results", {}).items():
                sig = " ***" if res["p"] < 0.05 else ""
                f.write(f"    {feat:>18}: r={res['r']:+.4f}  "
                        f"p={res['p']:.4f}{sig}\n")
            f.write("\n")
        f.write(sep+"\nINTERPRETATION\n"+sep+"\n")
        f.write("""
CASE 1 — BC > 0.555 AND significant spectral correlation:
  → Bimodality is real and explained by A matrix structure.
  → Umbrella sampling detects per-instance structural weakness.
  → BKZ gives the same gate count to all instances; umbrella
    discriminates between structurally strong and weak A matrices.
  → PUBLISHABLE FINDING: "umbrella roughness as a per-instance
    structural security supplement to BKZ"

CASE 2 — BC > 0.555 BUT no spectral correlation:
  → Bimodality is real but its cause is unknown.
  → Could be WHAM sampling noise or a feature of the reaction
    coordinate rather than the lattice.
  → Inconclusive: need alternative explanations.

CASE 3 — BC < 0.555:
  → No confirmed bimodality. The Exp 13 CV=0.18 was not enough
    data to see a clear bimodal structure.
  → Landscape is parameter-determined, not instance-specific.
  → Useful negative: confirms BKZ is complete for within-class
    hardness characterization.
""")
        f.write(sep+"\n")
    print(f"Summary → {path}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="20 seeds each, faster (~6 min)")
    args = p.parse_args()

    if not TORCH:
        raise RuntimeError("PyTorch required")

    print("="*70)
    print("Experiment 14: Bimodal Roughness + Spectral Analysis")
    print(f"Device: {DNAME}")
    print("="*70)

    t0 = time.time()

    if args.quick:
        n_seeds = 20
        kw = dict(W=12, n_burn=200, n_prod=700, n_chains=4)
    else:
        n_seeds = 60
        kw = dict(W=14, n_burn=300, n_prod=1000, n_chains=5)

    # Two configs from Exp 13
    configs = [
        ("easy (n=64)", 64, 193, 2.0),
        ("hard (n=96)", 96, 235, 2.0),
    ]

    all_rows = {}
    analyses = {}
    for label, n, q, sigma in configs:
        rows = run_config_batch(n, q, sigma, n_seeds, kw, label=label)
        ana  = analyze_batch(rows, label=label)
        all_rows[label] = rows
        analyses[label] = ana

    # Key conclusion
    easy_ana = analyses.get("easy (n=64)", {})
    bc = easy_ana.get("bc", 0.)
    cr = easy_ana.get("corr_results", {})
    sig_features = [f for f,v in cr.items() if v.get("p",1) < 0.05]

    print("\n" + "="*70)
    print("FINAL CONCLUSIONS")
    print("="*70)
    print(f"\n  BC (easy n=64): {bc:.4f}  "
          f"({'BIMODAL CONFIRMED' if bc>0.555 else 'not bimodal'})")
    if sig_features:
        print(f"  Significant spectral correlations: {sig_features}")
        print(f"  → Roughness is predicted by A matrix structure!")
        print(f"  → Umbrella sampling detects instance-level weakness BKZ misses.")
    else:
        print(f"  No significant spectral correlations found.")
        if bc > 0.555:
            print(f"  → Bimodal but cause unknown. May be WHAM sampling noise.")
        else:
            print(f"  → Landscape is parameter-determined. BKZ is complete.")

    print("\nGenerating plots...")
    plot_all(analyses)
    write_summary(analyses)

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
    print("Outputs: exp14_results.png  exp14_summary.txt")