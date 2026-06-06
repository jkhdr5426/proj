"""
Experiment 4 v3 — Two fixes:

FIX 1 — REPLACE freeze_ratio WITH mean_window_gap
FIX 2 — CORRECT CLUSTER LABELING
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from dataclasses import dataclass
from typing import List
import warnings, time, math
warnings.filterwarnings('ignore')

RNG = np.random.default_rng(2026)
BG  = '#0f1117'
PBG = '#1a1d27'
PHASE_COLORS = {"funnel":"#69ff94","rugged":"#ffd93d","cliff":"#ff6b6b","unknown":"#888"}

# LWE CORE

@dataclass
class LWEInstance:
    n: int; m: int; q: int; sigma: float
    A: np.ndarray; b: np.ndarray; s_true: np.ndarray

def generate_lwe(n, m, q, sigma):
    A = RNG.integers(0, q, size=(m, n))
    s = RNG.integers(0, q, size=(n,))
    e = np.round(RNG.normal(0, sigma, size=(m,))).astype(int)
    b = (A @ s + e) % q
    return LWEInstance(n=n, m=m, q=q, sigma=sigma, A=A, b=b, s_true=s)

def residual_norm(lwe, s):
    r = (lwe.A @ s - lwe.b) % lwe.q
    r = np.where(r > lwe.q // 2, r - lwe.q, r)
    return float(np.linalg.norm(r))

def umbrella_energy(xi, xc, k):
    return 0.5 * k * (xi - xc) ** 2

# SAMPLER

def mh_window(lwe, xi_center, k, T, n_steps, s_init=None, n_chains=3):
    n         = lwe.n
    bs        = max(1, n // 16)
    step_size = max(1, lwe.q // 64)
    all_xis   = []
    best_s    = None
    best_xi   = np.inf
    total_acc = 0
    total_prop = 0

    for _ in range(n_chains):
        if s_init is None:
            s = RNG.integers(0, lwe.q, size=(n,))
        else:
            noise = RNG.integers(-step_size, step_size + 1, size=(n,))
            s = (s_init + noise) % lwe.q

        xi_cur   = residual_norm(lwe, s)
        bias_cur = umbrella_energy(xi_cur, xi_center, k)
        xis = []; acc = 0

        for _ in range(n_steps):
            idxs   = RNG.choice(n, size=bs, replace=False)
            deltas = RNG.integers(-step_size, step_size + 1, size=(bs,))
            s_prop = s.copy()
            s_prop[idxs] = (s_prop[idxs] + deltas) % lwe.q
            xi_prop   = residual_norm(lwe, s_prop)
            bias_prop = umbrella_energy(xi_prop, xi_center, k)
            db = bias_prop - bias_cur
            if db <= 0 or RNG.random() < np.exp(-db / T):
                s, xi_cur, bias_cur = s_prop, xi_prop, bias_prop
                acc += 1
            xis.append(xi_cur)
            total_prop += 1

        total_acc += acc
        all_xis.extend(xis)
        if xi_cur < best_xi:
            best_xi = xi_cur; best_s = s.copy()

    return np.array(all_xis), best_s, total_acc / max(total_prop, 1)

def wham(window_xis, xi_centers, k, T, xi_bins, n_iter=300, tol=1e-7):
    n_win  = len(window_xis)
    n_bins = len(xi_bins) - 1
    bc     = 0.5 * (xi_bins[:-1] + xi_bins[1:])
    H = np.zeros((n_win, n_bins)); N = np.zeros(n_win)
    for i, xis in enumerate(window_xis):
        H[i], _ = np.histogram(xis, bins=xi_bins); N[i] = len(xis)
    bias = np.exp(-np.array([
        [umbrella_energy(b, xi_centers[i], k) / T for b in bc]
        for i in range(n_win)
    ]))
    f   = np.zeros(n_win); rho = np.ones(n_bins)
    for _ in range(n_iter):
        rho_old = rho.copy()
        denom   = (N[:, None] * bias * np.exp(f[:, None])).sum(axis=0)
        rho     = np.where(denom > 0, H.sum(axis=0) / denom, 0.0)
        Z_i     = (rho[None, :] * bias).sum(axis=1)
        f       = -np.log(np.where(Z_i > 0, Z_i, 1e-300))
        if np.max(np.abs(rho - rho_old)) < tol: break
    F = -np.log(np.where(rho > 0, rho, 1e-300)); F -= F.min()
    return bc, F

# RESULT — freeze_ratio replaced by window_gap

@dataclass
class Result:
    n: int; q: int; sigma: float; alpha: float
    xi_true: float; xi_random: float
    coverage:     float
    rugosity:     float
    window_gap:   float   # NEW: mean |mean(xi_traj) - xi_center| / xi_range
    mobility:     float
    bin_centers:  np.ndarray
    F:            np.ndarray
    window_means: List[float]
    window_acc:   List[float]
    bkz_beta:         float = 0.0
    bkz_log2gates:    float = 0.0
    bkz_attack_full:  bool  = False
    cluster_id:       int   = -1
    phase_label:      str   = "unknown"
    bkz_umbrella_gap: bool  = False

def run_config(n, q, sigma,
               n_windows=14, n_steps_burn=400,
               n_steps_prod=1200, k=0.006, T=1.0, n_chains=3):
    m   = 3 * n
    lwe = generate_lwe(n=n, m=m, q=q, sigma=sigma)
    alpha = q / (sigma * np.sqrt(n))

    xi_true   = residual_norm(lwe, lwe.s_true)
    xi_random = np.mean([residual_norm(lwe, RNG.integers(0, q, size=(n,)))
                         for _ in range(60)])

    xi_max     = xi_random * 1.03
    xi_min     = max(xi_true * 0.75, 1.0)
    xi_range   = xi_max - xi_min + 1e-9
    xi_centers = np.linspace(xi_max, xi_min, n_windows)

    window_xis   = []
    window_means = []
    window_acc   = []
    s_cur        = None

    for xc in xi_centers:
        _, s_cur, _ = mh_window(lwe, xc, k, T, n_steps_burn,
                                s_init=s_cur, n_chains=n_chains)
        xis, s_cur, acc = mh_window(lwe, xc, k, T, n_steps_prod,
                                    s_init=s_cur, n_chains=n_chains)
        window_xis.append(xis)
        window_means.append(float(xis.mean()))
        window_acc.append(acc)

    coverage   = 1.0
    window_gap = float(np.mean([
        abs(window_means[i] - xi_centers[i]) / xi_range
        for i in range(n_windows)
    ]))

    xi_bins = np.linspace(xi_min * 0.88, xi_max * 1.05, 60)
    bc, F   = wham(window_xis, xi_centers, k, T, xi_bins)

    valid    = np.isfinite(F)
    rugosity = (float(np.std(np.diff(F[valid]))) / (float(F[valid].max()) + 1e-9)
                if valid.sum() > 2 else 0.0)
    mobility = float(np.mean(window_acc))

    return Result(n=n, q=q, sigma=sigma, alpha=alpha,
                  xi_true=xi_true, xi_random=xi_random,
                  coverage=coverage, rugosity=rugosity,
                  window_gap=window_gap, mobility=mobility,
                  bin_centers=bc, F=F,
                  window_means=window_means, window_acc=window_acc)

# BKZ ESTIMATOR

def bkz_estimate(n, q, sigma, m_factor=3):
    m   = m_factor * n
    dim = n + m + 1
    log_vol = m * math.log2(max(q, 2))
    best_beta = dim; best_cost = float('inf'); attack_full = False
    closest_gap = float('inf'); closest_beta = dim
    for beta in range(40, min(dim + 1, 2000)):
        try:
            log_delta = (math.log(math.pi * beta) / beta
                         + 1 - math.log(2 * math.pi * math.e)) / (2 * (beta - 1))
        except: continue
        delta        = math.exp(log_delta)
        exponent     = (2 * beta - dim) * math.log2(delta)
        log2_svp_len = exponent + log_vol / dim
        log2_target  = 0.5 * math.log2(max(m, 1)) + math.log2(max(sigma, 0.01))
        gap = log2_svp_len - log2_target
        if abs(gap) < abs(closest_gap):
            closest_gap = gap; closest_beta = beta
        if gap <= 0:
            log2_cost = 0.292 * beta
            if log2_cost < best_cost:
                best_cost = log2_cost; best_beta = beta; attack_full = True
    if not attack_full:
        best_beta = closest_beta; best_cost = 0.292 * closest_beta
    return float(best_beta), float(best_cost), attack_full

def attach_bkz(results):
    print("\nComputing BKZ estimates...")
    print(f"  {'n':>4}  {'q':>5}  {'beta':>6}  {'log2gates':>10}  {'full_break':>10}")
    print("  " + "-" * 44)
    for r in results:
        r.bkz_beta, r.bkz_log2gates, r.bkz_attack_full = bkz_estimate(r.n, r.q, r.sigma)
        flag = "YES" if r.bkz_attack_full else "no (partial)"
        print(f"  {r.n:>4}  {r.q:>5}  {r.bkz_beta:>6.0f}  "
              f"{r.bkz_log2gates:>10.2f}  {flag:>10}")

# CLUSTERING — fixed labeling

def cluster_results(results, n_clusters=3):
    X = np.array([[r.rugosity, r.window_gap, r.mobility]
                  for r in results])
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km       = KMeans(n_clusters=n_clusters, n_init=30, random_state=42)
    labels   = km.fit_predict(X_scaled)
    sil      = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0.0
    centroids_orig = scaler.inverse_transform(km.cluster_centers_)

    rug_c = centroids_orig[:, 0]
    mob_c = centroids_orig[:, 2]

    # rugged = highest rugosity
    rugged_id = int(np.argmax(rug_c))
    remaining = [i for i in range(n_clusters) if i != rugged_id]
    # cliff = lowest mobility of remaining
    cliff_id  = remaining[int(np.argmin(mob_c[remaining]))]
    funnel_id = [i for i in remaining if i != cliff_id][0]

    cluster_names = {funnel_id:"funnel", cliff_id:"cliff", rugged_id:"rugged"}

    thresholds = {}
    for cid, name in cluster_names.items():
        mask = labels == cid
        thresholds[name] = {
            'rugosity':   (float(X[mask,0].min()), float(X[mask,0].max())),
            'window_gap': (float(X[mask,1].min()), float(X[mask,1].max())),
            'mobility':   (float(X[mask,2].min()), float(X[mask,2].max())),
            'n_configs':  int(mask.sum()),
        }

    for i, r in enumerate(results):
        r.cluster_id  = int(labels[i])
        r.phase_label = cluster_names[labels[i]]

    return labels, sil, centroids_orig, cluster_names, thresholds

# DISAGREEMENT DETECTION

def find_disagreements(results, bkz_hard_threshold=80.0):
    gaps_strong = []; gaps_soft = []
    for r in results:
        if r.phase_label != "rugged": continue
        if not r.bkz_attack_full:
            r.bkz_umbrella_gap = True; gaps_strong.append(r)
        elif r.bkz_log2gates > bkz_hard_threshold:
            r.bkz_umbrella_gap = True; gaps_soft.append(r)
    return gaps_strong, gaps_soft

def write_disagreements(gaps_strong, gaps_soft, path="lwe_disagreements_v3.txt"):
    with open(path, "w") as f:
        f.write("=" * 68 + "\n")
        f.write("BKZ vs UMBRELLA SAMPLING — DISAGREEMENT REPORT v3\n")
        f.write("=" * 68 + "\n\n")
        f.write(f"STRONG GAPS ({len(gaps_strong)} configs)\n")
        f.write("BKZ cannot fully break AND umbrella finds rugged landscape.\n\n")
        if gaps_strong:
            f.write(f"  {'n':>4}  {'q':>6}  {'alpha':>6}  "
                    f"{'rugosity':>9}  {'window_gap':>11}  "
                    f"{'mobility':>9}  {'log2gates':>10}\n")
            f.write("  " + "-" * 65 + "\n")
            for r in gaps_strong:
                f.write(f"  {r.n:>4}  {r.q:>6}  {r.alpha:>6.2f}  "
                        f"{r.rugosity:>9.4f}  {r.window_gap:>11.5f}  "
                        f"{r.mobility:>9.4f}  {r.bkz_log2gates:>10.2f}\n")
        else:
            f.write("  None found.\n")
        f.write(f"\nSOFT GAPS ({len(gaps_soft)} configs)\n")
        f.write("BKZ breaks but costs > 80 bits AND umbrella finds rugged.\n\n")
        if gaps_soft:
            for r in gaps_soft:
                f.write(f"  n={r.n}  q={r.q}  alpha={r.alpha:.2f}  "
                        f"rug={r.rugosity:.4f}  log2gates={r.bkz_log2gates:.2f}\n")
        else:
            f.write("  None found.\n")
    print(f"Disagreement report saved to {path}")


def plot_results(results, ns, alphas, sil, save="lwe_scale_results_v3.png"):
    fig = plt.figure(figsize=(20, 18))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)
    n_n = len(ns); n_a = len(alphas)

    def grid(attr):
        g = np.full((n_n, n_a), np.nan)
        for r in results:
            try:
                i = ns.index(r.n)
                j = min(range(n_a), key=lambda x: abs(alphas[x] - r.alpha))
                g[i, j] = getattr(r, attr)
            except: pass
        return g

    g_ph  = np.full((n_n, n_a), np.nan)
    g_bkz = grid('bkz_log2gates')
    g_rug = grid('rugosity')
    g_mob = grid('mobility')
    g_gap = grid('window_gap')
    pnum  = {"funnel":0,"rugged":1,"cliff":2}

    for r in results:
        try:
            i = ns.index(r.n)
            j = min(range(n_a), key=lambda x: abs(alphas[x] - r.alpha))
            g_ph[i,j] = pnum.get(r.phase_label, -1)
        except: pass

    xl = [f"{a:.0f}" for a in alphas]
    yl = [str(x) for x in ns]

    def styled(ax, title, xl='', yl='', tc='white'):
        ax.set_facecolor(PBG)
        ax.set_title(title, color=tc, fontsize=9, pad=6)
        ax.set_xlabel(xl, color='white', fontsize=8)
        ax.set_ylabel(yl, color='white', fontsize=8)
        ax.tick_params(colors='white', labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor('#444')

    def hm(ax, data, cmap, title, tc='white', vmin=None, vmax=None):
        im = ax.imshow(data.T, aspect='auto', origin='lower', cmap=cmap,
                       vmin=vmin, vmax=vmax,
                       extent=[-0.5, n_n-0.5, -0.5, n_a-0.5])
        styled(ax, title, 'dimension n', 'alpha', tc)
        ax.set_xticks(range(n_n)); ax.set_xticklabels(yl, fontsize=7)
        ax.set_yticks(range(n_a)); ax.set_yticklabels(xl, fontsize=7)
        return im

    # Row 0
    ax0 = fig.add_subplot(gs[0, :2])
    cmap_ph = LinearSegmentedColormap.from_list('ph',
                ['#69ff94','#ffd93d','#ff6b6b'], N=3)
    hm(ax0, g_ph, cmap_ph,
       f'PHASE DIAGRAM v3  (silhouette={sil:.4f})\n'
       f'F=Funnel  R=Rugged  C=Cliff  |  white border = BKZ gap',
       vmin=0, vmax=2)
    for i in range(n_n):
        for j in range(n_a):
            v = g_ph[i,j]
            if not np.isnan(v):
                ax0.text(i, j, ['F','R','C'][int(v)],
                         ha='center', va='center',
                         color='black', fontsize=9, fontweight='bold')
    for r in results:
        if r.bkz_umbrella_gap:
            try:
                i = ns.index(r.n)
                j = min(range(n_a), key=lambda x: abs(alphas[x] - r.alpha))
                ax0.add_patch(plt.Rectangle((i-0.5,j-0.5),1,1,
                              fill=False, edgecolor='white', lw=2.5))
            except: pass
    ax0.legend(handles=[
        Patch(color='#69ff94', label='Funnel'),
        Patch(color='#ffd93d', label='Rugged'),
        Patch(color='#ff6b6b', label='Cliff'),
    ], fontsize=7, facecolor=PBG, labelcolor='white', loc='upper right')

    ax_wg = fig.add_subplot(gs[0, 2])
    cmap_wg = LinearSegmentedColormap.from_list('wg',
               ['#0f1117','#00d4ff','#ffd93d'])
    im_wg = hm(ax_wg, g_gap, cmap_wg,
               'Window Gap (NEW)\n|mean(xi) - xi_center| / xi_range',
               tc='#00d4ff')
    plt.colorbar(im_wg, ax=ax_wg, shrink=0.85)

    # Row 1
    ax_bkz = fig.add_subplot(gs[1, 0])
    cmap_bkz = LinearSegmentedColormap.from_list('bkz',
                ['#0f3460','#00d4ff','#ff6b6b'])
    im_bkz = hm(ax_bkz, g_bkz, cmap_bkz, 'BKZ log2(gates)', tc='#00d4ff')
    for i in range(n_n):
        for j in range(n_a):
            v = g_bkz[i,j]
            if not np.isnan(v):
                ax_bkz.text(i,j,f'{v:.0f}', ha='center', va='center',
                            color='white', fontsize=6, fontweight='bold')
    plt.colorbar(im_bkz, ax=ax_bkz, shrink=0.85)

    ax_rug = fig.add_subplot(gs[1, 1])
    cmap_rug = LinearSegmentedColormap.from_list('rug',
                ['#0f1117','#ffd93d','#ff4444'])
    im_rug = hm(ax_rug, g_rug, cmap_rug, 'Rugosity', tc='#ffd93d')
    plt.colorbar(im_rug, ax=ax_rug, shrink=0.85)

    ax_mob = fig.add_subplot(gs[1, 2])
    cmap_mob = LinearSegmentedColormap.from_list('mob',
                ['#ff6b6b','#ffd93d','#69ff94'])
    im_mob = hm(ax_mob, g_mob, cmap_mob,
                'Mobility (acceptance rate)', tc='#69ff94')
    plt.colorbar(im_mob, ax=ax_mob, shrink=0.85)

    # Row 2
    cmap_n = plt.cm.plasma

    ax_wga = fig.add_subplot(gs[2, 0])
    ax_wga.set_facecolor(PBG)
    for i, n in enumerate(ns):
        col = cmap_n(i / len(ns))
        ax_wga.plot(alphas, g_gap[i,:], 'o-', color=col,
                    lw=1.8, ms=4, label=f'n={n}')
    styled(ax_wga, 'Window Gap vs alpha', 'alpha', 'window_gap', '#00d4ff')
    ax_wga.legend(fontsize=6, labelcolor='white', facecolor=PBG, ncol=2)

    ax_rn = fig.add_subplot(gs[2, 1])
    ax_rn.set_facecolor(PBG)
    for i, n in enumerate(ns):
        col = cmap_n(i / len(ns))
        ax_rn.plot(alphas, g_rug[i,:], 's-', color=col,
                   lw=1.8, ms=4, label=f'n={n}')
    styled(ax_rn, 'Rugosity vs alpha', 'alpha', 'rugosity', '#ffd93d')
    ax_rn.legend(fontsize=6, labelcolor='white', facecolor=PBG, ncol=2)

    ax_sc = fig.add_subplot(gs[2, 2])
    ax_sc.set_facecolor(PBG)
    for r in results:
        c  = PHASE_COLORS.get(r.phase_label, '#888')
        mk = '^' if r.bkz_umbrella_gap else 'o'
        sz = 100 if r.bkz_umbrella_gap else 50
        ax_sc.scatter(r.bkz_log2gates, r.rugosity,
                      color=c, marker=mk, s=sz,
                      alpha=0.85, edgecolors='white', lw=0.5)
    ax_sc.axvline(80, color='white', lw=1, linestyle='--', alpha=0.5)
    styled(ax_sc, 'BKZ Gates vs Rugosity\n(triangle = gap)',
           'log2(BKZ gates)', 'rugosity')
    ax_sc.legend(handles=[Patch(color=PHASE_COLORS[p], label=p)
                           for p in ["funnel","rugged","cliff"]],
                 fontsize=7, labelcolor='white', facecolor=PBG)

    fig.suptitle(
        'Experiment 4 v3 — window_gap + corrected cluster labels\n'
        'clustering on [rugosity, window_gap, mobility]',
        color='white', fontsize=13, fontweight='bold', y=1.005)
    plt.savefig(save, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"Plot saved to {save}")
    plt.close()

# CONCLUSIONS

def print_conclusions(results, gaps_strong, gaps_soft, sil, thresholds):
    counts = {p: sum(1 for r in results if r.phase_label==p)
              for p in ["funnel","rugged","cliff"]}
    total  = sum(counts.values())
    rugged = [r for r in results if r.phase_label=="rugged"]
    gap_vals = [r.window_gap for r in results]

    sep = "=" * 65
    print(f"\n{sep}")
    print("  EXPERIMENT 4 v3 — CONCLUSIONS")
    print(sep)
    print(f"\nSilhouette = {sil:.4f}  "
          f"{'[STRONG >0.5]' if sil>0.5 else '[MODERATE]' if sil>0.3 else '[WEAK]'}")
    print(f"\nPhase distribution ({total} configs):")
    for p, c in counts.items():
        print(f"  {p:>8}: {c:>3}  ({100*c/max(total,1):.0f}%)")
    print(f"\nWindow gap range: [{min(gap_vals):.5f}, {max(gap_vals):.5f}]")
    spread = max(gap_vals) - min(gap_vals)
    print(f"  {'LIVE — metric varies meaningfully.' if spread > 0.005 else 'FLAT — may need stronger k.'}")
    print(f"\nBKZ full breaks:  {sum(1 for r in results if r.bkz_attack_full)} / {total}")
    print(f"BKZ partial only: {sum(1 for r in results if not r.bkz_attack_full)} / {total}")
    print(f"\nStrong gaps (BKZ fails + rugged): {len(gaps_strong)}")
    print(f"Soft gaps   (BKZ hard + rugged):  {len(gaps_soft)}")
    if gaps_strong:
        print("\n  STRONG GAP CONFIGS:")
        for r in gaps_strong:
            print(f"    n={r.n}  q={r.q}  alpha={r.alpha:.1f}"
                  f"  rug={r.rugosity:.4f}  gap={r.window_gap:.5f}"
                  f"  mob={r.mobility:.4f}  beta={r.bkz_beta:.0f}")
    if rugged:
        print(f"\nRugged band:")
        print(f"  alpha: [{min(r.alpha for r in rugged):.1f}, {max(r.alpha for r in rugged):.1f}]")
        print(f"  n:     [{min(r.n for r in rugged)}, {max(r.n for r in rugged)}]")
    print(f"\nThresholds (from k-means):")
    for phase in ["funnel","rugged","cliff"]:
        if phase not in thresholds: continue
        t = thresholds[phase]
        print(f"  {phase.upper()} ({t['n_configs']} configs):")
        print(f"    rugosity   [{t['rugosity'][0]:.4f}, {t['rugosity'][1]:.4f}]")
        print(f"    window_gap [{t['window_gap'][0]:.5f}, {t['window_gap'][1]:.5f}]")
        print(f"    mobility   [{t['mobility'][0]:.4f}, {t['mobility'][1]:.4f}]")
    print(f"\nNext steps:")
    print("  1. Scale to n=128 (GPU recommended)")
    print("  2. Replace approx BKZ with real lattice-estimator")
    print("  3. Test Kyber-512: n=256, q=3329, sigma~1")
    print(f"\n{sep}")


if __name__ == "__main__":
    ns     = [24, 28, 32, 36, 40, 48, 56, 64]
    alphas = [3.0, 5.0, 8.0, 12.0, 18.0, 26.0, 36.0]
    sigma  = 2.0

    print("=" * 65)
    print("  Experiment 4 v3: window_gap + fixed cluster labels")
    print(f"  {len(ns)} x {len(alphas)} = {len(ns)*len(alphas)} configs")
    print("=" * 65)
    print(f"\n{'n':>4}  {'a':>5}  {'q':>5}  {'rug':>7}  {'gap':>8}  {'mob':>6}")
    print("-" * 44)

    results = []; t0 = time.time()
    for n in ns:
        for alpha in alphas:
            q = int(alpha * sigma * np.sqrt(n))
            if q < n + 2: q = n + 2
            if q % 2 == 0: q += 1
            t1 = time.time()
            try:
                r = run_config(n, q, sigma)
                results.append(r)
                print(f"{n:>4}  {alpha:>5.1f}  {q:>5}  {r.rugosity:>7.4f}  "
                      f"{r.window_gap:>8.5f}  {r.mobility:>6.3f}  "
                      f"({time.time()-t1:.1f}s)")
            except Exception as e:
                print(f"{n:>4}  {alpha:>5.1f}  ERROR: {e}")

    print(f"\n{len(results)} configs in {time.time()-t0:.1f}s")
    print("\nClustering on [rugosity, window_gap, mobility]...")
    labels, sil, centroids, cluster_names, thresholds = cluster_results(results)
    print(f"Silhouette: {sil:.4f}")
    for p in ["funnel","rugged","cliff"]:
        print(f"  {p}: {sum(1 for r in results if r.phase_label==p)}")

    attach_bkz(results)

    gaps_strong, gaps_soft = find_disagreements(results, bkz_hard_threshold=80.0)
    print(f"\n{len(gaps_strong)+len(gaps_soft)} total gaps "
          f"({len(gaps_strong)} strong, {len(gaps_soft)} soft)")
    write_disagreements(gaps_strong, gaps_soft)
    plot_results(results, ns, alphas, sil)
    print_conclusions(results, gaps_strong, gaps_soft, sil, thresholds)
    print(f"Total time: {time.time()-t0:.1f}s")