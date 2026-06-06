"""
Experiment 5 — GPU-Accelerated Umbrella Sampling
===
Hardware target: NVIDIA 5070 Ti (16GB VRAM)
CPU:             AMD 7800 X3D

Key changes from v3:
  - PyTorch CUDA batched MH: all chains × all windows run in parallel
  - Scales to n = 64, 96, 128, 192, 256
  - At n=256 we approach Kyber-512 territory (n=256, q=3329)
  - Same metrics: rugosity, window_gap, mobility
  - Same BKZ estimator + clustering + disagreement detection
  - Falls back to CPU if CUDA not available

GPU batching strategy:
  - Shape: (n_windows × n_chains, n) tensor on GPU
  - All proposals computed in one batched matmul
  - Acceptance computed with vectorised torch.where
  - Only xi values (scalars) transferred back to CPU for WHAM
  - Residual norm: || A @ s - b ||  via batched matmul on GPU

Memory estimate at n=256, m=768, batch=42×8=336 chains:
  A matrix:   768 × 256 × 4 bytes = 768 KB
  s batch:    336 × 256 × 4 bytes = 344 KB
  Total per config: ~2 MB  >>>>  well within 16 GB
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
from typing import List, Tuple
import warnings, time, math
warnings.filterwarnings('ignore')

#  PyTorch import with CUDA check ─
try:
    import torch
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        GPU_NAME = torch.cuda.get_device_name(0)
        VRAM_GB  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU detected: {GPU_NAME}  ({VRAM_GB:.1f} GB VRAM)")
    else:
        DEVICE = torch.device('cpu')
        print("No CUDA GPU detected — running on CPU.")
        print("Install PyTorch with CUDA: https://pytorch.org/get-started/locally/")
except ImportError:
    raise ImportError("PyTorch required. Install: pip install torch --index-url "
                      "https://download.pytorch.org/whl/cu128")

RNG = np.random.default_rng(2027)
BG  = '#0f1117'
PBG = '#1a1d27'
PHASE_COLORS = {"funnel":"#69ff94","rugged":"#ffd93d","cliff":"#ff6b6b","unknown":"#888"}

# LWE GENERATION (CPU — done once per config)

@dataclass
class LWEInstance:
    n: int; m: int; q: int; sigma: float
    A: np.ndarray; b: np.ndarray; s_true: np.ndarray
    # GPU tensors (filled after .to_gpu())
    A_gpu: object = None
    b_gpu: object = None

    def to_gpu(self):
        self.A_gpu = torch.tensor(self.A, dtype=torch.float32, device=DEVICE)
        self.b_gpu = torch.tensor(self.b, dtype=torch.float32, device=DEVICE)
        return self

def generate_lwe(n, m, q, sigma):
    A = RNG.integers(0, q, size=(m, n))
    s = RNG.integers(0, q, size=(n,))
    e = np.round(RNG.normal(0, sigma, size=(m,))).astype(int)
    b = (A @ s + e) % q
    return LWEInstance(n=n, m=m, q=q, sigma=sigma, A=A, b=b, s_true=s)

# GPU-BATCHED RESIDUAL NORM
# Input:  s_batch (B, n) float32 tensor on GPU
# Output: norms   (B,)  float32 tensor on GPU

def residual_norm_gpu(lwe: LWEInstance, s_batch: torch.Tensor) -> torch.Tensor:
    """
    r = (A @ s.T - b).T  mod q,  centered to (-q/2, q/2]
    norm = ||r||  per row
    s_batch shape: (B, n)
    """
    q = lwe.q
    # (m, n) @ (n, B) >>>> (m, B) >>>> (B, m)
    r = (lwe.A_gpu @ s_batch.T - lwe.b_gpu.unsqueeze(1)) % q
    r = torch.where(r > q // 2, r - q, r)
    return torch.norm(r.T, dim=1)   # (B,)

def residual_norm_cpu(lwe: LWEInstance, s: np.ndarray) -> float:
    r = (lwe.A @ s - lwe.b) % lwe.q
    r = np.where(r > lwe.q // 2, r - lwe.q, r)
    return float(np.linalg.norm(r))

# GPU-BATCHED MH WINDOW
# Runs n_windows × n_chains chains simultaneously on GPU.
# All proposals are computed in one batched step.

def umbrella_energy_gpu(xi: torch.Tensor, xc: torch.Tensor, k: float) -> torch.Tensor:
    return 0.5 * k * (xi - xc) ** 2

def mh_all_windows_gpu(
    lwe: LWEInstance,
    xi_centers: np.ndarray,    # (n_windows,)
    k: float,
    T: float,
    n_steps: int,
    n_chains: int = 8,
    s_init_batch: torch.Tensor = None,  # (n_windows × n_chains, n)
    step_size: int = None,
    block_size: int = None,
) -> Tuple[np.ndarray, torch.Tensor, np.ndarray]:
    """
    Returns:
      xi_trajs  (n_windows, n_chains * n_steps) — all xi trajectories
      s_final   (n_windows × n_chains, n)       — final states
      acc_rates (n_windows,)                    — acceptance rates
    """
    n_win = len(xi_centers)
    n     = lwe.n
    q     = lwe.q
    B     = n_win * n_chains      # total chains

    if step_size is None:
        step_size = max(1, q // 64)
    if block_size is None:
        block_size = max(1, n // 16)

    xc_gpu = torch.tensor(
        np.repeat(xi_centers, n_chains), dtype=torch.float32, device=DEVICE
    )   # (B,)

    # Initialise s_batch
    if s_init_batch is None:
        s_batch = torch.randint(0, q, (B, n), dtype=torch.float32, device=DEVICE)
    else:
        noise   = torch.randint(-step_size, step_size + 1, (B, n),
                                dtype=torch.float32, device=DEVICE)
        s_batch = (s_init_batch + noise) % q

    xi_cur   = residual_norm_gpu(lwe, s_batch)          # (B,)
    bias_cur = umbrella_energy_gpu(xi_cur, xc_gpu, k)   # (B,)

    # Storage: xi per chain per step
    # We only keep xi (scalar) not s to save GPU memory
    xi_records = torch.zeros((B, n_steps), dtype=torch.float32, device=DEVICE)
    acc_count  = torch.zeros(B, dtype=torch.float32, device=DEVICE)

    for step in range(n_steps):
        # Block-coordinate proposal: flip `block_size` random coords per chain
        # col_idx: (B, block_size) random column indices
        col_idx = torch.stack([
            torch.randperm(n, device=DEVICE)[:block_size] for _ in range(B)
        ])   # (B, block_size)

        deltas = torch.randint(-step_size, step_size + 1,
                               (B, block_size), dtype=torch.float32, device=DEVICE)

        s_prop = s_batch.clone()
        s_prop.scatter_add_(1, col_idx, deltas)
        s_prop = s_prop % q

        xi_prop   = residual_norm_gpu(lwe, s_prop)
        bias_prop = umbrella_energy_gpu(xi_prop, xc_gpu, k)

        db       = bias_prop - bias_cur                  # (B,)
        log_u    = torch.log(torch.rand(B, device=DEVICE) + 1e-30)
        accept   = (db <= 0) | (log_u < -db / T)        # (B,) bool

        # Update accepted chains
        s_batch  = torch.where(accept.unsqueeze(1), s_prop, s_batch)
        xi_cur   = torch.where(accept, xi_prop,   xi_cur)
        bias_cur = torch.where(accept, bias_prop, bias_cur)

        xi_records[:, step] = xi_cur
        acc_count += accept.float()

    # Move xi records to CPU and reshape
    xi_cpu     = xi_records.cpu().numpy()   # (B, n_steps)
    acc_rates  = (acc_count / n_steps).cpu().numpy()   # (B,)

    # Reshape to (n_windows, n_chains * n_steps)
    xi_trajs   = xi_cpu.reshape(n_win, n_chains, n_steps
                                ).reshape(n_win, n_chains * n_steps)
    # Per-window mean acceptance rate
    acc_by_win = acc_rates.reshape(n_win, n_chains).mean(axis=1)

    return xi_trajs, s_batch, acc_by_win

# WHAM (CPU — fast enough, operates on histograms not raw data)

def wham(window_xis, xi_centers, k, T, xi_bins, n_iter=300, tol=1e-7):
    n_win  = len(window_xis)
    n_bins = len(xi_bins) - 1
    bc     = 0.5 * (xi_bins[:-1] + xi_bins[1:])
    H = np.zeros((n_win, n_bins)); N = np.zeros(n_win)
    for i, xis in enumerate(window_xis):
        H[i], _ = np.histogram(xis, bins=xi_bins); N[i] = len(xis)
    bias = np.exp(-np.array([
        [0.5 * k * (b - xi_centers[i])**2 / T for b in bc]
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

# RESULT

@dataclass
class Result:
    n: int; q: int; sigma: float; alpha: float
    xi_true: float; xi_random: float
    rugosity:    float
    window_gap:  float
    mobility:    float
    bin_centers: np.ndarray
    F:           np.ndarray
    bkz_beta:         float = 0.0
    bkz_log2gates:    float = 0.0
    bkz_attack_full:  bool  = False
    cluster_id:       int   = -1
    phase_label:      str   = "unknown"
    bkz_umbrella_gap: bool  = False

def run_config(n, q, sigma,
               n_windows=16, n_steps_burn=600, n_steps_prod=2000,
               k=0.006, T=1.0, n_chains=8) -> Result:
    m   = 3 * n
    lwe = generate_lwe(n=n, m=m, q=q, sigma=sigma).to_gpu()
    alpha = q / (sigma * np.sqrt(n))

    # Baseline xi values (CPU)
    xi_true   = residual_norm_cpu(lwe, lwe.s_true)
    xi_random = float(np.mean([
        residual_norm_cpu(lwe, RNG.integers(0, q, size=(n,)))
        for _ in range(40)
    ]))

    xi_max     = xi_random * 1.03
    xi_min     = max(xi_true * 0.75, 1.0)
    xi_range   = xi_max - xi_min + 1e-9
    xi_centers = np.linspace(xi_max, xi_min, n_windows)

    #  Burn-in (all windows in parallel) 
    _, s_batch, _ = mh_all_windows_gpu(
        lwe, xi_centers, k, T, n_steps_burn, n_chains=n_chains)

    #  Production (all windows in parallel) ─
    xi_trajs, s_batch, acc_by_win = mh_all_windows_gpu(
        lwe, xi_centers, k, T, n_steps_prod,
        n_chains=n_chains, s_init_batch=s_batch)

    #  Metrics 
    window_means = xi_trajs.mean(axis=1)   # (n_windows,)
    window_gap   = float(np.mean(np.abs(window_means - xi_centers) / xi_range))
    mobility     = float(acc_by_win.mean())

    # WHAM
    xi_bins = np.linspace(xi_min * 0.88, xi_max * 1.05, 70)
    bc, F   = wham(list(xi_trajs), xi_centers, k, T, xi_bins)

    valid    = np.isfinite(F)
    rugosity = (float(np.std(np.diff(F[valid]))) / (float(F[valid].max()) + 1e-9)
                if valid.sum() > 2 else 0.0)

    # Free GPU tensors
    del lwe.A_gpu, lwe.b_gpu, s_batch, xi_trajs
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    return Result(n=n, q=q, sigma=sigma, alpha=alpha,
                  xi_true=xi_true, xi_random=xi_random,
                  rugosity=rugosity, window_gap=window_gap, mobility=mobility,
                  bin_centers=bc, F=F)

# BKZ ESTIMATOR

def bkz_estimate(n, q, sigma, m_factor=3):
    m   = m_factor * n
    dim = n + m + 1
    log_vol = m * math.log2(max(q, 2))
    best_beta = dim; best_cost = float('inf'); attack_full = False
    closest_gap = float('inf'); closest_beta = dim
    for beta in range(40, min(dim + 1, 3000)):
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
    print(f"  {'n':>5}  {'q':>6}  {'beta':>6}  {'log2gates':>10}  {'full':>12}")
    print("  " + "-" * 48)
    for r in results:
        r.bkz_beta, r.bkz_log2gates, r.bkz_attack_full = bkz_estimate(r.n, r.q, r.sigma)
        flag = "YES (breakable)" if r.bkz_attack_full else "no  (partial)"
        print(f"  {r.n:>5}  {r.q:>6}  {r.bkz_beta:>6.0f}  "
              f"{r.bkz_log2gates:>10.2f}  {flag:>12}")

# CLUSTERING

def cluster_results(results, n_clusters=3):
    X = np.array([[r.rugosity, r.window_gap, r.mobility] for r in results])
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km       = KMeans(n_clusters=n_clusters, n_init=30, random_state=42)
    labels   = km.fit_predict(X_scaled)
    sil      = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0.0
    centroids_orig = scaler.inverse_transform(km.cluster_centers_)
    rug_c = centroids_orig[:, 0]; mob_c = centroids_orig[:, 2]
    rugged_id = int(np.argmax(rug_c))
    remaining = [i for i in range(n_clusters) if i != rugged_id]
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
        r.cluster_id = int(labels[i]); r.phase_label = cluster_names[labels[i]]
    return labels, sil, centroids_orig, cluster_names, thresholds

# DISAGREEMENT DETECTION

def find_disagreements(results, bkz_hard_threshold=100.0):
    gaps_strong = []; gaps_soft = []
    for r in results:
        if r.phase_label != "rugged": continue
        if not r.bkz_attack_full:
            r.bkz_umbrella_gap = True; gaps_strong.append(r)
        elif r.bkz_log2gates > bkz_hard_threshold:
            r.bkz_umbrella_gap = True; gaps_soft.append(r)
    return gaps_strong, gaps_soft

def write_disagreements(gaps_strong, gaps_soft, path="lwe_disagreements_v5.txt"):
    with open(path, "w") as f:
        f.write("=" * 72 + "\n")
        f.write("BKZ vs UMBRELLA SAMPLING — DISAGREEMENT REPORT v5 (GPU scale)\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"STRONG GAPS ({len(gaps_strong)} configs)\n")
        f.write("BKZ cannot fully break AND umbrella finds rugged landscape.\n\n")
        if gaps_strong:
            f.write(f"  {'n':>5}  {'q':>6}  {'alpha':>6}  "
                    f"{'rugosity':>9}  {'window_gap':>11}  "
                    f"{'mobility':>9}  {'log2gates':>10}\n")
            f.write("  " + "-" * 66 + "\n")
            for r in gaps_strong:
                f.write(f"  {r.n:>5}  {r.q:>6}  {r.alpha:>6.2f}  "
                        f"{r.rugosity:>9.4f}  {r.window_gap:>11.5f}  "
                        f"{r.mobility:>9.4f}  {r.bkz_log2gates:>10.2f}\n")
        else:
            f.write("  None found at this scale.\n")
        f.write(f"\nSOFT GAPS ({len(gaps_soft)} configs)\n")
        if gaps_soft:
            for r in gaps_soft:
                f.write(f"  n={r.n}  q={r.q}  alpha={r.alpha:.2f}  "
                        f"log2gates={r.bkz_log2gates:.2f}\n")
        else:
            f.write("  None found.\n")
    print(f"Disagreement report >>>> {path}")


def plot_results(results, ns, alphas, sil,
                 save="lwe_gpu_results.png"):
    fig = plt.figure(figsize=(22, 18))
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

    xl = [f"{a:.0f}" for a in alphas]; yl = [str(x) for x in ns]

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

    # Row 0: Phase diagram + window gap
    ax0 = fig.add_subplot(gs[0, :2])
    cmap_ph = LinearSegmentedColormap.from_list('ph',['#69ff94','#ffd93d','#ff6b6b'],N=3)
    hm(ax0, g_ph, cmap_ph,
       f'PHASE DIAGRAM — GPU Scale (n=64–256)  |  silhouette={sil:.4f}\n'
       f'F=Funnel  R=Rugged  C=Cliff  |  white border = BKZ gap',
       vmin=0, vmax=2)
    for i in range(n_n):
        for j in range(n_a):
            v = g_ph[i,j]
            if not np.isnan(v):
                ax0.text(i,j,['F','R','C'][int(v)],
                         ha='center',va='center',color='black',
                         fontsize=9,fontweight='bold')
    for r in results:
        if r.bkz_umbrella_gap:
            try:
                i = ns.index(r.n)
                j = min(range(n_a), key=lambda x: abs(alphas[x] - r.alpha))
                ax0.add_patch(plt.Rectangle((i-0.5,j-0.5),1,1,
                              fill=False,edgecolor='white',lw=2.5))
            except: pass
    ax0.legend(handles=[
        Patch(color='#69ff94',label='Funnel'),
        Patch(color='#ffd93d',label='Rugged'),
        Patch(color='#ff6b6b',label='Cliff'),
    ], fontsize=7, facecolor=PBG, labelcolor='white', loc='upper right')

    ax_wg = fig.add_subplot(gs[0, 2])
    cmap_wg = LinearSegmentedColormap.from_list('wg',['#0f1117','#00d4ff','#ffd93d'])
    im_wg = hm(ax_wg, g_gap, cmap_wg, 'Window Gap\n(reachability metric)', tc='#00d4ff')
    plt.colorbar(im_wg, ax=ax_wg, shrink=0.85)

    # Row 1
    ax_bkz = fig.add_subplot(gs[1, 0])
    cmap_bkz = LinearSegmentedColormap.from_list('bkz',['#0f3460','#00d4ff','#ff6b6b'])
    im_bkz = hm(ax_bkz, g_bkz, cmap_bkz, 'BKZ log2(gates)', tc='#00d4ff')
    for i in range(n_n):
        for j in range(n_a):
            v = g_bkz[i,j]
            if not np.isnan(v):
                ax_bkz.text(i,j,f'{v:.0f}',ha='center',va='center',
                            color='white',fontsize=6,fontweight='bold')
    plt.colorbar(im_bkz, ax=ax_bkz, shrink=0.85)

    ax_rug = fig.add_subplot(gs[1, 1])
    cmap_rug = LinearSegmentedColormap.from_list('rug',['#0f1117','#ffd93d','#ff4444'])
    im_rug = hm(ax_rug, g_rug, cmap_rug, 'Rugosity', tc='#ffd93d')
    plt.colorbar(im_rug, ax=ax_rug, shrink=0.85)

    ax_mob = fig.add_subplot(gs[1, 2])
    cmap_mob = LinearSegmentedColormap.from_list('mob',['#ff6b6b','#ffd93d','#69ff94'])
    im_mob = hm(ax_mob, g_mob, cmap_mob, 'Mobility', tc='#69ff94')
    plt.colorbar(im_mob, ax=ax_mob, shrink=0.85)

    # Row 2: line plots + scatter
    cmap_n = plt.cm.plasma

    ax_rn = fig.add_subplot(gs[2, 0])
    ax_rn.set_facecolor(PBG)
    for i, n in enumerate(ns):
        col = cmap_n(i / len(ns))
        ax_rn.plot(alphas, g_rug[i,:], 'o-', color=col, lw=1.8, ms=4, label=f'n={n}')
    styled(ax_rn, 'Rugosity vs alpha\n(does rugged band persist at n=256?)',
           'alpha', 'rugosity', '#ffd93d')
    ax_rn.legend(fontsize=6, labelcolor='white', facecolor=PBG, ncol=2)

    ax_bn = fig.add_subplot(gs[2, 1])
    ax_bn.set_facecolor(PBG)
    for i, n in enumerate(ns):
        col = cmap_n(i / len(ns))
        ax_bn.plot(alphas, g_bkz[i,:], 's-', color=col, lw=1.8, ms=4, label=f'n={n}')
    ax_bn.axhline(128, color='white', lw=1.5, linestyle='--', alpha=0.7,
                  label='NIST Level 1 (128 bit)')
    ax_bn.axhline(192, color='#ffd93d', lw=1, linestyle=':', alpha=0.7,
                  label='NIST Level 3 (192 bit)')
    styled(ax_bn, 'BKZ log2(gates) vs alpha\n(NIST security levels marked)',
           'alpha', 'log2(gates)', '#00d4ff')
    ax_bn.legend(fontsize=6, labelcolor='white', facecolor=PBG, ncol=1)

    ax_sc = fig.add_subplot(gs[2, 2])
    ax_sc.set_facecolor(PBG)
    for r in results:
        c  = PHASE_COLORS.get(r.phase_label, '#888')
        mk = '^' if r.bkz_umbrella_gap else 'o'
        sz = 120 if r.bkz_umbrella_gap else 55
        ax_sc.scatter(r.bkz_log2gates, r.rugosity,
                      color=c, marker=mk, s=sz,
                      alpha=0.85, edgecolors='white', lw=0.5)
    ax_sc.axvline(128, color='white', lw=1, linestyle='--', alpha=0.5)
    styled(ax_sc, 'BKZ Gates vs Rugosity\n(triangle = gap, dashed = NIST L1)',
           'log2(BKZ gates)', 'rugosity')
    ax_sc.legend(handles=[Patch(color=PHASE_COLORS[p], label=p)
                           for p in ["funnel","rugged","cliff"]],
                 fontsize=7, labelcolor='white', facecolor=PBG)

    fig.suptitle(
        f'Experiment 5 — GPU-Accelerated Umbrella Sampling  '
        f'({GPU_NAME if DEVICE.type=="cuda" else "CPU"})\n'
        f'n = 64 – 256  |  Approaching Kyber-512 territory (n=256, q=3329)',
        color='white', fontsize=12, fontweight='bold', y=1.005)
    plt.savefig(save, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"Plot saved >>>> {save}")
    plt.close()

# CONCLUSIONS

def print_conclusions(results, gaps_strong, gaps_soft, sil, thresholds):
    counts = {p: sum(1 for r in results if r.phase_label==p)
              for p in ["funnel","rugged","cliff"]}
    total  = sum(counts.values())
    rugged = [r for r in results if r.phase_label=="rugged"]
    sep    = "=" * 68

    print(f"\n{sep}")
    print("  EXPERIMENT 5 — GPU SCALE CONCLUSIONS")
    print(sep)
    print(f"\nDevice: {GPU_NAME if DEVICE.type == 'cuda' else 'CPU'}")
    print(f"Silhouette = {sil:.4f}  "
          f"{'[STRONG >0.5]' if sil>0.5 else '[MODERATE]' if sil>0.3 else '[WEAK]'}")
    print(f"\nPhase distribution ({total} configs):")
    for p, c in counts.items():
        print(f"  {p:>8}: {c:>3}  ({100*c/max(total,1):.0f}%)")
    print(f"\nBKZ full breaks:  {sum(1 for r in results if r.bkz_attack_full)} / {total}")
    print(f"BKZ partial only: {sum(1 for r in results if not r.bkz_attack_full)} / {total}")
    print(f"\nStrong gaps (BKZ fails + rugged): {len(gaps_strong)}")
    print(f"Soft   gaps (BKZ hard + rugged):  {len(gaps_soft)}")
    if gaps_strong:
        print("\n  STRONG GAP CONFIGS:")
        for r in gaps_strong:
            print(f"    n={r.n:<4} q={r.q:<6} alpha={r.alpha:<6.1f}"
                  f" rug={r.rugosity:.4f} gap={r.window_gap:.4f}"
                  f" mob={r.mobility:.4f} beta={r.bkz_beta:.0f}")
    if rugged:
        print(f"\nRugged band at GPU scale:")
        print(f"  alpha: [{min(r.alpha for r in rugged):.1f},"
              f" {max(r.alpha for r in rugged):.1f}]")
        print(f"  n:     [{min(r.n for r in rugged)},"
              f" {max(r.n for r in rugged)}]")
    print(f"\nKey question: Do strong gaps appear at n≥128?")
    large_gaps = [r for r in gaps_strong if r.n >= 128]
    if large_gaps:
        print(f"  YES — {len(large_gaps)} gap configs at n≥128.")
        print("  This is the publishable result.")
        print("  BKZ-hard instances with rugged free energy landscape exist")
        print("  at cryptographically relevant dimensions.")
    else:
        print("  Not yet. Rugged phase may exist but BKZ also fails cleanly.")
        print("  Try: increase alpha range, or use real lattice-estimator.")
    print(f"\nComparison to Kyber-512 (n=256, q=3329, sigma~1):")
    kyber_alpha = 3329 / (1.0 * np.sqrt(256))
    print(f"  Kyber-512 alpha ≈ {kyber_alpha:.1f}")
    kyber_in_rugged = any(
        abs(r.alpha - kyber_alpha) < 20 and r.phase_label == "rugged"
        for r in results
    )
    print(f"  Closest configs in rugged band: "
          f"{'YES — near Kyber alpha range' if kyber_in_rugged else 'NO — outside range tested'}")
    print(f"\n{sep}\n")


if __name__ == "__main__":
    # Cryptographically approaching Kyber territory
    # Kyber-512: n=256, q=3329, sigma~1, alpha~208
    # We sweep up to n=256 and alpha up to 208 (Kyber's exact ratio)
    ns     = [64, 96, 128, 160, 192, 256]
    alphas = [3.0, 8.0, 18.0, 36.0, 72.0, 128.0, 208.0]
    sigma  = 2.0

    total = len(ns) * len(alphas)
    print(f"\n{'='*68}")
    print(f"  Experiment 5: GPU-Accelerated Umbrella Sampling")
    print(f"  Device : {GPU_NAME if DEVICE.type == 'cuda' else 'CPU'}")
    print(f"  Scale  : n = {ns[0]} – {ns[-1]}")
    print(f"  Configs: {len(ns)} x {len(alphas)} = {total}")
    print(f"{'='*68}\n")
    print(f"{'n':>5}  {'a':>6}  {'q':>6}  {'rug':>7}  {'gap':>8}  {'mob':>6}  {'time':>6}")
    print("-" * 52)

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
                print(f"{n:>5}  {alpha:>6.1f}  {q:>6}  {r.rugosity:>7.4f}  "
                      f"{r.window_gap:>8.5f}  {r.mobility:>6.3f}  "
                      f"{time.time()-t1:>5.1f}s")
            except Exception as e:
                print(f"{n:>5}  {alpha:>6.1f}  ERROR: {e}")

    elapsed = time.time() - t0
    print(f"\n{len(results)} configs in {elapsed:.1f}s  "
          f"({elapsed/max(len(results),1):.1f}s per config)")

    print("\nClustering...")
    labels, sil, centroids, cluster_names, thresholds = cluster_results(results)
    print(f"Silhouette: {sil:.4f}")
    for p in ["funnel","rugged","cliff"]:
        print(f"  {p}: {sum(1 for r in results if r.phase_label==p)}")

    attach_bkz(results)

    gaps_strong, gaps_soft = find_disagreements(results, bkz_hard_threshold=100.0)
    print(f"\n{len(gaps_strong)+len(gaps_soft)} total gaps "
          f"({len(gaps_strong)} strong, {len(gaps_soft)} soft)")
    write_disagreements(gaps_strong, gaps_soft)

    plot_results(results, ns, alphas, sil)
    print_conclusions(results, gaps_strong, gaps_soft, sil, thresholds)
    print(f"Total wall time: {time.time()-t0:.1f}s")