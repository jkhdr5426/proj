"""
Experiment 11: Fine Boundary Scan at n=32
===================
Run:  sage -python experiment11.py

Why n=32:
    Experiment 10 showed USVP at sigma=0.5,1.0 and BDD at sigma=2.0,3.0,4.0
    for n=32. The crossover is somewhere between sigma=1.0 and sigma=2.0.
    This experiment finds it precisely with 15 sigma values.

Two fixes from Exp 10:
    1. Barrier is now normalized by xi_range (so barrier=1.0 means the
       free energy spans the full xi range — comparable across configs)
    2. dom_gap threshold tightened to 1.5 bits for "near-boundary"

New metric: roughness_norm = roughness / mean_roughness_this_run
    Normalised roughness shows relative peaks more clearly.

Hypothesis:
    Roughness peaks at the exact sigma where USVP and BDD have equal cost.
    if it does: this is the publishable finding.
    if it doesn't: the reaction coordinate needs rethinking (discuss in paper).

Also runs the same scan at n=64 and n=96 for comparison.
"""

from __future__ import annotations

import json, math, os, re, sys, time, warnings
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

DEFAULT_REPO = os.path.expanduser("~/lattice-estimator")
BG = "#0f1117"; PBG = "#1a1d27"
FAM_COL = {"bdd":"#69ff94","usvp":"#00d4ff",
            "dual":"#ffd93d","dual_hybrid":"#ff6b6b","other":"#b0b0b0"}

try:
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DNAME  = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    TORCH  = True
except Exception:
    torch = None; DEVICE = "cpu"; DNAME = "CPU"; TORCH = False

if os.path.isdir(DEFAULT_REPO) and DEFAULT_REPO not in sys.path:
    sys.path.insert(0, DEFAULT_REPO)

ESTIMATOR = False
LWE = ND = RC = None
try:
    from estimator import LWE, ND, RC   # type: ignore
    ESTIMATOR = True
    RCM = getattr(RC, "BDGL16", None)
except Exception:
    RCM = None

JOBS = max(1, min(16, os.cpu_count() or 1))

# ESTIMATOR

def parse_rop(text):
    m = re.search(r"rop:\s*≈?\s*2\^([0-9]+(?:\.[0-9]+)?)", text)
    return float(m.group(1)) if m else None

def parse_beta(text):
    m = re.search(r"\bβ:\s*([0-9]+(?:\.[0-9]+)?)", text)
    return float(m.group(1)) if m else None

def fam(name):
    n = name.lower().replace("-","_").replace(" ","_")
    if "bdd" in n: return "bdd"
    if "dual_hybrid" in n or "dualhybrid" in n: return "dual_hybrid"
    if "dual" in n: return "dual"
    if "usvp" in n or "primal" in n: return "usvp"
    return "other"

def run_est(n, q, sigma, jobs=JOBS):
    m  = 3 * n
    try:    Xs = ND.UniformMod(q)
    except: Xs = ND.Uniform(0, q)
    try:    Xe = ND.DiscreteGaussian(stddev=sigma)
    except: Xe = ND.DiscreteGaussian(sigma)
    params = LWE.Parameters(n=n, q=q, Xs=Xs, Xe=Xe, m=m)
    kw = {"jobs": jobs}
    if RCM: kw["red_cost_model"] = RCM
    try:    result = LWE.estimate(params, **kw)
    except TypeError:
        try:    result = LWE.estimate(params, jobs=jobs)
        except TypeError: result = LWE.estimate(params)

    fam_best = {}
    if hasattr(result, "items"):
        for k, v in result.items():
            t = str(v); r = parse_rop(t)
            if r is None: continue
            f = fam(str(k))
            if f not in fam_best or r < fam_best[f]["rop"]:
                fam_best[f] = {"rop": r, "beta": parse_beta(t)}

    valid = sorted([(f,v["rop"]) for f,v in fam_best.items()], key=lambda x:x[1])
    best_f = valid[0][0] if valid else None
    best_b = valid[0][1] if valid else None
    best_beta = fam_best[best_f]["beta"] if best_f else None
    dgap = float(valid[1][1]-valid[0][1]) if len(valid)>=2 else None
    return {"best_family":best_f,"best_bits":best_b,"best_beta":best_beta,
            "dominance_gap":dgap,"fam_best":fam_best}

# LWE + GPU SAMPLER

@dataclass
class LWE_:
    n:int; m:int; q:int; sigma:float
    A:np.ndarray; b:np.ndarray; s:np.ndarray
    Ag:Any=None; bg:Any=None
    def gpu(self):
        self.Ag=torch.tensor(self.A,dtype=torch.float32,device=DEVICE)
        self.bg=torch.tensor(self.b,dtype=torch.float32,device=DEVICE)
        return self

def gen(n,m,q,sigma,seed):
    rng=np.random.default_rng(seed)
    A=rng.integers(0,q,size=(m,n),dtype=np.int64)
    s=rng.integers(0,q,size=(n,),dtype=np.int64)
    e=np.round(rng.normal(0,sigma,m)).astype(np.int64)
    return LWE_(n=n,m=m,q=q,sigma=sigma,A=A,b=(A@s+e)%q,s=s)

def xi_cpu(lwe,s):
    r=(lwe.A@s-lwe.b)%lwe.q; r=np.where(r>lwe.q//2,r-lwe.q,r)
    return float(np.linalg.norm(r))

def xi_gpu(lwe,sb):
    q=lwe.q; r=(lwe.Ag@sb.T-lwe.bg.unsqueeze(1))%q
    r=torch.where(r>q//2,r-q,r); return torch.norm(r.T,dim=1)

def mh(lwe,centers,k,T,steps,nc=6,s0=None):
    W=len(centers); n,q=lwe.n,lwe.q; B=W*nc
    st=max(1,q//64); bl=max(1,n//16)
    xc=torch.tensor(np.repeat(np.asarray(centers,float),nc),
                    dtype=torch.float32,device=DEVICE)
    sb=(torch.randint(0,q,(B,n),dtype=torch.float32,device=DEVICE)
        if s0 is None else (s0+torch.randint(-st,st+1,(B,n),
        dtype=torch.float32,device=DEVICE))%q)
    xc_cur=xi_gpu(lwe,sb); bc_cur=0.5*k*(xc_cur-xc)**2
    xr=torch.zeros((B,steps),dtype=torch.float32,device=DEVICE)
    ac=torch.zeros(B,dtype=torch.float32,device=DEVICE)
    for i in range(steps):
        col=torch.randint(0,n,(B,bl),device=DEVICE)
        d=torch.randint(-st,st+1,(B,bl),dtype=torch.float32,device=DEVICE)
        sp=sb.clone(); sp.scatter_add_(1,col,d); sp=sp%q
        xp=xi_gpu(lwe,sp); bp=0.5*k*(xp-xc)**2; db=bp-bc_cur
        ok=(db<=0)|(torch.log(torch.rand(B,device=DEVICE).clamp_min(1e-30))<-db/T)
        sb=torch.where(ok.unsqueeze(1),sp,sb)
        xc_cur=torch.where(ok,xp,xc_cur); bc_cur=torch.where(ok,bp,bc_cur)
        xr[:,i]=xc_cur; ac+=ok.float()
    tr=xr.cpu().numpy().reshape(W,nc,steps).reshape(W,nc*steps)
    aw=(ac/steps).cpu().numpy().reshape(W,nc).mean(1)
    return tr,sb,aw

def wham(wxis,centers,k,T,bins,it=400,tol=1e-8):
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

XI_REF=500.

def feats(bc,F,xi_range):
    v=np.isfinite(F)
    out=dict(barrier_raw=None,barrier_norm=None,roughness=None,
             entropy=None,n_peaks=0)
    if v.sum()<5: return out
    Fv=F[v]-np.nanmin(F[v])
    rho=np.exp(-Fv); p=rho/rho.sum()
    bar_raw=float(np.nanmax(Fv))
    bar_norm=bar_raw/max(xi_range,1.)
    diff=np.diff(Fv)
    rough=float(np.std(diff)/(bar_raw+1e-9)) if len(diff)>1 else 0.
    ent=float(-np.sum(p*np.log(p+1e-300))/math.log(len(p)+1e-300))
    ps=np.convolve(p,np.ones(5)/5,mode="same")
    pks=[i for i in range(1,len(ps)-1)
         if ps[i]>=ps[i-1] and ps[i]>ps[i+1]]
    out.update(barrier_raw=bar_raw,barrier_norm=bar_norm,
               roughness=rough,entropy=ent,n_peaks=len(pks))
    return out

def run_point(n,q,sigma,seed,W=16,nb=400,np_=1200,nc=6,k=0.006,T=1.,jobs=JOBS):
    lwe=gen(n,3*n,q,sigma,seed).gpu()
    xi_t=xi_cpu(lwe,lwe.s)
    rng2=np.random.default_rng(seed+99999)
    xi_r=float(np.mean([xi_cpu(lwe,rng2.integers(0,q,n,dtype=np.int64))
                         for _ in range(24)]))
    xi_max=xi_r*1.03; xi_min=max(xi_t*0.75,1.)
    xi_rng=xi_max-xi_min+1e-9
    k_sc=max(k*(XI_REF/xi_rng)**2,1e-10)
    cen=np.linspace(xi_max,xi_min,W)
    _,s0,_=mh(lwe,cen,k_sc,T,nb,nc)
    tr,_,_=mh(lwe,cen,k_sc,T,np_,nc,s0)
    bins=np.linspace(xi_min*0.85,xi_max*1.08,80)
    bc,F=wham(list(tr),cen,k_sc,T,bins)
    ft=feats(bc,F,xi_rng)
    sec=run_est(n,q,sigma,jobs)
    try:
        del lwe.Ag,lwe.bg
        if TORCH and torch: torch.cuda.empty_cache()
    except: pass
    return dict(n=n,q=q,sigma=sigma,alpha=q/(sigma*math.sqrt(n)),
                xi_rng=xi_rng,k_sc=k_sc,
                bc=bc.tolist(),F=F.tolist(),feats=ft,sec=sec,seed=seed)

# MAIN SWEEP: fine sigma scan at multiple n values

# sigmas chosen to bracket the USVP/BDD crossover at n=32
# (crossover was between sigma=1.0 and sigma=2.0 in Exp 10)
SIGMAS_FINE = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.5, 3.0, 4.0]
NS_SCAN     = [32, 64, 96]
ALPHA_FIXED = 12.0
N_SEEDS     = 3   # more seeds for stability

def main():
    if not TORCH:  raise RuntimeError("PyTorch required")
    if not ESTIMATOR: raise RuntimeError("lattice-estimator required")

    total = len(NS_SCAN)*len(SIGMAS_FINE)
    print("="*75)
    print("Experiment 11: Fine Boundary Scan (USVP/BDD crossover)")
    print(f"Device: {DNAME}")
    print(f"Sweep: {len(NS_SCAN)} n × {len(SIGMAS_FINE)} sigma = {total} configs")
    print(f"Fixed alpha = {ALPHA_FIXED}, seeds = {N_SEEDS}")
    print("="*75)
    print(f"\n{'n':>4}  {'sigma':>6}  {'q':>5}  {'family':>12}  "
          f"{'bits':>8}  {'dgap':>6}  {'rough':>7}  {'bar_norm':>9}  "
          f"{'boundary':>9}")
    print("-"*80)

    all_results = []
    t0 = time.time()

    for n in NS_SCAN:
        for sigma in SIGMAS_FINE:
            q = int(ALPHA_FIXED * sigma * math.sqrt(n))
            q = max(q, n+2)
            if q%2==0: q+=1

            seed_rows = []
            for seed in range(N_SEEDS):
                try:
                    r = run_point(n, q, sigma, seed=seed*31+n,
                                  W=16, nb=300, np_=1000, nc=6, jobs=JOBS)
                    seed_rows.append(r)
                except Exception as e:
                    print(f"  ERR n={n} sigma={sigma} s={seed}: {e}")

            if not seed_rows: continue

            # aggregate
            def mn(path):
                vals=[]
                for row in seed_rows:
                    cur=row
                    for k in path:
                        cur=cur.get(k) if isinstance(cur,dict) else None
                    if cur is not None:
                        try:
                            v=float(cur)
                            if np.isfinite(v): vals.append(v)
                        except: pass
                return float(np.mean(vals)) if vals else None

            def sd(path):
                vals=[]
                for row in seed_rows:
                    cur=row
                    for k in path:
                        cur=cur.get(k) if isinstance(cur,dict) else None
                    if cur is not None:
                        try:
                            v=float(cur)
                            if np.isfinite(v): vals.append(v)
                        except: pass
                return float(np.std(vals)) if len(vals)>=2 else 0.

            agg=dict(seed_rows[0])
            agg["seed_rows"]=seed_rows
            agg["feats"]={
                "roughness":   mn(["feats","roughness"]),
                "roughness_sd":sd(["feats","roughness"]),
                "barrier_norm":mn(["feats","barrier_norm"]),
                "barrier_raw": mn(["feats","barrier_raw"]),
                "entropy":     mn(["feats","entropy"]),
                "n_peaks":     int(round(mn(["feats","n_peaks"]) or 0)),
            }
            fams=[r["sec"]["best_family"] for r in seed_rows if r["sec"]["best_family"]]
            agg["sec"]["best_family"]=max(set(fams),key=fams.count) if fams else "other"
            agg["sec"]["best_bits"]  =mn(["sec","best_bits"])
            agg["sec"]["dominance_gap"]=mn(["sec","dominance_gap"])
            all_results.append(agg)

            f=agg["sec"]["best_family"] or "other"
            bits=agg["sec"]["best_bits"]
            dgap=agg["sec"]["dominance_gap"]
            rough=agg["feats"]["roughness"]
            bar=agg["feats"]["barrier_norm"]
            near="NEAR" if dgap is not None and dgap < 1.5 else ""
            print(f"{n:>4}  {sigma:>6.2f}  {q:>5}  {f:>12}  "
                  f"{'n/a' if bits is None else f'2^{bits:.1f}':>8}  "
                  f"{'n/a' if dgap is None else f'{dgap:.2f}':>6}  "
                  f"{'n/a' if rough is None else f'{rough:.4f}':>7}  "
                  f"{'n/a' if bar is None else f'{bar:.4f}':>9}  "
                  f"{near:>9}")

    print(f"\n{len(all_results)} configs in {(time.time()-t0)/60:.1f} min")

    #  Per-n analysis ─
    print("\n" + "="*70)
    print("ANALYSIS BY N")
    print("="*70)
    for n in NS_SCAN:
        rows=[r for r in all_results if r["n"]==n]
        if not rows: continue
        families=[r["sec"]["best_family"] for r in rows]
        roughness=np.array([r["feats"]["roughness"] or np.nan for r in rows])
        dgaps=np.array([r["sec"]["dominance_gap"] or np.nan for r in rows])
        sigmas_n=np.array([r["sigma"] for r in rows])

        # find transition: where family changes
        transitions=[]
        for i in range(len(rows)-1):
            if families[i] != families[i+1]:
                transitions.append((rows[i]["sigma"], rows[i+1]["sigma"],
                                     families[i], families[i+1]))

        print(f"\n  n={n}:")
        if transitions:
            for tr in transitions:
                s1,s2,f1,f2=tr
                print(f"    Family transition: {f1} >>>> {f2} between sigma={s1:.2f} and sigma={s2:.2f}")
                # roughness at transition
                mask_near=np.abs(sigmas_n-0.5*(s1+s2)) < 0.8
                rough_near=roughness[mask_near & np.isfinite(roughness)]
                rough_far=roughness[(~mask_near) & np.isfinite(roughness)]
                if len(rough_near)>0 and len(rough_far)>0:
                    print(f"    Roughness near transition: {np.mean(rough_near):.4f} ± {np.std(rough_near):.4f}")
                    print(f"    Roughness far from trans:  {np.mean(rough_far):.4f} ± {np.std(rough_far):.4f}")
                    diff=np.mean(rough_near)-np.mean(rough_far)
                    print(f"    Difference:                {diff:+.4f} "
                          f"({'HIGHER near boundary' if diff>0 else 'LOWER near boundary'})")
        else:
            print(f"    No family transition found (all {families[0]})")
            print(f"    Roughness range: [{np.nanmin(roughness):.4f}, {np.nanmax(roughness):.4f}]")

        # correlation at this n
        valid=np.isfinite(roughness) & np.isfinite(dgaps)
        if valid.sum() >= 3:
            corr=float(np.corrcoef(dgaps[valid], roughness[valid])[0,1])
            print(f"    corr(dom_gap, roughness) = {corr:.4f}")

    #  KEY TEST: roughness at transition vs away 
    print("\n" + "="*70)
    print("KEY TEST: Is roughness highest at the USVP/BDD boundary?")
    print("="*70)

    # classify each point as "at boundary" if its family differs from neighbours
    fams_all=[r["sec"]["best_family"] for r in all_results]
    at_boundary=[]
    for i,r in enumerate(all_results):
        n_v=r["n"]
        # compare to prev and next within same n
        same_n=[j for j,rr in enumerate(all_results) if rr["n"]==n_v]
        pos=same_n.index(i)
        is_bd=False
        if pos>0 and fams_all[same_n[pos-1]]!=fams_all[i]: is_bd=True
        if pos<len(same_n)-1 and fams_all[same_n[pos+1]]!=fams_all[i]: is_bd=True
        at_boundary.append(is_bd)

    rough_at=[all_results[i]["feats"]["roughness"]
              for i in range(len(all_results))
              if at_boundary[i] and all_results[i]["feats"]["roughness"]]
    rough_away=[all_results[i]["feats"]["roughness"]
                for i in range(len(all_results))
                if not at_boundary[i] and all_results[i]["feats"]["roughness"]]

    print(f"  At-boundary configs:  {len(rough_at)}")
    print(f"  Away-from-boundary:   {len(rough_away)}")
    if rough_at and rough_away:
        print(f"  Mean roughness AT boundary:   {np.mean(rough_at):.4f} ± {np.std(rough_at):.4f}")
        print(f"  Mean roughness AWAY boundary: {np.mean(rough_away):.4f} ± {np.std(rough_away):.4f}")
        diff=np.mean(rough_at)-np.mean(rough_away)
        print(f"  Difference: {diff:+.4f}")
        if len(rough_at)>=2 and len(rough_away)>=2:
            t,p=scipy_stats.ttest_ind(rough_at,rough_away)
            print(f"  t={t:.3f},  p={p:.4f}")
            if p<0.05:
                print("  *** SIGNIFICANT: roughness peaks at family boundary ***")
                print("  *** This is the publishable finding ***")
            elif p<0.15:
                print("  ~ Marginal signal (p<0.15). More data needed.")
            else:
                print("  Not significant. Roughness does not peak at boundary.")
                print("  >>>> Conclusion: residual-norm xi does not encode family info.")
                print("  >>>> Next step: use dual-lattice projection as 2nd coordinate.")

    #  PLOTS 
    fig=plt.figure(figsize=(20,14)); fig.patch.set_facecolor(BG)
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.35)

    def sty(ax,title,xl="",yl="",tc="white"):
        ax.set_facecolor(PBG)
        ax.set_title(title,color=tc,fontsize=9,pad=6)
        ax.set_xlabel(xl,color="white",fontsize=8)
        ax.set_ylabel(yl,color="white",fontsize=8)
        ax.tick_params(colors="white",labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#444")

    colors_n={32:"#69ff94",64:"#ffd93d",96:"#00d4ff"}

    # 1. Roughness vs sigma, per n
    ax0=fig.add_subplot(gs[0,:2]); ax0.set_facecolor(PBG)
    for n in NS_SCAN:
        rows=[r for r in all_results if r["n"]==n]
        if not rows: continue
        sigs=[r["sigma"] for r in rows]
        rugs=[r["feats"]["roughness"] or np.nan for r in rows]
        rugs_sd=[r["feats"]["roughness_sd"] or 0 for r in rows]
        fams=[r["sec"]["best_family"] for r in rows]
        col=colors_n[n]
        ax0.errorbar(sigs,rugs,yerr=rugs_sd,fmt="o-",color=col,
                     lw=2,ms=5,capsize=3,elinewidth=1,label=f"n={n}")
        # mark family transitions
        for i in range(len(rows)-1):
            if fams[i]!=fams[i+1]:
                mid=0.5*(sigs[i]+sigs[i+1])
                ax0.axvline(mid,color=col,lw=2,linestyle="--",alpha=0.7)
                ax0.text(mid,max(r for r in rugs if np.isfinite(r))*1.02,
                         f"n={n}\n{fams[i]}>>>>{fams[i+1]}",
                         ha="center",va="bottom",color=col,fontsize=6)
    sty(ax0,"MAIN PLOT: Roughness vs sigma (per n)\n"
             "dashed lines = attack-family transition points",
         "sigma","roughness","#ffd93d")
    ax0.legend(fontsize=8,labelcolor="white",facecolor=PBG)

    # 2. Normalised barrier vs sigma (should be flat if artifact is fixed)
    ax1=fig.add_subplot(gs[0,2]); ax1.set_facecolor(PBG)
    for n in NS_SCAN:
        rows=[r for r in all_results if r["n"]==n]
        if not rows: continue
        sigs=[r["sigma"] for r in rows]
        bars=[r["feats"]["barrier_norm"] or np.nan for r in rows]
        ax1.plot(sigs,bars,"o-",color=colors_n[n],lw=2,ms=4,label=f"n={n}")
    sty(ax1,"Normalised Barrier vs sigma\n(barrier / xi_range — should not be ~700 anymore)",
        "sigma","barrier / xi_range","#00d4ff")
    ax1.legend(fontsize=7,labelcolor="white",facecolor=PBG)

    # 3. dominance_gap vs sigma
    ax2=fig.add_subplot(gs[1,0]); ax2.set_facecolor(PBG)
    for n in NS_SCAN:
        rows=[r for r in all_results if r["n"]==n]
        if not rows: continue
        sigs=[r["sigma"] for r in rows]
        dgaps=[r["sec"]["dominance_gap"] or np.nan for r in rows]
        ax2.plot(sigs,dgaps,"o-",color=colors_n[n],lw=2,ms=4,label=f"n={n}")
    ax2.axhline(1.5,color="white",lw=1,linestyle="--",alpha=0.6,
                label="near-boundary (1.5 bits)")
    sty(ax2,"Dominance Gap vs sigma\n(smaller = more competitive families)",
        "sigma","dominance gap (bits)")
    ax2.legend(fontsize=7,labelcolor="white",facecolor=PBG)

    # 4. F(xi) profiles at transition vs away (n=32)
    ax3=fig.add_subplot(gs[1,1]); ax3.set_facecolor(PBG)
    rows32=[r for r in all_results if r["n"]==32]
    fams32=[r["sec"]["best_family"] for r in rows32]
    # find boundary configs
    bd_idx=[i for i in range(len(rows32))
            if (i>0 and fams32[i]!=fams32[i-1])
            or (i<len(rows32)-1 and fams32[i]!=fams32[i+1])]
    away_idx=[i for i in range(len(rows32)) if i not in bd_idx]
    cmap_bd=plt.cm.plasma; cmap_aw=plt.cm.Blues
    for k,i in enumerate(bd_idx[:3]):
        r=rows32[i]; bc=np.asarray(r["bc"]); F=np.asarray(r["F"])
        v=np.isfinite(F)
        if v.sum()<4: continue
        f=r["sec"]["best_family"] or "?"
        ax3.plot(bc[v],F[v],lw=2.5,color=cmap_bd(k/4),
                 label=f"σ={r['sigma']:.2f} [{f}] BOUNDARY")
    for k,i in enumerate(away_idx[:2]):
        r=rows32[i]; bc=np.asarray(r["bc"]); F=np.asarray(r["F"])
        v=np.isfinite(F)
        if v.sum()<4: continue
        f=r["sec"]["best_family"] or "?"
        ax3.plot(bc[v],F[v],lw=1.5,linestyle="--",color=cmap_aw(0.5+0.2*k),
                 label=f"σ={r['sigma']:.2f} [{f}] away")
    sty(ax3,"F(ξ) Profiles at n=32\nsolid=boundary, dashed=away",
        "ξ (residual norm)","F(ξ)")
    ax3.legend(fontsize=6,labelcolor="white",facecolor=PBG)

    # 5. Summary
    ax4=fig.add_subplot(gs[1,2]); ax4.set_facecolor(PBG); ax4.axis("off")
    lines=["EXPERIMENT 11 summmarryyyy",""]
    for n in NS_SCAN:
        rows=[r for r in all_results if r["n"]==n]
        fams_n=[r["sec"]["best_family"] for r in rows]
        trs=[(rows[i]["sigma"],rows[i+1]["sigma"],fams_n[i],fams_n[i+1])
             for i in range(len(rows)-1) if fams_n[i]!=fams_n[i+1]]
        lines.append(f"  n={n}:")
        if trs:
            for s1,s2,f1,f2 in trs:
                lines.append(f"    {f1}>>>>{f2} between σ={s1:.2f} and σ={s2:.2f}")
        else:
            lines.append(f"    No transition (all {fams_n[0] if fams_n else '?'})")

    lines += [
        "",
        "ROUGHNESS AT BOUNDARY:",
        f"  n={len(rough_at)} configs",
        f"  mean = {np.mean(rough_at):.4f}" if rough_at else "  no data",
        "",
        "ROUGHNESS AWAY:",
        f"  n={len(rough_away)} configs",
        f"  mean = {np.mean(rough_away):.4f}" if rough_away else "  no data",
        "",
        "CONCLUSION:",
    ]
    if rough_at and rough_away and len(rough_at)>=2:
        t,p=scipy_stats.ttest_ind(rough_at,rough_away)
        if p<0.05:
            lines.append("  SIGNIFICANT (p<0.05)")
            lines.append("  Roughness peaks at boundary")
        elif p<0.15:
            lines.append(f"  Marginal (p={p:.3f})")
        else:
            lines.append(f"  Not significant (p={p:.3f})")
            lines.append("  xi does not encode family info")
    ax4.text(0.04,0.97,"\n".join(lines),transform=ax4.transAxes,
             va="top",ha="left",fontsize=7.5,color="white",fontfamily="monospace",
             bbox=dict(boxstyle="round",facecolor="#0f1117",alpha=0.9))
    sty(ax4,"Summary")

    fig.suptitle("Experiment 11 — Fine Boundary Scan (USVP/BDD crossover)\n"
                 "Does roughness peak at the attack-family transition?",
                 color="white",fontsize=12,fontweight="bold",y=1.005)
    plt.savefig("exp11_results.png",dpi=150,bbox_inches="tight",facecolor=BG)
    print("\nPlot >>>> exp11_results.png")
    plt.close()

    # summary file
    with open("exp11_summary.txt","w") as f:
        f.write("="*70+"\n")
        f.write("EXPERIMENT 11 — FINE BOUNDARY SCAN\n")
        f.write("="*70+"\n\n")
        f.write(f"n scanned: {NS_SCAN}\n")
        f.write(f"sigma scanned: {SIGMAS_FINE}\n")
        f.write(f"Fixed alpha: {ALPHA_FIXED}\n\n")
        f.write("PER-CONFIG RESULTS\n"+"-"*50+"\n")
        f.write(f"{'n':>4}  {'sigma':>6}  {'q':>5}  {'family':>12}  "
                f"{'bits':>8}  {'dgap':>6}  {'rough':>7}  {'bar_norm':>9}\n")
        f.write("-"*70+"\n")
        for r in all_results:
            sec = r["sec"]; ft = r["feats"]
            bits_str  = "n/a" if sec["best_bits"]     is None else f"2^{sec['best_bits']:.1f}"
            dgap_str  = "n/a" if sec["dominance_gap"] is None else f"{sec['dominance_gap']:.2f}"
            rough_str = "n/a" if ft["roughness"]      is None else f"{ft['roughness']:.4f}"
            bar_str   = "n/a" if ft["barrier_norm"]   is None else f"{ft['barrier_norm']:.4f}"
            f.write(f"{r['n']:>4}  {r['sigma']:>6.2f}  {r['q']:>5}  "
                    f"{(sec['best_family'] or 'n/a'):>12}  "
                    f"{bits_str:>8}  {dgap_str:>6}  "
                    f"{rough_str:>7}  {bar_str:>9}\n")
        f.write("\nKEY RESULT\n"+"-"*50+"\n")
        if rough_at and rough_away and len(rough_at)>=2:
            t,p=scipy_stats.ttest_ind(rough_at,rough_away)
            f.write(f"  Roughness at boundary:   {np.mean(rough_at):.4f} ± {np.std(rough_at):.4f}\n")
            f.write(f"  Roughness away:          {np.mean(rough_away):.4f} ± {np.std(rough_away):.4f}\n")
            f.write(f"  Difference:              {np.mean(rough_at)-np.mean(rough_away):+.4f}\n")
            f.write(f"  p-value:                 {p:.4f}\n")
            verdict=("SIGNIFICANT" if p<0.05 else "marginal" if p<0.15 else "not significant")
            f.write(f"  Verdict: {verdict}\n")
        f.write("="*70+"\n")
    print("Summary >>>> exp11_summary.txt")
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--quick",action="store_true",
                   help="Fast test: only n=32, 6 sigma values, 1 seed")
    args=p.parse_args()
    if args.quick:
        NS_SCAN=[32]; SIGMAS_FINE=[0.7,1.1,1.5,1.9,2.5,4.0]; N_SEEDS=1
    main()