# -*- coding: utf-8 -*-
"""시각정밀 매칭 풀링 회귀 + 부트스트랩 신뢰구간 + 산포 개선 확인. 최종 그림."""
import sys, numpy as np, pandas as pd
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
rng = np.random.RandomState(0)
sys.stdout.reconfigure(encoding="utf-8")

d = pd.read_csv("pooled_matched3.csv")
d["dt_min"] = pd.to_numeric(d.get("dt_min"), errors="coerce")
print(f"풀링 표본 n={len(d)}  관측소 {d.station.nunique()}개  τ 0–{d.tau.max():.0f}")
if d["dt_min"].notna().any():
    print(f"매칭 시각오차 Δt: 중앙 {d.dt_min.median():.0f}분, 90%내 {d.dt_min.quantile(.9):.0f}분 (작을수록 co-location 정확)")

def rat(t,a): return 1/(1+a*t)
def bl(t,b):  return np.exp(-b*t)
def fit_one(t,T):
    pr,_=curve_fit(rat,t,T,p0=[.1],bounds=(0,5),maxfev=10000)
    pb,_=curve_fit(bl,t,T,p0=[.05],bounds=(0,2),maxfev=10000)
    rr=np.sqrt(np.mean((rat(t,pr[0])-T)**2)); rb=np.sqrt(np.mean((bl(t,pb[0])-T)**2))
    return (pr[0],rat,rr,'rat') if rr<rb else (pb[0],bl,rb,'bl')

def fit(sub,col,B=1000):
    m=sub[["tau",col]].dropna(); t=m.tau.values; T=np.clip(m[col].values,0,1.4)
    if len(t)<8: return None
    a,f,rmse,best=fit_one(t,T)
    boot=[]
    for _ in range(B):
        idx=rng.randint(0,len(t),len(t))
        try:
            if best=='rat': p,_=curve_fit(rat,t[idx],T[idx],p0=[a],bounds=(0,5),maxfev=5000)
            else: p,_=curve_fit(bl,t[idx],T[idx],p0=[a],bounds=(0,2),maxfev=5000)
            boot.append(p[0])
        except Exception: pass
    lo,hi=np.percentile(boot,[2.5,97.5]) if boot else (np.nan,np.nan)
    return dict(a=a,f=f,rmse=rmse,best=best,n=len(t),lo=lo,hi=hi)

BANDS=[("UVB305","T305","#6a3d9a"),("UVB310","T310","#1f78b4"),
       ("W324","T324","#33a02c"),("UVA380","T380","#e31a1c")]
print("\n=== 밴드별 T(τ) 적합 + 95% 부트스트랩 CI ===")
print(f"{'밴드':8s}{'n':>5s}{'모델':>6s}{'계수':>8s}{'95%CI':>18s}{'RMSE':>8s}  T10  T20  T40")
POOL={}
for nm,col,_ in BANDS:
    r=fit(d,col)
    if not r: continue
    POOL[nm]=r
    ci=f"[{r['lo']:.3f},{r['hi']:.3f}]"
    print(f"{nm:8s}{r['n']:5d}{r['best']:>6s}{r['a']:8.4f}{ci:>18s}{r['rmse']:8.3f}  "
          f"{r['f'](10,r['a']):.2f} {r['f'](20,r['a']):.2f} {r['f'](40,r['a']):.2f}")

print("\n=== 위도별 UVB305 계수(일관성) ===")
for st in sorted(d.station.unique(), key=lambda s:d[d.station==s].lat.iloc[0]):
    r=fit(d[d.station==st],"T305",B=300); lat=d[d.station==st].lat.iloc[0]
    if r: print(f"  {st:10s}({lat:+5.1f}°) a={r['a']:.4f}  95%CI[{r['lo']:.3f},{r['hi']:.3f}] n={r['n']}")

print("\n=== τ에 따른 투과율/차단율: UVB(305) vs UVA(380) ===")
u=POOL.get("UVB305"); a=POOL.get("UVA380")
print(f"{'τ':>4s}{'UVB투과':>9s}{'UVA투과':>9s}{'차단UVB%':>9s}{'차단UVA%':>9s}")
for t in [1,2,5,10,20,40]:
    tu=u['f'](t,u['a']); ta=a['f'](t,a['a'])
    print(f"{t:4d}{tu:9.2f}{ta:9.2f}{100*(1-tu):9.0f}{100*(1-ta):9.0f}")

# 그림
fig,ax=plt.subplots(figsize=(8.2,5.4))
xx=np.linspace(0,max(60,d.tau.max()),200)
for nm,col,c in BANDS:
    m=d[["tau",col]].dropna()
    ax.scatter(m.tau,np.clip(m[col],0,1.4),s=16,color=c,alpha=.4,edgecolor="none")
    r=POOL.get(nm)
    if r:
        ax.plot(xx,r['f'](xx,r['a']),color=c,lw=2.3,label=f"{nm}  a={r['a']:.3f} [{r['lo']:.3f},{r['hi']:.3f}]")
ax.set_xlabel("Cloud optical depth τ (MODIS Aqua, independent, overpass-matched)")
ax.set_ylabel("Cloud transmittance  T = all-sky / clear-sky")
ax.set_title("UV cloud transmittance vs cloud optical depth — 4 stations pooled, time-precise\n(ground spectral UV ÷ ozone-corrected clear-sky; 95% bootstrap CI)")
ax.set_ylim(0,1.2); ax.set_xlim(0,max(60,d.tau.max())); ax.grid(alpha=.25); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("uv_transmittance_final.png",dpi=150)
print("\n그림 저장: uv_transmittance_final.png")
