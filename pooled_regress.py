# -*- coding: utf-8 -*-
"""다관측소 풀링 회귀: T(τ) 밴드별 + 위도별 일관성 + UVB vs UVA. 그림 저장."""
import sys, numpy as np, pandas as pd
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8")

d = pd.read_csv("pooled_matched.csv")
print(f"풀링 표본 n={len(d)}  관측소 {d.station.nunique()}개  τ 0–{d.tau.max():.0f}")
def rat(t,a): return 1/(1+a*t)
def bl(t,b):  return np.exp(-b*t)

def fit(sub, col):
    m = sub[["tau",col]].dropna(); t=m.tau.values; T=np.clip(m[col].values,0,1.4)
    if len(t)<8: return None
    pr,_=curve_fit(rat,t,T,p0=[.1],bounds=(0,5),maxfev=10000)
    pb,_=curve_fit(bl,t,T,p0=[.05],bounds=(0,2),maxfev=10000)
    rr=np.sqrt(np.mean((rat(t,pr[0])-T)**2)); rb=np.sqrt(np.mean((bl(t,pb[0])-T)**2))
    return dict(a=pr[0],b=pb[0],rr=rr,rb=rb,best='rat' if rr<rb else 'bl',
                bp=pr[0] if rr<rb else pb[0], f=rat if rr<rb else bl, n=len(t))

BANDS=[("UVB305","T305","#6a3d9a"),("UVB310","T310","#1f78b4"),
       ("W324","T324","#33a02c"),("UVA380","T380","#e31a1c")]
print("\n=== 풀링(4관측소) 밴드별 T(τ) 최적적합 ===")
print(f"{'밴드':8s}{'n':>5s}{'모델':>6s}{'계수':>9s}{'RMSE':>8s}  T(τ=10) T(τ=20) T(τ=40)")
POOL={}
for nm,col,_ in BANDS:
    r=fit(d,col)
    if not r: continue
    POOL[nm]=r
    print(f"{nm:8s}{r['n']:5d}{r['best']:>6s}{r['bp']:9.4f}{min(r['rr'],r['rb']):8.3f}   "
          f"{r['f'](10,r['bp']):6.2f}  {r['f'](20,r['bp']):6.2f}  {r['f'](40,r['bp']):6.2f}")

# 위도별 일관성(UVB305)
print("\n=== 위도별 UVB305 계수 a (일관성 확인; 비슷하면 관계가 보편적) ===")
for st in sorted(d.station.unique(), key=lambda s: d[d.station==s].lat.iloc[0]):
    r=fit(d[d.station==st],"T305")
    lat=d[d.station==st].lat.iloc[0]
    if r: print(f"  {st:10s}({lat:+5.1f}°)  a={r['a']:.4f}  n={r['n']}")

# UVB vs UVA
print("\n=== τ에 따른 투과율: UVB(305, 4관측소 풀링) vs UVA(380, Lauder) ===")
u=POOL.get("UVB305"); a=POOL.get("UVA380")
print(f"{'τ':>4s}{'UVB305':>9s}{'UVA380':>9s}   차단UVB%  차단UVA%")
for t in [1,2,5,10,20,40]:
    tu=u['f'](t,u['bp']); ta=a['f'](t,a['bp'])
    print(f"{t:4d}{tu:9.2f}{ta:9.2f}   {100*(1-tu):7.0f}  {100*(1-ta):7.0f}")

# --- 그림 ---
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5.3))
smap={"MaunaLoa":"o","Tateno":"^","Uccle":"s","Lauder":"D","Ushuaia":"v"}
xx=np.linspace(0,66,200)
# 왼쪽: 밴드별 풀링
for nm,col,c in BANDS:
    m=d[["tau",col]].dropna()
    ax1.scatter(m.tau,np.clip(m[col],0,1.4),s=16,color=c,alpha=.4,edgecolor="none")
    r=POOL.get(nm)
    if r: ax1.plot(xx,r['f'](xx,r['bp']),color=c,lw=2.3,label=f"{nm} (a/b={r['bp']:.3f})")
ax1.set_title("Pooled 4 stations — T vs cloud optical depth (per band)")
ax1.set_xlabel("MODIS cloud optical depth τ"); ax1.set_ylabel("Cloud transmittance T")
ax1.set_ylim(0,1.2); ax1.set_xlim(0,66); ax1.grid(alpha=.25); ax1.legend(fontsize=8)
# 오른쪽: 위도별 UVB305 산점 + 곡선(일관성)
for st in sorted(d.station.unique(), key=lambda s:d[d.station==s].lat.iloc[0]):
    sub=d[d.station==st]; lat=sub.lat.iloc[0]
    ax2.scatter(sub.tau,np.clip(sub.T305,0,1.4),s=26,marker=smap.get(st,"o"),
                alpha=.7,label=f"{st} ({lat:+.0f}°)")
r=POOL["UVB305"]; ax2.plot(xx,r['f'](xx,r['bp']),"k-",lw=2.3,label=f"pooled fit (a={r['bp']:.3f})")
ax2.set_title("UVB(305 nm) across latitudes — consistency")
ax2.set_xlabel("MODIS cloud optical depth τ"); ax2.set_ylabel("T (UVB 305)")
ax2.set_ylim(0,1.2); ax2.set_xlim(0,66); ax2.grid(alpha=.25); ax2.legend(fontsize=8)
fig.suptitle("Cloud UV transmittance vs optical depth — ground spectral UV ÷ ozone-corrected clear-sky, independent MODIS τ",fontsize=11)
fig.tight_layout(); fig.savefig("uv_transmittance_pooled.png",dpi=150)
print("\n그림 저장: uv_transmittance_pooled.png")
