# -*- coding: utf-8 -*-
"""가시광선 vs UVB vs UVA '차단율' 비교 그림(흰 배경, 한글). Lauder 관측."""
import sys, numpy as np, pandas as pd
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.stdout.reconfigure(encoding="utf-8")
for fp in [r"C:\Windows\Fonts\malgun.ttf"]:
    try: font_manager.fontManager.addfont(fp)
    except Exception: pass
plt.rcParams["font.family"]="Malgun Gothic"; plt.rcParams["axes.unicode_minus"]=False

d=pd.read_csv("lauder3_vis.csv")
def rat(t,a): return 1/(1+a*t)
def fit(col):
    m=d[["tau",col]].dropna(); t=m.tau.values; T=np.clip(m[col].values,0,1.4)
    p,_=curve_fit(rat,t,T,p0=[.1],bounds=(0,5),maxfev=10000); return t,T,p[0]

PAPER="#ffffff"; INK="#2a2433"; GRID="#ece8f4"
# 가시광=차분한 청록/슬레이트, UVB=진보라, UVA=자홍
BANDS=[("가시광선 (눈에 보이는 빛)","T440","#3f7c8c"),
       ("UVB (자외선 B)","T305","#5b3a9e"),
       ("UVA (자외선 A)","T380","#c05a90")]

fig,ax=plt.subplots(figsize=(8.6,5.4))
fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
xx=np.linspace(0,60,200)
print("차단율 적합:")
for name,col,c in BANDS:
    t,T,a=fit(col)
    block=100*(1-np.clip(T,0,1))       # 차단율(%)
    ax.scatter(t,block,s=16,color=c,alpha=.28,edgecolor="none")
    ax.plot(xx,100*(1-rat(xx,a)),color=c,lw=2.8,label=name,solid_capstyle="round")
    print(f"  {name}: a={a:.3f}  τ=40 차단율 {100*(1-rat(40,a)):.0f}%")

ax.annotate("먹구름이라도\n자외선의 20~30%는 통과", xy=(40,100*(1-rat(40,0.10))),
            xytext=(41,44), color=INK, fontsize=11.5, ha="left",
            arrowprops=dict(arrowstyle="->",color="#948fa4",lw=1.3))
ax.set_xlabel("구름 두께  (두꺼울수록 오른쪽) →", fontsize=12.5, color=INK)
ax.set_ylabel("차단율 (%)\n(구름이 막은 비율)", fontsize=12.5, color=INK)
ax.set_title("구름은 가시광선도 자외선도 비슷하게 막는다", fontsize=15.5, color=INK, pad=14, weight="bold")
ax.set_ylim(0,100); ax.set_xlim(0,60)
ax.grid(True,color=GRID,lw=.9); ax.set_axisbelow(True)
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["left","bottom"]: ax.spines[s].set_color("#d5cee2")
ax.tick_params(colors="#948fa4")
leg=ax.legend(fontsize=11, frameon=True, facecolor=PAPER, edgecolor="#e8e4f0", loc="lower right", title="파장(빛의 종류)")
leg.get_title().set_color(INK); leg.get_title().set_fontsize(10)
for txt in leg.get_texts(): txt.set_color(INK)
fig.tight_layout(); fig.savefig("uv_three.png", dpi=150, facecolor=PAPER)
print("저장 uv_three.png")
