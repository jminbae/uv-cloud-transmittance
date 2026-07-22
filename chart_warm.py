# -*- coding: utf-8 -*-
"""따뜻한 톤 + 한글 라벨로 투과율 그림 재생성(일반인용)."""
import sys, numpy as np, pandas as pd
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.stdout.reconfigure(encoding="utf-8")

# 한글 폰트(윈도우 맑은 고딕)
for fp in [r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\malgunsl.ttf"]:
    try: font_manager.fontManager.addfont(fp)
    except Exception: pass
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

d = pd.read_csv("pooled_matched3.csv")
def rat(t,a): return 1/(1+a*t)
def fit(col):
    m=d[["tau",col]].dropna(); t=m.tau.values; T=np.clip(m[col].values,0,1.4)
    p,_=curve_fit(rat,t,T,p0=[.1],bounds=(0,5),maxfev=10000); return t,T,p[0]

# 따뜻한 아이보리 배경 + UV 보라 계열 선(UV=보라 너머의 빛). UVA는 가시광에 가까워 살짝 따뜻한 보라.
PAPER="#f7f3ec"; INK="#2a231b"; GRID="#e2d9c8"
BANDS=[("UVB (305nm)","T305","#5b3a9e"),("UVB (310nm)","T310","#7d5bc6"),
       ("보라 근처 (324nm)","T324","#a493d1"),("UVA (380nm)","T380","#c05a90")]

fig,ax=plt.subplots(figsize=(8.6,5.4))
fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
xx=np.linspace(0,60,200)
for name,col,c in BANDS:
    t,T,a=fit(col)
    ax.scatter(t,T,s=15,color=c,alpha=.28,edgecolor="none")
    ax.plot(xx,rat(xx,a),color=c,lw=2.6,label=f"{name}",solid_capstyle="round")

# 안내 주석
ax.annotate("맑음: 거의 다 통과", xy=(1.5,0.96), xytext=(14,1.06), color=INK, fontsize=11,
            arrowprops=dict(arrowstyle="->",color="#8a7d6b",lw=1.2))
ax.annotate("먹구름(맨 오른쪽):\nUV의 약 30%만 통과", xy=(40,rat(40,0.062)), xytext=(45,0.62),
            color=INK, fontsize=11, ha="left",
            arrowprops=dict(arrowstyle="->",color="#8a7d6b",lw=1.2))

ax.set_xlabel("구름 두께  (두꺼울수록 오른쪽) →", fontsize=12.5, color=INK)
ax.set_ylabel("자외선 통과율\n(구름 있을 때 ÷ 없었다면)", fontsize=12.5, color=INK)
ax.set_title("구름이 두꺼울수록 자외선은 얼마나 통과할까", fontsize=15.5, color=INK, pad=14, weight="bold")
ax.set_ylim(0,1.18); ax.set_xlim(0,60)
ax.grid(True,color=GRID,lw=.9); ax.set_axisbelow(True)
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["left","bottom"]: ax.spines[s].set_color("#c9bda8")
ax.tick_params(colors="#8a7d6b")
leg=ax.legend(fontsize=10.5, frameon=True, facecolor=PAPER, edgecolor="#e0d7c6", loc="upper right")
for txt in leg.get_texts(): txt.set_color(INK)
fig.tight_layout()
fig.savefig("uv_warm.png", dpi=150, facecolor=PAPER)
print("저장 uv_warm.png | 한글 렌더 확인 필요")
