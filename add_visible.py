# -*- coding: utf-8 -*-
"""Lauder 가시광(440nm) 투과율을 계산해 기존 매칭(pooled_matched3의 Lauder행: τ,T305,T380)에 붙임.
   → 가시광 vs UVB vs UVA 차단율 비교용."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
import fetch_ground_uv as FG

# 1) Lauder 지상자료 재추출(440 포함)
FG.fetch_ground_uv("Lauder", [2016,2017,2018], "ground_Lauder_vis.csv", stride=1, verbose=True)
g = pd.read_csv("ground_Lauder_vis.csv")
for c in ["uv_324","vis_440","sza"]:
    g[c] = pd.to_numeric(g[c], errors="coerce")
g = g.dropna(subset=["sza","uv_324","vis_440"])
g = g[(g.sza>=10)&(g.sza<=80)&(g.uv_324>0)&(g.vis_440>0)]
g["mu"] = np.cos(np.radians(g.sza.values))

# 2) 청천(맑은 스캔) 판정 = 오존둔감 324 SZA-포락선
x=g.sza.values; e=g.uv_324.values; cx=[];cy=[]
for lo in np.arange(10,82,2):
    m=(x>=lo)&(x<lo+2)
    if m.sum()>=8: cx.append(lo+1); cy.append(np.percentile(e[m],95))
env=np.interp(x,np.array(cx),np.array(cy)); clear=e>=0.9*env
print(f"맑은 스캔 {clear.sum()}개")

# 3) 가시광 청천 적합(오존 무관): ln E = a+b·lnμ+c·μ
mu=g.mu.values
X=np.column_stack([np.ones_like(mu),np.log(mu),mu])
y=g.vis_440.values
b,*_=np.linalg.lstsq(X[clear],np.log(y[clear]),rcond=None)
g["T440"]=np.clip(y/np.exp(X@b),0,1.4)
g["date"]=g["date"].astype(str); g["sza_r"]=g["sza"].round(1)

# 4) 기존 매칭의 Lauder행에 (date,sza)로 T440 붙이기
p=pd.read_csv("pooled_matched3.csv"); p=p[p.station=="Lauder"].copy()
p["date"]=p["date"].astype(str); p["sza_r"]=p["sza"].round(1)
vis=g.groupby(["date","sza_r"])["T440"].mean().reset_index()
m=p.merge(vis,on=["date","sza_r"],how="left")
m=m.dropna(subset=["T440"])
out=m[["date","sza","tau","T305","T380","T440"]].copy()
out.to_csv("lauder3_vis.csv",index=False)
print(f"\n매칭된 Lauder 관측 {len(out)}개 (τ·UVB·UVA·가시광 모두 있는 행)")
print(out.sort_values("tau").head(3).to_string(index=False))
print(out.sort_values("tau").tail(3).to_string(index=False))
