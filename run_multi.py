# -*- coding: utf-8 -*-
"""다관측소 파이프라인: 지상 분광 UV(오존보정 청천) ↔ MODIS L2 τ 매칭을 관측소별로.
   결과를 pooled_matched.csv에 증분 저장(재개 가능). 비극지 5개 위도 트랜섹트.
"""
import os, sys, numpy as np, pandas as pd, requests
sys.stdout.reconfigure(encoding="utf-8")
import fetch_ground_uv as FG
from fetch_modis_tau import modis_tau
API="https://api.woudc.org"

# (표시명, WOUDC platform_name, lat, lon, 지상연도들, UVA380가능)
STATIONS=[  # (표시명, WOUDC명, lat, lon, 연도, UVA380, stride)
 ("MaunaLoa","Mauna Loa", 19.536,-155.576,[2018],False, 8),
 ("Tateno",  "Tateno",    36.058, 140.126,[2019],False, 8),
 ("Uccle",   "Uccle",     50.80,    4.35, [2022],False,16),
 ("Lauder",  "Lauder",   -45.038, 169.684,[2018],True,  1),
 ("Ushuaia", "Ushuaia",  -54.85,  -68.31, [2004],True,  8),
]
DAYS_PER_STN=20
POOLED="pooled_matched.csv"

def pull_ozone(station, years):
    s=requests.Session(); s.headers.update({"User-Agent":"uv/1.0"}); rows=[]; off=0
    ys=set(years)
    while True:
        r=s.get(f"{API}/collections/totalozone/items",
                params={"f":"json","station_name":station,"limit":1000,"offset":off,"sortby":"observation_date"},timeout=60)
        fs=r.json().get("features",[])
        if not fs: break
        for f in fs:
            p=f["properties"]; d=str(p.get("observation_date"))[:10]; o=p.get("daily_columno3")
            if d[:4].isdigit() and int(d[:4]) in ys and o and o>100: rows.append((d,float(o)))
        off+=1000
        if len(fs)<1000: break
    if not rows: return None
    oz=pd.DataFrame(rows,columns=["date","o3"]).groupby("date").mean().reset_index()
    oz["doy"]=pd.to_datetime(oz["date"]).dt.dayofyear
    return oz

def transmittance(ground_csv, station, years, uva):
    df=pd.read_csv(ground_csv)
    for c in ["uvb_305","uvb_310","uv_324","uva_380","sza","wlmax"]:
        df[c]=pd.to_numeric(df.get(c),errors="coerce")
    df=df.dropna(subset=["sza","uvb_305","uv_324"])
    df=df[(df.sza>=10)&(df.sza<=80)&(df.uvb_305>0)&(df.uv_324>0)]
    df["doy"]=pd.to_datetime(df["date"]).dt.dayofyear
    df["mu"]=np.cos(np.radians(df.sza.values))
    # 오존: 연도별 보간(흐린 날 결측 대비)
    oz=pull_ozone(station,years)
    if oz is None:  # 오존 없으면 계절중앙값으로 대체
        df["o3"]=300.0
    else:
        o3=np.full(367,np.nan)
        for _,r in oz.iterrows(): o3[int(r.doy)]=r.o3
        idx=np.arange(1,367); good=np.isfinite(o3[1:367])
        df["o3"]=np.interp(df.doy.values, idx[good], o3[1:367][good])
    # 맑은 스캔: 오존둔감 324 SZA-포락선
    x=df.sza.values; e=df.uv_324.values; cx=[];cy=[]
    for lo in np.arange(10,82,2):
        m=(x>=lo)&(x<lo+2)
        if m.sum()>=8: cx.append(lo+1); cy.append(np.percentile(e[m],95))
    if len(cx)<3: return None
    env=np.interp(x,np.array(cx),np.array(cy)); clear=e>=0.9*env
    if clear.sum()<30: return None
    mu=df.mu.values; o3v=df.o3.values
    bands={"T305":"uvb_305","T310":"uvb_310","T324":"uv_324"}
    if uva: bands["T380"]="uva_380"
    X=np.column_stack([np.ones_like(mu),np.log(mu),mu,o3v/mu])
    for tb,col in bands.items():
        y=df[col].values.copy()
        if tb=="T380":  # 풀분광 아닌 스캔(파장부족)은 제외
            y=np.where(df.wlmax.values>=379, y, np.nan)
        ok=clear&np.isfinite(y)&(y>0)
        if ok.sum()<20:
            df[tb]=np.nan; continue
        b,*_=np.linalg.lstsq(X[ok],np.log(y[ok]),rcond=None)
        df[tb]=np.clip(y/np.exp(X@b),0,1.4)
    return df

# --- 재개용: 이미 처리한 (station,date) ---
done=set()
if os.path.exists(POOLED):
    try:
        p=pd.read_csv(POOLED); done=set(zip(p.station,p.date.astype(str)))
    except Exception: pass
cols=["station","lat","date","sza","T305","T310","T324","T380","tau"]
first=not os.path.exists(POOLED)

for disp,wname,lat,lon,years,uva,stride in STATIONS:
    print(f"\n===== {disp} ({lat:.1f}) years={years} UVA={uva} stride={stride} =====",flush=True)
    gcsv=f"ground_{disp}.csv"
    if not os.path.exists(gcsv):
        try: FG.fetch_ground_uv(wname, years, gcsv, stride=stride, verbose=True)
        except SystemExit as e: print("  지상 추출 실패:",e); continue
        except Exception as e: print("  지상 오류:",e); continue
    tdf=transmittance(gcsv, wname, years, uva)
    if tdf is None: print("  투과율 계산 실패(맑은스캔 부족 등)"); continue
    tdf["date"]=tdf["date"].astype(str)
    noon=tdf.loc[tdf.groupby("date")["sza"].idxmin()].sort_values("date").reset_index(drop=True)
    step=max(1,len(noon)//DAYS_PER_STN)
    sample=list(noon["date"])[::step][:DAYS_PER_STN]
    print(f"  투과율 {len(tdf)}스캔 / {len(noon)}일 → 표본 {len(sample)}일",flush=True)
    for d in sample:
        if (disp,d) in done: continue
        row=noon[noon.date==d].iloc[0]
        try: tau,n=modis_tau(d,lat,lon,verbose=False)
        except Exception as e: print(f"    {d} MODIS오류 {str(e)[:50]}"); continue
        if not np.isfinite(tau): continue
        rec={"station":disp,"lat":lat,"date":d,"sza":round(float(row.sza),1),
             "T305":round(float(row.T305),3) if np.isfinite(row.T305) else "",
             "T310":round(float(row.T310),3) if np.isfinite(row.T310) else "",
             "T324":round(float(row.T324),3) if np.isfinite(row.T324) else "",
             "T380":round(float(row.get("T380",np.nan)),3) if ("T380" in row and np.isfinite(row.T380)) else "",
             "tau":round(float(tau),2)}
        pd.DataFrame([rec])[cols].to_csv(POOLED,mode="a",header=first,index=False); first=False
        print(f"    {disp} {d} τ={tau:5.1f} T305={rec['T305']} T380={rec['T380']}",flush=True)
print("\n=== 전체 완료 ===",flush=True)
