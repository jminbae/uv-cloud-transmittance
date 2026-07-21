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
STATIONS=[  # (표시명, WOUDC명, lat, lon, 연도들, UVA380, stride)
 ("MaunaLoa","Mauna Loa", 19.536,-155.576,[2017,2018],False, 8),
 ("Tateno",  "Tateno",    36.058, 140.126,[2019,2020],False, 8),
 ("Uccle",   "Uccle",     50.80,    4.35, [2022,2023],False,16),
 ("Lauder",  "Lauder",   -45.038, 169.684,[2016,2017,2018],True, 1),
 ("Ushuaia", "Ushuaia",  -54.85,  -68.31, [2003,2004,2006,2007],True, 8),
]
DAYS_PER_STN=32
POOLED="pooled_matched3.csv"

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
    env=np.interp(x,np.array(cx),np.array(cy)); clear=e>=0.85*env   # 흐린 관측소(Ushuaia) 대비 완화
    if clear.sum()<10: return None
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

def th_hours(s):
    try: return int(str(s)[:2]) + int(str(s)[3:5])/60.0
    except Exception: return np.nan

# --- 재개용: 이미 처리한 (station,date) ---
done=set()
if os.path.exists(POOLED):
    try:
        p=pd.read_csv(POOLED); done=set(zip(p.station,p.date.astype(str)))
    except Exception: pass
cols=["station","lat","date","sza","dt_min","T305","T310","T324","T380","tau"]
first=not os.path.exists(POOLED)

for disp,wname,lat,lon,years,uva,stride in STATIONS:
    print(f"\n===== {disp} ({lat:.1f}) years={years} UVA={uva} stride={stride} =====",flush=True)
    gcsv=f"ground3_{disp}.csv"   # 다년·UTC오프셋 포함 재추출본
    if not os.path.exists(gcsv):
        try: FG.fetch_ground_uv(wname, years, gcsv, stride=stride, verbose=True)
        except SystemExit as e: print("  지상 추출 실패:",e); continue
        except Exception as e: print("  지상 오류:",e); continue
    tdf=transmittance(gcsv, wname, years, uva)
    if tdf is None: print("  투과율 계산 실패(맑은스캔 부족 등)"); continue
    tdf["date"]=tdf["date"].astype(str)
    tdf["th"]=tdf["time_local"].map(th_hours)
    tdf["uoff"]=pd.to_numeric(tdf.get("utcoffset"),errors="coerce")
    tdf=tdf.dropna(subset=["th","uoff"])
    tdf["scan_utc"]=(tdf["th"]-tdf["uoff"])%24.0    # 스캔별 UTC(서머타임 자동 반영)
    days=sorted(tdf["date"].unique())
    step=max(1,len(days)//DAYS_PER_STN)
    sample=days[::step][:DAYS_PER_STN]
    print(f"  투과율 {len(tdf)}스캔 / {len(days)}일 → 표본 {len(sample)}일",flush=True)
    for d in sample:
        if (disp,d) in done: continue
        try: tau,n,over_h=modis_tau(d,lat,lon,verbose=False)
        except Exception as e: print(f"    {d} MODIS오류 {str(e)[:50]}"); continue
        if not np.isfinite(tau): continue
        day=tdf[tdf["date"]==d].copy()
        if len(day)==0: continue
        if np.isfinite(over_h):
            day["dt"]=np.minimum((day["scan_utc"]-over_h)%24.0,(over_h-day["scan_utc"])%24.0)  # UTC 원형거리(시)
            best=day.loc[day["dt"].idxmin()]; dt=float(best["dt"])
        else:
            best=day.loc[day["sza"].idxmin()]; dt=np.nan
        if np.isfinite(dt) and dt>0.67:   # 통과시각 ±40분 내 스캔 없음 → 건너뜀
            continue
        def R(v): return round(float(v),3) if (v==v) else ""
        rec={"station":disp,"lat":lat,"date":d,"sza":round(float(best.sza),1),
             "dt_min":round(dt*60,0) if np.isfinite(dt) else "",
             "T305":R(best.T305),"T310":R(best.T310),"T324":R(best.T324),
             "T380":R(best.T380) if "T380" in best else "","tau":round(float(tau),2)}
        pd.DataFrame([rec])[cols].to_csv(POOLED,mode="a",header=first,index=False); first=False
        print(f"    {disp} {d} τ={tau:5.1f} Δt={rec['dt_min']}분 T305={rec['T305']} T380={rec['T380']}",flush=True)
print("\n=== 전체 완료 ===",flush=True)
