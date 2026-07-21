# -*- coding: utf-8 -*-
"""지상 분광 UV 추출기 (WOUDC Spectral) — 관측소·연도별로 스캔당 밴드 조도를 뽑아 CSV.
   전천(all-sky) 실측: 스캔마다 305/310(UVB)·324·380nm(UVA) + 홍반적산(총 UV) + 태양천정각.
   → 다음 단계에서 청천(모델/포락선)·MODIS τ와 결합해 밴드별 구름 차단율 회귀.
"""
import os, sys, time, requests, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

API = "https://api.woudc.org"
TARGETS = (305, 310, 324, 380)


def _records(platform_name, years, verbose=True):
    """WOUDC data_records에서 해당 관측소·연도의 Spectral 파일 목록(URL) 조회(페이지네이션)."""
    out = []
    for ds in ("Spectral_1.0", "Spectral_2.0"):
        offset = 0
        while True:
            r = requests.get(f"{API}/collections/data_records/items",
                             params={"f": "json", "dataset_id": ds, "platform_name": platform_name,
                                     "limit": 500, "offset": offset, "sortby": "timestamp_date"},
                             timeout=60)
            feats = r.json().get("features", [])
            if not feats:
                break
            for f in feats:
                p = f["properties"]
                y = str(p.get("timestamp_date", ""))[:4]
                if y.isdigit() and int(y) in years:
                    out.append((p.get("url"), p.get("instrument_model"), p.get("platform_id")))
            offset += 500
            if len(feats) < 500:
                break
    if verbose:
        print(f"[{platform_name}] Spectral 파일 {len(out)}개 (연도 {sorted(years)})")
    return out


def _parse_file(text):
    """WOUDC Spectral Extended-CSV 한 파일 → 스캔 리스트. 각 스캔: lat,lon,date,time,sza,cie,밴드."""
    lines = text.splitlines()
    scans, i, cur, loc = [], 0, {}, {}
    while i < len(lines):
        ln = lines[i].strip()
        if ln == "#LOCATION":
            v = lines[i + 2].split(",")
            try: loc = {"lat": float(v[0]), "lon": float(v[1]), "alt": float(v[2])}
            except Exception: loc = {}
            i += 3; continue
        if ln == "#TIMESTAMP":
            v = lines[i + 2].split(","); cur = {"date": v[1], "time": v[2]}; i += 3; continue
        if ln == "#GLOBAL_SUMMARY":
            h = lines[i + 1].split(","); v = lines[i + 2].split(","); d = dict(zip(h, v))
            try: cur["sza"] = float(d.get("ZenAngle"))
            except Exception: cur["sza"] = np.nan
            cur["cie"] = d.get("IntCIE"); i += 3; continue
        if ln == "#GLOBAL":
            wl, ir, j = [], [], i + 2
            while j < len(lines) and lines[j].strip() and not lines[j].startswith("#"):
                p = lines[j].split(",")
                try: wl.append(float(p[0])); ir.append(float(p[1]))
                except Exception: pass
                j += 1
            if wl and cur:
                wl = np.array(wl); ir = np.array(ir)
                row = {**loc, **cur, "wlmax": float(wl.max())}
                for t in TARGETS:
                    row[f"e{t}"] = float(ir[np.argmin(np.abs(wl - t))])
                scans.append(row)
            i = j; continue
        i += 1
    return scans


def fetch_ground_uv(platform_name, years, out_csv, max_files=None, verbose=True):
    if isinstance(years, int): years = {years}
    years = set(years)
    recs = _records(platform_name, years, verbose)
    if max_files: recs = recs[:max_files]
    rows = []
    for k, (url, model, sid) in enumerate(recs):
        if not url: continue
        try:
            txt = requests.get(url, timeout=120).text
        except Exception as e:
            if verbose: print(f"  ! 다운로드 실패 {url}: {e}"); continue
        for sc in _parse_file(txt):
            sc["instrument"] = model; sc["stn_id"] = sid
            rows.append(sc)
        if verbose and (k + 1) % 5 == 0:
            print(f"  … {k+1}/{len(recs)} 파일, 누적 {len(rows)} 스캔")
    if not rows:
        raise SystemExit(f"[{platform_name}] 스캔 0개.")
    df = pd.DataFrame(rows)
    df = df.rename(columns={"e305": "uvb_305", "e310": "uvb_310", "e324": "uv_324", "e380": "uva_380",
                            "cie": "erythemal", "date": "date", "time": "time_local"})
    df["station"] = platform_name
    cols = ["station", "stn_id", "lat", "lon", "alt", "date", "time_local", "sza",
            "uvb_305", "uvb_310", "uv_324", "uva_380", "erythemal", "wlmax", "instrument"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(out_csv, index=False)
    if verbose:
        print(f"[{platform_name}] 저장: {out_csv} ({len(df)} 스캔, {df['date'].nunique()}일, "
              f"파장최대 {df['wlmax'].max():.0f}nm)")
    return out_csv


if __name__ == "__main__":
    # 파일럿: Lauder(45S, 풀분광 380nm) 2018년 — 우선 소량으로 모듈 검증
    fetch_ground_uv("Lauder", 2018, "ground_uv_lauder_2018.csv", max_files=3)
