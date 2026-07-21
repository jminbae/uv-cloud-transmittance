# -*- coding: utf-8 -*-
"""MODIS Aqua(MYD06_L2) 구름광학두께 τ를 특정 지점·날짜에서 읽기(독립 τ, ~13:30 통과).
   지점 최근접 5km 셀 주변 1km COT 평균. earthaccess 인증(.netrc/_netrc) 사용."""
import os, glob, sys, numpy as np
import earthaccess
from pyhdf.SD import SD, SDC

_AUTH = None
def _login():
    global _AUTH
    if _AUTH is None:
        _AUTH = earthaccess.login(strategy="netrc")
    return _AUTH

def _read_scaled(sd, name):
    v = sd.select(name); a = v.get().astype("float64"); at = v.attributes()
    fv = at.get("_FillValue")
    if fv is not None: a[a == fv] = np.nan
    off = at.get("add_offset", 0.0) or 0.0
    sc = at.get("scale_factor", 1.0) or 1.0
    return sc * (a - off)

def modis_tau(date, lat, lon, dldir="modis_tmp", max_km=8.0, keep=False, verbose=True):
    """date 'YYYY-MM-DD' 하루의 Aqua 그래뉼에서 (lat,lon) 최근접 τ 평균. 없으면 nan."""
    _login()
    os.makedirs(dldir, exist_ok=True)
    res = earthaccess.search_data(short_name="MYD06_L2", version="6.1",
                                  temporal=(f"{date}T00:00:00", f"{date}T23:59:59"),
                                  bounding_box=(lon-0.2, lat-0.2, lon+0.2, lat+0.2))
    if not res:
        if verbose: print(f"  {date}: 그래뉼 없음")
        return np.nan, 0, np.nan
    # 주간 통과 그래뉼만 선별: 야간 통과는 구름광학두께(COT) 없음 → τ=0 오판·시각 어긋남.
    # 그래뉼명 A{YYYYDDD}.{HHMM}(UTC)에 lon/15을 더해 현지시각 추정, 주간(7~17시)만.
    def _oh(g):
        try:  # 파일명은 data_links에 (GranuleUR은 'LAADS:숫자'라 못 씀)
            b = g.data_links()[0].split("/")[-1]; tok = b.split(".A")[1].split(".")
            return int(tok[1][:2]) + int(tok[1][2:])/60.0
        except Exception: return np.nan
    cand = []
    for g in res:
        oh = _oh(g)
        if not np.isfinite(oh):
            cand.append((99.0, g, np.nan)); continue   # 파싱실패 → 최후순위
        olocal = (oh + lon/15.0) % 24.0
        if 7.0 <= olocal <= 17.0:                        # 주간 통과만(야간 제외)
            cand.append((abs(olocal - 13.5), g, oh))
    cand.sort(key=lambda x: x[0])   # 현지 정오(13.5시)에 가까운 주간 통과 우선
    for _key, g, oh_meta in cand:
        files = earthaccess.download(g, dldir)
        got = None; over_h = oh_meta
        for fp in files:
            fp = os.fspath(fp)
            if not fp.endswith(".hdf"):
                continue
            try:
                sd = SD(fp, SDC.READ)
                la = sd.select("Latitude").get(); lo = sd.select("Longitude").get()
                d2 = (la-lat)**2 + (lo-lon)**2
                i, j = np.unravel_index(np.argmin(d2), la.shape)
                dk = 111.0*np.sqrt((la[i,j]-lat)**2 + ((lo[i,j]-lon)*np.cos(np.radians(lat)))**2)
                if dk <= max_km:
                    cot = _read_scaled(sd, "Cloud_Optical_Thickness")  # 1km
                    n5, m5 = la.shape
                    bi, bj = i*(cot.shape[0]//n5), j*(cot.shape[1]//m5)
                    blk = cot[bi:bi+5, bj:bj+5]
                    valid = blk[np.isfinite(blk)]
                    # 구름 미검출(clear)이면 COT는 fill → 유효 없으면 τ=0(맑음)
                    got = float(np.mean(valid)) if valid.size else 0.0
                    try:  # 파일명 MYD06_L2.AYYYYDDD.HHMM... 에서 통과 UTC 시각(시)
                        tok = os.path.basename(fp).split(".")[2]
                        over_h = int(tok[:2]) + int(tok[2:4])/60.0
                    except Exception: over_h = np.nan
                sd.end()
            except Exception as e:
                if verbose: print(f"    판독오류 {os.path.basename(fp)}: {str(e)[:60]}")
            finally:
                if not keep:
                    try: os.remove(fp)
                    except Exception: pass
        if got is not None:
            return got, 1, over_h   # (τ, 그래뉼수, 통과UTC시각[시])
    return np.nan, 0, np.nan


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    LAT, LON = -45.038, 169.684   # Lauder
    for date, label in [("2018-12-25","맑았던 날"),("2018-12-22","흐렸던 날")]:
        t, n, oh = modis_tau(date, LAT, LON)
        print(f"{date} ({label}): MODIS τ = {t:.2f}  통과UTC {oh:.2f}h  (그래뉼 {n}개)")
