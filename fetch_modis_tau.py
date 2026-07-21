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
        return np.nan, 0
    for g in res:
        files = earthaccess.download(g, dldir)
        got = None
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
                sd.end()
            except Exception as e:
                if verbose: print(f"    판독오류 {os.path.basename(fp)}: {str(e)[:60]}")
            finally:
                if not keep:
                    try: os.remove(fp)
                    except Exception: pass
        if got is not None:
            return got, 1      # 지점을 덮은 첫 그래뉼에서 종료(다운로드 절감)
    return np.nan, 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    LAT, LON = -45.038, 169.684   # Lauder
    for date, label in [("2018-12-25","맑았던 날"),("2018-12-22","흐렸던 날")]:
        t, n = modis_tau(date, LAT, LON)
        print(f"{date} ({label}): MODIS τ = {t:.2f}  (그래뉼 {n}개)")
