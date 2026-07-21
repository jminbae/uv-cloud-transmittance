# 구름 광학두께에 따른 UV 차단율 (Cloud UV transmittance vs cloud optical depth)

지상 분광 UV 실측을 오존보정 청천으로 나눈 **투과율**을, 독립 위성(MODIS) **구름 광학두께 τ**에 대해 회귀하여 **UVA·UVB·전체 UV**의 구름 차단율을 규명한 분석.

## 📄 보고서 (웹)
[**analysis report →**](https://jminbae.github.io/uv-cloud-transmittance/) *(GitHub Pages)*

## 핵심 결론
- 구름은 UVA·UVB를 **거의 같은 비율로(회색)** 차단하며, 차단율은 τ에 따라 매끄럽게 커진다.
- 투과율은 산란형 **T ≈ 1/(1+a·τ)** 로 잘 기술됨 (a ≈ 0.05).
- **먹구름(τ≈40)에서 UV의 약 60–66% 차단**. UVA는 UVB보다 아주 조금 덜 막힘.
- 청천의 오존 흡수로 UVB만 강하게 걸러지는 맑은 하늘과는 정반대의 성질.

## 방법 (네 가지 설계 결정)
1. **순환논리 차단** — τ를 산출물 자체가 아니라 독립 위성 **MODIS/Aqua MYD06**에서.
2. **직접측정 전천** — 위성 역산 대신 **지상 분광복사계(WOUDC)** 305/310/324/380nm.
3. **오존보정 청천** — 동시측정 총오존으로 `ln E = a+b·lnμ+c·μ+d·(O₃/μ)` 적합.
4. **통과시각 정밀매칭** — 지상 스캔을 위성 통과시각(±수분)에 맞춤.

## 데이터 (무료·공개)
- 지상 UV·총오존: [WOUDC](https://woudc.org) OGC API
- 구름 τ: NASA [MODIS/Aqua MYD06_L2](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MYD06_L2) (earthaccess, NASA Earthdata 로그인)

## 파이프라인
| 파일 | 역할 |
|---|---|
| `fetch_ground_uv.py` | WOUDC 분광 UV 추출 (스캔당 밴드·SZA·UTC오프셋) |
| `run_multi.py` | 다관측소: 오존보정 청천 → 투과율 → MODIS τ 매칭 (증분·재개) |
| `fetch_modis_tau.py` | MODIS MYD06 τ 지점 추출 + 통과시각 |
| `pooled_regress.py` | 풀링 회귀 + 그림 |

재현: NASA Earthdata 계정(`~/.netrc`) 후 `python run_multi.py` → `python pooled_regress.py`.

## 한계
단일 지점 트랜섹트(열대·남반구 지상 관측소 부재), 관측소별 계수 편차, 청천은 모델 반사실. 자세한 내용은 보고서 참조.
