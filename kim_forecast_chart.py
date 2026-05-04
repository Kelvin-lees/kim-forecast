"""
KIM 모델 예측 결과를 활용한 한반도 종합 예보 차트 생성

입력 변수:
    - 강수량 (precipitation, mm)
    - 해면기압 (mean sea level pressure, hPa)
    - 10m 바람 (u10, v10, m/s)
    - 최하층 운량 (low cloud cover, %)
    - 2m 기온 (temperature, °C)

출력:
    예측 3일까지 3시간 간격 → 시각별 PNG 한 장씩
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import scipy.ndimage as ndimage
import matplotlib.patheffects as patheffects


# ============================================================
# 1. 설정 (Configuration)
# ============================================================

# 한반도 도메인 (경도 min, 경도 max, 위도 min, 위도 max)
KOREA_EXTENT = [123.0, 132.0, 32.0, 43.0]

# 입력 및 출력 디렉토리
INPUT_DIR = "/Users/lees/nwp/aiml/esaia"
OUTPUT_DIR = "./outputs/kim_charts"

# 변수명 매핑 (KIM 출력 파일의 실제 변수명에 맞게 수정 필요)
VAR_NAMES = {
    "precip": "prec_acc",     # precipitation amount (kg/m2)
    "mslp":   "psl",          # mean sea level pressure (Pa or hPa)
    "u10":    "u10m",         # 10m u-wind (m/s)
    "v10":    "v10m",         # 10m v-wind (m/s)
    "lcc":    "lcld",         # low cloud cover
    "t2m":    "t2m",          # 2m temperature (K or °C)
    "topo":   "topo",         # topography (m)
    "slmsk":  "slmsk",        # sea-land mask
}

# 시각화 파라미터
PRECIP_LEVELS = [0.1, 0.5, 1, 2, 5, 10, 20, 30, 50, 70, 100]   # mm
MSLP_LEVELS   = np.arange(960, 1060, 2)                         # hPa, 2hPa 간격
TEMP_LEVELS   = np.arange(-30, 45, 3)                           # °C, 2°C 간격
WIND_THRESHOLD = 5.0       # m/s 미만은 마스킹
CLOUD_THRESHOLD = 60.0     # % 이상만 해칭
WIND_SKIP = 6              # 바람 벡터 솎기 간격 (격자점 단위)


# ============================================================
# 2. 데이터 로드 및 단위 정규화
# ============================================================

def load_kim_data(filepath: str) -> xr.Dataset:
    """
    KIM 출력 파일을 읽어 xarray Dataset으로 반환.
    GRIB2 / NetCDF 자동 감지.
    """
    if filepath.endswith((".grib", ".grib2", ".gb2")):
        ds = xr.open_dataset(filepath, engine="cfgrib")
    else:
        ds = xr.open_dataset(filepath)
    return ds


def normalize_units(ds: xr.Dataset) -> xr.Dataset:
    """
    변수별 단위를 표준 단위로 변환.
    - 기압: Pa → hPa
    - 기온: K → °C
    - 운량: 0~1 → 0~100 (%)
    """
    # 기압
    mslp_var = VAR_NAMES["mslp"]
    if mslp_var in ds and ds[mslp_var].max() > 2000:   # Pa로 추정
        ds[mslp_var] = ds[mslp_var] / 100.0
        ds[mslp_var].attrs["units"] = "hPa"

    # 기온
    t2m_var = VAR_NAMES["t2m"]
    if t2m_var in ds and ds[t2m_var].max() > 200:      # K로 추정
        ds[t2m_var] = ds[t2m_var] - 273.15
        ds[t2m_var].attrs["units"] = "degC"

    # 운량
    lcc_var = VAR_NAMES["lcc"]
    if lcc_var in ds and ds[lcc_var].max() <= 1.0:     # 0~1로 추정
        ds[lcc_var] = ds[lcc_var] * 100.0
        ds[lcc_var].attrs["units"] = "%"

    return ds


def subset_korea(ds: xr.Dataset) -> xr.Dataset:
    """한반도 도메인으로 잘라내기."""
    lon_min, lon_max, lat_min, lat_max = KOREA_EXTENT

    # 좌표 이름 자동 감지 (longitude/lon/lons, latitude/lat/lats)
    lon_name = "longitude" if "longitude" in ds.coords else ("lons" if "lons" in ds.coords else "lon")
    lat_name = "latitude"  if "latitude"  in ds.coords else ("lats" if "lats" in ds.coords else "lat")

    # 위도 정렬 방향 확인 (북→남 또는 남→북)
    lat_vals = ds[lat_name].values
    if lat_vals[0] > lat_vals[-1]:   # 북→남으로 정렬된 경우
        ds = ds.sel({lat_name: slice(lat_max, lat_min),
                     lon_name: slice(lon_min, lon_max)})
    else:
        ds = ds.sel({lat_name: slice(lat_min, lat_max),
                     lon_name: slice(lon_min, lon_max)})
    return ds


# ============================================================
# 3. 단일 시각 차트 그리기
# ============================================================

def plot_single_timestep(ds_t: xr.Dataset, valid_time, init_time, fhour: int,
                          output_path: str):
    """
    단일 예측 시각에 대해 한반도 종합 차트 한 장 생성.

    Parameters
    ----------
    ds_t : xr.Dataset
        해당 시각의 데이터 (시간 차원 없음)
    valid_time : datetime-like
        예측 유효 시각
    init_time : datetime-like
        모델 초기 시각
    fhour : int
        예측 선행 시간 (시간 단위)
    output_path : str
        PNG 저장 경로
    """
    # 좌표 추출
    lon_name = "longitude" if "longitude" in ds_t.coords else ("lons" if "lons" in ds_t.coords else "lon")
    lat_name = "latitude"  if "latitude"  in ds_t.coords else ("lats" if "lats" in ds_t.coords else "lat")
    lons = ds_t[lon_name].values
    lats = ds_t[lat_name].values
    LON, LAT = np.meshgrid(lons, lats)

    # 변수 추출 및 크기가 1인 차원(time_ini 등) 제거
    precip = np.squeeze(ds_t[VAR_NAMES["precip"]].values)
    mslp   = np.squeeze(ds_t[VAR_NAMES["mslp"]].values)
    t2m    = np.squeeze(ds_t[VAR_NAMES["t2m"]].values)
    lcc    = np.squeeze(ds_t[VAR_NAMES["lcc"]].values)
    u10    = np.squeeze(ds_t[VAR_NAMES["u10"]].values)
    v10    = np.squeeze(ds_t[VAR_NAMES["v10"]].values)
    topo   = np.squeeze(ds_t[VAR_NAMES["topo"]].values)
    slmsk  = np.squeeze(ds_t[VAR_NAMES["slmsk"]].values)

    # ----- Figure 생성 -----
    fig = plt.figure(figsize=(11, 11))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(KOREA_EXTENT, crs=ccrs.PlateCarree())

    # 지리 정보
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="black")
    # 행정구역(도) 경계선 추가
    provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none')
    ax.add_feature(provinces, edgecolor='gray', linewidth=0.4, alpha=0.7, zorder=0.8)
    # ax.add_feature(cfeature.BORDERS,    linewidth=0.6, edgecolor="black") # 휴전선 등 국경선 제거
    ax.add_feature(cfeature.LAND.with_scale("50m"),
                   facecolor="#ffffff", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"),
                   facecolor="#e8f1f5", zorder=0)

    # 격자선
    gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                      color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    # ----- (0.5) 지형 높이 (육지 마스킹) 백그라운드 -----
    topo_masked = np.where(slmsk == 1, topo, np.nan)
    ax.pcolormesh(LON, LAT, topo_masked,
                  cmap="Greys", vmin=0, vmax=1800,
                  transform=ccrs.PlateCarree(),
                  zorder=0.5, alpha=0.8)

    # ----- (1) 강수량 음영 (배경) -----
    # 일반적인 기상 레이더(Radar) 강수량 색상 테이블 적용
    radar_colors = [
        "#00ECEC", # 0.1 - 0.5 (Cyan)
        "#01A0F6", # 0.5 - 1 (Light Blue)
        "#0000F6", # 1 - 2 (Blue)
        "#00FF00", # 2 - 5 (Green)
        "#00C800", # 5 - 10 (Dark Green)
        "#FFFF00", # 10 - 20 (Yellow)
        "#FF9000", # 20 - 30 (Orange)
        "#FF0000", # 30 - 50 (Red)
        "#D60000", # 50 - 70 (Dark Red)
        "#FF00FF"  # 70 - 100 (Magenta)
    ]
    precip_cmap = mcolors.ListedColormap(radar_colors)
    precip_cmap.set_over("#800080")  # 100 이상은 Purple
    precip_norm = mcolors.BoundaryNorm(PRECIP_LEVELS, precip_cmap.N)

    # 0.1mm 미만은 투명
    precip_masked = np.where(precip >= PRECIP_LEVELS[0], precip, np.nan)
    cf = ax.contourf(LON, LAT, precip_masked,
                     levels=PRECIP_LEVELS,
                     cmap=precip_cmap, norm=precip_norm,
                     extend="max", transform=ccrs.PlateCarree(),
                     zorder=1, alpha=0.5)
                     
    # 강수량 구분을 위한 흰색 등치선
    ax.contour(LON, LAT, precip_masked,
               levels=PRECIP_LEVELS,
               colors="white",
               linewidths=0.5,
               transform=ccrs.PlateCarree(),
               zorder=1.1, alpha=0.8)

    # ----- (2) 운량 70% 이상 네이비색 점선 해칭 -----
    old_hatch_color = plt.rcParams['hatch.color']
    old_hatch_lw = plt.rcParams['hatch.linewidth']
    plt.rcParams['hatch.color'] = 'navy'
    plt.rcParams['hatch.linewidth'] = 0.5

    cloud_mask = np.where(lcc >= CLOUD_THRESHOLD, 1, np.nan)
    ax.contourf(LON, LAT, cloud_mask,
                levels=[0.5, 1.5],
                colors="none",
                hatches=["....."],
                transform=ccrs.PlateCarree(),
                zorder=2)

    # 원래 설정으로 복구
    plt.rcParams['hatch.color'] = old_hatch_color
    plt.rcParams['hatch.linewidth'] = old_hatch_lw

    # ----- (3) 기온 등치선 (빨간 점선) -----
    t2m_smooth = ndimage.gaussian_filter(t2m, sigma=1.5)

    # ----- (3.5) 기온 33도 이상 폭염 구역 빨간 빗금 해칭 -----
    old_hatch_color = plt.rcParams['hatch.color']
    old_hatch_lw = plt.rcParams['hatch.linewidth']
    plt.rcParams['hatch.color'] = 'red'
    plt.rcParams['hatch.linewidth'] = 0.5

    heat_mask = np.where(t2m_smooth >= 33.0, 1, np.nan)
    ax.contourf(LON, LAT, heat_mask,
                levels=[0.5, 1.5],
                colors="none",
                hatches=["////"],
                transform=ccrs.PlateCarree(),
                zorder=2.5)

    # 원래 설정으로 복구
    plt.rcParams['hatch.color'] = old_hatch_color
    plt.rcParams['hatch.linewidth'] = old_hatch_lw

    cs_temp = ax.contour(LON, LAT, t2m_smooth,
                         levels=TEMP_LEVELS,
                         colors="red",
                         linestyles="--",
                         linewidths=1.2,
                         transform=ccrs.PlateCarree(),
                         zorder=3, alpha=0.8)
    ax.clabel(cs_temp, inline=True, fontsize=8, fmt="%d°C")

    # ----- (4) 해면기압 등압선 (검은 실선) -----
    mslp_smooth = ndimage.gaussian_filter(mslp, sigma=1.5)
    cs_mslp = ax.contour(LON, LAT, mslp_smooth,
                         levels=MSLP_LEVELS,
                         colors="black",
                         linestyles="-",
                         linewidths=1.5,
                         transform=ccrs.PlateCarree(),
                         zorder=4)
    ax.clabel(cs_mslp, inline=True, fontsize=9, fmt="%d")

    # ----- (4.5) 저기압(L) 중심 표시 -----
    neighborhood_size = 40
    local_min = (ndimage.minimum_filter(mslp_smooth, size=neighborhood_size, mode='nearest') == mslp_smooth)

    # 가장자리(경계선)에서 발생하는 가짜 극값 제거 (테두리 5격자)
    local_min[:5, :] = False; local_min[-5:, :] = False
    local_min[:, :5] = False; local_min[:, -5:] = False

    # 저기압(L) - 빨간색
    for y, x in zip(*np.where(local_min)):
        ax.text(LON[y, x], LAT[y, x], "L", color="red", fontsize=15, fontweight="bold",
                ha="center", va="center", transform=ccrs.PlateCarree(),
                path_effects=[patheffects.withStroke(linewidth=2, foreground="white")], zorder=6)

    # ----- (5) 바람 벡터 (5m/s 이상) -----
    wspd = np.sqrt(u10 ** 2 + v10 ** 2)
    u_masked = np.where(wspd >= WIND_THRESHOLD, u10, np.nan)
    v_masked = np.where(wspd >= WIND_THRESHOLD, v10, np.nan)

    skip = (slice(None, None, WIND_SKIP), slice(None, None, WIND_SKIP))
    ax.barbs(LON[skip], LAT[skip],
             u_masked[skip], v_masked[skip],
             length=5.5, linewidth=0.6,
             barbcolor="#1a1a1a", flagcolor="#1a1a1a",
             transform=ccrs.PlateCarree(),
             zorder=5)

    # ----- 컬러바 (강수량) -----
    cbar = plt.colorbar(cf, ax=ax, orientation="horizontal",
                        pad=0.06, shrink=0.7, aspect=35,
                        ticks=PRECIP_LEVELS)
    cbar.set_label("Precipitation (mm / 3h)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # ----- 제목 및 정보 -----
    init_str  = _fmt_time(init_time)
    valid_str = _fmt_time(valid_time)
    title = (f"KIM Forecast — Korea\n"
             f"Init: {init_str}  |  Valid: {valid_str}  |  F+{fhour:03d}h")
    ax.set_title(title, fontsize=12, loc="left", pad=10)

    # ----- 범례 (텍스트) -----
    legend_text = (
        "■ Color shading & White contour: Precip\n"
        "─ Black solid: MSLP (hPa, 2hPa) / L(Red)\n"
        "┄ Red dashed: Temperature (°C, 3°C)\n"
        "//// Red hatch: Temp ≥ 33°C\n"
        "··· Navy hatch: Low cloud ≥ 60%\n"
        "↗ Wind barbs: ≥ 5 m/s"
    )
    ax.text(0.02, 0.02, legend_text,
            transform=ax.transAxes,
            fontsize=8, verticalalignment="bottom",
            bbox=dict(facecolor="white", alpha=0.85,
                      edgecolor="gray", boxstyle="round,pad=0.4"))

    # 저장
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {output_path}")


def _fmt_time(t) -> str:
    """numpy/pandas/datetime → 'YYYY-MM-DD HH UTC' 문자열."""
    if hasattr(t, "values"):
        t = t.values
    t = np.datetime64(t, "h")
    dt = datetime.utcfromtimestamp(
        (t - np.datetime64("1970-01-01T00", "h")).astype(int) * 3600
    )
    return dt.strftime("%Y-%m-%d %H UTC")


# ============================================================
# 4. 메인 루프 (전체 예측 시각 처리)
# ============================================================

def generate_all_charts(input_dir: str,
                         output_dir: str = OUTPUT_DIR,
                         max_fhour: int = 72,
                         step_hour: int = 3):
    """
    KIM 출력 파일 전체에 대해 시각별 차트 생성.

    Parameters
    ----------
    input_dir : str
        KIM 출력 파일들이 위치한 디렉토리
    output_dir : str
        저장 디렉토리
    max_fhour : int
        최대 선행 시간 (시간), 기본 72h (3일)
    step_hour : int
        시간 간격 (시간), 기본 3h
    """
    import glob
    os.makedirs(output_dir, exist_ok=True)

    search_pattern = os.path.join(input_dir, "g576_v091_easia*.nc")
    file_list = sorted(glob.glob(search_pattern))
    
    if not file_list:
        print(f"No files matching {search_pattern} found.")
        return

    print(f"Found {len(file_list)} files. Generating charts...")

    # 첫 번째 파일에서 초기 시각(init_time) 추출
    ds_init = load_kim_data(file_list[0])
    
    time_name = "valid_time" if "valid_time" in ds_init.coords else (
                "time" if "time" in ds_init.coords else "step")
    
    # 시간 추출 (보통 time의 첫번째 값 혹은 attribute의 initial_time 활용)
    if "initial_time" in ds_init.attrs:
        init_time_str = ds_init.attrs["initial_time"]
        # e.g. "MON APR 27 00:00:00 2026"
        from datetime import datetime
        try:
            init_time = datetime.strptime(init_time_str[4:], "%b %d %H:%M:%S %Y")
            init_time = np.datetime64(init_time)
        except ValueError:
            init_time = ds_init[time_name].values[0]
    else:
        init_time = ds_init[time_name].values[0]

    for filepath in file_list:
        ds = load_kim_data(filepath)
        ds = normalize_units(ds)
        ds = subset_korea(ds)

        times = ds[time_name].values
        # 파일 하나당 보통 하나의 timestep이 들어있다고 가정
        for i, t in enumerate(times):
            # 선행 시간 계산
            delta_h = int((np.datetime64(t) - np.datetime64(init_time))
                          / np.timedelta64(1, "h"))

            # 3일 초과 또는 3시간 간격이 아니면 스킵
            if delta_h > max_fhour:
                break
            if delta_h % step_hour != 0:
                continue

            ds_t = ds.isel({time_name: i})
            
            # precipitation이 3h 누적이 아니라면 처리가 필요할 수 있으나, 현재는 파일 내 변수값을 그대로 표시
            out_name = f"kim_korea_F{delta_h:03d}h.png"
            out_path = os.path.join(output_dir, out_name)

            print(f"[{os.path.basename(filepath)}] F+{delta_h:03d}h")
            plot_single_timestep(ds_t,
                                 valid_time=t, init_time=init_time,
                                 fhour=delta_h, output_path=out_path)

    print(f"\nDone. Charts saved to: {output_dir}")


# ============================================================
# 5. 실행 진입점
# ============================================================

if __name__ == "__main__":
    print(f"Generating charts from input directory: {INPUT_DIR}")
    generate_all_charts(input_dir=INPUT_DIR,
                        output_dir=OUTPUT_DIR,
                        max_fhour=72,
                        step_hour=3)
