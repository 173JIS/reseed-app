"""
ReSeed — 드론 시드볼 살포 의사결정 시스템 (Python/Streamlit 버전)
변환일: 2026-06-22  |  원본: app_v5.R (R/Shiny)
"""
import math
import io
import base64
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rasterio.transform
from rasterio.warp import transform_bounds
import folium
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ── 경로 설정 ──────────────────────────────────────────────
BASE     = Path(__file__).parent
DATA     = BASE / "data"
LIB_CSV  = DATA / "seedball_library.csv"
ZONE_TIF = DATA / "n3a_situations_4326.tif"
DEMO_TIF = DATA / "result_small.tif"

# ── 상수 ───────────────────────────────────────────────────
ZONE_CODE = ["S1", "S2", "S3", "S4"]
ZONE_NAME = {
    "S1": "긴급 안정화 구역", "S2": "개척 파종 구역",
    "S3": "천이 촉진 구역",   "S4": "하층 보완 구역",
}
ZONE_DESC = {
    "S1": "맨땅·급경사·침식 위험 — 빨리 덮어 흙을 잡아야 하는 곳",
    "S2": "맨땅·완경사 — 처음 식물을 들이는 곳",
    "S3": "풀·작은나무 단계 — 숲으로 키워갈 곳",
    "S4": "큰나무 우거짐 — 그늘 아래를 채울 곳",
}
ZONE_PAL     = {"S1": "#c62828", "S2": "#ef6c00", "S3": "#fbc02d", "S4": "#2e7d32"}
ZONE_PAL_RGB = {"S1": (198,40,40), "S2": (239,108,0), "S3": (251,192,45), "S4": (46,125,50)}
ZONE_VAL_MAP = {1: "S1", 2: "S2", 3: "S3", 4: "S4"}
IDEAL_FORM   = {"S1":["초본","관목"],"S2":["초본","관목"],"S3":["관목","교목"],"S4":["관목","초본"]}
SIT_PREF     = {"S1":"R","S2":"R","S3":"C","S4":"S"}
PREF_KOR     = {"R":"개척형","C":"경쟁형","S":"내성형"}

# ── 페이지 설정 ────────────────────────────────────────────
st.set_page_config(page_title="🌱 ReSeed", layout="wide", page_icon="🌱")
st.markdown("""
<style>
.header-bar{background:#1a4d2e;color:#fff;padding:12px 18px;border-radius:6px;margin-bottom:16px;}
.header-bar h2{margin:0;font-size:20px;}
.header-bar .sub{font-size:12px;opacity:.85;}
.zone-card{border-radius:6px;padding:12px 14px;margin:4px;}
.score-note{font-size:11px;color:#888;background:#fafafa;padding:4px 8px;border-radius:3px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-bar">
  <h2>🌱 ReSeed — 드론 시드볼 살포 의사결정 시스템</h2>
  <div class="sub">드론 영상과 위치를 넣으면 → 대상지를 복원 구역으로 나누고 → 구역마다 뿌리기 좋은 식물을 추천합니다.</div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# 데이터 로딩
# ════════════════════════════════════════════════════════════
@st.cache_data
def load_library() -> pd.DataFrame:
    if not LIB_CSV.exists():
        st.error("data/seedball_library.csv 파일이 없습니다.")
        st.stop()
    df = pd.read_csv(LIB_CSV, encoding="utf-8-sig")
    for col in ["c_percent","s_percent","r_percent","op_establishment","op_safe_growth","op_sourcing"]:
        if col not in df.columns:
            df[col] = np.nan
    return df

def reload_library():
    """캐시 초기화 후 재로딩 (종 추가/삭제 후 호출)"""
    load_library.clear()
    st.session_state["lib"] = load_library()


# ════════════════════════════════════════════════════════════
# 추천 엔진
# ════════════════════════════════════════════════════════════
def _form_match(form: str | float, zone: str) -> float:
    if pd.isna(form):
        return 0.5
    return 1.0 if form in IDEAL_FORM[zone] else 0.3

def _csr_match(c, s, r, zone: str) -> float:
    if any(pd.isna(v) for v in [c, s, r]):
        return 0.5
    v = {"R": float(r), "C": float(c), "S": float(s)}[SIT_PREF[zone]] / 100
    return round(0.4 + 0.6 * v, 2)

def recommend_for(lib: pd.DataFrame, zone: str, n: int = 8) -> pd.DataFrame:
    df = lib.copy()
    df["환경적합"] = df.apply(
        lambda row: round((_form_match(row.get("form_grp"), zone) +
                           _csr_match(row.get("c_percent"), row.get("s_percent"),
                                      row.get("r_percent"), zone)) / 2, 2), axis=1)
    df["현장실행"] = ((df["op_establishment"].fillna(0.5) +
                      df["op_safe_growth"].fillna(0.5)) / 2).round(2)
    df["추천점수"] = ((0.5 * df["환경적합"] + 0.5 * df["현장실행"]) * 100).round().astype(int)
    df["추천이유"] = df.apply(lambda r: _tag_reason(r, zone), axis=1)
    df = df.sort_values(["추천점수","op_safe_growth"], ascending=[False,False]).reset_index(drop=True)
    df["순위"] = range(1, len(df) + 1)
    return df.head(n)

def _tag_reason(row, zone: str) -> str:
    fm = _form_match(row.get("form_grp"), zone)
    fit = f"{ZONE_NAME[zone].replace(' 구역','')}에 맞음" if fm >= 1 else "부분 적합"
    c, s, r = row.get("c_percent"), row.get("s_percent"), row.get("r_percent")
    if any(pd.isna(v) for v in [c, s, r]):
        strat = "전략 미상"
    else:
        dom = ["C","S","R"][int(np.argmax([float(c), float(s), float(r)]))]
        strat = PREF_KOR[dom] + ("(이 구역 강점)" if dom == SIT_PREF[zone] else "")
    es = row.get("op_establishment", 0.5) or 0.5
    sg = row.get("op_safe_growth", 0.7) or 0.7
    est = "빨리 정착" if es >= 0.8 else ("보통 정착" if es >= 0.6 else "느린 정착")
    saf = "안 퍼짐(안전)" if sg >= 0.8 else ("⚠ 잘 퍼짐 주의" if sg <= 0.5 else "보통")
    return f"{fit} · {strat} · {est} · {saf}"


# ════════════════════════════════════════════════════════════
# GIS 유틸
# ════════════════════════════════════════════════════════════
def read_tif_meta(path: Path) -> dict:
    with rasterio.open(path) as src:
        bnds = src.bounds
        if src.crs and src.crs.to_epsg() != 4326:
            b = transform_bounds(src.crs, "EPSG:4326", *bnds)
        else:
            b = (bnds.left, bnds.bottom, bnds.right, bnds.top)
    lon_c = (b[0] + b[2]) / 2
    lat_c = (b[1] + b[3]) / 2
    m_lon = 111320 * math.cos(math.radians(lat_c))
    m_lat = 111320
    w_m = (b[2] - b[0]) * m_lon
    h_m = (b[3] - b[1]) * m_lat
    return {"bounds": b, "lon_c": lon_c, "lat_c": lat_c,
            "w_m": w_m, "h_m": h_m, "area_ha": w_m * h_m / 10_000,
            "nbands": src.count, "width": src.width, "height": src.height}

def make_mission(bounds: tuple, spacing_m: float = 2, margin: float = 0.04,
                 marker_cap: int = 400) -> dict:
    xmn, ymn, xmx, ymx = bounds
    lat_c = (ymn + ymx) / 2
    m_lat = 111320
    m_lon = 111320 * math.cos(math.radians(lat_c))
    lr = (xmx - xmn) * margin
    tr = (ymx - ymn) * margin
    x0, x1, y0, y1 = xmn + lr, xmx - lr, ymn + tr, ymx - tr
    dlat = spacing_m / m_lat
    ys = list(np.arange(y0, y1 + dlat * 0.5, dlat))
    if len(ys) < 2:
        ys = [y0, y1]
    path_pts = []
    for j, y in enumerate(ys):
        path_pts += [[y, x0], [y, x1]] if j % 2 == 0 else [[y, x1], [y, x0]]
    dlon = spacing_m / m_lon
    xs = list(np.arange(x0, x1 + dlon * 0.5, dlon))
    n_way = len(ys) * len(xs)
    grid = [(y, x) for y in ys for x in xs]
    if len(grid) > marker_cap:
        step = max(1, len(grid) // marker_cap)
        grid = grid[::step][:marker_cap]
    return {"path": path_pts, "markers": grid,
            "n_lines": len(ys), "n_way": n_way, "spacing_m": spacing_m}

@st.cache_data
def zone_tif_overlay() -> tuple[str | None, tuple | None]:
    """구역 TIF → base64 PNG (folium ImageOverlay용)"""
    if not ZONE_TIF.exists():
        return None, None
    with rasterio.open(ZONE_TIF) as src:
        data = src.read(1)
        bnds = src.bounds
        b = (bnds.left, bnds.bottom, bnds.right, bnds.top)
    h, w = data.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for val, s in ZONE_VAL_MAP.items():
        mask = data == val
        rgb = ZONE_PAL_RGB[s]
        rgba[mask, 0] = rgb[0]
        rgba[mask, 1] = rgb[1]
        rgba[mask, 2] = rgb[2]
        rgba[mask, 3] = 140  # 반투명
    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}", b

def sample_zone_at_points(markers: list[tuple]) -> list[str | None]:
    """드롭점 좌표 → 해당 구역 코드 반환"""
    if not ZONE_TIF.exists():
        return [None] * len(markers)
    with rasterio.open(ZONE_TIF) as src:
        coords = [(lon, lat) for lat, lon in markers]
        vals = [v[0] for v in src.sample(coords)]
    return [ZONE_VAL_MAP.get(int(v), None) if not np.isnan(v) else None for v in vals]


# ════════════════════════════════════════════════════════════
# 지도 생성
# ════════════════════════════════════════════════════════════
def build_map(meta: dict, lib: pd.DataFrame, spacing_m: float = 2) -> str:
    lat_c, lon_c = meta["lat_c"], meta["lon_c"]
    b = meta["bounds"]

    m = folium.Map(location=[lat_c, lon_c], zoom_start=15, tiles=None)
    folium.TileLayer("Esri.WorldImagery", name="위성", attr="Esri").add_to(m)
    folium.TileLayer("OpenStreetMap", name="일반 지도").add_to(m)

    # 대상지 경계
    folium.Rectangle([[b[1],b[0]],[b[3],b[2]]],
                     color="#f1c40f", weight=2, fill=False,
                     name="대상지").add_to(m)

    # 구역 오버레이
    png_data, zone_bnds = zone_tif_overlay()
    if png_data and zone_bnds:
        folium.raster_layers.ImageOverlay(
            image=png_data,
            bounds=[[zone_bnds[1], zone_bnds[0]], [zone_bnds[3], zone_bnds[2]]],
            opacity=0.5, name="복원 구역",
        ).add_to(m)

    # 드론 경로
    ms = make_mission(b, spacing_m)
    path_fg = folium.FeatureGroup(name="드론 경로", show=True)
    folium.PolyLine(ms["path"], color="#1e88e5", weight=1.5, opacity=0.8).add_to(path_fg)
    path_fg.add_to(m)

    # 드롭점 (구역별 색상 + 뿌릴 종 라벨)
    tops = {s: (recommend_for(lib, s, 1)["name_kor"].iloc[0]
                if len(recommend_for(lib, s, 1)) > 0 else "-")
            for s in ZONE_CODE}
    zone_codes = sample_zone_at_points(ms["markers"])
    drop_fg = folium.FeatureGroup(name="드롭점(뿌릴 종)", show=True)
    for (lat, lon), s in zip(ms["markers"], zone_codes):
        color = ZONE_PAL.get(s, "#888888") if s else "#888888"
        label = f"{ZONE_NAME.get(s,'구역 밖')} → {tops.get(s,'-')} 뿌리기" if s else "구역 밖"
        folium.CircleMarker(
            [lat, lon], radius=3, color="white", weight=0.5,
            fill=True, fill_color=color, fill_opacity=0.95,
            tooltip=label,
        ).add_to(drop_fg)
    drop_fg.add_to(m)

    # 범례
    legend = ('<div style="position:fixed;bottom:30px;right:10px;z-index:1000;'
              'background:white;padding:10px;border-radius:6px;font-size:12px;'
              'border:1px solid #ccc;min-width:180px;">'
              '<b>🎨 어디에 무엇을</b><br>')
    for s in ZONE_CODE:
        legend += (f'<span style="background:{ZONE_PAL[s]};display:inline-block;'
                   f'width:12px;height:12px;margin-right:5px;border-radius:2px;"></span>'
                   f'{ZONE_NAME[s].replace(" 구역","")} → <b>{tops.get(s,"-")}</b><br>')
    legend += '</div>'
    m.get_root().html.add_child(folium.Element(legend))

    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])
    return m._repr_html_()


# ════════════════════════════════════════════════════════════
# 점수 차트 (Plotly)
# ════════════════════════════════════════════════════════════
def score_chart(df: pd.DataFrame) -> go.Figure:
    names = df["name_kor"].tolist()[::-1]
    env_  = (df["환경적합"] * 50).tolist()[::-1]
    est_  = (df["op_establishment"].fillna(0.5) * 25).tolist()[::-1]
    saf_  = (df["op_safe_growth"].fillna(0.5) * 25).tolist()[::-1]
    scores = df["추천점수"].tolist()[::-1]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="환경 적합", y=names, x=env_, orientation="h",
                         marker_color="#2e7d32", hovertemplate="%{x:.0f}점<extra>환경 적합</extra>"))
    fig.add_trace(go.Bar(name="정착",     y=names, x=est_, orientation="h",
                         marker_color="#1e88e5", hovertemplate="%{x:.0f}점<extra>정착</extra>"))
    fig.add_trace(go.Bar(name="안전",     y=names, x=saf_, orientation="h",
                         marker_color="#ef6c00", hovertemplate="%{x:.0f}점<extra>안전</extra>"))
    # 총점 텍스트
    fig.add_trace(go.Scatter(
        x=[e+es+sa+2 for e,es,sa in zip(env_,est_,saf_)], y=names,
        mode="text", text=[f"{s}점" for s in scores],
        textfont=dict(size=11, color="#333"), showlegend=False,
    ))
    fig.update_layout(
        barmode="stack", height=max(220, len(df) * 36 + 60),
        margin=dict(l=0, r=50, t=20, b=20),
        legend=dict(orientation="h", y=1.12, x=0),
        xaxis=dict(range=[0, 108], title="추천 점수 (100점 만점)"),
        plot_bgcolor="white",
    )
    return fig


# ════════════════════════════════════════════════════════════
# 다운로드 헬퍼
# ════════════════════════════════════════════════════════════
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue().encode("utf-8-sig")

def path_to_csv_bytes(ms: dict) -> bytes:
    rows = [{"순번": i+1, "위도": round(lat,7), "경도": round(lon,7)}
            for i, (lat, lon) in enumerate(ms["path"])]
    return df_to_csv_bytes(pd.DataFrame(rows))


# ════════════════════════════════════════════════════════════
# 탭 구성
# ════════════════════════════════════════════════════════════
if "lib" not in st.session_state:
    st.session_state["lib"] = load_library()

tab1, tab2 = st.tabs(["🗺 드론 영상 분석 & 추천", "🌿 종 라이브러리 관리"])


# ════════════════════════════════════════════════════════════
# TAB 1 — 분석 & 추천
# ════════════════════════════════════════════════════════════
with tab1:
    col_in, col_out = st.columns([1, 2])

    # ── 입력 패널 ────────────────────────────────────────────
    with col_in:
        st.subheader("① 입력")
        uploaded = st.file_uploader(
            "📂 드론 영상 올리기 (GeoTIFF)",
            type=["tif", "tiff"],
            help="정사영상 GeoTIFF 파일. EPSG:4326 또는 UTM 가능.",
        )
        use_demo = st.checkbox("데모 영상 사용 (result_small.tif)", value=True)
        spacing_m = st.number_input("🚁 드론 비행 간격 (m) — 작을수록 촘촘",
                                    min_value=1, max_value=20, value=2, step=1)
        zone_sel = st.selectbox(
            "🗺 구역 선택 (상세 추천 볼 구역)",
            options=ZONE_CODE,
            format_func=lambda s: ZONE_NAME[s],
            index=3,
        )
        st.caption(f"👉 {ZONE_DESC[zone_sel]}")

        # TIF 결정
        tif_path: Path | None = None
        tif_bytes: bytes | None = None
        if uploaded:
            tif_bytes = uploaded.read()
            tif_path = Path(uploaded.name)
        elif use_demo and DEMO_TIF.exists():
            tif_path = DEMO_TIF

        # 메타 표시
        if tif_path or tif_bytes:
            try:
                if tif_bytes:
                    with rasterio.open(io.BytesIO(tif_bytes)) as src:
                        bnds = src.bounds
                        crs_epsg = src.crs.to_epsg() if src.crs else None
                        nc, nr, nb = src.width, src.height, src.count
                    if crs_epsg and crs_epsg != 4326:
                        b = transform_bounds(f"EPSG:{crs_epsg}", "EPSG:4326", *bnds)
                    else:
                        b = (bnds.left, bnds.bottom, bnds.right, bnds.top)
                    lon_c = (b[0]+b[2])/2; lat_c = (b[1]+b[3])/2
                    m_lon = 111320*math.cos(math.radians(lat_c)); m_lat=111320
                    meta = {"bounds":b,"lon_c":lon_c,"lat_c":lat_c,
                            "w_m":(b[2]-b[0])*m_lon,"h_m":(b[3]-b[1])*m_lat,
                            "area_ha":(b[2]-b[0])*m_lon*(b[3]-b[1])*m_lat/10000,
                            "nbands":nb,"width":nc,"height":nr}
                else:
                    meta = read_tif_meta(tif_path)
                st.session_state["meta"] = meta
                ms = make_mission(meta["bounds"], spacing_m)
                st.session_state["ms"] = ms
                st.info(
                    f"📍 중심: {meta['lat_c']:.5f}°N, {meta['lon_c']:.5f}°E  \n"
                    f"📐 범위: {meta['w_m']:.0f} m × {meta['h_m']:.0f} m ≈ **{meta['area_ha']:.2f} ha**  \n"
                    f"🎞 크기: {meta['nbands']}밴드 · {meta['width']}×{meta['height']}px  \n"
                    f"🚁 비행: 라인 {ms['n_lines']}개 · 드롭점 약 {ms['n_way']:,}개"
                )
            except Exception as e:
                st.error(f"영상 읽기 실패: {e}")
                st.session_state.pop("meta", None)

    # ── 결과 패널 ────────────────────────────────────────────
    with col_out:
        st.subheader("② 결과")
        lib = st.session_state["lib"]

        if "meta" not in st.session_state:
            st.info("← 왼쪽에서 드론 영상을 선택하면 결과가 여기에 나타납니다.")
        else:
            meta = st.session_state["meta"]
            ms   = st.session_state["ms"]

            # 지도
            with st.spinner("🛰 지도 생성 중…"):
                map_html = build_map(meta, lib, spacing_m)
            st.components.v1.html(map_html, height=440, scrolling=False)

            st.divider()

            # ── 전체 구역 요약 카드 ──────────────────────────
            st.markdown("**🗺 전체 복원 구역 요약 — 구역별 추천 Top 3**")
            card_cols = st.columns(4)
            for i, s in enumerate(ZONE_CODE):
                top3 = recommend_for(lib, s, 3)
                color = ZONE_PAL[s]
                with card_cols[i]:
                    html = (f'<div style="background:#fafafa;border-left:4px solid {color};'
                            f'border-radius:4px;padding:10px 12px;">'
                            f'<div style="font-weight:bold;color:{color};font-size:13px;margin-bottom:4px;">'
                            f'{ZONE_NAME[s]}</div>'
                            f'<div style="font-size:10px;color:#666;margin-bottom:6px;">{ZONE_DESC[s]}</div>')
                    for _, row in top3.iterrows():
                        html += (f'<div style="display:flex;justify-content:space-between;'
                                 f'font-size:12px;padding:2px 0;border-bottom:1px solid #eee;">'
                                 f'<span>{int(row["순위"])}. {row["name_kor"]} ({row["form_grp"]})</span>'
                                 f'<span style="font-weight:bold;color:{color};">{row["추천점수"]}점</span>'
                                 f'</div>')
                    html += '</div>'
                    st.markdown(html, unsafe_allow_html=True)

            st.divider()

            # ── 구역별 상세 추천 ─────────────────────────────
            st.markdown(f"**🌱 [{ZONE_NAME[zone_sel]}] 추천 식물 상세**")
            rec = recommend_for(lib, zone_sel, 8)

            # 점수 차트
            st.plotly_chart(score_chart(rec), use_container_width=True)

            # 종 테이블
            disp_cols = ["순위","name_kor","form_grp","추천점수","추천이유"]
            disp_names = {"순위":"순위","name_kor":"식물 이름","form_grp":"생활형",
                          "추천점수":"추천 점수","추천이유":"추천 이유"}
            st.dataframe(
                rec[disp_cols].rename(columns=disp_names),
                use_container_width=True, hide_index=True,
                column_config={"추천 점수": st.column_config.ProgressColumn(
                    "추천 점수", min_value=0, max_value=100, format="%d점")},
            )

            # 선택 종 상세
            sel_name = st.selectbox(
                "종을 선택하면 점수 근거가 보여요",
                options=rec["name_kor"].tolist(),
                index=0,
            )
            sel = rec[rec["name_kor"] == sel_name].iloc[0]
            c1, c2 = st.columns(2)
            with c1:
                st.metric("추천 점수", f"{sel['추천점수']}점")
                env_lbl = "잘 맞음" if sel["환경적합"]>=0.75 else ("보통" if sel["환경적합"]>=0.5 else "잘 안 맞음")
                st.metric("환경 적합", env_lbl, f"{sel['환경적합']*100:.0f}/100")
            with c2:
                est_v = sel.get("op_establishment", 0.5) or 0.5
                saf_v = sel.get("op_safe_growth", 0.7) or 0.7
                st.metric("자리잡기 (정착)", "빠름" if est_v>=0.8 else ("보통" if est_v>=0.6 else "느림"),
                          f"{est_v*100:.0f}/100")
                st.metric("번짐 안전성", "안전" if saf_v>=0.8 else ("주의" if saf_v<=0.5 else "보통"),
                          f"{saf_v*100:.0f}/100")
            c_v = sel.get("c_percent"); r_v = sel.get("r_percent"); s_v = sel.get("s_percent")
            if not pd.isna(c_v):
                st.caption(f"생태전략 CSR: C{c_v:.0f} · S{s_v:.0f} · R{r_v:.0f}  "
                           f"(구역 선호 {PREF_KOR[SIT_PREF[zone_sel]]})")
            else:
                st.caption("CSR 데이터 없음 → 중립 0.5 처리")
            st.caption(":orange[조달 점수는 데이터 조사 중 — 현재 점수 미반영]")

            st.divider()

            # ── 내보내기 ─────────────────────────────────────
            st.markdown("**📤 내보내기**")
            dl1, dl2, dl3 = st.columns(3)
            today = datetime.date.today().strftime("%Y%m%d")
            with dl1:
                st.download_button(
                    "추천 결과 CSV",
                    data=df_to_csv_bytes(rec[["순위","name_kor","name_sci","form_grp","추천점수","추천이유"]]),
                    file_name=f"ReSeed_추천_{ZONE_NAME[zone_sel]}_{today}.csv",
                    mime="text/csv",
                )
            with dl2:
                st.download_button(
                    "드론 비행경로 CSV",
                    data=path_to_csv_bytes(ms),
                    file_name=f"ReSeed_비행경로_{today}.csv",
                    mime="text/csv",
                )
            with dl3:
                all_rows = []
                for s in ZONE_CODE:
                    df_s = recommend_for(lib, s, 3).copy()
                    df_s["구역코드"] = s
                    df_s["구역명"] = ZONE_NAME[s]
                    all_rows.append(df_s)
                df_all = pd.concat(all_rows)[["구역코드","구역명","순위","name_kor","name_sci","form_grp","추천점수","추천이유"]]
                st.download_button(
                    "전체 구역 요약 CSV",
                    data=df_to_csv_bytes(df_all),
                    file_name=f"ReSeed_전체구역_{today}.csv",
                    mime="text/csv",
                )

            st.markdown('<p class="score-note">뿌릴 수 있는 자생식물 44종 중 선별. 정착·안전은 내 형질DB의 CSR 실측 기반 (일부 생활형 추정). 조달·지형 데이터 추가 시 자동 반영.</p>',
                        unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 2 — 종 라이브러리 관리
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🌿 종 라이브러리 관리")
    st.caption(f"현재 등록 종: **{len(st.session_state['lib'])}종** | 파일: `data/seedball_library.csv`")
    lib = st.session_state["lib"]

    # ── 현재 목록 ────────────────────────────────────────────
    with st.expander("📋 현재 등록 종 목록 보기", expanded=True):
        view_cols = ["name_kor","name_sci","form_grp","c_percent","s_percent","r_percent",
                     "op_establishment","op_safe_growth","op_sourcing"]
        view_names = {"name_kor":"국문명","name_sci":"학명","form_grp":"생활형",
                      "c_percent":"C%","s_percent":"S%","r_percent":"R%",
                      "op_establishment":"정착","op_safe_growth":"안전","op_sourcing":"조달"}
        st.dataframe(lib[view_cols].rename(columns=view_names),
                     use_container_width=True, hide_index=True)

    st.divider()

    # ── 새 종 추가 (단일) ────────────────────────────────────
    st.markdown("### ➕ 새 종 추가")
    st.caption("국문명과 생활형은 필수. CSR 값을 넣을수록 추천 품질이 높아집니다.")

    with st.form("add_species_form", clear_on_submit=True):
        fc1, fc2 = st.columns(2)
        with fc1:
            new_kor   = st.text_input("국문명 *", placeholder="예) 복사나무")
            new_sci   = st.text_input("학명",     placeholder="예) Prunus tomentosa")
            new_form  = st.selectbox("생활형 *", ["교목","관목","초본","덩굴"])
            new_memo  = st.text_input("조달처 메모", placeholder="예) 국립종자원 기증 2026-06")
        with fc2:
            st.markdown("**CSR 값** (없으면 빈칸 — 중립 처리)")
            new_c = st.number_input("C% (경쟁성)", min_value=0, max_value=100, value=0, step=1)
            new_s = st.number_input("S% (내성)",   min_value=0, max_value=100, value=0, step=1)
            new_r = st.number_input("R% (개척성)", min_value=0, max_value=100, value=0, step=1)
            csr_known = st.checkbox("위 CSR 값 사용 (체크 해제 시 중립 처리)")

        submitted = st.form_submit_button("✅ 추가하기", type="primary")
        if submitted:
            if not new_kor.strip():
                st.error("국문명은 필수입니다.")
            elif new_kor.strip() in lib["name_kor"].values:
                st.warning(f"'{new_kor}' 은 이미 등록되어 있습니다.")
            else:
                c_val = float(new_c) if csr_known else np.nan
                s_val = float(new_s) if csr_known else np.nan
                r_val = float(new_r) if csr_known else np.nan
                # op 점수 자동 계산 (CSR 있으면 파생, 없으면 생활형 prior)
                if csr_known and r_val > 0:
                    op_est = round(min(max(0.45 + 0.55*(r_val/100), 0.30), 1.00), 2)
                    op_saf = round(min(max(1.00 - 0.60*(c_val/100), 0.40), 1.00), 2)
                else:
                    est_prior = {"교목":0.40,"관목":0.60,"초본":0.80,"덩굴":0.50}
                    op_est = est_prior.get(new_form, 0.60)
                    op_saf = 0.70
                new_id = f"SB{len(lib)+1:03d}"
                new_row = {
                    "seedball_id": new_id, "plant_id": new_id,
                    "name_kor": new_kor.strip(), "name_sci": new_sci.strip(),
                    "form_grp": new_form, "height_cl": np.nan,
                    "form_grp_ref": np.nan, "form_match_ref": np.nan,
                    "confidence": "사용자입력", "stock": 0,
                    "channel_cnt": 0,
                    "op_sourcing": 0.5,
                    "op_establishment": op_est, "op_safe_growth": op_saf,
                    "c_percent": c_val, "s_percent": s_val, "r_percent": r_val,
                    "strategy_class": np.nan,
                    "op_establishment_src": "CSR_R" if csr_known else "form_prior",
                    "op_establishment_qual": "사용자입력",
                    "op_safe_growth_src": "CSR_C" if csr_known else "form_prior",
                    "op_safe_growth_qual": "사용자입력",
                    "op_sourcing_src": new_memo or "사용자입력",
                    "op_mean": round((op_est+op_saf)/2, 2),
                    "op_db_source": "web_입력",
                }
                updated = pd.concat([lib, pd.DataFrame([new_row])], ignore_index=True)
                updated.to_csv(LIB_CSV, index=False, encoding="utf-8-sig")
                reload_library()
                st.success(f"✅ '{new_kor}' 추가 완료! (ID: {new_id}  |  추천에 즉시 반영됩니다)")
                st.rerun()

    st.divider()

    # ── CSV 일괄 업로드 ──────────────────────────────────────
    st.markdown("### 📥 CSV 일괄 업로드")
    st.caption("여러 종을 한꺼번에 추가할 때 사용. 아래 양식을 참고해 CSV를 만들어 업로드하세요.")

    tmpl = pd.DataFrame([{
        "name_kor":"복사나무","name_sci":"Prunus tomentosa","form_grp":"관목",
        "c_percent":25,"s_percent":30,"r_percent":45,
        "op_sourcing_src":"국립종자원 기증 2026-06",
    }])
    st.download_button("📄 양식 CSV 다운로드",
                       data=df_to_csv_bytes(tmpl),
                       file_name="ReSeed_종추가_양식.csv", mime="text/csv")

    csv_upload = st.file_uploader("CSV 파일 업로드 (위 양식 참고)",
                                  type=["csv"], key="bulk_upload")
    if csv_upload:
        try:
            new_df = pd.read_csv(csv_upload, encoding="utf-8-sig")
            required = {"name_kor","form_grp"}
            missing = required - set(new_df.columns)
            if missing:
                st.error(f"필수 컬럼이 없습니다: {missing}")
            else:
                dups = set(new_df["name_kor"]) & set(lib["name_kor"])
                if dups:
                    st.warning(f"이미 등록된 종 제외: {dups}")
                    new_df = new_df[~new_df["name_kor"].isin(dups)]
                if len(new_df) == 0:
                    st.info("추가할 새 종이 없습니다.")
                else:
                    st.dataframe(new_df, use_container_width=True, hide_index=True)
                    if st.button(f"✅ {len(new_df)}종 추가 확정"):
                        # 기본값 채우기
                        for col in ["c_percent","s_percent","r_percent"]:
                            if col not in new_df.columns:
                                new_df[col] = np.nan
                        for col in ["op_establishment","op_safe_growth","op_sourcing"]:
                            if col not in new_df.columns:
                                new_df[col] = 0.5
                        new_df["seedball_id"] = [f"SB{len(lib)+i+1:03d}" for i in range(len(new_df))]
                        new_df["confidence"] = "CSV일괄입력"
                        new_df["op_db_source"] = "web_csv_업로드"
                        updated = pd.concat([lib, new_df], ignore_index=True)
                        updated.to_csv(LIB_CSV, index=False, encoding="utf-8-sig")
                        reload_library()
                        st.success(f"✅ {len(new_df)}종 추가 완료!")
                        st.rerun()
        except Exception as e:
            st.error(f"CSV 읽기 실패: {e}")

    st.divider()

    # ── 종 삭제 ──────────────────────────────────────────────
    st.markdown("### 🗑 종 삭제")
    del_name = st.selectbox("삭제할 종 선택", options=["선택 안 함"] + lib["name_kor"].tolist())
    if del_name != "선택 안 함":
        if st.button(f"🗑 '{del_name}' 삭제", type="secondary"):
            updated = lib[lib["name_kor"] != del_name].copy()
            updated.to_csv(LIB_CSV, index=False, encoding="utf-8-sig")
            reload_library()
            st.success(f"'{del_name}' 삭제 완료.")
            st.rerun()
