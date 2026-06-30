"""
ReSeed — 드론 시드볼 살포 의사결정 시스템 (Python/Streamlit 버전)
변환일: 2026-06-22  |  원본: app_v5.R (R/Shiny)
"""
from __future__ import annotations
import math
import io
import base64
import datetime
import tempfile
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rasterio.transform
from rasterio.warp import transform_bounds
import folium
from branca.element import MacroElement
from jinja2 import Template as JinjaTemplate
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors
from reportlab.lib.units import mm as rl_mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

# ── 경로 설정 ──────────────────────────────────────────────
BASE       = Path(__file__).parent
DATA       = BASE / "data"
LIB_CSV    = DATA / "seedball_library.csv"
ZONE_TIF   = DATA / "n3a_situations_4326.tif"
DEMO_TIF   = DATA / "result_small.tif"
# TraitX DB: 환경변수 > data/ 로컬 복사본 > Windows 개발 경로 순 fallback
_traitx_win = Path(r"C:\Users\Intern\OneDrive\Desktop\Rstudio\결과물\trait_kor_mean.xlsx")
TRAITX_DB   = Path(os.environ.get("TRAITX_DB", str(_traitx_win)))
if not TRAITX_DB.exists():
    TRAITX_DB = DATA / "trait_kor_mean.xlsx"

# ── 상수 ───────────────────────────────────────────────────
ZONE_CODE = ["S1", "S2", "S3", "S4"]
ZONE_NAME = {
    "S1": "긴급 안정화 구역", "S2": "개척 파종 구역",
    "S3": "천이 촉진 구역",   "S4": "하층 보완 구역",
}
ZONE_DESC = {
    "S1": "집중 나지·고황폐도 — 주변 식생 없고 토양 노출 심각, 즉시 피복 필요",
    "S2": "분산 나지·중황폐도 — 인접 식생 존재, 개척 파종으로 자연 회복 유도",
    "S3": "풀·작은나무 단계 — 숲으로 키워갈 곳",
    "S4": "큰나무 우거짐 — 그늘 아래를 채울 곳",
}
ZONE_PAL     = {"S1": "#c62828", "S2": "#ef6c00", "S3": "#fbc02d", "S4": "#2e7d32"}
ZONE_PAL_RGB = {"S1": (198,40,40), "S2": (239,108,0), "S3": (251,192,45), "S4": (46,125,50)}
ZONE_VAL_MAP = {1: "S1", 2: "S2", 3: "S3", 4: "S4"}
IDEAL_FORM   = {"S1":["초본"],"S2":["초본","관목"],"S3":["관목","교목"],"S4":["관목","초본"]}
SIT_PREF     = {"S1":"R","S2":"C","S3":"C","S4":"S"}  # S2: 완경사 개척 → 경쟁형 전환
PREF_KOR     = {"R":"개척형","C":"경쟁형","S":"내성형"}
# 구역별 (환경적합 가중치, 현장실행 가중치)
ZONE_SCORE_W = {
    "S1": (0.35, 0.65),  # 침식 긴급: 정착 속도 최우선
    "S2": (0.60, 0.40),  # 개척 파종: 생태 적합 우선
    "S3": (0.65, 0.35),  # 천이 촉진: 장기 군락 형성
    "S4": (0.50, 0.50),  # 하층 보완: 균형
}

# ── PDF 한글 폰트 등록 (Windows: Malgun / Linux: Nanum) ───────────────
_KR_FONT, _KR_BOLD = "Helvetica", "Helvetica-Bold"
_font_candidates = [
    (Path("C:/Windows/Fonts/malgun.ttf"),   Path("C:/Windows/Fonts/malgunbd.ttf")),
    (Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
     Path("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf")),
]
for _fp, _fb in _font_candidates:
    if _fp.exists() and _fb.exists():
        try:
            pdfmetrics.registerFont(TTFont("_KR",  str(_fp)))
            pdfmetrics.registerFont(TTFont("_KRB", str(_fb)))
            _KR_FONT, _KR_BOLD = "_KR", "_KRB"
        except Exception:
            pass
        break


def make_summary_pdf(rec_cache: dict,
                     meta: dict | None = None,
                     zone_frac: dict | None = None,
                     exg_mean: float | None = None,
                     veg_cover: float | None = None,
                     zone_png: str | None = None,
                     logo_path: str | None = None) -> bytes:
    """현재 분석 결과 → 1장 PDF 보고서 bytes"""
    import base64 as _b64
    buf = io.BytesIO()
    c   = rl_canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    M = 14 * rl_mm

    INK   = rl_colors.HexColor("#023047")
    BLUE  = rl_colors.HexColor("#0000FF")
    LGRAY = rl_colors.HexColor("#F4F6F8")
    MGRAY = rl_colors.HexColor("#B0BEC5")
    SGRAY = rl_colors.HexColor("#546e7a")
    WHITE = rl_colors.white
    C_ENV = rl_colors.HexColor("#2e7d32")
    C_EST = rl_colors.HexColor("#1e88e5")
    C_SAF = rl_colors.HexColor("#ef6c00")
    _ZC   = {s: rl_colors.HexColor(ZONE_PAL[s]) for s in ZONE_CODE}

    today_str = datetime.date.today().strftime("%Y-%m-%d")
    kw = (W - 2*M - 9*rl_mm) / 4

    def section(y_, title_):
        c.setFillColor(INK); c.setFont(_KR_BOLD, 9.5)
        c.drawString(M, y_, title_)
        c.setStrokeColor(MGRAY); c.setLineWidth(0.4)
        c.line(M, y_ - 1.5*rl_mm, W - M, y_ - 1.5*rl_mm)
        return y_ - 7*rl_mm

    # ── 헤더 배너 ──────────────────────────────────────────
    c.setFillColor(INK)
    c.rect(0, H - 26*rl_mm, W, 26*rl_mm, fill=1, stroke=0)
    c.setFillColor(WHITE); c.setFont(_KR_BOLD, 15)
    c.drawString(M, H - 13*rl_mm, "ReSeed  드론 시드볼 파종 의사결정 보고서")
    c.setFont(_KR_FONT, 8)
    area_str = f"{meta['area_ha']:.2f} ha  |  " if meta else ""
    c.drawString(M, H - 21*rl_mm, f"{area_str}{today_str}  |  Invalab")
    c.setFillColor(BLUE)
    c.rect(W - 26*rl_mm, H - 26*rl_mm, 26*rl_mm, 26*rl_mm, fill=1, stroke=0)
    c.setFillColor(WHITE); c.setFont(_KR_BOLD, 7.5)
    c.drawCentredString(W - 13*rl_mm, H - 13*rl_mm, "v1.0")
    c.setFont(_KR_FONT, 6.5)
    c.drawCentredString(W - 13*rl_mm, H - 19*rl_mm, "ReSeed / Invalab")

    # ── KPI 박스 ──────────────────────────────────────────
    y = H - 40*rl_mm
    for i, s in enumerate(ZONE_CODE):
        x       = M + i * (kw + 3*rl_mm)
        frac    = zone_frac.get(s, 0.0) if zone_frac else None
        val_str = f"{frac*100:.1f}%" if frac is not None else "—"
        c.setFillColor(LGRAY)
        c.roundRect(x, y - 14*rl_mm, kw, 14*rl_mm, 2.5*rl_mm, fill=1, stroke=0)
        c.setFillColor(_ZC[s])
        c.roundRect(x, y - 5*rl_mm, kw, 5*rl_mm, 2.5*rl_mm, fill=1, stroke=0)
        c.setFillColor(WHITE); c.setFont(_KR_BOLD, 7)
        c.drawCentredString(x + kw/2, y - 3.5*rl_mm, f"{s}  {ZONE_NAME[s]}")
        c.setFillColor(INK); c.setFont(_KR_BOLD, 13)
        c.drawCentredString(x + kw/2, y - 10.5*rl_mm, val_str)

    # ── 구역별 추천 종 테이블 ─────────────────────────────
    y -= 20*rl_mm
    y = section(y, "구역별 추천 종  (Top 3)")
    ROW_H = 8.5*rl_mm
    for i, s in enumerate(ZONE_CODE):
        x    = M + i * (kw + 3*rl_mm)
        top3 = rec_cache[s].head(3)
        c.setFillColor(_ZC[s])
        c.roundRect(x, y - 5.5*rl_mm, kw, 5.5*rl_mm, 2*rl_mm, fill=1, stroke=0)
        c.setFillColor(WHITE); c.setFont(_KR_BOLD, 7.5)
        c.drawCentredString(x + kw/2, y - 3.8*rl_mm, ZONE_NAME[s])
        row_y = y - 5.5*rl_mm
        for _, row in top3.iterrows():
            bg = LGRAY if int(row["순위"]) % 2 == 1 else WHITE
            c.setFillColor(bg)
            c.rect(x, row_y - ROW_H, kw, ROW_H, fill=1, stroke=0)
            c.setFillColor(INK); c.setFont(_KR_BOLD, 7.5)
            c.drawString(x + 2*rl_mm, row_y - 3.5*rl_mm,
                         f"{int(row['순위'])}. {row['name_kor']}")
            c.setFillColor(_ZC[s]); c.setFont(_KR_BOLD, 7)
            c.drawRightString(x + kw - 2*rl_mm, row_y - 3.5*rl_mm,
                              f"{row['추천점수']}점")
            c.setFillColor(MGRAY); c.setFont(_KR_FONT, 6)
            c.drawString(x + 2*rl_mm, row_y - 7.5*rl_mm, row.get("form_grp", ""))
            row_y -= ROW_H

    table_bottom = y - 5.5*rl_mm - 3 * ROW_H

    # ── 식물 분포도 + 구역 면적 막대그래프 ───────────────────
    y = table_bottom - 6*rl_mm
    y = section(y, "식물 분포도  &  구역 면적 비율")

    VIZ_H  = 68*rl_mm
    left_w = (W - 2*M) * 0.52 - 3*rl_mm
    rx     = M + left_w + 6*rl_mm
    rw     = W - M - rx

    if zone_png and zone_png.startswith("data:image/png;base64,"):
        try:
            raw = _b64.b64decode(zone_png.split(",", 1)[1])
            ir  = ImageReader(io.BytesIO(raw))
            iw, ih = ir.getSize()
            asp    = iw / ih if ih > 0 else 1.0
            dw     = min(left_w, VIZ_H * asp)
            dh     = dw / asp
            c.drawImage(ir, M, y - dh, dw, dh,
                        preserveAspectRatio=True, mask="auto")
            c.setFillColor(MGRAY); c.setFont(_KR_FONT, 6.5)
            c.drawString(M, y - dh - 3*rl_mm,
                         "▲ S1 빨강(긴급안정화)  S2 주황(개척파종)  S3 노랑(천이촉진)  S4 초록(하층보완)")
        except Exception:
            c.setFillColor(LGRAY)
            c.roundRect(M, y - VIZ_H, left_w, VIZ_H, 2*rl_mm, fill=1, stroke=0)
            c.setFillColor(MGRAY); c.setFont(_KR_FONT, 8)
            c.drawCentredString(M + left_w / 2, y - VIZ_H / 2, "TIF 분석 후 표시")
    else:
        c.setFillColor(LGRAY)
        c.roundRect(M, y - VIZ_H, left_w, VIZ_H, 2*rl_mm, fill=1, stroke=0)
        c.setFillColor(MGRAY); c.setFont(_KR_FONT, 8)
        c.drawCentredString(M + left_w / 2, y - VIZ_H / 2, "TIF 분석 후 표시")

    lw2 = 19*rl_mm
    bmax = rw - lw2 - 10*rl_mm
    brh  = VIZ_H / 4
    for i, s in enumerate(ZONE_CODE):
        frac = zone_frac.get(s, 0.0) if zone_frac else 0.0
        by   = y - (i + 0.5) * brh
        bh   = brh * 0.48
        c.setFillColor(_ZC[s]); c.setFont(_KR_BOLD, 8)
        c.drawString(rx, by + bh * 0.2, s)
        c.setFillColor(INK); c.setFont(_KR_FONT, 7)
        c.drawString(rx + 7*rl_mm, by + bh * 0.2, ZONE_NAME[s].replace(" 구역", ""))
        tx = rx + lw2
        c.setFillColor(LGRAY)
        c.roundRect(tx, by - bh * 0.3, bmax, bh * 0.6, 1*rl_mm, fill=1, stroke=0)
        fw = max(bmax * frac, 2*rl_mm)
        c.setFillColor(_ZC[s])
        c.roundRect(tx, by - bh * 0.3, fw, bh * 0.6, 1*rl_mm, fill=1, stroke=0)
        c.setFillColor(INK); c.setFont(_KR_BOLD, 8)
        c.drawString(tx + bmax + 2*rl_mm, by - bh * 0.1, f"{frac*100:.1f}%")

    # ── 추천 점수 구성 (수평 스택 막대, 2×2 패널) ─────────────
    y = y - VIZ_H - 6*rl_mm
    y = section(y, "추천 점수 구성  (환경 적합 · 정착 · 안전 세부 점수, 100점 만점)")

    # 패널 치수
    _GAP    = 5*rl_mm
    _PW     = (W - 2*M - _GAP) / 2    # 패널 너비 ≈88 mm
    _LBL    = 20*rl_mm                 # 종명 라벨
    _SCR    = 10*rl_mm                 # 우측 총점 칸
    _BAR    = _PW - _LBL - _SCR       # 바 실제 너비
    _SP     = 5.8*rl_mm               # 종 행 높이
    _PHDR   = 5.5*rl_mm               # 구역 헤더 높이
    _AXH    = 4*rl_mm                 # X축 높이
    _NSHOW  = 7                       # 구역당 최대 종 수
    _PH     = _PHDR + _NSHOW * _SP + _AXH   # 패널 전체 높이 ≈47 mm

    # 남은 공간 부족 시 새 페이지
    _needed = 2 * _PH + _GAP + 10*rl_mm
    if y - _needed < M + 28*rl_mm:
        c.showPage()
        y = H - M

    for _pr in range(2):
        for _pc in range(2):
            _zi  = _pr * 2 + _pc
            _s   = ZONE_CODE[_zi]
            _top = rec_cache[_s].head(_NSHOW)
            _n   = len(_top)
            _ph  = _PHDR + _n * _SP + _AXH   # 실제 패널 높이

            _px  = M + _pc * (_PW + _GAP)
            _py  = y - _pr * (_PH + _GAP)

            # 패널 배경
            c.setFillColor(LGRAY)
            c.roundRect(_px, _py - _ph, _PW, _ph, 2*rl_mm, fill=1, stroke=0)

            # 구역 헤더
            c.setFillColor(_ZC[_s])
            c.roundRect(_px, _py - _PHDR, _PW, _PHDR, 2*rl_mm, fill=1, stroke=0)
            c.setFillColor(WHITE); c.setFont(_KR_BOLD, 7.5)
            c.drawCentredString(_px + _PW/2, _py - _PHDR*0.65, ZONE_NAME[_s])

            # X축 눈금선
            _axy = _py - _PHDR - _n * _SP
            for _xv in [25, 50, 75, 100]:
                _gx = _px + _LBL + (_xv / 100) * _BAR
                c.setStrokeColor(MGRAY); c.setLineWidth(0.25)
                c.line(_gx, _py - _PHDR, _gx, _axy)
                c.setFillColor(MGRAY); c.setFont(_KR_FONT, 4.8)
                c.drawCentredString(_gx, _axy - _AXH * 0.68, str(_xv))

            # 0점 기준선
            c.setStrokeColor(INK); c.setLineWidth(0.5)
            c.line(_px + _LBL, _py - _PHDR, _px + _LBL, _axy)

            # 종별 행
            for _ri, (_, _row) in enumerate(_top.iterrows()):
                _ry  = _py - _PHDR - _ri * _SP
                _es  = float(_row.get("환경적합",         0.5))
                _ets = float(_row.get("op_establishment",  0.5))
                _ss  = float(_row.get("op_safe_growth",    0.5))
                _tot = int(_row["추천점수"])
                _ew  = _es  * 50 / 100 * _BAR
                _tw  = _ets * 25 / 100 * _BAR
                _sw  = _ss  * 25 / 100 * _BAR
                _bh  = _SP * 0.58
                _by  = _ry - _SP + (_SP - _bh) / 2

                # 행 배경
                c.setFillColor(WHITE if _ri % 2 == 0 else rl_colors.HexColor("#EEF2F5"))
                c.rect(_px, _ry - _SP, _PW, _SP, fill=1, stroke=0)

                # 종명
                c.setFillColor(INK); c.setFont(_KR_FONT, 6.2)
                c.drawString(_px + 1.5*rl_mm, _ry - _SP*0.62, _row["name_kor"][:8])

                # 환경 적합 바 (초록)
                _bx = _px + _LBL
                c.setFillColor(C_ENV); c.rect(_bx, _by, _ew, _bh, fill=1, stroke=0)
                if _ew > 6*rl_mm:
                    c.setFillColor(WHITE); c.setFont(_KR_BOLD, 5)
                    c.drawCentredString(_bx + _ew/2, _by + _bh*0.2, str(round(_es*50)))

                # 정착 바 (파랑)
                c.setFillColor(C_EST); c.rect(_bx+_ew, _by, _tw, _bh, fill=1, stroke=0)
                if _tw > 4*rl_mm:
                    c.setFillColor(WHITE); c.setFont(_KR_BOLD, 5)
                    c.drawCentredString(_bx+_ew + _tw/2, _by + _bh*0.2, str(round(_ets*25)))

                # 안전 바 (주황)
                c.setFillColor(C_SAF); c.rect(_bx+_ew+_tw, _by, _sw, _bh, fill=1, stroke=0)
                if _sw > 4*rl_mm:
                    c.setFillColor(WHITE); c.setFont(_KR_BOLD, 5)
                    c.drawCentredString(_bx+_ew+_tw + _sw/2, _by + _bh*0.2, str(round(_ss*25)))

                # 총점 (우측 칸)
                c.setFillColor(_ZC[_s]); c.setFont(_KR_BOLD, 7)
                c.drawCentredString(_px + _PW - _SCR/2, _ry - _SP*0.62, f"{_tot}점")

    # 범례
    _leg_y = y - 2*_PH - _GAP - 4*rl_mm
    for _li, (_lt, _lc) in enumerate([("환경 적합 (CSR·생활형)", C_ENV),
                                       ("정착 (SLA·초기 성장)",  C_EST),
                                       ("안전 (LDMC·생존력)",    C_SAF)]):
        _lx = M + _li * 60*rl_mm
        c.setFillColor(_lc); c.rect(_lx, _leg_y-3*rl_mm, 5*rl_mm, 3*rl_mm, fill=1, stroke=0)
        c.setFillColor(INK); c.setFont(_KR_FONT, 6.5)
        c.drawString(_lx + 6*rl_mm, _leg_y - 1.5*rl_mm, _lt)

    chart_bottom = _leg_y - 5*rl_mm

    # ── 용어 해설 ──────────────────────────────────────────
    y = chart_bottom - 5*rl_mm
    y = section(y, "용어 해설")

    GLOSSARY = [
        ("ExG (식생지수)",      "2×초록 − 빨강 − 파랑. 양수=식생, 음수=나지."),
        ("CSR 전략",           "C=경쟁형 · S=내성형 · R=개척형. 식물 생태전략 분류."),
        ("SLA (비엽면적)",     "잎 면적÷건조중량. 높을수록 초기 성장 빠름."),
        ("LDMC (잎건조질량비)", "잎 건조중량÷생체중량. 높을수록 내구성 강함."),
        ("op_sourcing",        "공급 채널 가용성. 현재 임시값, 추후 실데이터 대체 예정."),
        ("ExG 평균 해석",      "0.1 이상=식생 풍부 / 0~0.1=보통 / 0 미만=나지 우세."),
    ]
    half = (W - 2*M - 4*rl_mm) / 2
    for idx, (term, desc) in enumerate(GLOSSARY):
        gx = M if idx % 2 == 0 else M + half + 4*rl_mm
        gy = y - (idx // 2) * 8.5*rl_mm
        c.setFillColor(BLUE); c.setFont(_KR_BOLD, 7)
        c.drawString(gx, gy, term)
        c.setFillColor(SGRAY); c.setFont(_KR_FONT, 6.8)
        c.drawString(gx, gy - 3.8*rl_mm, desc)

    glossary_bottom = y - ((len(GLOSSARY) + 1) // 2) * 8.5*rl_mm

    # ── 분석 통계 한 줄 ───────────────────────────────────
    y = glossary_bottom - 4*rl_mm
    if exg_mean is not None:
        c.setFillColor(SGRAY); c.setFont(_KR_FONT, 7)
        c.drawString(M, y,
            f"영상 분석: ExG 평균 {exg_mean:.3f}  |  식생 피복률 {veg_cover*100:.1f}%"
            "  |  ※ RGB 드론 기반 추정, LiDAR 미사용")
        y -= 5*rl_mm

    # ── 주의사항 박스 ──────────────────────────────────────
    y -= 2*rl_mm
    c.setFillColor(LGRAY)
    c.roundRect(M, y - 11*rl_mm, W - 2*M, 11*rl_mm, 2*rl_mm, fill=1, stroke=0)
    c.setFillColor(SGRAY); c.setFont(_KR_FONT, 6.8)
    c.drawString(M + 3*rl_mm, y - 4*rl_mm,
        "⚠ ReSeed는 의사결정 지원 도구입니다. 최종 복원 계획은 생태 복원 전문가의 현장 검토를 거쳐 수립하세요.")
    c.drawString(M + 3*rl_mm, y - 8.5*rl_mm,
        "추천 종: 59종 라이브러리 기준 (SLA·LDMC·CSR 실형질). op_sourcing 현재 임시값 사용 중.")

    # ── 푸터 + 로고 ────────────────────────────────────────
    FOOTER_H = 12*rl_mm
    c.setFillColor(INK)
    c.rect(0, 0, W, FOOTER_H, fill=1, stroke=0)

    # 로고: 이미지 있으면 삽입, 없으면 텍스트 배지
    logo_drawn = False
    if logo_path and Path(logo_path).exists():
        try:
            lr = ImageReader(logo_path)
            lw, lh = lr.getSize()
            logo_h = FOOTER_H * 0.75
            logo_w = logo_h * lw / lh
            c.drawImage(lr, M, (FOOTER_H - logo_h) / 2,
                        logo_w, logo_h, preserveAspectRatio=True, mask="auto")
            logo_drawn = True
        except Exception:
            pass
    if not logo_drawn:
        c.setFillColor(BLUE)
        c.roundRect(M, 1.8*rl_mm, 24*rl_mm, 8.5*rl_mm, 1.5*rl_mm, fill=1, stroke=0)
        c.setFillColor(WHITE); c.setFont(_KR_BOLD, 9.5)
        c.drawCentredString(M + 12*rl_mm, 4.8*rl_mm, "INVALAB")

    c.setFillColor(WHITE); c.setFont(_KR_FONT, 7)
    c.drawString(M + 27*rl_mm, 5*rl_mm, "ReSeed Project  |  내부용")
    c.drawRightString(W - M, 5*rl_mm, today_str)

    c.save()
    return buf.getvalue()


# ── 페이지 설정 ────────────────────────────────────────────
st.set_page_config(page_title="🌱 ReSeed", layout="wide", page_icon="🌱")
st.markdown("""
<style>
.header-bar{background:#1a4d2e;color:#fff;padding:12px 18px;border-radius:6px;margin-bottom:16px;}
.header-bar h2{margin:0;font-size:20px;}
.header-bar .sub{font-size:12px;opacity:.85;}
.zone-card{border-radius:6px;padding:12px 14px;margin:4px;}
.score-note{font-size:11px;color:#888;background:#fafafa;padding:4px 8px;border-radius:3px;}
/* 로딩 중 흰 화면 대신 옅은 배경 유지 */
.stApp, [data-testid="stAppViewContainer"] {background-color:#f5f7f5 !important;}
[data-testid="stMain"] > div:first-child {background-color:#f5f7f5;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-bar">
  <h2>🌱 ReSeed — 드론 시드볼 살포 의사결정 시스템</h2>
  <div class="sub">드론 영상과 위치를 넣으면 → 대상지를 복원 구역으로 나누고 → 구역마다 뿌리기 좋은 식물을 추천합니다.</div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TraitX 자동 갱신 — 앱 시작 시 TraitX가 라이브러리보다 새로우면 재빌드
# ════════════════════════════════════════════════════════════
def _auto_rebuild():
    try:
        from build_op_scores import needs_rebuild, rebuild
        if needs_rebuild(TRAITX_DB, LIB_CSV):
            with st.spinner("TraitX DB 갱신 감지 — 라이브러리 자동 업데이트 중..."):
                info = rebuild(TRAITX_DB, LIB_CSV)
            st.toast(
                f"라이브러리 업데이트 완료 ({info['traitx_file']} 기준, "
                f"CSR {info['n_csr_matched']}/{info['n_lib']}종)",
                icon="✅",
            )
            if "lib" in st.session_state:
                del st.session_state["lib"]
            load_library.clear()
    except Exception as e:
        st.warning(f"라이브러리 자동 갱신 실패: {e}")

_auto_rebuild()

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
    if "is_alien" not in df.columns:
        df["is_alien"] = False
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
    ideal_forms = IDEAL_FORM[zone]
    df_native = lib[~lib["is_alien"].fillna(False).astype(bool)].copy()
    df_pool = df_native[df_native["form_grp"].isin(ideal_forms)].copy()
    if len(df_pool) < n:
        df_pool = df_native.copy()
    df = df_pool.copy()

    # C1: CSR 저가중치 — 생활형 60% : CSR 40% (CSR 데이터 불확실성 반영)
    df["환경적합"] = df.apply(
        lambda row: round(
            _form_match(row.get("form_grp"), zone) * 0.6 +
            _csr_match(row.get("c_percent"), row.get("s_percent"),
                       row.get("r_percent"), zone) * 0.4, 2), axis=1)
    df["현장실행"] = ((df["op_establishment"].fillna(0.5) +
                      df["op_safe_growth"].fillna(0.5)) / 2).round(2)
    w_e, w_o = ZONE_SCORE_W[zone]
    df["추천점수"] = ((w_e * df["환경적합"] + w_o * df["현장실행"]) * 100).round().astype(int)
    df = df.sort_values(["추천점수","op_safe_growth"], ascending=[False,False]).reset_index(drop=True)

    # C4: 생활형 + 속(genus) 다양성 동시 보장
    max_per_form = max(2, -(-n // len(ideal_forms))) if len(ideal_forms) >= 2 else n
    MAX_PER_GENUS = 2
    picked, form_counts, genus_counts = [], {}, {}
    for _, row in df.iterrows():
        fg  = row.get("form_grp", "")
        sci = str(row.get("name_sci", "") or "")
        genus = sci.split()[0] if sci.strip() else ""
        # 생활형 캡
        if len(ideal_forms) >= 2 and fg in ideal_forms:
            if form_counts.get(fg, 0) >= max_per_form:
                continue
        # 속(genus) 캡 — 같은 속은 최대 MAX_PER_GENUS종
        if genus and genus_counts.get(genus, 0) >= MAX_PER_GENUS:
            continue
        form_counts[fg]   = form_counts.get(fg, 0) + 1
        if genus:
            genus_counts[genus] = genus_counts.get(genus, 0) + 1
        picked.append(row)
        if len(picked) >= n:
            break
    if picked:
        df = pd.DataFrame(picked).reset_index(drop=True)

    df["추천이유"] = df.apply(lambda r: _tag_reason(r, zone), axis=1)
    df["순위"] = range(1, len(df) + 1)
    return df.head(n)

def alien_in_pool(lib: pd.DataFrame, zone: str) -> pd.DataFrame:
    """구역 생활형 풀 내 외래종 목록 (카드 하단 회색 참고 표시용)"""
    ideal_forms = IDEAL_FORM[zone]
    df_alien = lib[lib["is_alien"].fillna(False).astype(bool)].copy()
    df_pool = df_alien[df_alien["form_grp"].isin(ideal_forms)].copy()
    if df_pool.empty:
        return pd.DataFrame()
    df_pool["환경적합"] = df_pool.apply(
        lambda row: round((_form_match(row.get("form_grp"), zone) +
                           _csr_match(row.get("c_percent"), row.get("s_percent"),
                                      row.get("r_percent"), zone)) / 2, 2), axis=1)
    df_pool["현장실행"] = ((df_pool["op_establishment"].fillna(0.5) +
                           df_pool["op_safe_growth"].fillna(0.5)) / 2).round(2)
    w_e, w_o = ZONE_SCORE_W[zone]
    df_pool["참고점수"] = ((w_e * df_pool["환경적합"] + w_o * df_pool["현장실행"]) * 100).round().astype(int)
    return df_pool.sort_values("참고점수", ascending=False).reset_index(drop=True)

def _tag_reason(row, zone: str) -> str:
    tags = []
    form = row.get("form_grp") or ""
    if _form_match(form, zone) >= 1.0:
        tags.append(f"{form} ✓")
    c, s, r = row.get("c_percent"), row.get("s_percent"), row.get("r_percent")
    if not any(pd.isna(v) for v in [c, s, r]):
        dom = ["C", "S", "R"][int(np.argmax([float(c), float(s), float(r)]))]
        tags.append(f"{PREF_KOR[dom]} {'✓' if dom == SIT_PREF[zone] else '―'}")
    es = row.get("op_establishment", 0.5) or 0.5
    sg = row.get("op_safe_growth", 0.7) or 0.7
    if es >= 0.75:
        tags.append("빠른 정착")
    elif es < 0.55:
        tags.append("정착 느림")
    if sg >= 0.8:
        tags.append("확산 안전")
    elif sg <= 0.5:
        tags.append("⚠ 확산 주의")
    return " · ".join(tags) if tags else "구역 적합"


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

@st.cache_data
def classify_zones_from_tif(tif_path_str: str) -> dict:
    """업로드된 RGB(A) TIF에서 S1~S4 구역을 실시간 분류.
    반환: {"zone_png": str(base64 data url), "exg_png": str, "veg_png": str,
           "bounds": tuple, "zone_frac": dict, "exg_mean": float, "veg_cover": float}
    """
    MAX_CLF = 1500  # 분류용 최대 픽셀
    MAX_VIZ = 600   # 시각화용 최대 픽셀

    # ── 1단계: 메타데이터 + 오버뷰 확인 ─────────────────────────
    with rasterio.open(tif_path_str) as src:
        bnds = src.bounds
        if src.crs and src.crs.to_epsg() != 4326:
            b = transform_bounds(src.crs, "EPSG:4326", *bnds)
        else:
            b = (bnds.left, bnds.bottom, bnds.right, bnds.top)
        h_orig, w_orig = src.height, src.width
        n = src.count
        ovrs = src.overviews(1)  # 오버뷰 배율 목록 e.g. [2, 4, 8, 16, 32, 64]

    # ── 2단계: 오버뷰 활용(빠름) vs 직접 다운샘플(느림) ──────────
    ovr_level = None
    if ovrs:
        target_factor = max(h_orig, w_orig) / MAX_CLF
        for i, f in enumerate(ovrs):
            if f >= target_factor:
                ovr_level = i
                break
        if ovr_level is None:
            ovr_level = len(ovrs) - 1  # 가장 거친 오버뷰

    if ovr_level is not None:
        with rasterio.open(tif_path_str, overview_level=ovr_level) as src:
            out_h, out_w = src.height, src.width
            bands = src.read().astype(np.float32)
    else:
        scale = min(1.0, MAX_CLF / max(h_orig, w_orig))
        out_h = max(1, int(h_orig * scale))
        out_w = max(1, int(w_orig * scale))
        with rasterio.open(tif_path_str) as src:
            bands = src.read(
                out_shape=(n, out_h, out_w),
                resampling=rasterio.enums.Resampling.average
            ).astype(np.float32)

    # 정규화 (0~1)
    def norm_band(b_arr):
        mn, mx = b_arr.min(), b_arr.max()
        if mx - mn < 1e-6:
            return np.zeros_like(b_arr)
        return (b_arr - mn) / (mx - mn)

    R = norm_band(bands[0])
    G = norm_band(bands[1])
    B = norm_band(bands[2]) if n >= 3 else norm_band(bands[1])
    A = (bands[3] > 0) if n >= 4 else (R + G + B > 0.01)

    # Valid mask
    valid = A if n >= 4 else (R + G + B > 0.01)

    # ExG 식생지수
    exg = 2 * G - R - B
    exg = np.clip(exg, -1, 1)

    # 식생 마스크
    veg = (exg > 0.05) & valid
    bare = (~veg) & valid

    # CHM proxy — ExG 양수 부분만 사용 (음수=나지, 높을수록 키 큰 식생)
    chm = np.where(exg > 0, exg * 14.0, 0.0)  # 0~14m

    # Layer 분류
    layer = np.zeros_like(chm, dtype=np.int32)
    layer[valid & (chm <= 0.3)] = 0   # 나지
    layer[valid & (chm > 0.3) & (chm <= 1.5)] = 1  # 저초
    layer[valid & (chm > 1.5) & (chm <= 5.0)] = 2  # 중간
    layer[valid & (chm > 5.0)] = 3    # 교목

    # 나지 밀도 — 5×5 윈도우 내 나지 픽셀 비율
    from numpy.lib.stride_tricks import sliding_window_view
    pad = 2
    bare_f32 = bare.astype(np.float32)
    bare_padded = np.pad(bare_f32, pad, mode='edge')
    windows = sliding_window_view(bare_padded, (5, 5))
    gb = windows.mean(axis=(-2, -1))  # [0, 1]

    # bd: 나지 중심 픽셀 또는 주변 나지 밀도 ≥ 50%
    bd = (layer == 0) | (gb >= 0.5)

    # S1 vs S2 구분 — 영상 기반 황폐도 점수 (LiDAR 없이 RGB만으로)
    # ① gb : 주변 나지 밀도 → 높을수록 식생 씨앗 유입 차단, 표면 노출 집중
    # ② exg_neg : ExG 음수 강도 → 강할수록 고반사율 나지(재·노출암·심한 교란 토양)
    exg_neg = np.clip(-exg, 0, 1)
    s1_score = 0.6 * gb + 0.4 * exg_neg

    # 나지 픽셀 중 황폐도 상위 35% → S1(긴급안정화), 나머지 → S2(개척파종)
    bd_pixels = valid & bd
    s1_thresh = float(np.percentile(s1_score[bd_pixels], 65)) if bd_pixels.sum() > 10 else 0.5
    s1_cond = bd & (s1_score >= s1_thresh)

    # 구역 분류
    zone = np.zeros((out_h, out_w), dtype=np.int32)
    zone[valid & s1_cond]            = 1  # S1: 긴급안정화
    zone[valid & bd & ~s1_cond]      = 2  # S2: 개척파종
    zone[valid & ~bd & (layer == 3)] = 4  # S4: 하층보완
    zone[valid & ~bd & (layer < 3)]  = 3  # S3: 천이촉진

    # 구역별 픽셀 비율
    total_valid = valid.sum()
    zone_frac = {}
    for v, s in ZONE_VAL_MAP.items():
        cnt = (zone == v).sum()
        zone_frac[s] = float(cnt / total_valid) if total_valid > 0 else 0.0

    # 통계
    exg_mean = float(exg[valid].mean()) if valid.any() else 0.0
    veg_cover = float(veg.sum() / total_valid) if total_valid > 0 else 0.0

    def _make_overlay_png(rgba_arr: np.ndarray) -> str:
        """RGBA numpy array → base64 PNG data URL, max MAX_VIZ px"""
        h_a, w_a = rgba_arr.shape[:2]
        sc = min(1.0, MAX_VIZ / max(h_a, w_a))
        if sc < 1.0:
            new_h, new_w = max(1, int(h_a * sc)), max(1, int(w_a * sc))
            img = Image.fromarray(rgba_arr, "RGBA").resize((new_w, new_h), Image.NEAREST)
        else:
            img = Image.fromarray(rgba_arr, "RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    # 구역 PNG
    zone_rgba = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    for val, s in ZONE_VAL_MAP.items():
        mask = zone == val
        rgb = ZONE_PAL_RGB[s]
        zone_rgba[mask, 0] = rgb[0]
        zone_rgba[mask, 1] = rgb[1]
        zone_rgba[mask, 2] = rgb[2]
        zone_rgba[mask, 3] = 140
    zone_png = _make_overlay_png(zone_rgba)

    # ExG PNG (초록 그라디언트)
    exg_disp = ((exg + 1) / 2 * 255).astype(np.uint8)
    exg_rgba = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    exg_rgba[:, :, 1] = exg_disp  # Green channel
    exg_rgba[:, :, 3] = np.where(valid, 160, 0).astype(np.uint8)
    exg_png = _make_overlay_png(exg_rgba)

    # 식생마스크 PNG
    veg_rgba = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    veg_rgba[veg, 0] = 46; veg_rgba[veg, 1] = 125; veg_rgba[veg, 2] = 50; veg_rgba[veg, 3] = 160   # 초록
    veg_rgba[bare, 0] = 239; veg_rgba[bare, 1] = 108; veg_rgba[bare, 2] = 0; veg_rgba[bare, 3] = 160  # 주황
    veg_png = _make_overlay_png(veg_rgba)

    return {
        "zone_png": zone_png,
        "exg_png": exg_png,
        "veg_png": veg_png,
        "bounds": b,
        "zone_frac": zone_frac,
        "exg_mean": exg_mean,
        "veg_cover": veg_cover,
        "zone_arr": zone,   # lat/lon → 픽셀 색인용 (드롭점 구역 결정)
    }

def sample_zone_at_points(markers: list[tuple],
                          zone_arr: np.ndarray | None = None,
                          bounds: tuple | None = None) -> list[str | None]:
    """드롭점 좌표 → 해당 구역 코드 반환. zone_arr 있으면 실시간 분류 결과 우선 사용."""
    if zone_arr is not None and bounds is not None:
        h, w = zone_arr.shape
        xmn, ymn, xmx, ymx = bounds
        dx, dy = xmx - xmn, ymx - ymn
        result = []
        for lat, lon in markers:
            col = int((lon - xmn) / dx * (w - 1))
            row_ = int((ymx - lat) / dy * (h - 1))
            if 0 <= row_ < h and 0 <= col < w:
                v = int(zone_arr[row_, col])
                result.append(ZONE_VAL_MAP.get(v, None) if v > 0 else None)
            else:
                result.append(None)
        return result
    if not ZONE_TIF.exists():
        return [None] * len(markers)
    with rasterio.open(ZONE_TIF) as src:
        coords = [(lon, lat) for lat, lon in markers]
        vals = [v[0] for v in src.sample(coords)]
    return [ZONE_VAL_MAP.get(int(v), None) if not np.isnan(v) else None for v in vals]


# ════════════════════════════════════════════════════════════
# 지도 생성
# ════════════════════════════════════════════════════════════
def build_map(meta: dict, lib: pd.DataFrame, spacing_m: float = 2,
              layers: dict | None = None, tops: dict | None = None) -> str:
    lat_c, lon_c = meta["lat_c"], meta["lon_c"]
    b = meta["bounds"]

    m = folium.Map(location=[lat_c, lon_c], zoom_start=15, tiles=None)
    folium.TileLayer("OpenStreetMap", name="일반 지도").add_to(m)
    folium.TileLayer("Esri.WorldImagery", name="위성", attr="Esri").add_to(m)  # 마지막 = 기본

    # 대상지 경계
    folium.Rectangle([[b[1],b[0]],[b[3],b[2]]],
                     color="#f1c40f", weight=2, fill=False,
                     name="대상지").add_to(m)

    # 구역 오버레이 — 실시간 분류 결과 또는 fallback(데모용 고정 TIF)
    if layers is not None:
        zone_png = layers.get("zone_png")
        zone_bnds = layers.get("bounds")
        if zone_png and zone_bnds:
            folium.raster_layers.ImageOverlay(
                image=zone_png,
                bounds=[[zone_bnds[1], zone_bnds[0]], [zone_bnds[3], zone_bnds[2]]],
                opacity=0.55, name="복원 구역 (실시간)",
            ).add_to(m)
        exg_png = layers.get("exg_png")
        if exg_png and zone_bnds:
            folium.raster_layers.ImageOverlay(
                image=exg_png,
                bounds=[[zone_bnds[1], zone_bnds[0]], [zone_bnds[3], zone_bnds[2]]],
                opacity=0.6, name="ExG 식생지수", show=False,
            ).add_to(m)
        veg_png = layers.get("veg_png")
        if veg_png and zone_bnds:
            folium.raster_layers.ImageOverlay(
                image=veg_png,
                bounds=[[zone_bnds[1], zone_bnds[0]], [zone_bnds[3], zone_bnds[2]]],
                opacity=0.6, name="식생 마스크", show=False,
            ).add_to(m)
    else:
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
    if tops is None:
        tops = {s: (recommend_for(lib, s, 1)["name_kor"].iloc[0]
                    if len(recommend_for(lib, s, 1)) > 0 else "-")
                for s in ZONE_CODE}
    zone_arr  = layers.get("zone_arr")  if layers else None
    zone_bnds = layers.get("bounds")    if layers else None
    zone_codes = sample_zone_at_points(ms["markers"], zone_arr=zone_arr, bounds=zone_bnds)
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

    # 범례 — MacroElement로 leaflet 맵 컨테이너 안에 오버레이
    legend_rows = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0;">'
        f'<span style="background:{ZONE_PAL[s]};width:12px;height:12px;'
        f'border-radius:2px;flex-shrink:0;"></span>'
        f'<span>{ZONE_NAME[s].replace(" 구역","")} → <b>{tops.get(s,"-")}</b></span></div>'
        for s in ZONE_CODE
    )
    legend_html = (
        '<div style="position:absolute;bottom:30px;right:10px;z-index:9999;'
        'background:rgba(255,255,255,0.93);padding:10px 12px;border-radius:7px;'
        'font-size:12px;border:1px solid #ccc;min-width:170px;'
        'box-shadow:0 2px 6px rgba(0,0,0,.18);">'
        '<b style="font-size:13px;">🎨 어디에 무엇을</b>'
        f'{legend_rows}</div>'
    )
    macro = MacroElement()
    macro._template = JinjaTemplate("{% macro html(this, kwargs) %}" + legend_html + "{% endmacro %}")
    macro.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])
    return m


# ════════════════════════════════════════════════════════════
# 점수 차트 (Plotly)
# ════════════════════════════════════════════════════════════
def score_chart(df: pd.DataFrame, alien_df: pd.DataFrame | None = None) -> go.Figure:
    # ── 자생종 ──────────────────────────────────────────────
    names  = df["name_kor"].tolist()[::-1]
    env_   = (df["환경적합"] * 50).tolist()[::-1]
    est_   = (df["op_establishment"].fillna(0.5) * 25).tolist()[::-1]
    saf_   = (df["op_safe_growth"].fillna(0.5) * 25).tolist()[::-1]
    scores = df["추천점수"].tolist()[::-1]

    has_alien = alien_df is not None and not alien_df.empty

    # 외래종이 있으면 구분 줄 + 외래종 항목을 맨 아래(y축 앞)에 추가
    if has_alien:
        sep        = ["── 참고: 외래종 ──"]
        a_names    = alien_df["name_kor"].tolist()
        a_env      = (alien_df["환경적합"] * 50).tolist()
        a_est      = (alien_df["op_establishment"].fillna(0.5) * 25).tolist()
        a_saf      = (alien_df["op_safe_growth"].fillna(0.5) * 25).tolist()
        a_scores   = alien_df["참고점수"].tolist()
        all_names  = a_names + sep + names
        all_env    = a_env  + [0] + env_
        all_est    = a_est  + [0] + est_
        all_saf    = a_saf  + [0] + saf_
        all_scores = a_scores + [None] + scores
        # 색: 외래종=회색, 구분=투명, 자생종=각 색
        env_colors = ["#bdbdbd"]*len(a_names) + ["rgba(0,0,0,0)"] + ["#2e7d32"]*len(names)
        est_colors = ["#bdbdbd"]*len(a_names) + ["rgba(0,0,0,0)"] + ["#1e88e5"]*len(names)
        saf_colors = ["#bdbdbd"]*len(a_names) + ["rgba(0,0,0,0)"] + ["#ef6c00"]*len(names)
        n_rows     = len(all_names)
    else:
        all_names  = names
        all_env, all_est, all_saf, all_scores = env_, est_, saf_, scores
        env_colors = ["#2e7d32"] * len(names)
        est_colors = ["#1e88e5"] * len(names)
        saf_colors = ["#ef6c00"] * len(names)
        n_rows     = len(names)

    fig = go.Figure()
    # 호버: 각 구성요소 상세 설명 포함
    env_hover = "<b>%{y}</b><br>환경 적합: %{x:.0f}점<br><i>CSR 생태전략 + 생활형 매칭</i><extra></extra>"
    est_hover = "<b>%{y}</b><br>정착: %{x:.0f}점<br><i>SLA 기반 초기 정착력</i><extra></extra>"
    saf_hover = "<b>%{y}</b><br>안전: %{x:.0f}점<br><i>LDMC 기반 장기 생존력</i><extra></extra>"

    fig.add_trace(go.Bar(name="환경 적합 (CSR·생활형)", y=all_names, x=all_env, orientation="h",
                         marker_color=env_colors, hovertemplate=env_hover))
    fig.add_trace(go.Bar(name="정착 (SLA·초기 성장)",   y=all_names, x=all_est, orientation="h",
                         marker_color=est_colors, hovertemplate=est_hover))
    fig.add_trace(go.Bar(name="안전 (LDMC·생존력)",     y=all_names, x=all_saf, orientation="h",
                         marker_color=saf_colors, hovertemplate=saf_hover))

    # 총점 레이블 — 막대 끝에 짧게 "86점"만 표시, 상세는 호버로
    label_x = [e+es+sa+1 for e,es,sa in zip(all_env, all_est, all_saf)]
    label_t = [f"{s}점" if s is not None else "" for s in all_scores]
    label_c = (["#999"]*len(a_names) + ["rgba(0,0,0,0)"] + ["#333"]*len(names)) if has_alien else ["#333"]*len(names)
    fig.add_trace(go.Scatter(
        x=label_x, y=all_names, mode="text", text=label_t,
        textposition="middle right",
        textfont=dict(size=12, color=label_c, family="Arial Black"),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.update_layout(
        barmode="stack", height=max(220, n_rows * 36 + 60),
        margin=dict(l=0, r=55, t=20, b=20),
        legend=dict(orientation="h", y=1.14, x=0, font=dict(size=11)),
        xaxis=dict(range=[0, 108], title="추천 점수 (100점 만점)", fixedrange=True),
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

def drop_points_csv_bytes(markers: list, zone_codes: list, rec_cache: dict) -> bytes:
    """파종 드롭점 → 좌표 + 구역 + 추천 Top3 종 CSV"""
    rows = []
    for i, ((lat, lon), z) in enumerate(zip(markers, zone_codes)):
        top3 = rec_cache[z].head(3)["name_kor"].tolist() if z and z in rec_cache else []
        rows.append({
            "순번": i + 1,
            "위도": round(lat, 7),
            "경도": round(lon, 7),
            "구역코드": z or "구역밖",
            "구역명": ZONE_NAME.get(z, "구역 밖"),
            "1순위종": top3[0] if len(top3) > 0 else "",
            "2순위종": top3[1] if len(top3) > 1 else "",
            "3순위종": top3[2] if len(top3) > 2 else "",
        })
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
        local_path_str = st.text_input(
            "📁 드론 영상 경로 (GeoTIFF)",
            placeholder=r"예: Z:\1.Project\현장\result.tif",
            help="이 PC에 있는 파일 경로. 크기 제한 없음.",
        )
        uploaded = st.file_uploader(
            "또는 파일 직접 업로드 (외부 PC에서 접속 시)",
            type=["tif", "tiff"],
            help="외부 기기에서 접속해 자신의 TIF를 올릴 때 사용합니다.",
        )
        use_demo = False
        if local_path_str and Path(local_path_str).exists():
            fsize_mb = Path(local_path_str).stat().st_size / 1e6
            st.success(f"✅ {Path(local_path_str).name}  ({fsize_mb:.0f} MB)")
            # 대용량 파일: 오버뷰 상태 표시
            if fsize_mb > 200:
                try:
                    with rasterio.open(local_path_str) as _s:
                        _ovrs = _s.overviews(1)
                    if not _ovrs:
                        st.warning(
                            f"⚠ 대용량 파일({fsize_mb:.0f}MB)에 오버뷰가 없습니다. "
                            "첫 분석에 수 분이 걸릴 수 있습니다. "
                            "아래 버튼으로 1회 전처리하면 이후 1초 이내로 빨라집니다."
                        )
                        if st.button("⚡ 오버뷰 생성 (1회 전처리)", key="build_ovr"):
                            with st.spinner(f"오버뷰 생성 중… ({fsize_mb:.0f}MB, 1~3분 소요)"):
                                with rasterio.open(local_path_str, "r+") as _ds:
                                    _ds.build_overviews(
                                        [2, 4, 8, 16, 32, 64],
                                        rasterio.enums.Resampling.nearest,
                                    )
                                    _ds.update_tags(ns="rio_overview", resampling="nearest")
                                classify_zones_from_tif.clear()
                            st.success("✅ 오버뷰 생성 완료! 빠른 분석을 사용합니다.")
                            st.rerun()
                    else:
                        st.caption(f"⚡ 오버뷰 {len(_ovrs)}레벨 확인 — 빠른 분석 사용")
                except Exception:
                    pass
        elif local_path_str:
            st.error("⚠ 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        elif uploaded:
            st.success("✅ 업로드된 파일로 분석합니다.")
        else:
            use_demo = st.checkbox("📍 데모 영상으로 보기 (result_small.tif)", value=True)
        spacing_m = st.number_input("🚁 드론 비행 간격 (m) — 작을수록 촘촘",
                                    min_value=1, max_value=20, value=2, step=1)
        zone_sel = st.selectbox(
            "🗺 구역 선택 (상세 추천 볼 구역)",
            options=ZONE_CODE,
            format_func=lambda s: ZONE_NAME[s],
            index=3,
        )
        st.caption(f"👉 {ZONE_DESC[zone_sel]}")

        # TIF 결정 — 경로 > 업로드 > 데모 순 우선순위
        tif_path: Path | None = None
        if local_path_str and Path(local_path_str).exists():
            tif_path = Path(local_path_str)
        elif uploaded:
            up_name = uploaded.name
            if st.session_state.get("_up_name") != up_name:
                with st.spinner(f"📂 {up_name} 저장 중…"):
                    suffix = Path(up_name).suffix or ".tif"
                    old_tmp = st.session_state.get("_tmp_tif")
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    tmp.write(uploaded.read())
                    tmp.close()
                    if old_tmp:
                        try: os.unlink(old_tmp)
                        except: pass
                    st.session_state["_tmp_tif"] = tmp.name
                    st.session_state["_up_name"] = up_name
            tif_path = Path(st.session_state["_tmp_tif"])
        elif use_demo and DEMO_TIF.exists():
            tif_path = DEMO_TIF

        # 실시간 구역 분류 (데모·업로드 동일하게 처리, 캐시로 데모는 빠름)
        if tif_path:
            tif_path_str = str(tif_path)
            is_demo = tif_path == DEMO_TIF
            status_label = "📍 데모 영상 분석 중…" if is_demo else "🔬 영상 분석 중…"
            with st.status(status_label, expanded=True) as _sts:
                try:
                    _sts.write("🛰 픽셀에 따른 지형 분석 중…")
                    layers_result = classify_zones_from_tif(tif_path_str)
                    _sts.write("🗺 복원 구역(S1–S4) 경계 확정 중…")
                    st.session_state["layers"] = layers_result
                    _sts.write("🌱 분석 결과에 따른 수종 선정 중…")
                    _sts.update(label="✅ 영상 분석 완료", state="complete", expanded=False)
                except Exception as e:
                    _sts.update(label="❌ 분류 실패", state="error", expanded=False)
                    st.warning(f"구역 분류 실패 (기본 표시로 대체): {e}")
                    st.session_state["layers"] = None

        # 메타 표시
        if tif_path:
            try:
                with st.spinner("📡 영상 정보 읽는 중…"):
                    meta = read_tif_meta(tif_path)
                st.session_state["meta"] = meta
                ms = make_mission(meta["bounds"], spacing_m)
                st.session_state["ms"] = ms
                st.success(
                    f"✅ **{meta['area_ha']:.2f} ha** 대상지 로드 완료 | 드롭점 약 {ms['n_way']:,}개"
                )
                with st.expander("📐 상세 정보"):
                    st.caption(f"중심: {meta['lat_c']:.5f}°N, {meta['lon_c']:.5f}°E")
                    st.caption(f"범위: {meta['w_m']:.0f}m × {meta['h_m']:.0f}m")
                    st.caption(f"크기: {meta['nbands']}밴드 · {meta['width']}×{meta['height']}px")
                    st.caption(f"비행: 라인 {ms['n_lines']}개 · 간격 {spacing_m}m")
            except Exception as e:
                st.error(f"영상 읽기 실패: {e}")
                st.session_state.pop("meta", None)

    # ── 결과 패널 ────────────────────────────────────────────
    with col_out:
        st.subheader("② 결과")
        lib = st.session_state["lib"]
        # 추천 캐시 — 한 번만 계산해서 카드·차트·CSV 전부 재사용
        rec_cache   = {s: recommend_for(lib, s, 8) for s in ZONE_CODE}
        alien_cache = {s: alien_in_pool(lib, s) for s in ZONE_CODE}
        tops = {s: (rec_cache[s]["name_kor"].iloc[0] if len(rec_cache[s]) > 0 else "-")
                for s in ZONE_CODE}

        if "meta" not in st.session_state:
            st.info("← 왼쪽에서 드론 영상을 선택하면 결과가 여기에 나타납니다.")
        else:
            meta = st.session_state["meta"]
            ms   = st.session_state["ms"]

            # 파종지점 구역 결정 (drop_points CSV + 지도 드롭점 공유)
            _layers = st.session_state.get("layers")
            _zone_arr  = _layers.get("zone_arr")  if _layers else None
            _zone_bnds = _layers.get("bounds")    if _layers else None
            zone_codes_for_markers = sample_zone_at_points(
                ms["markers"], zone_arr=_zone_arr, bounds=_zone_bnds)

            # 지도
            with st.spinner("🛰 지도 생성 중…"):
                folium_map = build_map(meta, lib, spacing_m,
                                      layers=_layers,
                                      tops=tops)
            components.html(folium_map._repr_html_(), height=520, scrolling=False)
            with st.expander("🗺 지도 레이어 설명"):
                st.markdown(
                    "지도 우상단 레이어 컨트롤에서 각 레이어를 켜고 끌 수 있습니다.\n\n"
                    "- **복원 구역** — S1(빨강·긴급안정화) / S2(주황·개척파종) / S3(노랑·천이촉진) / S4(초록·하층보완) 구역 경계\n"
                    "- **ExG 식생지수** — ExG = 2×G − R − B. 초록이 진할수록 식생 밀도 높음. 음수(어두움)는 나지\n"
                    "- **식생 마스크** — ExG > 0.05 픽셀을 초록(식생), 나머지를 주황(나지)으로 표시. "
                    "구역 분류의 기초 레이어로, 초록 면적이 넓을수록 S3·S4 비중이 높아짐\n"
                    "- **드론 경로** — 설정한 비행 간격(m)으로 생성된 왕복(boustrophedon) 비행 경로\n"
                    "- **드롭점** — 시드볼을 투하할 격자 지점. 최대 400점 표시"
                )

            st.divider()

            # ── 분석 레이어 요약 패널 ─────────────────────────
            _layers = st.session_state.get("layers")
            if _layers is not None:
                zone_frac = _layers.get("zone_frac", {})
                exg_mean = _layers.get("exg_mean", 0.0)
                veg_cover = _layers.get("veg_cover", 0.0)

                st.markdown("**🔬 실시간 분석 결과**")
                frac_cols = st.columns(4)
                for i, s in enumerate(ZONE_CODE):
                    frac = zone_frac.get(s, 0.0)
                    with frac_cols[i]:
                        st.metric(
                            label=f"{ZONE_NAME[s].replace(' 구역','')} ({s})",
                            value=f"{frac*100:.1f}%",
                            help=ZONE_DESC[s],
                        )

                with st.expander("📊 분석 레이어 상세"):
                    ec1, ec2 = st.columns(2)
                    with ec1:
                        exg_lbl = "높음(식생 풍부)" if exg_mean > 0.1 else ("보통" if exg_mean > 0 else "낮음(나지 우세)")
                        st.metric("ExG 평균 (식생지수)", f"{exg_mean:.3f}", help="ExG = 2G-R-B. 양수일수록 식생 많음")
                        st.caption(f"해석: {exg_lbl}")
                    with ec2:
                        st.metric("식생 피복률", f"{veg_cover*100:.1f}%", help="ExG>0.05 픽셀 비율")
                        st.caption("레이어 토글: 지도 우상단 레이어 컨트롤에서 ExG·식생마스크 켜기")
                    st.info(
                        "📐 **분석 방법 안내** — 구역 분류는 RGB 드론 영상의 식생지수(ExG)와 "
                        "나지 밀도만으로 수행됩니다. S3·S4의 층위 구분은 ExG 강도로 추정하며, "
                        "실제 수관 높이 측정(LiDAR)보다 정밀도가 낮을 수 있습니다. "
                        "현장 판단과 함께 참고 자료로 활용하세요.",
                        icon="ℹ️",
                    )

            st.divider()

            # ── 전체 구역 요약 카드 ──────────────────────────
            st.markdown("**🗺 전체 복원 구역 요약 — 구역별 추천 Top 3**")
            card_cols = st.columns(4)
            for i, s in enumerate(ZONE_CODE):
                top3 = rec_cache[s].head(3)
                color = ZONE_PAL[s]
                with card_cols[i]:
                    html = (f'<div style="background:#fafafa;border-left:4px solid {color};'
                            f'border-radius:4px;padding:10px 12px;">'
                            f'<div style="font-weight:bold;color:{color};font-size:13px;margin-bottom:4px;">'
                            f'{ZONE_NAME[s]}</div>'
                            f'<div style="font-size:12px;color:#555;margin-bottom:8px;">{ZONE_DESC[s]}</div>')
                    for _, row in top3.iterrows():
                        html += (f'<div style="display:flex;justify-content:space-between;'
                                 f'font-size:13px;padding:3px 0;border-bottom:1px solid #eee;">'
                                 f'<span>{int(row["순위"])}. {row["name_kor"]} ({row["form_grp"]})</span>'
                                 f'<span style="font-weight:bold;color:{color};">{row["추천점수"]}점</span>'
                                 f'</div>')
                    # 외래종 참고 섹션 (구역 풀에 외래종이 있을 때만)
                    aliens = alien_cache.get(s, pd.DataFrame())
                    if not aliens.empty:
                        html += ('<div style="margin-top:8px;padding:5px 7px;'
                                 'background:#f0f0f0;border-radius:4px;">'
                                 '<div style="font-size:10px;color:#999;margin-bottom:3px;">'
                                 '참고 (외래종 — 조달 용이하나 확산 주의)</div>')
                        for _, a in aliens.iterrows():
                            html += (f'<div style="font-size:11px;color:#aaa;padding:1px 0;">'
                                     f'{a["name_kor"]} ({a["form_grp"]}) — {a["참고점수"]}점</div>')
                        html += '</div>'
                    html += '</div>'
                    st.markdown(html, unsafe_allow_html=True)

            st.divider()

            # ── 구역별 상세 추천 ─────────────────────────────
            st.markdown(f"**🌱 [{ZONE_NAME[zone_sel]}] 추천 식물 상세**")
            rec = rec_cache[zone_sel]

            # 점수 차트 (외래종 회색 막대 포함)
            alien_rec = alien_cache.get(zone_sel, pd.DataFrame())
            st.plotly_chart(score_chart(rec, alien_rec))
            with st.expander("📖 점수 구성 설명"):
                st.markdown(
                    "- 🟩 **환경 적합 (최대 50점)** — 구역 CSR 생태전략과 생활형이 맞는 정도. "
                    "S1(개척R)·S2(경쟁C)·S3(경쟁C)·S4(내성S) 기준으로 종의 CSR 비율을 비교. "
                    "※ S1은 극단적 나지라 빠른 피복이 목적인 개척형(R), "
                    "S2는 인접 식생이 있어 경쟁 속에서도 살아남는 경쟁형(C)이 장기 군락 형성에 유리.\n"
                    "- 🟦 **정착 (최대 25점)** — SLA(비엽면적) 기반 초기 정착력. "
                    "값이 클수록 잎이 얇고 빠른 광합성 → 개방지 조기 정착에 유리.\n"
                    "- 🟧 **안전 (최대 25점)** — LDMC(잎 건조질량비) 기반 장기 생존력. "
                    "값이 클수록 잎이 두껍고 질겨 건조·척박 환경에서도 생존 안정."
                )

            # ── 자생종 추천 표 ─────────────────────────────
            disp_cols  = ["순위","name_kor","form_grp","추천점수","추천이유"]
            disp_names = {"순위":"순위","name_kor":"식물 이름",
                          "form_grp":"생활형","추천점수":"추천 점수","추천이유":"추천 이유"}
            st.dataframe(
                rec[disp_cols].rename(columns=disp_names),
                hide_index=True,
                column_config={"추천 점수": st.column_config.ProgressColumn(
                    "추천 점수", min_value=0, max_value=100, format="%d점")},
            )

            # ── 외래종 참고 표 ──────────────────────────────
            if not alien_rec.empty:
                st.markdown(
                    '<p style="font-size:12px;color:#888;margin:12px 0 4px 0;">'
                    '⚠ 참고 — 외래종 (조달 용이하나 확산 주의 / 자생종 우선 후 보조 활용 권고)</p>',
                    unsafe_allow_html=True,
                )
                a_disp = alien_rec.copy()
                a_disp["순위"] = [f"참고 {i+1}" for i in range(len(a_disp))]
                a_disp_cols  = ["순위","name_kor","form_grp","참고점수"]
                a_disp_names = {"순위":"순위","name_kor":"식물 이름",
                                "form_grp":"생활형","참고점수":"참고 점수"}
                st.dataframe(
                    a_disp[a_disp_cols].rename(columns=a_disp_names),
                    hide_index=True,
                    column_config={"참고 점수": st.column_config.ProgressColumn(
                        "참고 점수", min_value=0, max_value=100, format="%d점")},
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
                env_lbl = "잘 맞음" if sel["환경적합"]>=0.75 else ("부분 적합" if sel["환경적합"]>=0.5 else "적합도 낮음")
                st.metric("환경 적합도", env_lbl, f"{sel['환경적합']*100:.0f}/100")
            with c2:
                est_v = sel.get("op_establishment", 0.5) or 0.5
                saf_v = sel.get("op_safe_growth", 0.7) or 0.7
                st.metric("정착 속도", "빠름" if est_v>=0.8 else ("보통" if est_v>=0.6 else "느림"),
                          f"{est_v*100:.0f}/100")
                st.metric("확산 안전성", "안전" if saf_v>=0.8 else ("⚠ 주의" if saf_v<=0.5 else "보통"),
                          f"{saf_v*100:.0f}/100")
            c_v = sel.get("c_percent"); r_v = sel.get("r_percent"); s_v = sel.get("s_percent")
            # 💡 한 줄 추천 이유 (시연용)
            _zone_why = {
                "S1": "나지 밀도·황폐도가 높아 흙 유실 차단이 시급합니다. 빠르게 퍼지는 개척형이 1순위입니다.",
                "S2": "나지이지만 인접 식생이 있어 자연 유입도 기대됩니다. 첫 식물 정착으로 토양 형성을 시작합니다.",
                "S3": "이미 풀·관목이 있습니다. 경쟁력 있는 식물로 숲 천이를 촉진합니다.",
                "S4": "큰 나무 그늘 아래입니다. 적은 빛에서도 버티는 내성 식물이 필요합니다.",
            }
            if not pd.isna(c_v):
                dom = ["C","S","R"][int(np.argmax([float(c_v), float(s_v), float(r_v)]))]
                dom_str = (f" **{sel_name}**은(는) {PREF_KOR[dom]} 전략이 강해 이 구역 조건과 잘 맞습니다."
                           if dom == SIT_PREF[zone_sel]
                           else f" **{sel_name}**은(는) {PREF_KOR[dom]} 전략 종으로, 정착·안전 점수로 순위가 결정됩니다.")
                st.info(f"💡 {_zone_why[zone_sel]}{dom_str}")
                st.caption(f"생태전략 CSR: C{c_v:.0f} · S{s_v:.0f} · R{r_v:.0f}  (구역 선호: {PREF_KOR[SIT_PREF[zone_sel]]})")
            else:
                st.info(f"💡 {_zone_why[zone_sel]}")
                st.caption("CSR 데이터 없음 → 중립 0.5 처리")
            st.caption("📦 추천 목록은 현지 조달 가능 수준으로 검토된 59종 내에서 선별됩니다. 단, 수급 가능 여부는 시기·공급처 상황에 따라 변동될 수 있으므로 참고사항으로 활용하세요.")

            st.divider()

            # ── 내보내기 ─────────────────────────────────────
            st.markdown("**📤 내보내기**")
            dl1, dl2, dl3, dl4 = st.columns(4)
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
                _pdf_zone_frac = _layers.get("zone_frac") if _layers else None
                _pdf_exg       = _layers.get("exg_mean")  if _layers else None
                _pdf_veg       = _layers.get("veg_cover") if _layers else None
                _pdf_zone_png  = _layers.get("zone_png")  if _layers else None
                st.download_button(
                    "전체 구역 요약 PDF",
                    data=make_summary_pdf(
                        rec_cache,
                        meta=meta,
                        zone_frac=_pdf_zone_frac,
                        exg_mean=_pdf_exg,
                        veg_cover=_pdf_veg,
                        zone_png=_pdf_zone_png,
                    ),
                    file_name=f"ReSeed_전체구역_{today}.pdf",
                    mime="application/pdf",
                )
            with dl4:
                st.download_button(
                    "📍 파종지점 CSV",
                    data=drop_points_csv_bytes(ms["markers"], zone_codes_for_markers, rec_cache),
                    file_name=f"ReSeed_파종지점_{today}.csv",
                    mime="text/csv",
                    help="드롭점마다 좌표·구역·추천 Top3 종 포함",
                )

            st.markdown('<p class="score-note">추천 점수 = 환경 적합(CSR 전략 + 생활형) + 현장 실행력(정착·안전 생장). 조달 가능성은 59종 목록 선별 단계에서 이미 반영되어 점수에 별도 포함하지 않습니다. 수급 가능 여부는 시기·공급처 상황에 따라 변동될 수 있습니다. ⚠ 표시 = 귀화식물(외래종) — 현장 확산 모니터링 권장.</p>',
                        unsafe_allow_html=True)

            with st.expander("📐 점수 산출 방식", expanded=False):
                st.markdown("""
**총점 100점 만점** — 환경 적합(50점) + 정착 속도(25점) + 확산 안전(25점)

| 항목 | 배점 | 산출 방법 |
|---|---|---|
| **환경 적합** | 50점 | 생활형 일치도×0.6 + CSR 생태전략×0.4 (C1: CSR 저가중치) |
| **정착 속도** | 25점 | 비엽면적(SLA) 기반 정착력 지수 — 높을수록 초기 피복 빠름 |
| **확산 안전** | 25점 | 잎 건조질량비(LDMC) 기반 생존력 지수 — 높을수록 장기 생존 안정 |

**구역별 가중치** (환경 적합 : 현장 실행)
- S1 긴급 안정화: 35 : 65 — 침식 대응이 급하므로 정착 속도 우선
- S2 개척 파종: 60 : 40 — 올바른 생태형 선택이 더 중요
- S3 천이 촉진: 65 : 35 — 장기 군락 구성 고려
- S4 하층 보완: 50 : 50 — 균형

**구역별 CSR 선호 전략**
- S1 개척R — 극단적 나지·고황폐도. 경쟁이 없는 환경에서 빠르게 피복하는 개척형 우선
- S2 경쟁C — 인접 식생이 있어 자원 경쟁 발생. 경쟁 속에서도 살아남아 군락 형성하는 경쟁형 유리
- S3 경쟁C — 풀·관목 단계. 천이를 이끌 경쟁력 강한 종이 필요
- S4 내성S — 수관 그늘. 낮은 빛에서 버티는 내성형(스트레스 내성) 적합

CSR 데이터가 없는 종은 중립값(0.5) 처리됩니다.
""")



# ════════════════════════════════════════════════════════════
# 앱 하단 고정 — 주의사항
# ════════════════════════════════════════════════════════════
st.divider()
st.markdown("""
<div style="font-size:11px;color:#9e9e9e;line-height:1.8;padding:4px 0 12px 0;">
<strong>⚠ 주의사항</strong><br>
ReSeed는 드론 영상 분석을 바탕으로 복원 계획 수립을 돕는 <strong>의사결정 지원 도구</strong>입니다.
제시된 구역 분류와 식물 추천은 알고리즘이 산출한 참고 정보이며,
현장의 토양·지형·미기후·수계·법적 보호종 등 실제 조건에 따라 결과가 달라질 수 있습니다.
라이브러리 데이터의 한계 및 모델 불확실성이 존재하므로
출력값을 확신하거나 맹신하지 마시고,
<strong>최종 복원 계획과 시공 결정은 반드시 생태 복원 전문가의 현장 검토를 거쳐 이루어지시기 바랍니다.</strong>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 2 — 종 라이브러리 관리
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🌿 종 라이브러리 관리")
    st.caption(f"현재 등록 종: **{len(st.session_state['lib'])}종** | 파일: `data/seedball_library.csv`")
    lib = st.session_state["lib"]

    # ── 현재 목록 ────────────────────────────────────────────
    with st.expander("📋 현재 등록 종 목록 보기", expanded=True):
        lib_disp = lib.copy()
        lib_disp["외래종여부"] = lib_disp["is_alien"].apply(lambda x: "⚠ 외래종" if x else "")
        view_cols = ["name_kor","외래종여부","name_sci","form_grp","c_percent","s_percent","r_percent",
                     "op_establishment","op_safe_growth"]
        view_names = {"name_kor":"국문명","외래종여부":"주의","name_sci":"학명","form_grp":"생활형",
                      "c_percent":"C%","s_percent":"S%","r_percent":"R%",
                      "op_establishment":"정착 속도","op_safe_growth":"생장 안전"}
        st.dataframe(lib_disp[view_cols].rename(columns=view_names),
                     hide_index=True)

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
            new_alien = st.checkbox("⚠ 외래종 (귀화식물)", value=False)
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
            elif csr_known and (new_c + new_s + new_r) != 100:
                st.error(f"CSR 합이 {new_c + new_s + new_r}입니다. C + S + R = 100 이어야 합니다. (현재 C:{new_c} + S:{new_s} + R:{new_r})")
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
                    "is_alien": bool(new_alien),
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
                    st.dataframe(new_df, hide_index=True)
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
