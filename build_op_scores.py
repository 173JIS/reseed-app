# -*- coding: utf-8 -*-
"""
build_op_scores.py — ReSeed 시드볼 라이브러리 자동 갱신
  - TraitX DB(trait_kor_mean.xlsx)가 라이브러리보다 새로울 때마다 호출됨
  - 갱신 항목: SLA/LDMC → op_establishment/op_safe_growth
               csr_result → c_percent / s_percent / r_percent / strategy_class
  - 단독 실행: python build_op_scores.py
  - 모듈 호출: from build_op_scores import rebuild
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np

_BASE     = Path(__file__).parent
_traitx_win = Path(r"C:\Users\Intern\OneDrive\Desktop\Rstudio\결과물\trait_kor_mean.xlsx")
TRAITX_DB = Path(os.environ.get("TRAITX_DB", str(_traitx_win)))
if not TRAITX_DB.exists():
    TRAITX_DB = _BASE / "data" / "trait_kor_mean.xlsx"
LIB_CSV   = Path(os.environ.get("LIB_CSV", str(_BASE / "data" / "seedball_library.csv")))


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi == lo:
        return s.apply(lambda _: 0.5)
    return ((s - lo) / (hi - lo)).round(3)


def rebuild(traitx_path: Path = TRAITX_DB, lib_path: Path = LIB_CSV) -> dict:
    """
    TraitX 최신값으로 라이브러리 갱신. 변경 내용을 dict로 반환.
    """
    # ── 1. TraitX 로드 ────────────────────────────────────────
    trait = pd.read_excel(traitx_path, sheet_name="trait_kor_mean")[
        ["name_kor", "SLA", "LDMC"]
    ]
    csr = pd.read_excel(traitx_path, sheet_name="csr_result")[
        ["name_kor", "c_percent", "s_percent", "r_percent", "strategy_class"]
    ]

    # ── 2. 라이브러리 로드 ────────────────────────────────────
    lib = pd.read_csv(lib_path, encoding="utf-8-sig")

    # ── 3. 형질·CSR 병합 ──────────────────────────────────────
    merged = (
        lib
        .merge(trait.rename(columns={"SLA": "_SLA", "LDMC": "_LDMC"}),
               on="name_kor", how="left")
        .merge(csr.rename(columns={
               "c_percent": "_c", "s_percent": "_s",
               "r_percent": "_r", "strategy_class": "_strat"}),
               on="name_kor", how="left")
    )

    n_sla  = merged["_SLA"].notna().sum()
    n_csr  = merged["_c"].notna().sum()
    print(f"[build_op_scores] 매칭: SLA/LDMC {n_sla}/{len(lib)}종 | CSR {n_csr}/{len(lib)}종")

    # ── 4. op 점수 재계산 (라이브러리 내 범위 기준) ─────────────
    sla_vals  = pd.to_numeric(merged["_SLA"],  errors="coerce")
    ldmc_vals = pd.to_numeric(merged["_LDMC"], errors="coerce")

    lib["op_establishment"] = _minmax(sla_vals).fillna(0.5).values
    lib["op_safe_growth"]   = _minmax(ldmc_vals).fillna(0.5).values

    # ── 5. CSR 값 갱신 ────────────────────────────────────────
    for col_new, col_lib in [("_c","c_percent"),("_s","s_percent"),
                              ("_r","r_percent"),("_strat","strategy_class")]:
        # 컬럼 타입을 float/object로 먼저 변환 (기존 int64 → float 호환)
        if col_lib in ["c_percent","s_percent","r_percent"]:
            lib[col_lib] = lib[col_lib].astype(float)
        mask = merged[col_new].notna()
        lib.loc[mask, col_lib] = merged.loc[mask, col_new].values

    # ── 6. op_establishment_src 갱신 ──────────────────────────
    lib["op_db_source"] = traitx_path.name

    # ── 7. 저장 ───────────────────────────────────────────────
    lib.to_csv(lib_path, index=False, encoding="utf-8-sig")
    print(f"[build_op_scores] 저장 완료 → {lib_path}")

    return {
        "n_lib": len(lib),
        "n_sla_matched": int(n_sla),
        "n_csr_matched": int(n_csr),
        "traitx_file": traitx_path.name,
    }


def needs_rebuild(traitx_path: Path = TRAITX_DB, lib_path: Path = LIB_CSV) -> bool:
    """TraitX가 라이브러리보다 새로우면 True."""
    if not traitx_path.exists() or not lib_path.exists():
        return False
    return traitx_path.stat().st_mtime > lib_path.stat().st_mtime


if __name__ == "__main__":
    info = rebuild()
    print(f"완료: {info}")
