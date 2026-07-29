# -*- coding: utf-8 -*-
"""
compare_floor_off.py — floor_drift_off の On/Off を simulate_capped_triage で比較する

確認する3点:
  1. キャップ起因の追加遅れ … 上位N枠が空いて縮んだか
  2. 検知＝なし の件数      … floor 単位のドリフトを止めて取りこぼしが出ていないか（最重要）
  3. 月間発火数・積み残し    … 負荷が下がったか、積み残しが単調増加していないか

効率について
------------
prep_static（先行機種カーブ・ベースライン推定）は floor_drift_off に依存しないので
両条件で共有できる。依存するのは replay_unit を通る build_initial_caches の方だけ。
よって static は1回、init_caches は条件ごとに作り直す。

使い方
------
    python compare_floor_off.py
  もしくは
    import compare_floor_off as cf
    cf.compare(units, cfg, labels, top_n=20)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import state_logic_cusum as s
from simulate_capped_triage import (prep_static, build_initial_caches,
                                    simulate_capped_triage)

PANEL_PATH = "panel.csv"
LABELS_PATH = "labels.csv"


def _one(units, cfg, labels, static, top_n, assumed, verbose=False):
    init = build_initial_caches(*static, cfg, verbose=verbose)
    monthly, incidents, details, _ = simulate_capped_triage(
        units, cfg, top_n=top_n, labels=labels, assumed_disposition=assumed,
        collect_details=True, verbose=verbose, _static=static, _init_caches=init)
    return monthly, incidents, details


def compare(units: pd.DataFrame, cfg: dict, labels: pd.DataFrame,
            top_n: int = 20, assumed: str = "対策中", verbose: bool = False):
    """floor_drift_off False/True を並べて3指標を出す。"""
    static = prep_static(units, cfg, verbose=verbose)

    out = {}
    for flag in (False, True):
        c = dict(cfg); c["floor_drift_off"] = flag
        out[flag] = _one(units, c, labels, static, top_n, assumed, verbose)

    (m0, i0, d0), (m1, i1, d1) = out[False], out[True]

    def _med(x):
        x = pd.to_numeric(x, errors="coerce").dropna()
        return float(x.median()) if len(x) else float("nan")

    def _trend(m):
        """発火候補の月次推移の傾き（正なら増加傾向＝誤報が沈んでいない）。"""
        y = m["発火候補"].to_numpy(dtype=float)
        if len(y) < 3:
            return float("nan")
        return float(np.polyfit(np.arange(len(y)), y, 1)[0])

    rows = [
        ("① キャップ起因の追加遅れ 中央", _med(i0["キャップ起因の追加遅れ"]),
                                          _med(i1["キャップ起因の追加遅れ"])),
        ("   処理遅れ月 中央（負=販社より早い）", _med(i0["処理遅れ月"]), _med(i1["処理遅れ月"])),
        ("   インボックス遅れ月 中央", _med(i0["インボックス遅れ月"]), _med(i1["インボックス遅れ月"])),
        ("   販社より早い 件数", int((i0["販社より早い"] == "○").sum()),
                                 int((i1["販社より早い"] == "○").sum())),
        ("② 検知＝なし 件数  ★要注視", int((i0["検知"] == "なし").sum()),
                                        int((i1["検知"] == "なし").sum())),
        ("   処理＝なし 件数", int((i0["処理"] == "なし").sum()),
                               int((i1["処理"] == "なし").sum())),
        ("   ラベル総数", len(i0), len(i1)),
        ("③ 月間 発火候補 中央", _med(m0["発火候補"]), _med(m1["発火候補"])),
        ("   月間 積み残し 中央", _med(m0["積み残し"]), _med(m1["積み残し"])),
        ("   月間 要対応計 中央", _med(m0["要対応計"]), _med(m1["要対応計"])),
        ("   発火候補の傾き(件/月)", round(_trend(m0), 3), round(_trend(m1), 3)),
        ("   二軍水準 中央", _med(m0["二軍水準"]), _med(m1["二軍水準"])),
    ]

    print("=" * 84)
    print(f"floor_drift_off 比較   top_n={top_n}  assumed_disposition={assumed}")
    print("=" * 84)
    print(f"{'指標':<38}{'False(従来)':>16}{'True(対処A)':>16}")
    print("-" * 84)
    for name, a, b in rows:
        print(f"{name:<38}{str(a):>16}{str(b):>16}")

    print("\n【③ 月次推移（発火候補 / 積み残し）】")
    mm = m0[["年月", "発火候補", "積み残し"]].merge(
        m1[["年月", "発火候補", "積み残し"]], on="年月", suffixes=("_False", "_True"))
    print(mm.tail(18).to_string(index=False))

    print("\n【判定の目安】")
    print("  ② が増えていない → 対処Aは無害。①③の改善をそのまま享受してよい")
    print("  ② が増えている   → floor 単位のドリフトで拾えていた本物がある。")
    print("                     対処C（Gamma-Poissonで λ0 を埋める）の検討へ")
    print("  発火候補の傾きが正 → 誤報が沈んでいない。R/h の見直しが必要")
    print("\n  ※ 発火が減った分だけ誤報予算に余裕ができている。")
    print("     この後 backtest_cusum で h を下げる方向に決め直すこと。")
    return dict(monthly_off=m0, monthly_on=m1, inc_off=i0, inc_on=i1,
                details_off=d0, details_on=d1)


def label_breakdown(i0: pd.DataFrame, i1: pd.DataFrame):
    """②が増えたとき、どのラベルが消えたかを特定する。"""
    k = ["事業コード", "開発コード", "部番", "販社報告月"]
    j = i0[k + ["検知", "処理遅れ月"]].merge(
        i1[k + ["検知", "処理遅れ月"]], on=k, suffixes=("_False", "_True"))
    lost = j[(j["検知_False"] == "あり") & (j["検知_True"] == "なし")]
    print(f"\n対処Aで検知できなくなったラベル: {len(lost)} 件")
    if len(lost):
        print(lost.to_string(index=False))
    return lost


if __name__ == "__main__":
    cfg = dict(s.CONFIG)
    # ↓ 実運用の確定値に置き換えて使う
    cfg.update(R=1.5, h=2.5, alpha_spike=0.005, min_count=3, burst_window=0)

    panel = s._prepare_panel(pd.read_csv(PANEL_PATH), cfg)
    units = s.aggregate_units(panel)
    labels = pd.read_csv(LABELS_PATH, encoding="utf-8-sig")

    res = compare(units, cfg, labels, top_n=20, assumed="対策中")
    label_breakdown(res["inc_off"], res["inc_on"])
