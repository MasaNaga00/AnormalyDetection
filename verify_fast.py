# -*- coding: utf-8 -*-
"""
verify_fast.py
==============
simulate_historical_load_fast が元の simulate_historical_load と
「毎月の件数内訳」で一致する（差0）ことを確認し、処理時間も比較する。

使い方:
    python verify_fast.py                       # 既定 test_panel.csv
    python verify_fast.py 実データpanel.csv     # 実データで確認
"""
from __future__ import annotations
import sys
import time
import pandas as pd
import state_logic_cusum as s
import simulate_historical_load as slow
import simulate_historical_load_fast as fast


def run(panel_path: str):
    raw = pd.read_csv(panel_path)
    panel = s._prepare_panel(raw, s.CONFIG)
    units = s.aggregate_units(panel)
    cfg = dict(s.CONFIG)
    # 操作点は本番と同じにそろえる（必要なら書き換え）
    cfg.update(R=1.5, h=5.0, alpha_spike=0.005, min_count=3, burst_window=0)

    n_units = units[["biz", "dev", "part"]].drop_duplicates().shape[0]
    n_months = units["ym"].nunique()
    print(f"機種×部番={n_units} / 月数={n_months}")

    ok = True
    for disp in ("ノイズ", "対策中"):
        t0 = time.time()
        m_slow, _, _ = slow.simulate_monthly_load(units, cfg, assumed_disposition=disp,
                                                  verbose=False)
        t_slow = time.time() - t0

        t0 = time.time()
        m_fast, _, _ = fast.simulate_monthly_load(units, cfg, assumed_disposition=disp,
                                                  verbose=False)
        t_fast = time.time() - t0

        merged = m_slow.merge(m_fast, on="年月", suffixes=("_slow", "_fast"))
        cols = ["件数", "発火", "再評価", "保留継続", "対象単位数"]
        maxdiff = 0
        for c in cols:
            d = (merged[f"{c}_slow"] - merged[f"{c}_fast"]).abs().max()
            maxdiff = max(maxdiff, int(d) if pd.notna(d) else 0)
        status = "一致(差0)" if maxdiff == 0 else f"不一致 最大差={maxdiff}"
        ok = ok and (maxdiff == 0)
        print(f"[{disp}] {status} / 元={t_slow:.1f}秒 高速={t_fast:.1f}秒 "
              f"({t_slow / max(t_fast, 1e-9):.1f}倍)")

    print("結果:", "OK 差0で高速化成功" if ok else "NG 差分あり（要調査）")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "test_panel.csv")
