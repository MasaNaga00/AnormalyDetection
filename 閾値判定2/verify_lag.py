# -*- coding: utf-8 -*-
"""
verify_lag.py — 販社の報告遅れ対応の検証

確認すること
------------
1) 後方互換: horizon=None, revisit_months=0 なら従来と**差0**
2) 遅れている販社の異常月が、従来設定では**永久に検知されない**ことの再現
3) horizon + revisit_months を入れると、遅れて到着した月が正しく検知され、
   遅延月がついて出ること
4) 未確定月を切り落とすことでベースラインが汚れないこと
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import reporting_horizon as rh
import signal_c_dist as sc

COLS = dict(biz="事業コード", dev="開発コード", part="部番", dist="販社",
            ym="年月", monthly_use="月次使用数", cum_sales="累積販売台数")


def build_panel(last_by_dist: dict, spikes: dict | None = None,
                partial: dict | None = None, seed: int = 7) -> pd.DataFrame:
    """販社ごとに最終月が違うパネルを作る。

    last_by_dist : {販社: その販社の最終年月}
    spikes       : {(dev, part, dist, ym): 使用数}  異常を仕込む
    partial      : {(dist, ym): 倍率}  その月を部分受領（薄く）にする
    """
    rng = np.random.default_rng(seed)
    spikes = spikes or {}
    partial = partial or {}
    rows = []
    for dev in ("M01", "M02"):
        for part in ("P1", "P2"):
            for dist, last in last_by_dist.items():
                m = 202301
                while m <= last:
                    use = float(rng.poisson(6))
                    key = (dev, part, dist, m)
                    if key in spikes:
                        use = float(spikes[key])
                    if (dist, m) in partial:
                        use = round(use * partial[(dist, m)])
                    rows.append({COLS["biz"]: "E1", COLS["dev"]: dev,
                                 COLS["part"]: part, COLS["dist"]: dist,
                                 COLS["ym"]: m, COLS["monthly_use"]: use,
                                 COLS["cum_sales"]: 2000})
                    m = rh.shift_ym(m, 1)
    return pd.DataFrame(rows)


def fired(res, dev, part, dist, ym) -> bool:
    if res is None or res.empty:
        return False
    q = res[(res.dev == dev) & (res.part == part) &
            (res.dist == dist) & (res.ym == ym)]
    return bool(len(q) and q["alert_dist"].any())


def main():
    T = 202503

    # ---------------------------------------------------------------- 1
    print("=" * 66)
    print("1) 後方互換: 遅れの無いパネルで 旧設定 vs 新引数の既定 が差0か")
    flat = build_panel({"A": T, "B": T, "C": T},
                       spikes={("M01", "P1", "B", 202502): 40})
    old = sc.run_signal_c(flat, COLS, asof_ym=T, months_back=6)
    new = sc.run_signal_c(flat, COLS, asof_ym=T, months_back=6,
                          horizon=None, revisit_months=0)
    common = [c for c in old.columns if c in new.columns]
    same = old[common].reset_index(drop=True).equals(new[common].reset_index(drop=True))
    print(f"   判定行 {len(old)} / 発火 {int(old.alert_dist.sum())}  → 差0: {same}")
    assert same, "後方互換が壊れている"

    # ---------------------------------------------------------------- 2
    print("=" * 66)
    print("2) 販社Bが3ヶ月遅れ（202412まで）。202412にBで異常。")
    lag = build_panel({"A": 202503, "B": 202412, "C": 202502},
                      spikes={("M01", "P1", "B", 202412): 40},
                      partial={("A", 202503): 0.2, ("C", 202502): 0.3})

    print("\n   -- 受領状況 --")
    rep = rh.report(lag, COLS, revisit_months=3)
    print(rep.to_string(index=False))
    hz = rh.estimate_horizon(lag, COLS)
    print(f"\n   horizon = {hz}   global = {rh.global_horizon(hz)}")
    print(f"   再評価窓3ヶ月で足りない販社 = {rh.check_revisit(hz, T, 3)}")

    print("\n   -- 現行設定（T=202503 の1ヶ月だけ判定）--")
    cur = sc.run_signal_c(lag, COLS, asof_ym=T, months_back=0)
    n_b = 0 if cur.empty else int((cur.dist == "B").sum())
    print(f"   判定行 {len(cur)} / うち販社Bの行 {n_b}")
    print(f"   B/202412 の異常を検知したか: {fired(cur, 'M01', 'P1', 'B', 202412)}")
    print("   → Bは判定対象にすら入らない。翌月以降もTしか見ないので永久に未検定。")

    # ---------------------------------------------------------------- 3
    print("=" * 66)
    print("3) horizon + revisit_months=4 を入れる")
    fix = sc.run_signal_c(lag, COLS, asof_ym=T, months_back=0,
                          horizon=hz, revisit_months=4)
    print(f"   判定行 {len(fix)}")
    ok = fired(fix, "M01", "P1", "B", 202412)
    print(f"   B/202412 の異常を検知したか: {ok}")
    a = fix[fix.alert_dist]
    if len(a):
        print("\n   -- 発火一覧 --")
        print(a[["dev", "part", "dist", "ym", "run_ym", "遅延月",
                 "use", "expected", "O_E", "p"]].to_string(index=False))
    assert ok, "遅れて到着した月の異常が検知できていない"

    # ---------------------------------------------------------------- 4
    print("=" * 66)
    print("4) 未確定月の切り落とし: horizon より後ろが判定・ベースラインに入らないか")
    over = fix[fix.apply(lambda r: r["ym"] > hz[r["dist"]], axis=1)]
    print(f"   horizon超過の判定行: {len(over)} 件（0であるべき）")
    assert len(over) == 0

    # 部分受領を残した場合、A/202503(薄い月)が判定対象に入ってしまう
    nohz = sc.run_signal_c(lag, COLS, asof_ym=T, months_back=0,
                           horizon=None, revisit_months=4)
    bad = nohz[(nohz.dist == "A") & (nohz.ym == 202503)]
    print(f"   horizon無しだと A/202503（部分受領）が判定対象に: {len(bad)} 件")

    print("=" * 66)
    print("すべて期待どおり。")


if __name__ == "__main__":
    main()
