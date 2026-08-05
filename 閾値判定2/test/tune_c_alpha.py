# -*- coding: utf-8 -*-
"""
tune_c_alpha.py — 信号C の alpha / min_count を実データで決める

考え方
------
`run_signal_c` は「p <= alpha かつ use >= min_count」で発火する。
p の計算（条件付き二項）が重いので、**閾値を一番緩くして1回だけ計算し**、
その結果を保存してから alpha / min_count のスイープはフィルタで行う。

3ステップ
---------
  step1_scan()   … p値を1回だけ計算して scan_c.csv に保存（重い。数分かかる）
  step2_sweep()  … alpha × min_count の格子で月次件数を出す（軽い。何度でも）
  step3_labels() … 過去の既知例が拾えているかを確認（ラベルCSVがあれば）

使い方
------
    python tune_c_alpha.py panel.csv                    # step1 + step2
    python tune_c_alpha.py panel.csv labels.csv         # step3 まで

ラベルCSVの形式（列名は LABEL_COLS で変更可）
    事業コード, 開発コード, 部番, 発生年月   ← 販社列があれば使う（無くてもよい）
    発生年月は **年月（YYYYMM）**。年月日で書くと月がズレて評価を誤るので注意。
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

import signal_c_dist as sd

COLS = dict(biz="事業コード", dev="開発コード", part="部番", dist="販社",
            ym="年月", monthly_use="月次使用数", cum_sales="累積販売台数")

LABEL_COLS = dict(biz="事業コード", dev="開発コード", part="部番",
                  ym="発生年月", dist="販社")

MONTHS_BACK = 24        # 何ヶ月さかのぼって実測するか
SCAN_MIN_COUNT = 2      # p値を計算する最低使用数。1にすると重い代わりに全域を見られる
ALPHAS = [0.005, 0.002, 0.001, 5e-4, 2e-4, 1e-4, 5e-5]
MIN_COUNTS = [3, 4, 5, 7]
LOOKBACK_M = 6          # ラベル評価: 報告月の何ヶ月前まで遡って先行検知を認めるか


# ============================================================================
def step1_scan(panel_path: str, out: str = "scan_c.csv") -> pd.DataFrame:
    """p値を1回だけ計算して保存する。ここが唯一の重い処理。"""
    raw = pd.read_csv(panel_path, encoding="utf-8-sig")
    raw["年月"] = raw["年月"].astype(str).str.replace(r"\D", "", regex=True).astype(int)
    dist = raw[raw[COLS["dist"]].astype(str) != "ALL"].copy()
    print(f"販社別パネル: {len(dist)}行  "
          f"系列数={dist.groupby(['事業コード','開発コード','部番','販社']).ngroups}")

    # alpha=1.0 で「全部発火扱い」にして p を残す。min_count は最小に。
    res = sd.run_signal_c(dist, COLS, alpha=1.0, min_count=SCAN_MIN_COUNT,
                          months_back=MONTHS_BACK)
    res.to_csv(out, index=False, encoding="utf-8-sig")
    n_p = int(res["p"].notna().sum())
    print(f"判定行={len(res)}  p値を計算した行={n_p}  → {out} に保存")
    return res


# ============================================================================
def step2_sweep(scan: pd.DataFrame, alphas=ALPHAS, min_counts=MIN_COUNTS,
                budget: float | None = None) -> pd.DataFrame:
    """alpha × min_count で月あたり件数を出す。"""
    d = scan[scan["p"].notna()].copy()
    n_months = d["ym"].nunique()
    all_ym = sorted(d["ym"].unique())

    rows = []
    for a in alphas:
        for mc in min_counts:
            hit = d[(d["p"] <= a) & (d["use"] >= mc)]
            per = hit.groupby("ym").size().reindex(all_ym, fill_value=0)
            # 機種×部番に畳んだ件数（レビュー単位はこちら）
            unit = (hit.groupby(["ym", "biz", "dev", "part"]).size()
                       .groupby("ym").size().reindex(all_ym, fill_value=0))
            rows.append(dict(
                alpha=a, min_count=mc,
                総件数=int(len(hit)),
                月平均=round(per.mean(), 1), 月中央=int(per.median()),
                月P90=int(per.quantile(0.9)), 月最大=int(per.max()),
                単位月平均=round(unit.mean(), 1), 単位月最大=int(unit.max()),
                発火0の月=int((per == 0).sum()),
            ))
    tbl = pd.DataFrame(rows)
    if budget is not None:
        tbl["予算内"] = tbl["単位月平均"] <= budget
    print(f"\n=== alpha × min_count スイープ（直近{n_months}ヶ月）===")
    print("  月平均/月最大 = 販社別の発火行数、単位月平均 = 機種×部番に畳んだ件数")
    print(tbl.to_string(index=False))
    return tbl


def step2b_monthly(scan: pd.DataFrame, alpha: float, min_count: int) -> pd.Series:
    """選んだ操作点での月別件数の推移。立ち上がり月の塊を確認する。"""
    d = scan[(scan["p"].notna()) & (scan["p"] <= alpha) & (scan["use"] >= min_count)]
    per = (d.groupby(["ym", "biz", "dev", "part"]).size().groupby("ym").size())
    per = per.reindex(sorted(scan["ym"].unique()), fill_value=0)
    print(f"\n=== 月別件数（alpha={alpha}, min_count={min_count}、機種×部番単位）===")
    print(per.to_string())
    return per


# ============================================================================
def step3_labels(scan: pd.DataFrame, labels_path: str,
                 alphas=ALPHAS, min_counts=MIN_COUNTS,
                 lookback_m: int = LOOKBACK_M) -> pd.DataFrame:
    """既知の過去例が拾えるかを操作点ごとに評価する。

    報告月の lookback_m ヶ月前〜報告月 の範囲に発火があれば検知とみなす。
    遅れ月がマイナス = 販社報告より先に検知できた（成功）。
    """
    lab = pd.read_csv(labels_path, encoding="utf-8-sig")
    L = LABEL_COLS
    lab[L["ym"]] = lab[L["ym"]].astype(str).str.replace(r"\D", "", regex=True).str[:6].astype(int)
    d = scan[scan["p"].notna()].copy()
    all_ym = sorted(scan["ym"].unique())

    rows = []
    for a in alphas:
        for mc in min_counts:
            hit = d[(d["p"] <= a) & (d["use"] >= mc)]
            n_ok, delays = 0, []
            for r in lab.itertuples():
                b = getattr(r, L["biz"]); dev = getattr(r, L["dev"])
                part = getattr(r, L["part"]); ym = getattr(r, L["ym"])
                lo = sd._shift_ym(ym, -lookback_m)
                m = hit[(hit["biz"] == b) & (hit["dev"] == dev) &
                        (hit["part"] == part) & (hit["ym"] >= lo) & (hit["ym"] <= ym)]
                if L["dist"] in lab.columns:
                    dv = getattr(r, L["dist"], None)
                    if isinstance(dv, str) and dv.strip():
                        m = m[m["dist"] == dv.strip()]
                if len(m):
                    n_ok += 1
                    first = int(m["ym"].min())
                    delays.append(_months_between(ym, first))
            unit = (hit.groupby(["ym", "biz", "dev", "part"]).size().groupby("ym").size()
                       .reindex(all_ym, fill_value=0))
            rows.append(dict(alpha=a, min_count=mc,
                             検知=f"{n_ok}/{len(lab)}",
                             検知率=round(n_ok / max(len(lab), 1), 2),
                             遅れ月中央=int(np.median(delays)) if delays else None,
                             単位月平均=round(unit.mean(), 1)))
    tbl = pd.DataFrame(rows)
    print(f"\n=== 既知例の検知（ラベル{len(lab)}件、報告{lookback_m}ヶ月前まで遡って評価）===")
    print("  遅れ月がマイナス = 販社報告より先に検知できた")
    print(tbl.to_string(index=False))
    return tbl


def _months_between(a: int, b: int) -> int:
    ya, ma = divmod(int(a), 100)
    yb, mb = divmod(int(b), 100)
    return (yb * 12 + mb) - (ya * 12 + ma)


def missed_labels(scan: pd.DataFrame, labels_path: str, alpha: float,
                  min_count: int, lookback_m: int = LOOKBACK_M) -> pd.DataFrame:
    """選んだ操作点で取りこぼしたラベルを、その理由つきで出す。"""
    lab = pd.read_csv(labels_path, encoding="utf-8-sig")
    L = LABEL_COLS
    lab[L["ym"]] = lab[L["ym"]].astype(str).str.replace(r"\D", "", regex=True).str[:6].astype(int)
    d = scan.copy()
    rows = []
    for r in lab.itertuples():
        b = getattr(r, L["biz"]); dev = getattr(r, L["dev"])
        part = getattr(r, L["part"]); ym = getattr(r, L["ym"])
        lo = sd._shift_ym(ym, -lookback_m)
        w = d[(d["biz"] == b) & (d["dev"] == dev) & (d["part"] == part) &
              (d["ym"] >= lo) & (d["ym"] <= ym)]
        if w.empty:
            why = "監視レンジ外またはキー不一致（該当行なし）"
        elif w["p"].notna().sum() == 0:
            why = f"p値未計算（使用数が{SCAN_MIN_COUNT}未満、または期待値以下）"
        else:
            best = w.loc[w["p"].idxmin()]
            if best["p"] > alpha:
                why = f"p={best['p']:.2e} が alpha={alpha} を超える"
            elif best["use"] < min_count:
                why = f"使用数{int(best['use'])}件が min_count={min_count} 未満"
            else:
                continue        # 検知できている
        rows.append(dict(機種=dev, 部番=part, 発生年月=ym, 理由=why))
    out = pd.DataFrame(rows)
    print(f"\n=== 取りこぼし（alpha={alpha}, min_count={min_count}）: {len(out)}件 ===")
    if len(out):
        print(out.to_string(index=False))
    return out


# ============================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    scan = step1_scan(sys.argv[1])
    tbl = step2_sweep(scan)
    if len(sys.argv) > 2:
        step3_labels(scan, sys.argv[2])
