# -*- coding: utf-8 -*-
"""
inject_power.py — ラベルが無い品目で検出力を測る（合成注入）

実データの系列に既知の異常を人工的に注入し、現在の設定で拾えるかを測る。
「過去の実例が無い」品目でも、検出力のカーブが描ける。

使い方
------
    python inject_power.py panel.csv          # 信号C
    python inject_power.py panel.csv b        # 信号B

注入の考え方
------------
- **信号C**: ランダムに選んだ系列の最終月を k 倍する。倍率ごとの検知率を出す。
- **信号B**: ランダムに選んだ機種×SF の件数を全期間 k 倍する（恒常的な高水準）。

実データの分布・露出・ノイズをそのまま使うので、合成パネルより実態に近い。
「倍率 k 以上なら拾える」という保証水準が分かる。
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

import signal_c_dist as sd
import signal_b_peer as sb
import state_logic_cusum as sc
import settings as st

MULTS = [1.5, 2, 3, 5, 8, 12]
N_TRIAL = 60


# ============================================================================
def power_c(panel_path: str, mults=MULTS, n_trial: int = N_TRIAL, seed: int = 0):
    """信号C: 最終月に単月スパイクを注入して検知率を測る。"""
    rng = np.random.default_rng(seed)
    raw = pd.read_csv(panel_path, encoding="utf-8-sig")
    raw["年月"] = raw["年月"].astype(str).str.replace(r"\D", "", regex=True).astype(int)
    d = sd.prepare_dist_panel(raw[raw[st.COLS["dist"]] != st.ALL_TOKEN], st.COLS)

    # 判定に足る長さのある系列だけを母集団にする
    g = d.groupby(["biz", "dev", "part", "dist"])
    keys = [k for k, v in g.size().items() if v >= st.C_BASE_LEN + 1]
    if not keys:
        print("判定に足る長さの系列がありません"); return
    print(f"母集団 {len(keys)} 系列から {n_trial} 本を抽出して注入\n")

    rows = []
    for m in mults:
        hit = base_mean = 0
        for idx in rng.choice(len(keys), min(n_trial, len(keys)), replace=False):
            k = keys[idx]
            s = d[(d.biz == k[0]) & (d.dev == k[1]) &
                  (d.part == k[2]) & (d.dist == k[3])].sort_values("ym")
            u = s["use"].to_numpy(float).copy()
            f = s["fleet"].to_numpy(float)
            base = u[-1 - st.C_BASE_LEN:-1].mean()
            base_mean += base
            # 注入: 最終月をベースライン平均の m 倍に置き換える
            u[-1] = max(np.round(base * m), 1)
            a = sd._rolling_test(u, f, st.C_BASE_LEN, st.C_ALPHA, st.C_MIN_COUNT,
                                 st.C_MIN_BASE_MONTHS, st.C_MIN_BASE_COUNT,
                                 st.C_MIN_OE, st.C_MIN_EXCESS, st.C_EXCEED_HIST)[1]
            hit += int(a[-1])
        n = min(n_trial, len(keys))
        rows.append(dict(倍率=m, 検知率=round(hit / n, 3),
                         平常月平均=round(base_mean / n, 2)))
    t = pd.DataFrame(rows)
    print(f"=== 信号C の検出力（min_oe={st.C_MIN_OE}, alpha={st.C_ALPHA}, "
          f"min_count={st.C_MIN_COUNT}）===")
    print(t.to_string(index=False))
    print("\n読み方: 検知率が1.0に近づく倍率が『確実に拾える水準』。")
    print("        その倍率が実務的に許容できるかで設定の可否を判断する。")
    return t


# ============================================================================
def power_b(panel_path: str, mults=MULTS, n_trial: int = N_TRIAL, seed: int = 0):
    """信号B: 機種×SF の件数を全期間 k 倍して検知率を測る。"""
    rng = np.random.default_rng(seed)
    raw = pd.read_csv(panel_path, encoding="utf-8-sig")
    raw["年月"] = raw["年月"].astype(str).str.replace(r"\D", "", regex=True).astype(int)
    p_all = raw[raw[st.COLS["dist"]].astype(str) == st.ALL_TOKEN]

    cfg = dict(sc.CONFIG); cfg["cols"] = {**sc.CONFIG["cols"], **st.COLS}
    pb = sc._prepare_panel(p_all.copy(), cfg)

    u0 = sb.build_peer_units(pb, elapsed_cap=st.B_ELAPSED_CAP, require_full=True)
    key = ["biz", "sf", "nb"] if st.B_NB_STRAT else ["biz", "sf"]
    cand = u0.groupby(key).filter(lambda x: len(x) >= st.B_MIN_PEERS + 1)
    if cand.empty:
        print("ピアが足りる群がありません"); return
    print(f"判定対象 {len(cand)} 単位から {n_trial} 本を抽出して注入\n")

    rows = []
    for m in mults:
        hit = 0
        pick = rng.choice(len(cand), min(n_trial, len(cand)), replace=False)
        for i in pick:
            uu = u0.copy()
            uu["C"] = uu["C"].astype(float)
            tgt = cand.index[i]
            uu.loc[tgt, "C"] = uu.loc[tgt, "C"] * m
            r = sb._peer_compare(uu, key, st.B_MIN_PEERS)
            if r.empty:
                continue
            row = r[(r.biz == u0.loc[tgt, "biz"]) & (r.sf == u0.loc[tgt, "sf"])
                    & (r.dev == u0.loc[tgt, "dev"])]
            if len(row) and (row["O_E"].iloc[0] >= st.B_MIN_OE
                             and row["p"].iloc[0] <= st.B_ALPHA
                             and row["C"].iloc[0] >= st.B_MIN_COUNT
                             and row["C_peer"].iloc[0] >= st.B_MIN_PEER_COUNT):
                hit += 1
        rows.append(dict(倍率=m, 検知率=round(hit / len(pick), 3)))
    t = pd.DataFrame(rows)
    print(f"=== 信号B の検出力（min_oe={st.B_MIN_OE}, min_count={st.B_MIN_COUNT}, "
          f"min_peer_count={st.B_MIN_PEER_COUNT}）===")
    print(t.to_string(index=False))
    print("\n読み方: 2パスを通していないので、実運用ではこれより少し拾いやすい。")
    return t


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    if len(sys.argv) > 2 and sys.argv[2].lower() == "b":
        power_b(sys.argv[1])
    else:
        power_c(sys.argv[1])
