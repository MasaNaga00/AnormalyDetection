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
def power_c(panel_path: str, mults=MULTS, n_trial: int = N_TRIAL, seed: int = 0,
            min_oe=None, alpha=None, min_count=None, min_base_count=None,
            exceed_hist=None, min_excess=None, base_len=None, quiet=False):
    """信号C: 最終月に単月スパイクを注入して検知率を測る。

    パラメータを指定すると settings.py の値を上書きする（settings.py は変更しない）。
    None のものは settings.py の現在値を使う。"""
    P = dict(min_oe=st.C_MIN_OE if min_oe is None else min_oe,
             alpha=st.C_ALPHA if alpha is None else alpha,
             min_count=st.C_MIN_COUNT if min_count is None else min_count,
             min_base_count=st.C_MIN_BASE_COUNT if min_base_count is None else min_base_count,
             exceed_hist=st.C_EXCEED_HIST if exceed_hist is None else exceed_hist,
             min_excess=st.C_MIN_EXCESS if min_excess is None else min_excess,
             base_len=st.C_BASE_LEN if base_len is None else base_len)
    rng = np.random.default_rng(seed)
    raw = pd.read_csv(panel_path, encoding="utf-8-sig")
    raw["年月"] = raw["年月"].astype(str).str.replace(r"\D", "", regex=True).astype(int)
    d = sd.prepare_dist_panel(raw[raw[st.COLS["dist"]] != st.ALL_TOKEN], st.COLS)

    # 判定に足る長さのある系列だけを母集団にする
    g = d.groupby(["biz", "dev", "part", "dist"])
    keys = [k for k, v in g.size().items() if v >= P["base_len"] + 1]
    if not keys:
        print("判定に足る長さの系列がありません"); return
    if not quiet:
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
            base = u[-1 - P["base_len"]:-1].mean()
            base_mean += base
            # 注入: 最終月をベースライン平均の m 倍に置き換える
            u[-1] = max(np.round(base * m), 1)
            a = sd._rolling_test(u, f, P["base_len"], P["alpha"], P["min_count"],
                                 st.C_MIN_BASE_MONTHS, P["min_base_count"],
                                 P["min_oe"], P["min_excess"], P["exceed_hist"])[1]
            hit += int(a[-1])
        n = min(n_trial, len(keys))
        rows.append(dict(倍率=m, 検知率=round(hit / n, 3),
                         平常月平均=round(base_mean / n, 2)))
    t = pd.DataFrame(rows)
    if not quiet:
        print(f"=== 信号C の検出力（min_oe={P['min_oe']}, alpha={P['alpha']}, "
              f"min_count={P['min_count']}, exceed_hist={P['exceed_hist']}）===")
        print(t.to_string(index=False))
        print("\n読み方: 検知率が1.0に近づく倍率が『確実に拾える水準』。")
    return t


def alerts_c(panel_path: str, months_back: int = 24, **over) -> float:
    """同じパラメータで、注入なしの実データが毎月何件鳴るか（機種×部番単位）。"""
    P = dict(min_oe=st.C_MIN_OE, alpha=st.C_ALPHA, min_count=st.C_MIN_COUNT,
             min_base_count=st.C_MIN_BASE_COUNT, exceed_hist=st.C_EXCEED_HIST,
             min_excess=st.C_MIN_EXCESS, base_len=st.C_BASE_LEN)
    P.update({k: v for k, v in over.items() if v is not None})
    raw = pd.read_csv(panel_path, encoding="utf-8-sig")
    raw["年月"] = raw["年月"].astype(str).str.replace(r"\D", "", regex=True).astype(int)
    r = sd.run_signal_c(raw[raw[st.COLS["dist"]] != st.ALL_TOKEN], st.COLS,
                        base_len=P["base_len"], alpha=P["alpha"],
                        min_count=P["min_count"],
                        min_base_months=st.C_MIN_BASE_MONTHS,
                        min_base_count=P["min_base_count"],
                        min_oe=P["min_oe"], min_excess=P["min_excess"],
                        exceed_hist=P["exceed_hist"], months_back=months_back)
    if r.empty or not r["alert_dist"].any():
        return 0.0, 0
    hit = r[r["alert_dist"]]
    per = (hit.groupby(["ym", "biz", "dev", "part"]).size().groupby("ym").size()
              .reindex(sorted(r["ym"].unique()), fill_value=0))
    return round(per.mean(), 2), int(per.max())


def sweep_c(panel_path: str, param: str, values, mults=(2, 3, 5, 8),
            n_trial: int = N_TRIAL, months_back: int = 24, **fixed):
    """**これが操作点を決める主表。** 1つのパラメータを振り、
    各倍率の検知率と、実データでの月次発火件数を並べる。

        sweep_c("panel.csv", "min_oe", [1.5, 2.0, 3.0, 4.0])

    見方: 拾いたい倍率の検知率が十分で、単位月最大が予算に収まる行を選ぶ。
    """
    rows = []
    for v in values:
        kw = dict(fixed); kw[param] = v
        pw = power_c(panel_path, mults=list(mults), n_trial=n_trial, quiet=True, **kw)
        mean, mx = alerts_c(panel_path, months_back=months_back, **kw)
        d = {param: v}
        for m in mults:
            d[f"{m}倍"] = pw.loc[pw["倍率"] == m, "検知率"].iloc[0]
        d["単位月平均"] = mean; d["単位月最大"] = mx
        rows.append(d)
    t = pd.DataFrame(rows)
    print(f"=== 信号C: {param} スイープ（検知率 × 月次件数）===")
    print(t.to_string(index=False))
    print("\n  左側=拾えるか / 右側=見きれるか。両立する行を選ぶ。")
    return t


# ============================================================================
def power_b(panel_path: str, mults=MULTS, n_trial: int = N_TRIAL, seed: int = 0,
            min_oe=None, alpha=None, min_count=None, min_peer_count=None,
            min_peers=None, elapsed_cap=None, quiet=False):
    """信号B: 機種×SF の件数を全期間 k 倍して検知率を測る。

    パラメータを指定すると settings.py の値を上書きする（settings.py は変更しない）。"""
    P = dict(min_oe=st.B_MIN_OE if min_oe is None else min_oe,
             alpha=st.B_ALPHA if alpha is None else alpha,
             min_count=st.B_MIN_COUNT if min_count is None else min_count,
             min_peer_count=st.B_MIN_PEER_COUNT if min_peer_count is None else min_peer_count,
             min_peers=st.B_MIN_PEERS if min_peers is None else min_peers,
             elapsed_cap=st.B_ELAPSED_CAP if elapsed_cap is None else elapsed_cap)
    rng = np.random.default_rng(seed)
    raw = pd.read_csv(panel_path, encoding="utf-8-sig")
    raw["年月"] = raw["年月"].astype(str).str.replace(r"\D", "", regex=True).astype(int)
    p_all = raw[raw[st.COLS["dist"]].astype(str) == st.ALL_TOKEN]

    cfg = dict(sc.CONFIG); cfg["cols"] = {**sc.CONFIG["cols"], **st.COLS}
    pb = sc._prepare_panel(p_all.copy(), cfg)

    u0 = sb.build_peer_units(pb, elapsed_cap=P["elapsed_cap"], require_full=True)
    key = ["biz", "sf", "nb"] if st.B_NB_STRAT else ["biz", "sf"]
    cand = u0.groupby(key).filter(lambda x: len(x) >= P["min_peers"] + 1)
    if cand.empty:
        if not quiet:
            print("ピアが足りる群がありません")
        return pd.DataFrame()
    if not quiet:
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
            r = sb._peer_compare(uu, key, P["min_peers"])
            if r.empty:
                continue
            row = r[(r.biz == u0.loc[tgt, "biz"]) & (r.sf == u0.loc[tgt, "sf"])
                    & (r.dev == u0.loc[tgt, "dev"])]
            if len(row) and (row["O_E"].iloc[0] >= P["min_oe"]
                             and row["p"].iloc[0] <= P["alpha"]
                             and row["C"].iloc[0] >= P["min_count"]
                             and row["C_peer"].iloc[0] >= P["min_peer_count"]):
                hit += 1
        rows.append(dict(倍率=m, 検知率=round(hit / len(pick), 3)))
    t = pd.DataFrame(rows)
    if not quiet:
        print(f"=== 信号B の検出力（min_oe={P['min_oe']}, min_count={P['min_count']}, "
              f"min_peer_count={P['min_peer_count']}, cap={P['elapsed_cap']}）===")
        print(t.to_string(index=False))
        print("\n読み方: 2パスを通していないので、実運用ではこれより少し拾いやすい。")
    return t


def alerts_b(panel_path: str, **over):
    """同じパラメータで、注入なしの実データが立ち上げ初回に何件鳴るか。"""
    P = dict(min_oe=st.B_MIN_OE, alpha=st.B_ALPHA, min_count=st.B_MIN_COUNT,
             min_peer_count=st.B_MIN_PEER_COUNT, min_peers=st.B_MIN_PEERS,
             elapsed_cap=st.B_ELAPSED_CAP)
    P.update({k: v for k, v in over.items() if v is not None})
    raw = pd.read_csv(panel_path, encoding="utf-8-sig")
    raw["年月"] = raw["年月"].astype(str).str.replace(r"\D", "", regex=True).astype(int)
    cfg = dict(sc.CONFIG); cfg["cols"] = {**sc.CONFIG["cols"], **st.COLS}
    pb = sc._prepare_panel(raw[raw[st.COLS["dist"]].astype(str) == st.ALL_TOKEN].copy(), cfg)
    r = sb.run_signal_b(pb, elapsed_cap=P["elapsed_cap"], min_peers=P["min_peers"],
                        alpha_peer=P["alpha"], min_count=P["min_count"],
                        min_peer_count=P["min_peer_count"], min_oe=P["min_oe"])
    if r.empty:
        return 0, 0
    return int(r["alert_peer"].sum()), len(r)


def sweep_b(panel_path: str, param: str, values, mults=(1.5, 2, 3, 5),
            n_trial: int = N_TRIAL, **fixed):
    """**信号Bの操作点を決める主表。**

        sweep_b("panel.csv", "min_count", [10, 20, 30, 50])

    信号Bは発火すると12ヶ月抑制されるので、件数は「立ち上げ初回の総数」で見る。
    """
    rows = []
    for v in values:
        kw = dict(fixed); kw[param] = v
        pw = power_b(panel_path, mults=list(mults), n_trial=n_trial, quiet=True, **kw)
        n_alert, n_judged = alerts_b(panel_path, **kw)
        d = {param: v}
        for m in mults:
            d[f"{m}倍"] = (pw.loc[pw["倍率"] == m, "検知率"].iloc[0]
                          if len(pw) else None)
        d["初回発火数"] = n_alert; d["判定対象数"] = n_judged
        rows.append(d)
    t = pd.DataFrame(rows)
    print(f"=== 信号B: {param} スイープ（検知率 × 初回件数）===")
    print(t.to_string(index=False))
    print("\n  判定対象数が減りすぎていないかも見る（監視できない単位が増える）。")
    return t


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    if len(sys.argv) > 2 and sys.argv[2].lower() == "b":
        power_b(sys.argv[1])
    else:
        power_c(sys.argv[1])
