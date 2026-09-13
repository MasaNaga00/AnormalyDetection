# -*- coding: utf-8 -*-
"""
signal_b_peer.py — 信号B: 同一 (biz, SF-CODE) 群における機種間ピア比較

CUSUM（自己参照＝その部品自身の過去との比較）の構造的盲点である
「最初から高い水準で平坦に推移する部品」を、横断比較で拾うための検出器。

考え方
------
監視単位を「機種 × (biz, SF-CODE)」に集約する（部番を合算）。
これにより機種間で部番が違っても比較でき、部品対応表が不要になる。

  対象 : 機種X の (biz, SF) の (C_focal, E_focal)
  ピア : 同一 (biz, SF) の他機種をプールした (C_peer, E_peer)
  検定 : 条件付き二項（cusum_monitor.binom_sf を流用）
          H0: 両者のレートが等しい
          n = C_focal + C_peer を固定すると
          X ~ Binomial(n, E_focal / (E_focal + E_peer))
          p = P(X >= C_focal)

露出（E）の定義
--------------
既存トラックと同じ「累積販売台数の月合計（台数×月）」。
ただし **同一機種内で部番が複数あっても fleet は1回だけ数える**
（部番ごとに足すと保有台数が水増しされるため、月ごとに max を取る）。

nb（機種×SF内のユニーク部番数）の扱い
------------------------------------
1修理で複数部番を使う設計だと件数が系統的に膨らみ、横断比較では
そのまま「異常」に見える。診断の結果 nb で割る正規化は成立しなかったため
（nb=2→1.39, nb=3→3.66, nb=4→5.09 と比例しない）、
**nb が一致する機種同士だけを比較する（nb 層別）** ことで相殺する。

経過月の揃え方
--------------
発売年次の差がそのまま出るのを防ぐため、経過月 0〜elapsed_cap の
区間で切り、その区間を完走している機種だけを比較対象にする。
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

import cusum_monitor as cm


# ============================================================================
# 集計
# ============================================================================
def build_peer_units(panel: pd.DataFrame, elapsed_cap: int = 36,
                     require_full: bool = True) -> pd.DataFrame:
    """_prepare_panel 済みパネル（列: biz/dev/part/dist/ym/elapsed/use/fleet/sf）を
    機種 × (biz, SF) 単位に集約する。

    Returns: biz, sf, dev, C, E, nb, cover（観測できた最大経過月）
    """
    if "sf" not in panel.columns:
        raise ValueError("sf 列がありません。cfg['cols']['sf'] の指定を確認してください。")

    d = panel[panel["elapsed"] <= elapsed_cap].copy()
    d = d[d["sf"].notna()]

    # --- 分子 C: 部番も販社も合算 ---
    c = (d.groupby(["biz", "sf", "dev"], as_index=False)
           .agg(C=("use", "sum"), nb=("part", "nunique"), cover=("elapsed", "max")))

    # --- 分母 E: 機種の保有台数。部番で水増ししないよう月ごとに1回だけ数える ---
    #     販社は合算する（ALL行のみのパネルなら実質そのまま）
    fm = (d.groupby(["biz", "dev", "ym", "dist"], as_index=False)["fleet"].max()
            .groupby(["biz", "dev", "ym"], as_index=False)["fleet"].sum())
    e = fm.groupby(["biz", "dev"], as_index=False)["fleet"].sum().rename(columns={"fleet": "E"})

    u = c.merge(e, on=["biz", "dev"], how="left")
    u = u[u["E"] > 0]
    if require_full:
        u = u[u["cover"] >= elapsed_cap]
    return u.reset_index(drop=True)


# ============================================================================
# 検定
# ============================================================================
def _cond_binom_p(x: float, e_focal: float, c_peer: float, e_peer: float,
                  max_exact: int = 20000) -> float:
    """P(X >= x), X ~ Binom(n=x+c_peer, p=e_focal/(e_focal+e_peer))。
    n が大きすぎる場合は正規近似（連続補正つき）にフォールバック。"""
    n = int(round(x + c_peer))
    if n <= 0 or e_focal <= 0 or e_peer <= 0:
        return 1.0
    p = e_focal / (e_focal + e_peer)
    if n <= max_exact:
        return cm.binom_sf(int(round(x)), n, p)
    mu = n * p
    sd = math.sqrt(n * p * (1.0 - p))
    if sd <= 0:
        return 1.0
    z = (x - 0.5 - mu) / sd
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _peer_compare(u: pd.DataFrame, key: list, min_peers: int,
                  exclude: pd.Series | None = None) -> pd.DataFrame:
    """leave-one-out でピアと比較する1パス。
    exclude=True の機種はピアプールから除く（自分自身の判定は行う）。"""
    u = u.copy()
    w = u.copy()
    if exclude is not None:
        w = w.assign(C=np.where(exclude, 0.0, u["C"]),
                     E=np.where(exclude, 0.0, u["E"]),
                     cnt=np.where(exclude, 0, 1))
    else:
        w = w.assign(cnt=1)
    g = w.groupby(key)
    u["C_peer"] = g["C"].transform("sum") - w["C"]
    u["E_peer"] = g["E"].transform("sum") - w["E"]
    u["n_peers"] = g["cnt"].transform("sum") - w["cnt"]

    u = u[(u["n_peers"] >= min_peers) & (u["E_peer"] > 0)].reset_index(drop=True)
    if u.empty:
        return u
    u["peer_rate"] = u["C_peer"] / u["E_peer"]
    u["expected"] = u["peer_rate"] * u["E"]
    u["O_E"] = np.where(u["expected"] > 0, u["C"] / u["expected"], np.nan)
    u["p"] = [_cond_binom_p(r.C, r.E, r.C_peer, r.E_peer) for r in u.itertuples()]
    return u


def run_signal_b(panel: pd.DataFrame, elapsed_cap: int = 36, min_peers: int = 2,
                 alpha_peer: float = 0.005, min_count: int = 20,
                 min_peer_count: int = 20, min_oe: float = 1.5,
                 nb_strat: bool = True, require_full: bool = True,
                 two_pass: bool = True) -> pd.DataFrame:
    """信号Bの本体。全 (機種, biz, SF) について、同群のピアと比較した結果を返す。

    min_peers : ピア機種が何台以上そろっていれば判定するか（未満は監視不能＝沈黙）
    min_count : **対象機種の累積件数の下限。** 件数が少ないと純粋なノイズでも
                O/E が跳ねる（10件規模だとノイズだけで O/E=2.9 に達する）。
                注目度は O/E ベースなので、薄い単位が上位を占拠してしまう。
    min_peer_count : **ピアプール側の累積件数の下限。** 分母が薄いと期待値の
                推定が不安定になり、同じくO/Eが暴れる。対象と同水準にしておく。
    min_oe    : **主レバー**。ピア比がこの倍率以上のときだけ発火させる。
                件数Cが数百になるとp値は桁で吹き飛び、O/E=1.2 でも p<1e-3 になる。
                機種は正当な理由（設計世代・市場構成）でも差が出る＝過分散があるので、
                統計的有意性だけでは実務的に無意味な発火が混ざる。効果量で切る。
    alpha_peer: 補助レバー。件数が少ない群での偶然を落とすためのガード。
    nb_strat  : True なら nb が一致する機種同士だけを比較（推奨）
    two_pass  : 1パス目で発火した機種をピアプールから外して再計算する。
                同群に複数の異常機種があると基準が押し上げられて互いに
                打ち消し合うため、その汚染を1段だけ取り除く。
    """
    u0 = build_peer_units(panel, elapsed_cap=elapsed_cap, require_full=require_full)
    key = ["biz", "sf", "nb"] if nb_strat else ["biz", "sf"]

    u = _peer_compare(u0, key, min_peers)
    if u.empty:
        return u

    if two_pass:
        bad = ((u["p"] <= alpha_peer) & (u["C"] >= min_count)
               & (u["C_peer"] >= min_peer_count) & (u["O_E"] >= min_oe))
        if bad.any():
            flag = u0.merge(u.loc[bad, ["biz", "sf", "dev"]].assign(_x=True),
                            on=["biz", "sf", "dev"], how="left")["_x"].fillna(False)
            # 除外してピアが min_peers を割る群は1パス目の結果を使う
            u2 = _peer_compare(u0, key, min_peers, exclude=flag.to_numpy())
            if not u2.empty:
                keep = u.merge(u2[["biz", "sf", "dev"]].assign(_y=True),
                               on=["biz", "sf", "dev"], how="left")["_y"].isna()
                u = pd.concat([u[keep.to_numpy()], u2], ignore_index=True)

    u["alert_peer"] = ((u["p"] <= alpha_peer) & (u["C"] >= min_count)
                       & (u["C_peer"] >= min_peer_count) & (u["O_E"] >= min_oe))

    # 注目度は O/E ベース（p値は桁が飛びすぎて並び順の物差しにならない）。
    # min_oe で1.0、min_oe の3倍で上限3.0 になる線形スケール。
    u["注目度"] = np.where(
        u["O_E"] >= min_oe,
        np.minimum(1.0 + 2.0 * (u["O_E"] - min_oe) / max(2.0 * min_oe, 1e-9), 3.0),
        0.0).round(3)

    cols = ["biz", "sf", "dev", "nb", "n_peers", "C", "C_peer", "E", "expected",
            "O_E", "p", "peer_rate", "alert_peer", "注目度", "cover"]
    out = u[cols].sort_values(["alert_peer", "注目度"], ascending=[False, False])
    return out.reset_index(drop=True)


# ============================================================================
# 補助: 部番数そのものの外れ（設計変更の痕跡）
# ============================================================================
def rank_part_count_outliers(panel: pd.DataFrame, elapsed_cap: int = 36) -> pd.DataFrame:
    """同群のピアより部番数が多い機種を抽出する。
    「交換が多い→改良品番が追加される」という因果の痕跡を、件数とは独立に拾う。"""
    u = build_peer_units(panel, elapsed_cap=elapsed_cap, require_full=False)
    g = u.groupby(["biz", "sf"])["nb"]
    u["nb_中央"] = g.transform("median")
    u["nb_最大他"] = g.transform("max")
    u["群機種数"] = g.transform("size")
    r = u[(u["群機種数"] >= 3) & (u["nb"] > u["nb_中央"])].copy()
    r["超過"] = r["nb"] - r["nb_中央"]
    return r[["biz", "sf", "dev", "nb", "nb_中央", "群機種数", "超過", "C", "E"]] \
        .sort_values("超過", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import state_logic_cusum as s
    import gen_peer_panel as gp

    cfg = dict(s.CONFIG)
    raw = gp.build()
    panel = s._prepare_panel(raw, cfg)

    res = run_signal_b(panel, elapsed_cap=36, min_peers=2, nb_strat=True)
    print("=== 信号B 判定対象:", len(res), "／発火:", int(res["alert_peer"].sum()), "===")
    print(res.head(12).to_string(index=False))
