# -*- coding: utf-8 -*-
"""
tune_b_minoe.py — 信号B の min_oe / alpha を実データで決める

信号Cとの違い（決め方が構造的に違う理由）
----------------------------------------
- 信号Cは「単位×販社×月」を毎月検定する。alpha が月次の誤報件数に直結する。
- 信号Bは「機種×SF」を1回判定するだけで、値は月次でほぼ動かない。
  発火すると12ヶ月抑制されるので、効いてくるのは
  **立ち上げ初回の総件数**と、その後の細い流入（新機種が窓を完走した分）。

さらに、件数Cが数百になると p値は桁で飛ぶ（O/E=1.27でp=6e-4、O/E=2.48でp=1e-58）。
機種は正当な理由（設計世代・市場構成・使用環境）でも差が出る＝過分散があるため、
統計的有意性だけで切ると「有意だが実務的に無意味」な発火が大量に混ざる。
**主レバーは効果量 min_oe、alpha は件数の少ない群のガード**という役割分担にする。

使い方
------
    python tune_b_minoe.py panel.csv                # 件数の分布
    python tune_b_minoe.py panel.csv labels.csv     # 既知例の検知も評価
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

import state_logic_cusum as sc
import signal_b_peer as sb
import settings as st

MIN_OES = [1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]
ALPHAS = [0.05, 0.005, 0.001]


def scan(panel_path: str, elapsed_cap: int | None = None) -> pd.DataFrame:
    """全 (機種, SF) の O/E と p を1回だけ計算する。"""
    cap = elapsed_cap if elapsed_cap is not None else st.B_ELAPSED_CAP
    raw = pd.read_csv(panel_path, encoding="utf-8-sig")
    raw["年月"] = raw["年月"].astype(str).str.replace(r"\D", "", regex=True).astype(int)
    p_all = raw[raw[st.COLS["dist"]].astype(str) == st.ALL_TOKEN]

    cfg = dict(sc.CONFIG)
    cfg["cols"] = {**sc.CONFIG["cols"], **st.COLS}
    pb = sc._prepare_panel(p_all.copy(), cfg)

    # min_oe=0, alpha=0.9999 で全件の O/E と p を残す
    r = sb.run_signal_b(pb, elapsed_cap=cap, min_peers=st.B_MIN_PEERS,
                        alpha_peer=0.9999, min_count=st.B_MIN_COUNT,
                        min_oe=0.0, two_pass=False)
    r = r[r["p"].notna()].copy()
    r.to_csv("scan_b.csv", index=False, encoding="utf-8-sig")
    print(f"判定対象 (機種×SF): {len(r)}  中央C={int(r['C'].median())}  "
          f"→ scan_b.csv に保存")
    print(f"O/E の分布: "
          f"中央={r['O_E'].median():.2f}  P90={r['O_E'].quantile(0.9):.2f}  "
          f"P95={r['O_E'].quantile(0.95):.2f}  最大={r['O_E'].max():.2f}")
    return r


def sweep(s: pd.DataFrame, min_oes=MIN_OES, alphas=ALPHAS) -> pd.DataFrame:
    """min_oe × alpha で立ち上げ初回の発火件数を出す。"""
    rows = []
    for oe in min_oes:
        for a in alphas:
            hit = s[(s["O_E"] >= oe) & (s["p"] <= a) & (s["C"] >= st.B_MIN_COUNT)]
            rows.append(dict(min_oe=oe, alpha=a, 発火数=len(hit),
                             対象比=round(len(hit) / max(len(s), 1), 3),
                             最小O_E=round(hit["O_E"].min(), 2) if len(hit) else None,
                             機種数=hit["dev"].nunique()))
    t = pd.DataFrame(rows)
    print("\n=== min_oe × alpha（立ち上げ初回の発火件数、機種×SF単位）===")
    print("  発火するとその単位は12ヶ月抑制されるので、以後はこの数より大幅に減る")
    print(t.to_string(index=False))
    return t


def check_labels(s: pd.DataFrame, labels_path: str,
                 min_oes=MIN_OES, alphas=ALPHAS) -> pd.DataFrame:
    """既知例（機種×部番で与える）が拾えるかを操作点ごとに見る。

    信号Bの単位は機種×SFなので、ラベルの部番が属するSFに読み替えて判定する。
    """
    lab = pd.read_csv(labels_path, encoding="utf-8-sig")
    L = st.LABEL_COLS
    # 部番→SF の対応を scan から作れないので、パネル側で引く必要がある場合は
    # ラベルに SF 列を持たせてもよい。ここでは機種単位で当たりを見る。
    key = "SF" if "SF" in lab.columns else None

    rows = []
    for oe in min_oes:
        for a in alphas:
            hit = s[(s["O_E"] >= oe) & (s["p"] <= a) & (s["C"] >= st.B_MIN_COUNT)]
            n = 0
            for r in lab.itertuples():
                dev = getattr(r, L["dev"])
                m = hit[hit["dev"] == dev]
                if key:
                    m = m[m["sf"] == getattr(r, key)]
                if len(m):
                    n += 1
            rows.append(dict(min_oe=oe, alpha=a, 検知=f"{n}/{len(lab)}",
                             検知率=round(n / max(len(lab), 1), 2), 発火数=len(hit)))
    t = pd.DataFrame(rows)
    print(f"\n=== 既知例の検知（ラベル{len(lab)}件）===")
    if not key:
        print("  ※ ラベルに SF 列が無いので機種単位で判定（同機種の別SFでも○になる）")
    print(t.to_string(index=False))
    return t


def top_list(s: pd.DataFrame, min_oe: float, alpha: float, n: int = 30):
    """選んだ操作点で実際に何が出るかを見る。"""
    hit = s[(s["O_E"] >= min_oe) & (s["p"] <= alpha) & (s["C"] >= st.B_MIN_COUNT)]
    hit = hit.sort_values("O_E", ascending=False)
    print(f"\n=== min_oe={min_oe}, alpha={alpha} での発火 {len(hit)}件（上位{n}）===")
    print(hit.head(n)[["dev", "sf", "nb", "n_peers", "C", "O_E", "p"]].to_string(index=False))
    return hit


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    s = scan(sys.argv[1])
    sweep(s)
    if len(sys.argv) > 2:
        check_labels(s, sys.argv[2])


def oe_by_count(s: pd.DataFrame) -> pd.DataFrame:
    """件数帯別の O/E 分布。**件数下限を決めるための主要な表。**

    薄い単位は異常がなくても O/E が跳ねる。件数帯ごとの O/E の上側
    （P95・最大）を見て、ノイズで到達しうる水準を実データで確かめる。
    """
    d = s.copy()
    d["件数帯"] = pd.cut(d["C"], [0, 5, 10, 20, 50, 100, 1e9],
                      labels=["〜5", "6-10", "11-20", "21-50", "51-100", "100〜"])
    t = d.groupby("件数帯", observed=True).agg(
        単位数=("O_E", "size"), O_E中央=("O_E", "median"),
        P90=("O_E", lambda x: x.quantile(0.9)),
        P95=("O_E", lambda x: x.quantile(0.95)),
        最大=("O_E", "max"))
    print("\n=== 件数帯別の O/E 分布 ===")
    print("  薄い帯の P95・最大が大きいほど、その帯はノイズで上位に来る")
    print("  → その帯が消える件数を B_MIN_COUNT にする")
    print(t.round(2).to_string())
    return t


def count_floor_sweep(s: pd.DataFrame, floors=(3, 10, 20, 30, 50),
                      min_oe: float = 1.5, alpha: float = 0.005) -> pd.DataFrame:
    """件数下限を振って、発火件数と最小件数がどう動くかを見る。"""
    rows = []
    for mc in floors:
        hit = s[(s["C"] >= mc) & (s.get("C_peer", s["C"]) >= mc)
                & (s["O_E"] >= min_oe) & (s["p"] <= alpha)]
        rows.append(dict(下限=mc, 判定対象=int(((s["C"] >= mc)).sum()),
                         発火数=len(hit),
                         発火の最小件数=int(hit["C"].min()) if len(hit) else None,
                         発火の最大O_E=round(hit["O_E"].max(), 2) if len(hit) else None))
    t = pd.DataFrame(rows)
    print(f"\n=== 件数下限スイープ（min_oe={min_oe}, alpha={alpha}）===")
    print(t.to_string(index=False))
    return t
