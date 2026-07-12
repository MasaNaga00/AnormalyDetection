# -*- coding: utf-8 -*-
"""
analyze_capped_labels.py
========================
「インボックス入り月はあるが top-N処理月が空欄」のラベルを調べる。

やること2段:
  段1) キャップ起因の確定:
       top_n=N（予算）と top_n=None（無制限）で回し、N では処理月が空欄なのに
       無制限だと埋まるラベルを拾う。これが「検知はできているが予算Nに負けて
       処理まで回っていない」＝キャップ起因のラベル。
  段2) 何位で負けているか:
       そのラベルが「発火」として載った各月に、全発火候補を注目度降順で順位付けし、
       何位だったか＝N をいくつにすれば拾えたかを出す。

早期段階（安定化前）で載ったラベルは注目度(S/h)が安定期の派手なドリフトに
負けやすい。順位が分かれば「N をいくつ上げれば拾えるか」「別枠にすべきか」が
判断できる。
"""
import pandas as pd
import state_logic_cusum as s
from simulate_capped_triage import simulate_capped_triage


def analyze(units, cfg, labels, budget_n, assumed_disposition="対策中",
            lookback_m=6):
    """budget_n: 実運用で想定している月間予算N。"""
    # --- 予算Nと無制限で回す（無制限はキャップ影響を外した基準）---
    _, inc_n, _, _ = simulate_capped_triage(
        units, cfg, top_n=budget_n, assumed_disposition=assumed_disposition,
        labels=labels, lookback_m=lookback_m, verbose=False)
    _, inc_unl, det_unl, _ = simulate_capped_triage(
        units, cfg, top_n=None, assumed_disposition=assumed_disposition,
        labels=labels, lookback_m=lookback_m, collect_details=True, verbose=False)

    m_n = inc_n.set_index(["事業コード", "開発コード", "部番"])
    m_unl = inc_unl.set_index(["事業コード", "開発コード", "部番"])

    # --- 段1: キャップ起因ラベルの抽出 ---
    cap_rows = []
    for k in m_n.index:
        t_n = m_n.loc[k, "top-N処理月"]
        ib = m_n.loc[k, "インボックス入り月"]
        t_unl = m_unl.loc[k, "top-N処理月"] if k in m_unl.index else None
        if pd.isna(t_n) and pd.notna(ib):
            cause = "キャップ起因（無制限なら処理される）" if pd.notna(t_unl) \
                else "キャップ以外（無制限でも未処理）"
            cap_rows.append(dict(
                事業コード=k[0], 開発コード=k[1], 部番=k[2],
                インボックス入り月=ib, N処理月=t_n, 無制限処理月=t_unl, 判定=cause))
    cap = pd.DataFrame(cap_rows)

    print("=" * 90)
    print(f"予算 N={budget_n}: インボックス入りありだが処理月が空欄のラベル")
    print("=" * 90)
    if cap.empty:
        print("  該当なし（Nの範囲で全ラベルが処理まで回っている）")
        return cap, pd.DataFrame()
    print(cap.to_string(index=False))

    # --- 段2: 各ラベルが載った月の注目度順位 ---
    det_fire = det_unl[det_unl["載る理由"] == "発火"].copy()
    rank_rows = []
    cap_cause = cap[cap["判定"].str.startswith("キャップ起因")]
    for _, r in cap_cause.iterrows():
        key = (r["事業コード"], r["開発コード"], r["部番"])
        mine = det_fire[(det_fire["事業コード"] == key[0]) &
                        (det_fire["開発コード"] == key[1]) &
                        (det_fire["部番"] == key[2])]
        for ym in sorted(mine["シミュレーション年月"].unique()):
            month = det_fire[det_fire["シミュレーション年月"] == ym] \
                .sort_values("注目度", ascending=False).reset_index(drop=True)
            pos = month[(month["事業コード"] == key[0]) &
                        (month["開発コード"] == key[1]) &
                        (month["部番"] == key[2])].index
            if len(pos) == 0:
                continue
            rank = int(pos[0]) + 1
            rank_rows.append(dict(
                開発コード=key[1], 部番=key[2], 年月=ym,
                注目度=round(float(month.iloc[rank - 1]["注目度"]), 3),
                順位=rank, 発火総数=len(month),
                必要N=rank,  # このラベルをこの月に拾うのに必要なN
                経過月=month.iloc[rank - 1]["経過月"]))
    ranks = pd.DataFrame(rank_rows)

    print("\n" + "=" * 90)
    print("キャップ起因ラベルが「発火として載った月」の注目度順位")
    print("（必要N = その月にこのラベルを処理するのに要する予算N）")
    print("=" * 90)
    if not ranks.empty:
        print(ranks.to_string(index=False))
        # 各ラベルの最小必要N（一番順位が高かった月）
        best = ranks.groupby(["開発コード", "部番"])["必要N"].min().reset_index()
        best.columns = ["開発コード", "部番", "最小必要N"]
        print("\n--- 各ラベルを拾うのに要する最小N（最良月ベース）---")
        print(best.sort_values("最小必要N").to_string(index=False))
        print(f"\n  これらを全部拾うには N≥{int(best['最小必要N'].max())} が必要。")
        print(f"  半数を拾うには N≥{int(best['最小必要N'].median())} 目安。")
    return cap, ranks


if __name__ == "__main__":
    # ==== ここを実データに差し替え ====
    PANEL = "実パネル.csv"
    LABELS = "実ラベル.csv"
    BUDGET_N = 10          # 想定している月間予算N
    # ==================================

    cfg = dict(s.CONFIG)
    cfg.update(R=1.5, h=5.0, alpha_spike=0.005, min_count=3, burst_window=0)
    panel = s._prepare_panel(pd.read_csv(PANEL), cfg)
    units = s.aggregate_units(panel)
    labels = pd.read_csv(LABELS, encoding="utf-8-sig")
    analyze(units, cfg, labels, budget_n=BUDGET_N)
