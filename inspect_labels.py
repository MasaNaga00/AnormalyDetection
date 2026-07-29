# -*- coding: utf-8 -*-
"""
inspect_labels.py — 正解ラベルを1件ずつ、月次トレースと注目度順位で読む

ラベルが数件しかない段階では中央値の比較は意味を持たない（1件動くと飛ぶ）。
1件ずつ「いつ・どの検出器で・何位で」拾えたのかを読むのが正しい。

出すもの（ラベルごと）:
  - ベースライン窓の修理件数 C と lambda0（低頻度部番かどうか）
  - 販社報告月の前後の月次トレース
      使用数 / 期待故障数 / S / ドリフト発火 / スパイク発火 / p値 / 注目度
  - **その月の艦隊内での注目度順位**（何位なら上位N件に入れたか）
  - 検出器の別（drift か spike か）→ drift_min_baseline_count の判断材料

使い方
------
    import inspect_labels as il
    il.inspect(units, cfg, labels, thresholds=[None, 3, 5], around=8, top_n=20)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import cusum_monitor as cm
import state_logic_cusum as s


def _baseline_info(unit, cfg):
    lo = cfg["stable_start_m"]; hi = lo + cfg["baseline_len"]
    w = unit[(unit["elapsed"] >= lo) & (unit["elapsed"] < hi)]
    C = float(w["use"].sum()); E = float(w["fleet"].sum())
    raw = cm.estimate_lambda0(w["use"].to_numpy(), w["fleet"].to_numpy())
    return C, E, raw


def inspect(units: pd.DataFrame, cfg: dict, labels: pd.DataFrame,
            thresholds=(None, 3, 5), around: int = 8, top_n: int = 20,
            ledger: pd.DataFrame | None = None):
    if ledger is None:
        ledger = pd.DataFrame(columns=["事業コード", "開発コード", "部番",
                                       "判定年月", "記録日", "処置区分"])
    led = s.load_ledger(cfg, df=ledger)
    asof = int(units["ym"].max())

    lab = labels.copy()
    lab["発生年月"] = lab["発生年月"].map(s.to_yyyymm)

    for thr in thresholds:
        c = dict(cfg); c["drift_min_baseline_count"] = thr
        table, meta = s.evaluate_units(units, led, c, asof)
        if table.empty:
            print(f"drift_min_baseline_count={thr}: table が空"); continue

        # 月ごとの注目度と順位（発火した単位の中での順位）
        t = table.copy()
        t["注目度"] = [s.attention_score(r.S, r.h, r.p_spike, r.p_burst, c)
                       for r in t.itertuples()]
        t["順位"] = (t[t["total_alert"]].groupby("ym")["注目度"]
                     .rank(ascending=False, method="min"))
        n_fire = t[t["total_alert"]].groupby("ym").size().rename("その月の発火数")

        print("\n" + "=" * 100)
        print(f"drift_min_baseline_count = {thr}")
        print("=" * 100)

        for _, r in lab.iterrows():
            key = (r["事業コード"], r["開発コード"], r["部番"])
            rep = int(r["発生年月"])
            u = units[(units["biz"] == key[0]) & (units["dev"] == key[1])
                      & (units["part"] == key[2])]
            if u.empty:
                print(f"\n■ {key}  → パネルに無い（対象外）")
                continue
            C, E, raw = _baseline_info(u, c)
            mode = meta.get(key, {}).get("mode", "?")
            低頻度 = (thr is not None and C < thr)
            print(f"\n■ {key}  報告月={rep}  モード={mode}  "
                  f"ベースライン C={C:.0f} λ0(raw)={raw:.3e}"
                  f"{'  ★ドリフト停止対象' if 低頻度 else ''}")

            g = t[(t["biz"] == key[0]) & (t["dev"] == key[1])
                  & (t["part"] == key[2])].sort_values("ym")
            if g.empty:
                print("   監視レンジ内に行が無い（レンジ外）")
                continue
            lo_ym = s._add_months(rep, -around)
            hi_ym = s._add_months(rep, around)
            g = g[(g["ym"] >= lo_ym) & (g["ym"] <= hi_ym)]
            if g.empty:
                print("   報告月周辺に監視行が無い")
                continue

            g = g.merge(n_fire, left_on="ym", right_index=True, how="left")
            show = pd.DataFrame({
                "年月": g["ym"].astype(int),
                "経過月": g["elapsed"],
                "報告差": [s_ for s_ in (
                    (g["ym"].map(lambda x: (int(x)//100*12 + int(x) % 100)
                                 - (rep//100*12 + rep % 100))))],
                "使用数": g["use"].astype(int),
                "期待": g["mu0"].round(2),
                "S": g["S"].round(2),
                "drift": np.where(g["alert_drift"], "●", ""),
                "spike": np.where(g["alert_spike"], "●", ""),
                "p値": g["p_spike"].round(4),
                "注目度": g["注目度"].round(2),
                "順位": g["順位"].astype("Int64"),
                "発火数": g["その月の発火数"].astype("Int64"),
                f"上位{top_n}入": np.where(g["順位"].le(top_n).fillna(False), "○", ""),
            })
            print(show.to_string(index=False))

            fired = g[g["total_alert"]]
            if fired.empty:
                print("   → 期間内に発火なし")
            else:
                f0 = fired.iloc[0]
                kind = "+".join([n for n, v in (("drift", f0["alert_drift"]),
                                                ("spike", f0["alert_spike"]),
                                                ("burst", f0["alert_burst"])) if v])
                intop = fired[fired["順位"].le(top_n).fillna(False)]
                first_top = int(intop.iloc[0]["ym"]) if len(intop) else None
                print(f"   → 初回発火 {int(f0['ym'])}（{kind}、報告の"
                      f"{(int(f0['ym'])//100*12+int(f0['ym'])%100)-(rep//100*12+rep%100):+d}ヶ月）"
                      f" / 初回 上位{top_n}入り "
                      f"{first_top if first_top else 'なし（キャップで待機）'}")
    print("\n" + "=" * 100)
    print("読み方")
    print("=" * 100)
    print("・低頻度ラベル(C小)が spike で発火していた → ドリフト停止は検知を失わない。閾値を上げてよい")
    print("・drift のみで発火していた                → 停止すると本当に遅くなる。閾値は慎重に")
    print("・発火はしているのに『上位N入』が空       → 検知でなく triage 順位の問題。")
    print("                                            別枠運用かNの引き上げで対処する")
    print("・『順位』と『発火数』を見れば、必要なNが直接わかる")
