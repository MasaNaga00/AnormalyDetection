# -*- coding: utf-8 -*-
"""
locate_onset_elapsed.py
=======================
「存在○・cache○・窓内被り×」＝報告月が監視レンジ外、と判明したラベルについて、
報告月時点でその部番が経過月何ヶ月だったかを出し、どちらの境界で外れたかを確定する。

境界は2つ:
  下側: stable_start_m（安定期の監視開始。既定4）。報告が経過月<4 なら初期不良期で
        安定期レンジの手前 → 本来は安定化前カーブ(earlylife)の担当。
  上側: monitor_end_m（監視終了。既定60）。報告が経過月>60 なら末期で監視対象外。

出力:
  報告月経過月 : 報告月にその部番が経過月何ヶ月だったか（パネルから引く）
  監視レンジ   : [下側境界, 上側境界]（安定期は[stable_start_m, monitor_end_m]、
                安定化前は[0, monitor_end_m]）
  外れ方       : "下側(初期不良期)" / "上側(末期)" / "レンジ内(別要因)"
  近接データ月 : 報告月に最も近いパネル上の実在月とその経過月
"""
import numpy as np
import pandas as pd
import state_logic_cusum as s


def locate(panel_path, labels_path, cfg_base):
    labels = pd.read_csv(labels_path, encoding="utf-8-sig")
    cfg = dict(cfg_base)
    panel = s._prepare_panel(pd.read_csv(panel_path), cfg)
    units = s.aggregate_units(panel)

    lo_st = cfg["stable_start_m"]
    hi = cfg["monitor_end_m"]

    lab = labels.copy()
    lab["発生年月"] = lab["発生年月"].map(s.to_yyyymm)

    rows = []
    for _, r in lab.iterrows():
        key = (r["事業コード"], r["開発コード"], r["部番"])
        onset = int(r["発生年月"]) if pd.notna(r["発生年月"]) else None
        u = units[(units["biz"] == key[0]) & (units["dev"] == key[1]) &
                  (units["part"] == key[2])].sort_values("ym")
        rec = dict(dev=key[1], part=key[2], 報告月=onset)
        if u.empty or onset is None:
            rec.update(報告月経過月="-", モード="-", 監視レンジ="-",
                       外れ方="単位/報告月なし", 近接データ月="-")
            rows.append(rec); continue

        mode = s.classify_mode(u, cfg)
        lo = lo_st if mode == "安定期" else 0

        # 報告月ちょうどの行があればその経過月、無ければ最近接月から外挿
        exact = u[u["ym"] == onset]
        if not exact.empty:
            el = int(exact["elapsed"].iloc[0])
            near_ym = onset
            near_el = el
        else:
            # 最近接の実在月
            u2 = u.assign(d=(u["ym"].map(lambda y: abs(_mdiff(y, onset)))))
            nr = u2.sort_values("d").iloc[0]
            near_ym = int(nr["ym"]); near_el = int(nr["elapsed"])
            # 報告月の経過月を月差で補正
            el = near_el + _mdiff(onset, near_ym)

        rec["報告月経過月"] = el
        rec["モード"] = mode
        rec["監視レンジ"] = f"[{lo}, {hi}]"
        if el < lo:
            rec["外れ方"] = f"下側(初期不良期, <{lo})→安定化前カーブ担当"
        elif el > hi:
            rec["外れ方"] = f"上側(末期, >{hi})→監視対象外"
        else:
            rec["外れ方"] = "レンジ内(別要因—窓長/データ欠測を確認)"
        rec["近接データ月"] = f"{near_ym}(経過{near_el})"
        rows.append(rec)

    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 240, "display.max_columns", 20):
        print("=" * 100)
        print("報告月時点の経過月と監視レンジの突き合わせ")
        print("=" * 100)
        print(out.to_string(index=False))
    print("\n--- 外れ方の集計 ---")
    print(out["外れ方"].value_counts().to_string())
    return out


def _mdiff(a, b):
    ya, ma = divmod(int(a), 100)
    yb, mb = divmod(int(b), 100)
    return (ya * 12 + ma - 1) - (yb * 12 + mb - 1)


if __name__ == "__main__":
    cfg = dict(s.CONFIG)
    cfg.update(R=1.5, h=5.0, alpha_spike=0.005, min_count=3, burst_window=0)
    locate("scale_panel.csv", "scale_labels.csv", cfg)
