# -*- coding: utf-8 -*-
"""
check_label_keys.py
===================
trace_reset_gap で「トレース行なし＝監視レンジ外」が出たとき、その本当の理由を切り分ける。

ラベルの各キー (biz,dev,part) について次を確認する:
  存在      : units（集約後）にそのキーがあるか（無ければキー不一致＝文字列/型の問題）
  経過月範囲 : そのキーの elapsed の min..max
  年月範囲   : そのキーの ym の min..max
  モード     : 安定期 / 安定化前 / 安定化前(保留)
  cache      : 本番リプレイのキャッシュが空でないか（保留や監視レンジ外だと空）
  cache年月  : cache の ym の min..max（トレース窓がここに入っていないと行が出ない）
  報告月     : ラベルの発生年月
  窓内被り   : [報告月-window_before, 報告月+window_after] と cache年月 が重なるか

「存在=×」ならキー不一致。まず両者のキー集合を突き合わせる（末尾に候補表示）。
「存在=○ だが cache=空」なら保留(監視不能)か監視レンジ外。
「cache=○ だが 窓内被り=×」なら報告月が系列の外（データ期間と報告月がズレている）。
"""
import numpy as np
import pandas as pd
import state_logic_cusum as s
from simulate_capped_triage import prep_static, _replay_to_cache, month_diff


def check(panel_path, labels_path, cfg_base, window_before=8, window_after=4):
    labels = pd.read_csv(labels_path, encoding="utf-8-sig")
    cfg = dict(cfg_base)
    panel = s._prepare_panel(pd.read_csv(panel_path), cfg)
    units = s.aggregate_units(panel)

    unit_keys = set(map(tuple, units[["biz", "dev", "part"]].drop_duplicates().to_numpy()))
    unit_map, base_map, mode_map = prep_static(units, cfg, verbose=False)

    lab = labels.copy()
    lab["発生年月"] = lab["発生年月"].map(s.to_yyyymm)

    rows = []
    miss_keys = []
    for _, r in lab.iterrows():
        key = (r["事業コード"], r["開発コード"], r["部番"])
        onset = int(r["発生年月"]) if pd.notna(r["発生年月"]) else None
        rec = dict(事業=key[0], dev=key[1], part=key[2], 報告月=onset)

        exists = key in unit_keys
        rec["存在"] = "○" if exists else "×"
        if not exists:
            miss_keys.append(key)
            rec.update(経過月範囲="-", 年月範囲="-", モード="-",
                       cache="-", cache年月="-", 窓内被り="-")
            rows.append(rec); continue

        u = units[(units["biz"] == key[0]) & (units["dev"] == key[1]) &
                  (units["part"] == key[2])]
        rec["経過月範囲"] = f"{int(u['elapsed'].min())}..{int(u['elapsed'].max())}"
        rec["年月範囲"] = f"{int(u['ym'].min())}..{int(u['ym'].max())}"
        rec["モード"] = mode_map.get(key, "?")

        cache = _replay_to_cache(key, unit_map[key], base_map[key],
                                 mode_map[key], [], cfg)
        if cache is None:
            rec.update(cache="空", cache年月="-", 窓内被り="×")
            rows.append(rec); continue
        cyms = sorted(cache["ym_idx"].keys())
        rec["cache"] = f"{len(cyms)}月"
        rec["cache年月"] = f"{cyms[0]}..{cyms[-1]}"
        if onset is None:
            rec["窓内被り"] = "?"
        else:
            lo = s._add_months(onset, -window_before)
            hi = s._add_months(onset, window_after)
            overlap = any(lo <= y <= hi for y in cyms)
            rec["窓内被り"] = "○" if overlap else "×"
        rows.append(rec)

    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 240, "display.max_columns", 20):
        print("=" * 100)
        print("ラベルキーの監視レンジ診断")
        print("=" * 100)
        print(out.to_string(index=False))

    # キー不一致があれば候補を出す
    if miss_keys:
        print("\n" + "=" * 100)
        print(f"存在×のキー {len(miss_keys)}件 — キー不一致の可能性。近い候補を表示")
        print("=" * 100)
        for key in miss_keys:
            # 同じ dev,part で biz違い / 同じ part で dev違い などを探す
            cand = [k for k in unit_keys if k[1] == key[1] and k[2] == key[2]]
            cand2 = [k for k in unit_keys if k[2] == key[2]][:5]
            print(f"  ラベル {key}")
            print(f"    dev,part一致(biz違い): {cand[:5]}")
            if not cand:
                print(f"    part一致のみ: {cand2}")
            # 型やrepr（空白・全角混入の発見用）
            print(f"    repr: biz={key[0]!r} dev={key[1]!r} part={key[2]!r}")

    # 参考: units 側のキーサンプル（型・reprの突き合わせ用）
    print("\n--- units 側キーのサンプル（repr, 先頭5件） ---")
    for k in list(unit_keys)[:5]:
        print(f"  biz={k[0]!r} dev={k[1]!r} part={k[2]!r}")

    return out


if __name__ == "__main__":
    cfg = dict(s.CONFIG)
    cfg.update(R=1.5, h=5.0, alpha_spike=0.005, min_count=3, burst_window=0)
    check("scale_panel.csv", "scale_labels.csv", cfg)
