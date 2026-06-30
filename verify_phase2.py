# -*- coding: utf-8 -*-
"""フェーズ2検証: 先行機種からの安定化前カーブ結線。
(1)正常な新機種は鳴らない (2)異常な新機種は早期点灯 (3)先行機種不足は監視保留
(4)スケールは先行機種側で固定（監視対象を入れ替えてもカーブ不変）。"""
import math
import numpy as np
import pandas as pd
import state_logic_cusum as m

rng = np.random.default_rng(7)

def base_rate(t):
    return 0.0012 + 0.0050 * math.exp(-t / 3.0)

def fleet_at(t):
    return 300 + 180 * t

def gen(dev, part, sf, rank, n_months, start_ym, mult=1.0, jitter=1.0):
    rows = []
    for t in range(n_months):
        F = fleet_at(t)
        u = int(rng.poisson(base_rate(t) * mult * jitter * F))
        rows.append(dict(事業コード="E1", 開発コード=dev, 部番=part, 販社="D0",
                         **{"SF-コード": sf, "ランク": rank},
                         年月=m._add_months(start_ym, t), 経過月=t,
                         月次使用数=u, 累積販売台数=F))
    return rows

rows = []
rows += gen("DEV-L1", "PART-X", "SF-100", "B", 24, 202101, jitter=0.9)
rows += gen("DEV-L2", "PART-X", "SF-100", "B", 24, 202101, jitter=1.0)
rows += gen("DEV-L3", "PART-X", "SF-100", "B", 24, 202101, jitter=1.15)
rows += gen("DEV-OK",  "PART-X", "SF-100", "B", 9, 202401, mult=1.0)
rows += gen("DEV-NG",  "PART-X", "SF-100", "B", 9, 202401, mult=2.6)
rows += gen("DEV-S1", "PART-Y", "SF-200", "B", 24, 202101, jitter=1.0)
rows += gen("DEV-LONELY", "PART-Y", "SF-200", "B", 9, 202401, mult=2.6)

cfg = dict(m.CONFIG); cfg["h"]=4.0; cfg["alpha_spike"]=0.01; cfg["min_count"]=3; cfg["burst_window"]=3

panel = m._prepare_panel(pd.DataFrame(rows), cfg)
units = m.aggregate_units(panel)
ledger = m.load_ledger(cfg, df=pd.DataFrame(columns=["事業コード","開発コード","部番","判定年月","記録日","処置区分"]))
asof = int(units["ym"].max())

levels = m.build_earlylife_curves(units, cfg)
print("=== (0) 推定カーブ（採用された最細階層から SF-100, 経過月0-8）===")
print("構築された階層:", [lv["keys"] for lv in levels])
lv = next(l for l in levels if ("E1","SF-100","B") in l["counts"] or ("E1","SF-100") in l["counts"])
cur = lv["curve"]
c100 = cur[(cur["biz"]=="E1") & (cur["sf"]=="SF-100") & (cur["elapsed"]<=8)]
print(c100[lv["keys"]+["elapsed","lambda0_hat","n_models"]].to_string(index=False))
print(f"真のカーブ(参考): " + ", ".join(f"t{t}:{base_rate(t):.4f}" for t in range(0,9,2)))

table, meta = m.evaluate_units(units, ledger, cfg, asof)
print("\n=== (1)(2)(3) 安定化前 監視対象の判定 ===")
for dev in ["DEV-OK","DEV-NG","DEV-LONELY"]:
    sub = table[table["dev"]==dev].sort_values("ym")
    key = ("E1", dev, sub["part"].iloc[0])
    fired = sub[sub["total_alert"]]
    first = int(fired["elapsed"].min()) if not fired.empty else None
    held = meta[key]["mode"].endswith("(保留)")
    print(f"\n[{dev}] モード={meta[key]['mode']}  集団レベル={meta[key].get('earlylife_level')}  先行機種数={meta[key].get('leader_count')}")
    verdict = ("監視保留（鳴らさない）" if held else
               ("鳴らない(正常)" if first is None else f"経過月{first}で初点灯"))
    print(f"  → {verdict}")

units_noNG = units[units["dev"] != "DEV-NG"]
def sf100_curve(u):
    lv = m.build_earlylife_curves(u, cfg)
    l = next(x for x in lv if ("E1","SF-100","B") in x["counts"] or ("E1","SF-100") in x["counts"])
    c = l["curve"]; return c[(c["sf"]=="SF-100")&(c["elapsed"]<=8)]["lambda0_hat"].to_numpy()
print("\n=== (4) スケール固定の確認 ===")
print(f"DEV-NG(異常監視対象)を含む/含まないでSF-100カーブの最大差 = {np.max(np.abs(sf100_curve(units)-sf100_curve(units_noNG))):.2e}")
