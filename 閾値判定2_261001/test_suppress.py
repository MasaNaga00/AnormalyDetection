# -*- coding: utf-8 -*-
"""検出器別抑制の検証:
  A) 信号Bを記録 → 12ヶ月は信号Bだけ黙り、閾値/信号Cは生き続ける
  B) 再評価年月を空欄にしても翌月に再発火しない
  C) 保留は毎月出続ける
"""
import numpy as np, pandas as pd, unified_inbox as ui, state_logic_cusum as sc
import test_unified as tu   # パネル生成を流用

cfg = dict(ui.CONFIG); cfg.update(base_threshold_pct=1.5)
p_all, p_dist, cols = tu.p_all, tu.p_dist, tu.cols
T = int(p_all["年月"].max())

def row(dev, part, det, dispo, ym, rev=np.nan):
    d = {c: np.nan for c in ui.LEDGER_COLS}
    d.update({"記録日": "2024-05-01", "事業コード": "E1", "開発コード": dev, "部番": part,
              "検出器": det, "判定年月": ym, "処置区分": dispo, "再評価年月": rev})
    return d

def show(tag, led, asof):
    r = ui.build_unified_inbox(p_all, p_dist, led, cfg, cols, asof_ym=asof)
    ib = r["inbox"]
    print(f"--- {tag} (asof={asof}) ---")
    if ib.empty: print("  (空)"); return
    print(ib[["開発コード","検出器","統合注目度","抑制解除月"]].to_string(index=False))

# --- A/B: M3 を信号Bで記録、再評価年月は空欄 ---
led = ui.empty_ledger()
led.loc[0] = row("M3", "M3-P0", "信号B", "対策中", T)
show("M3を信号Bで記録した当月", led, T)

print("\n=== 抑制期間の推移（信号Bを判定年月Mで記録、再評価年月は空欄）===")
for back in (2, 5, 11, 12, 13):
    M = sc._add_months(T, -back)
    led = ui.empty_ledger(); led.loc[0] = row("M3", "M3-P0", "信号B", "対策中", M)
    r = ui.build_unified_inbox(p_all, p_dist, led, cfg, cols, asof_ym=T)
    m3 = r["inbox"][r["inbox"].開発コード == "M3"]
    det = m3["検出器"].iloc[0] if len(m3) else "(出ない)"
    print(f"  記録から{back:2d}ヶ月経過 → M3の検出器 = {det}")

print("\n=== 保留は毎月出るか ===")
for back in (0, 3, 20):
    M = sc._add_months(T, -back)
    led = ui.empty_ledger(); led.loc[0] = row("M3", "M3-P0", "信号B", "保留", M)
    r = ui.build_unified_inbox(p_all, p_dist, led, cfg, cols, asof_ym=T)
    m3 = r["inbox"][r["inbox"].開発コード == "M3"]
    print(f"  保留を{back:2d}ヶ月前に記録 → 状態={m3['状態'].iloc[0] if len(m3) else '(出ない)'}")

print("\n=== 信号Bを抑制中に販社スパイクが起きたら（M1で確認）===")
M = sc._add_months(T, -2)
led = ui.empty_ledger(); led.loc[0] = row("M1", "M1-P0", "信号B", "対策中", M)
r = ui.build_unified_inbox(p_all, p_dist, led, cfg, cols, asof_ym=T)
m1 = r["inbox"][r["inbox"].開発コード == "M1"]
print("  M1:", m1["検出器"].iloc[0] if len(m1) else "(出ない)", "← 信号Cは生きている必要あり")
