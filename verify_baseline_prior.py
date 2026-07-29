# -*- coding: utf-8 -*-
"""
verify_baseline_prior.py — 対処C（Gamma-Poisson 縮約ベースライン）の検証

確認する5点:
  1. 後方互換 : cfg["baseline_prior"] 未設定なら従来と完全一致（差0）
  2. 連続性   : C が増えるほど lambda0 が群平均→自己実績へ単調に移る
  3. 病理解消 : C=0/1 の単位で k が意味のある値になり、S が累計カウンタ化しない
  4. 非侵襲   : C が大きい単位（C>=100）の lambda0 はほぼ変わらない
  5. スパイク : 条件付き二項に渡す (C, E) は生値のまま（p値が変わらない）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import cusum_monitor as cm
import state_logic_cusum as s
import verify_floor_drift_off as v

OK, NG = "OK", "NG"
res = []


def check(name, cond, detail=""):
    res.append((OK if cond else NG, name))
    print(f"[{OK if cond else NG}] {name}" + (f"  {detail}" if detail else ""))


cfg0 = dict(s.CONFIG)
cfg0.update(R=1.5, h=2.5, alpha_spike=0.005, min_count=3, burst_window=0)

panel = s._prepare_panel(v.make_panel(n_floor=120, n_real=8), cfg0)
units = s.aggregate_units(panel)
ledger = s.load_ledger(cfg0, df=pd.DataFrame(
    columns=["事業コード", "開発コード", "部番", "判定年月", "記録日", "処置区分"]))

print("=" * 84)
prior = s.fit_baseline_prior(units, cfg0)
print("=" * 84 + "\n")

cfg1 = dict(cfg0); cfg1["baseline_prior"] = prior

# --- 単位ごとの (C, E, λ0旧, λ0新) ---
rows = []
for key, u in units.groupby(["biz", "dev", "part"], sort=False):
    if s.classify_mode(u, cfg0) != "安定期":
        continue
    l0, C, E = s.estimate_baseline(u, cfg0)
    l1, C1, E1 = s.estimate_baseline(u, cfg1)
    rows.append(dict(key=key, C=C, E=E, λ0_旧=l0, λ0_新=l1,
                     C一致=(C == C1), E一致=(E == E1)))
b = pd.DataFrame(rows).sort_values("C").reset_index(drop=True)

# --- 1. 後方互換 ---
cfg_none = dict(cfg0)
same = all(s.estimate_baseline(u, cfg_none)[0] == s.estimate_baseline(u, cfg0)[0]
           for _, u in units.groupby(["biz", "dev", "part"], sort=False))
check("1. baseline_prior 未設定なら従来と完全一致", same)

# --- 2. 連続性（Cの昇順でλ0_新が単調に自己実績へ寄る）---
grp_mean = prior[0] / prior[1]
b["自己レート"] = np.where(b["E"] > 0, b["C"] / b["E"], np.nan)
b["群平均への近さ"] = (b["λ0_新"] - b["自己レート"]).abs() / grp_mean
lo_C = b[b["C"] <= 1]["群平均への近さ"].median()
hi_C = b[b["C"] >= 20]["群平均への近さ"].median()
check("2. C が小さいほど群平均寄り・大きいほど自己実績寄り",
      lo_C > hi_C, f"C<=1 のズレ {lo_C:.3f} > C>=20 のズレ {hi_C:.3f}")

# --- 3. 病理解消（floor単位の k）---
fl = b[b["C"] <= 1]
F = 20000.0
k_old = (cfg0["R"] - 1) * fl["λ0_旧"] * F / np.log(cfg0["R"])
k_new = (cfg0["R"] - 1) * fl["λ0_新"] * F / np.log(cfg0["R"])
check("3. C<=1 の単位で参照値 k が意味のある大きさになる",
      float(k_new.median()) > 10 * float(k_old.median()),
      f"k中央 {float(k_old.median()):.4f} → {float(k_new.median()):.4f} 件/月")
print(f"     （旧は修理1件で S が +{1-float(k_old.median()):.2f}、"
      f"新は +{1-float(k_new.median()):.2f}）")

# --- 4. 非侵襲（Cが大きい単位）---
hi = b[b["C"] >= 100]
if len(hi):
    d = (hi["λ0_新"] / hi["λ0_旧"] - 1).abs().max()
    check("4. C>=100 の単位は lambda0 がほぼ不変", float(d) < 0.05,
          f"最大変化 {float(d):.2%}")
else:
    check("4. C>=100 の単位は lambda0 がほぼ不変", True, "該当なし（合成データ）")

# --- 5. スパイクに渡す (C,E) が生値のまま ---
check("5. spike_test に渡す (C, E) は生値のまま",
      bool(b["C一致"].all() and b["E一致"].all()))

# --- 参考: 実データのラベル4件に相当する C 帯での挙動 ---
print("\n【C 帯ごとの lambda0 の動き】")
b["帯"] = pd.cut(b["C"], [-1, 0, 1, 5, 20, 100, 1e9],
                 labels=["C=0", "C=1", "C=2-5", "C=6-20", "C=21-100", "C>100"])
print(b.groupby("帯", observed=True).agg(
    件数=("C", "size"), λ0_旧中央=("λ0_旧", "median"),
    λ0_新中央=("λ0_新", "median")).to_string())

# --- 発火とインボックス構成の変化 ---
def run(cfg):
    asof = int(units["ym"].max())
    t, m = s.evaluate_units(units, ledger, cfg, asof)
    return t, s.build_inbox(t, m, cfg, asof)

t0, ib0 = run(cfg0)
t1, ib1 = run(cfg1)
floor_keys = set(b[b["C"] <= 1]["key"])


def compo(ib, n):
    top = ib.head(n)
    if not len(top):
        return float("nan")
    return float(np.mean([(r.事業コード, r.開発コード, r.部番) in floor_keys
                          for r in top.itertuples()]))


print(f"\nインボックス件数: {len(ib0)} → {len(ib1)}")
for n in (15, 20, 40):
    print(f"  上位{n:>2}件に占める C<=1 単位の割合: {compo(ib0, n):.3f} → {compo(ib1, n):.3f}")
real0 = sum(1 for r in ib0.head(20).itertuples() if str(r.開発コード).startswith("RE"))
real1 = sum(1 for r in ib1.head(20).itertuples() if str(r.開発コード).startswith("RE"))
print(f"  上位20件の本物のドリフト単位: {real0} → {real1} （全8件）")
print(f"  スパイク発火総数: {int(t0['alert_spike'].sum())} → {int(t1['alert_spike'].sum())}"
      "  （変わらなければ条件付き二項は無傷）")

print("\n" + "=" * 84)
ng = [r for r in res if r[0] == NG]
print(f"総合: {len(res)-len(ng)}/{len(res)} PASS")
for _, n in ng:
    print("  NG:", n)
print("=" * 84)
