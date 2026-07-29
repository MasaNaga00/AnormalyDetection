"""
verify_exposure_diag.py
=======================
diagnose_exposure.py の検証。真の生成過程が分かっている3ケースで、診断が
「正しい結論」と「誤った推奨を出さないこと」の両方を満たすか確認する。

ケース1 退役あり・経年劣化なし  → cum は末期 O/E<1（見逃し方向）。真値カーネルを復元すべき
ケース2 退役なし・経年劣化なし  → cum のままで正しい。**変更不要と言うべき**（負の対照）
ケース3 退役なし・経年劣化あり  → 末期 O/E>1。「露出定義では直らない」と言うべき（負の対照）

実行: python verify_exposure_diag.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd

import diagnose_exposure as dx
import state_logic_cusum as s


def make_panel(surv_eta=None, surv_beta=2.0, aging_per_year=0.0,
               n_dev=20, n_part=5, n_month=180, seed=11) -> pd.DataFrame:
    """合成パネル。
    surv_eta=None       → 退役なし（S(a)=1）
    aging_per_year=0.2  → 経年で故障率が年20%ずつ上がる（h(a) 上昇）
    """
    rng = np.random.default_rng(seed)
    a = np.arange(n_month + 1, dtype=float)
    surv = (np.ones_like(a) if surv_eta is None
            else np.exp(-np.power(a / surv_eta, surv_beta)))
    haz = 3.0e-4 * np.power(1.0 + aging_per_year, a / 12.0)

    rows = []
    for d in range(n_dev):
        s_new = np.zeros(n_month)
        peak = rng.uniform(3000, 12000)
        for t in range(42):
            s_new[t] = peak * np.exp(-((t - 8) ** 2) / (2 * 12.0 ** 2))
        cum = np.cumsum(s_new)
        # 年齢別に台数を持ち、期待件数 = sum_a s_{t-a} S(a) h(a)
        mu_base = np.convolve(s_new, surv * haz[:len(surv)])[:n_month]
        for p in range(n_part):
            sc = rng.uniform(0.4, 2.0)
            for t in range(n_month):
                rows.append(dict(
                    事業コード="B1", 開発コード=f"DEV{d:02d}", 部番=f"P{p:02d}",
                    販社="SUM", 年月=dx._ym_seq(202001, t), 経過月=t,
                    月次使用数=int(rng.poisson(max(mu_base[t] * sc, 0.0))),
                    累積販売台数=float(cum[t]),
                    **{"SF-コード": f"SF{d % 4}", "ランク": f"R{d % 2}"}))
    return pd.DataFrame(rows)


def diag(panel, label, do_fit=True):
    cfg = dict(s.CONFIG)
    p = s._prepare_panel(panel, cfg)
    units = s.aggregate_units(p)
    us = dx.build_unit_sales(units, verbose=False)
    max_a = int(us["elapsed"].max())
    ks = dx.build_default_kernels(max_a + 2)
    best = None
    if do_fit:
        best, _ = dx.fit_kernel_grid(us, cfg, verbose=False)
        if best:
            ks["wbl_fit"] = dx.kernel_weibull(max_a + 2, best["eta"], best["beta"])
    oe, patho, meta = dx.compute_oe(us, cfg, ks, verbose=False)
    flat = dx.summarize_flatness(oe, meta)
    cum = flat[flat["kernel"] == "cum"].iloc[0]
    print(f"\n--- {label} ---")
    print(f"  cum: 末期O/E={cum['oe_tail']:.3f}  使用可能上限={cum['usable_max']}  "
          f"mad_log={cum['mad_log']:.3f}")
    if best:
        wf = flat[flat["kernel"] == "wbl_fit"]
        print(f"  当てはめ: eta={best['eta']:.0f} beta={best['beta']:.2f}"
              + (f"  mad_log={wf.iloc[0]['mad_log']:.3f}" if not wf.empty else ""))
    return cum, best, flat


print("=" * 78)
print("ケース1: 退役あり(ワイブル eta=108,beta=2.0) / 経年劣化なし")
print("  期待される結論: cum の末期 O/E<1（見逃し方向）、当てはめが eta≈108,beta≈2.0 を復元")
print("=" * 78)
c1, b1, f1 = diag(make_panel(surv_eta=108.0, surv_beta=2.0), "ケース1")
ok1 = (c1["oe_tail"] < 0.8) and b1 is not None and abs(b1["eta"] - 108) <= 12 \
      and abs(b1["beta"] - 2.0) <= 0.35
print(f"  判定: {'OK' if ok1 else '★NG'}  (末期O/E<0.8 かつ eta≈108 beta≈2.0)")

print("\n" + "=" * 78)
print("ケース2【負の対照】: 退役なし / 経年劣化なし → cum が正解")
print("  期待される結論: cum の末期 O/E≈1、mad_log が小さく『変更不要』")
print("=" * 78)
c2, b2, f2 = diag(make_panel(surv_eta=None), "ケース2")
ok2 = 0.85 <= c2["oe_tail"] <= 1.18 and c2["mad_log"] < 0.10
print(f"  判定: {'OK' if ok2 else '★NG'}  (末期O/E≈1 かつ mad_log<0.10 → 露出変更は不要と読める)")
alt2 = f2[(f2["kernel"] != "cum") & (f2["kernel"] != "wbl_fit")].sort_values("mad_log").iloc[0]
print(f"  参考: 既存定義の最良 {alt2['kernel']} mad_log={alt2['mad_log']:.3f} "
      f"(cum={c2['mad_log']:.3f} を下回らないこと)")
print(f"        → {'OK: cum が最良' if c2['mad_log'] <= alt2['mad_log'] else '★NG: 別定義が cum に勝った'}")

print("\n" + "=" * 78)
print("ケース3【負の対照】: 退役なし / 経年劣化あり(年+20%) → 露出では直らない")
print("  期待される結論: cum の末期 O/E>1.25（誤報方向）")
print("=" * 78)
c3, b3, f3 = diag(make_panel(surv_eta=None, aging_per_year=0.20), "ケース3")
ok3 = c3["oe_tail"] > 1.25
print(f"  判定: {'OK' if ok3 else '★NG'}  (末期O/E>1.25 → 『露出定義では直らない』分岐に入る)")

print("\n" + "=" * 78)
print(f"総合: {'全ケースOK' if (ok1 and ok2 and ok3) else '★NG あり'}")
print("=" * 78)
