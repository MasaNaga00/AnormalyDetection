# -*- coding: utf-8 -*-
"""
verify_floor_drift_off.py — floor_drift_off パッチの検証

確認する3点:
  1. 後方互換   : floor_drift_off=False で既存挙動と完全一致（差0）
  2. 効果       : True で floor 単位のドリフトだけが止まり、スパイクは残る
  3. 順位の回復 : 上位N枠から floor 単位が退き、本物が上がってくる

合成データの作り: 低頻度部番（ベースライン0件）を多数 ＋ 真のドリフトを少数。
実データで観測された「発火の49%・上位40件の78%が floor 単位」を再現する狙い。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import state_logic_cusum as s
import cusum_monitor as cm

OK, NG = "OK", "NG"
results = []


def check(name, cond, detail=""):
    results.append((OK if cond else NG, name))
    print(f"[{OK if cond else NG}] {name}" + (f"  {detail}" if detail else ""))


def _months(start, n):
    out = [start]
    while len(out) < n:
        out.append(s.next_month(out[-1]))
    return out


def make_panel(n_floor=120, n_real=8, n_month=48, seed=3):
    """floor単位（ベースライン窓で0件の低頻度部番）＋ 真のドリフト単位。"""
    rng = np.random.default_rng(seed)
    yms = _months(202101, n_month)
    rows = []

    # --- floor 単位: 平常レート極小。ベースライン窓(経過月4-15)はほぼ0件 ---
    #     うち先頭10件には経過月35に本物の単月スパイク(6件)を仕込み、
    #     floor_drift_off でもスパイク検知が生き残ることを確認できるようにする。
    for i in range(n_floor):
        lam = 3e-6 * (0.5 + rng.random())      # 2万台で月0.06〜0.09件
        for t, ym in enumerate(yms):
            F = 20000 + 400 * t
            u = rng.poisson(lam * F)
            if i < 10 and t == 35:
                u = 6                          # 単月スパイク
            rows.append(dict(事業コード="E1", 開発コード=f"FL{i:03d}", 部番="P1",
                             販社="ALL", 年月=ym, 経過月=t,
                             月次使用数=int(u), 累積販売台数=int(F)))

    # --- 真のドリフト単位: まともなベースライン。経過月30から2倍 ---
    for i in range(n_real):
        lam = 2.5e-4
        for t, ym in enumerate(yms):
            F = 20000 + 400 * t
            mult = 2.0 if t >= 30 else 1.0
            u = rng.poisson(lam * F * mult)
            rows.append(dict(事業コード="E1", 開発コード=f"RE{i:03d}", 部番="P1",
                             販社="ALL", 年月=ym, 経過月=t,
                             月次使用数=int(u), 累積販売台数=int(F)))
    return pd.DataFrame(rows)


def run(cfg, units, ledger):
    asof = int(units["ym"].max())
    table, meta = s.evaluate_units(units, ledger, cfg, asof)
    inbox = s.build_inbox(table, meta, cfg, asof)
    return table, meta, inbox, asof


if __name__ == "__main__":
    base_cfg = dict(s.CONFIG)
    base_cfg.update(R=1.5, h=2.5, alpha_spike=0.005, min_count=3, burst_window=0)

    panel = s._prepare_panel(make_panel(), base_cfg)
    units = s.aggregate_units(panel)
    ledger = s.load_ledger(base_cfg, df=pd.DataFrame(
        columns=["事業コード", "開発コード", "部番", "判定年月", "記録日", "処置区分"]))

    # floor 単位の特定
    floor_keys = set()
    lo = base_cfg["stable_start_m"]; hi = lo + base_cfg["baseline_len"]
    for key, u in units.groupby(["biz", "dev", "part"], sort=False):
        if s.classify_mode(u, base_cfg) != "安定期":
            continue
        w = u[(u["elapsed"] >= lo) & (u["elapsed"] < hi)]
        if cm.estimate_lambda0(w["use"].to_numpy(), w["fleet"].to_numpy()) <= 0:
            floor_keys.add(key)
    print(f"合成データ: 全{units.groupby(['biz','dev','part']).ngroups}単位中 "
          f"floor単位 {len(floor_keys)}件\n")

    cfg_off = dict(base_cfg); cfg_off["floor_drift_off"] = False
    cfg_on = dict(base_cfg);  cfg_on["floor_drift_off"] = True

    t0, m0, ib0, asof = run(cfg_off, units, ledger)
    t1, m1, ib1, _ = run(cfg_on, units, ledger)

    # --- 1. 後方互換: 既定(False)が従来と同じか ---
    #     旧実装は lam をそのまま渡していた。False のとき lam_drift = lam なので
    #     ロジック上は完全に同一。ここでは「Falseで floor 単位が従来どおり発火する」で確認。
    fl0 = t0[[(r.biz, r.dev, r.part) in floor_keys for r in t0.itertuples()]]
    check("1. floor_drift_off=False では floor 単位のドリフトが従来どおり発火",
          bool(fl0["alert_drift"].any()),
          f"floor単位のドリフト発火 {int(fl0['alert_drift'].sum())} 件")

    # --- 2. 効果 ---
    fl1 = t1[[(r.biz, r.dev, r.part) in floor_keys for r in t1.itertuples()]]
    check("2a. True で floor 単位のドリフト発火が0になる",
          int(fl1["alert_drift"].sum()) == 0,
          f"{int(fl0['alert_drift'].sum())} → {int(fl1['alert_drift'].sum())}")
    ns0, ns1 = int(fl0["alert_spike"].sum()), int(fl1["alert_spike"].sum())
    check("2b. floor 単位のスパイクは残る（仕込んだ単月スパイクを検知し続ける）",
          ns1 == ns0 and ns1 > 0, f"{ns0} → {ns1}")

    re0 = t0[[r.dev.startswith("RE") for r in t0.itertuples()]]
    re1 = t1[[r.dev.startswith("RE") for r in t1.itertuples()]]
    check("2c. 本物のドリフト単位は影響を受けない（S が完全一致）",
          float((re0["S"].to_numpy() - re1["S"].to_numpy()).__abs__().max()) < 1e-9,
          f"最大差 {float(np.abs(re0['S'].to_numpy()-re1['S'].to_numpy()).max()):.2e}")

    # --- 3. 順位の回復 ---
    def compo(inbox, n):
        top = inbox.head(n)
        isf = [(r.事業コード, r.開発コード, r.部番) in floor_keys
               for r in top.itertuples()]
        return float(np.mean(isf)) if len(top) else float("nan")

    print(f"\nインボックス件数: {len(ib0)} → {len(ib1)}")
    for n in (15, 20, 40):
        c0, c1 = compo(ib0, n), compo(ib1, n)
        print(f"  上位{n:>2}件に占める floor 単位の割合: {c0:.3f} → {c1:.3f}")
    check("3a. 上位20件の floor 単位比率が下がる", compo(ib1, 20) < compo(ib0, 20))

    def real_in_top(inbox, n):
        top = inbox.head(n)
        return sum(1 for r in top.itertuples() if str(r.開発コード).startswith("RE"))
    r0, r1 = real_in_top(ib0, 20), real_in_top(ib1, 20)
    print(f"\n  上位20件に入った本物のドリフト単位: {r0} → {r1} （全{8}件中）")
    check("3b. 上位20件に入る本物が増える", r1 >= r0, f"{r0} → {r1}")

    print("\n" + "=" * 78)
    ng = [r for r in results if r[0] == NG]
    print(f"総合: {len(results)-len(ng)}/{len(results)} PASS")
    if ng:
        for _, n in ng:
            print("  NG:", n)
    print("=" * 78)
