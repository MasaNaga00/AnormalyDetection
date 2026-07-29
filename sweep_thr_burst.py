# -*- coding: utf-8 -*-
"""
sweep_thr_burst.py — drift_min_baseline_count × burst_window を振って、
                      発火量とラベル捕捉を同時に見る

低頻度部番（ベースライン窓の修理件数 C が小さい単位）は、
  ・ドリフト: lambda0 が推定値と呼べず、k がほぼ0 → S が修理件数の累計カウンタ化
  ・単月スパイク: min_count に届かず鳴らない
という板挟みになる。バーストウィンドウ検定（数ヶ月合算の条件付き二項）が
この中間レンジの担当なので、ドリフトを切る代わりにバーストを入れて
拾い直せるかを確認する。

ラベルのキーは labels.csv から自動で読むので、手で埋める必要はない。
C はベースライン窓に依存するので、**本番で使う cfg のまま**実行すること。

使い方
------
    import pandas as pd, state_logic_cusum as s, sweep_thr_burst as sw

    cfg = dict(s.CONFIG)
    cfg.update(R=1.5, h=2.5, alpha_spike=0.005, min_count=3,
               stable_start_m=4, baseline_len=18, monitor_end_m=60)

    panel  = s._prepare_panel(pd.read_csv("panel.csv"), cfg)
    units  = s.aggregate_units(panel)
    labels = pd.read_csv("labels.csv", encoding="utf-8-sig")

    sw.report(units, cfg, labels,
              thresholds=[None, 1, 5, 10], bursts=[0, 3, 6], low_c=5)
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

import cusum_monitor as cm
import state_logic_cusum as s

LEDGER_COLS = ["事業コード", "開発コード", "部番", "判定年月", "記録日", "処置区分",
               "再評価年月", "上書きR", "上書きh", "新ベースライン値",
               "ベースライン窓起点", "ベースライン窓長"]


def empty_ledger() -> pd.DataFrame:
    """0行の台帳。load_ledger に df= で必ず明示すること
    （ledger_path=None かつ df=None だと内蔵デモ台帳が読まれる）。"""
    return pd.DataFrame(columns=LEDGER_COLS)


def baseline_counts(units: pd.DataFrame, cfg: dict) -> dict:
    """監視単位ごとのベースライン窓の修理件数 C。**cfg の窓設定に依存する**。"""
    lo = cfg["stable_start_m"]; hi = lo + cfg["baseline_len"]
    out = {}
    for k, u in units.groupby(["biz", "dev", "part"], sort=False):
        w = u[(u["elapsed"] >= lo) & (u["elapsed"] < hi)]
        out[k] = float(w["use"].sum())
    return out


def label_keys(labels: pd.DataFrame) -> list[tuple]:
    return [(r["事業コード"], r["開発コード"], r["部番"], s.to_yyyymm(r["発生年月"]))
            for _, r in labels.iterrows()]


def _mdiff(a, b):
    return (int(a) // 100 * 12 + int(a) % 100) - (int(b) // 100 * 12 + int(b) % 100)


def _first_hit(table: pd.DataFrame, key, rep_ym, lookback_m=6):
    """報告月の lookback_m ヶ月前以降で最初の発火。(年月, 検出器, 報告差) を返す。"""
    g = table[(table["biz"] == key[0]) & (table["dev"] == key[1])
              & (table["part"] == key[2]) & table["total_alert"]]
    if g.empty:
        return None
    floor_ym = s._add_months(int(rep_ym), -lookback_m)
    g = g[g["ym"] >= floor_ym].sort_values("ym")
    if g.empty:
        return None
    r = g.iloc[0]
    kind = "+".join([n for n, v in (("d", r["alert_drift"]), ("s", r["alert_spike"]),
                                    ("b", r["alert_burst"])) if v]) or "-"
    return (int(r["ym"]), kind, _mdiff(r["ym"], rep_ym))


def report(units: pd.DataFrame, cfg: dict, labels: pd.DataFrame,
           thresholds=(None, 1, 5, 10), bursts=(0, 3, 6),
           low_c: float = 5, lookback_m: int = 6):
    Cmap = baseline_counts(units, cfg)
    keys = label_keys(labels)
    asof = int(units["ym"].max())
    led = s.load_ledger(cfg, df=empty_ledger())

    print("=" * 96)
    print(f"ベースライン窓 = 経過月 {cfg['stable_start_m']}〜"
          f"{cfg['stable_start_m'] + cfg['baseline_len'] - 1}"
          f"   monitor_end_m={cfg['monitor_end_m']}  R={cfg['R']} h={cfg['h']}")
    print("=" * 96)
    print("ラベルのベースライン修理件数 C（窓を変えると動くので毎回確認）:")
    for biz, dev, part, rep in keys:
        C = Cmap.get((biz, dev, part))
        print(f"  {dev} / {part}  報告月={rep}  "
              + ("パネルに無い" if C is None else f"C={C:.0f}"))
    n_low = sum(1 for v in Cmap.values() if v < low_c)
    print(f"\n低頻度単位 (C<{low_c}): {n_low} / {len(Cmap)}")

    print("\n" + "=" * 96)
    hdr = (f"{'thr':>5}{'burst':>6} | {'低頻度 中/最大':>14}{'一軍 中/最大':>13} | "
           + "".join(f"{d[:6]:>9}" for _, d, _, _ in keys))
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for thr, bw in itertools.product(thresholds, bursts):
        c = dict(cfg)
        c["drift_min_baseline_count"] = thr
        c["burst_window"] = bw
        table, meta = s.evaluate_units(units, led, c, asof)
        if table.empty:
            continue
        f = table[table["total_alert"]].copy()
        f["C"] = [Cmap.get((r.biz, r.dev, r.part), 0.0) for r in f.itertuples()]
        lo_pm = f[f["C"] < low_c].groupby("ym").size()
        hi_pm = f[f["C"] >= low_c].groupby("ym").size()

        cells = []
        for biz, dev, part, rep in keys:
            h = _first_hit(table, (biz, dev, part), rep, lookback_m)
            cells.append("×" if h is None else f"{h[1]}{h[2]:+d}")

        print(f"{str(thr):>5}{bw:>6} | "
              f"{(lo_pm.median() if len(lo_pm) else 0):>6.0f}/{(lo_pm.max() if len(lo_pm) else 0):>7.0f}"
              f"{(hi_pm.median() if len(hi_pm) else 0):>6.0f}/{(hi_pm.max() if len(hi_pm) else 0):>6.0f} | "
              + "".join(f"{x:>9}" for x in cells))
        rows.append(dict(thr=thr, burst=bw,
                         低頻度中央=(lo_pm.median() if len(lo_pm) else 0),
                         低頻度最大=(lo_pm.max() if len(lo_pm) else 0),
                         一軍中央=(hi_pm.median() if len(hi_pm) else 0),
                         **{d: x for (_, d, _, _), x in zip(keys, cells)}))

    print("\n読み方:")
    print("  セルは『検出器 + 報告月との差』。d=ドリフト s=単月スパイク b=バースト")
    print("  マイナスなら販社より早い。× は期間内に発火なし")
    print("  ★狙い: ドリフトを切って低頻度の発火量を落としつつ、b でラベルを拾い直せる行")
    return pd.DataFrame(rows)
