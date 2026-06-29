"""
run_monthly.py
==============
毎月の運用ランナー。実データCSVを入れて、前処理 → モード自動振り分け →
異常検知 → 結果CSV出力 までを一気通貫で行う。

各監視単位を履歴の長さで2モードに自動振り分けする:
  - 安定期モード   : 自己ベースラインを取れるだけの履歴がある単位。
                     その単位自身の過去を物差しに、固定lambda0でCUSUM+スパイク監視。
  - 安定化前モード : 履歴が浅い若い単位。同じ集団(SF-コード×ランク等)の
                     先行機種から推定した期待カーブ lambda0(t) を物差しに監視。
  - 監視保留       : 安定期にも達せず、先行機種カーブも作れない単位。今月は監視外。

チューニング(操作点の決定)は別途 backtest.py で済ませてある前提で、
ここでは確定済みの操作点パラメータを CONFIG に置いて使う。

実運用での使い方:
  1. データマートから当月までの縦長パネルCSVを出力(INPUT_CSV)。
     SF-コード・ランク列があると安定化前モードの集団分けに使える。
  2. CONFIG のパラメータ(操作点・監視レンジ・振り分け閾値)を確定値に設定。
  3. python run_monthly.py を実行。
  4. 出力された alerts_*.csv を Tableau のデータソースにする。
     「基準モード」列で安定期/安定化前を区別できる。
"""

from __future__ import annotations
import pandas as pd
import numpy as np

import cusum_monitor as cm
from earlylife_baseline import estimate_earlylife_curve, attach_curve_to_unit

# ---- 確定済みの操作点(チューニング結果をここに固定する) ----
CONFIG = dict(
    stable_start_m=36,   # 安定期の入口(経過月)
    baseline_len=12,     # ベースライン窓の長さ(月)
    monitor_end_m=90,    # 妥当性ウィンドウの末尾(これ以降は監視しない)
    # ドリフト側(CUSUM)
    R=2.0,
    h=8.0,
    # スパイク側(単月/バースト検定)
    alpha_spike=0.001,
    min_count=3,
    burst_window=2,
)

# ---- 安定化前モードの設定 ----
EARLYLIFE = dict(
    enabled=True,
    group_keys=["事業コード", "SF-コード", "ランク"],  # 先行機種をプールする集団キー
    curve_max_m=35,       # カーブを推定する経過月の上限(安定化するあたり)
    min_leaders=2,        # カーブ推定に必要な先行機種の最低数
    # 安定化前モードはスパイク/ドリフトを別感度にしたい場合ここで上書き
    R=2.0,
    h=6.0,
    alpha_spike=0.005,
    min_count=3,
    burst_window=2,
)

INPUT_CSV = "mixed_input.csv"
OUT_PREFIX = "alerts"


def fill_zero_months(
    df: pd.DataFrame,
    unit_keys: list[str],
    col_keizoku="経過月",
    col_usage="月次使用数",
    col_fleet="累積販売台数",
) -> pd.DataFrame:
    """各監視単位について、観測されている経過月レンジを連続にし、
    欠落月を 月次使用数=0 / 累積販売台数=前方埋め で補完する。"""
    out = []
    static_cols = [c for c in df.columns
                   if c not in (col_keizoku, col_usage, col_fleet)]
    for key, g in df.groupby(unit_keys, dropna=False):
        g = g.sort_values(col_keizoku)
        full = range(int(g[col_keizoku].min()), int(g[col_keizoku].max()) + 1)
        g = g.set_index(col_keizoku).reindex(full)
        g[col_usage] = g[col_usage].fillna(0)
        g[col_fleet] = g[col_fleet].ffill()
        for c in static_cols:
            g[c] = g[c].ffill().bfill()
        g.index.name = col_keizoku
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def classify_unit(max_m: int, cfg: dict) -> str:
    """単位の最大経過月から、どのモードで監視できるかを判定する。

    安定期モードの最低要件: ベースライン窓を取り切れること。すなわち
      max_m >= stable_start_m + baseline_len - 1
    これに満たない単位は、安定化前モードの候補(=若い単位)。
    """
    need = cfg["stable_start_m"] + cfg["baseline_len"] - 1
    if max_m >= need:
        return "stable"
    return "earlylife"


def run(level: str = "agg"):
    """level='agg' なら販社合算(機種×部番)、'sha' なら販社別(機種×部番×販社)。"""
    raw = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    print(f"[入力] {INPUT_CSV}: {len(raw)} 行, "
          f"{raw['開発コード'].nunique()} 機種, 販社 {sorted(raw['販社'].unique())}")

    if level == "agg":
        unit_keys = cm.UNIT_KEYS_AGG_SHA
        agg = cm.aggregate_over_sha(raw)
        # 集団キー(SF-コード・ランク)を機種側から復元(合算で消えるため)
        if EARLYLIFE["enabled"]:
            meta_cols = [c for c in EARLYLIFE["group_keys"]
                         if c in raw.columns and c not in cm.UNIT_KEYS_AGG_SHA]
            if meta_cols:
                meta = (raw[cm.UNIT_KEYS_AGG_SHA + meta_cols]
                        .drop_duplicates(subset=cm.UNIT_KEYS_AGG_SHA + meta_cols))
                agg = agg.merge(meta, on=cm.UNIT_KEYS_AGG_SHA, how="left")
        data = fill_zero_months(agg, unit_keys)
        tag = "agg_機種x部番"
    else:
        unit_keys = cm.UNIT_KEYS_WITH_SHA
        data = fill_zero_months(raw, unit_keys)
        tag = "sha_機種x部番x販社"

    # --- 各単位の最大経過月でモードを判定 ---
    max_m = data.groupby(unit_keys, dropna=False)["経過月"].max()
    mode_map = {k: classify_unit(int(v), CONFIG) for k, v in max_m.items()}
    n_stable = sum(1 for v in mode_map.values() if v == "stable")
    n_early = sum(1 for v in mode_map.values() if v == "earlylife")
    print(f"[振り分け] 安定期 {n_stable} 単位 / 安定化前(候補) {n_early} 単位 "
          f"(level={level})")

    # --- 安定化前モード用: 先行機種(安定期に達した単位)からカーブを推定 ---
    curve = None
    group_keys = EARLYLIFE["group_keys"]
    can_earlylife = (
        level == "agg" and EARLYLIFE["enabled"] and n_early > 0
        and all(k in data.columns for k in group_keys)
    )
    if can_earlylife:
        # 先行機種 = 安定期モードに分類された単位(履歴が十分=形が見えている)
        stable_units = {k for k, v in mode_map.items() if v == "stable"}
        leader_mask = data.apply(
            lambda r: tuple(r[k] for k in unit_keys) in stable_units
            if len(unit_keys) > 1 else r[unit_keys[0]] in stable_units, axis=1)
        leaders = data[leader_mask]
        if len(leaders):
            curve = estimate_earlylife_curve(
                leaders, group_keys=group_keys,
                max_keizoku=EARLYLIFE["curve_max_m"],
            )

    # --- 監視実行 ---
    results = []
    skipped = 0
    for key, g in data.groupby(unit_keys, dropna=False):
        g = g.sort_values("経過月")
        mode = mode_map[key]

        if mode == "stable":
            res = cm.monitor_unit(
                g, stable_start_m=CONFIG["stable_start_m"],
                baseline_len=CONFIG["baseline_len"],
                monitor_end_m=CONFIG["monitor_end_m"],
                R=CONFIG["R"], h=CONFIG["h"],
                alpha_spike=CONFIG["alpha_spike"],
                min_count=CONFIG["min_count"],
                burst_window=CONFIG["burst_window"],
            )
        else:  # earlylife
            if curve is None:
                skipped += 1
                continue
            # この単位の集団カーブが引けるか確認
            n_leaders = _count_leaders(curve, g, group_keys)
            if n_leaders < EARLYLIFE["min_leaders"]:
                skipped += 1
                continue
            lam_curve = attach_curve_to_unit(g, curve, group_keys)
            if np.all(lam_curve <= 0):
                skipped += 1
                continue
            res = cm.monitor_unit(
                g, stable_start_m=0, baseline_len=0,
                monitor_end_m=EARLYLIFE["curve_max_m"],
                R=EARLYLIFE["R"], h=EARLYLIFE["h"],
                alpha_spike=EARLYLIFE["alpha_spike"],
                min_count=EARLYLIFE["min_count"],
                burst_window=EARLYLIFE["burst_window"],
                lambda0_curve=lam_curve,
            )
        if len(res):
            results.append(res)

    result = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    out_csv = f"{OUT_PREFIX}_{tag}.csv"
    result.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # --- サマリ ---
    print(f"\n[結果] {out_csv} に保存")
    if len(result):
        n_units = result.groupby(unit_keys, dropna=False).ngroups
        by_mode = result.groupby("基準モード", dropna=False).apply(
            lambda d: d.groupby(unit_keys, dropna=False).ngroups, include_groups=False)
        print(f"  監視した単位数: {n_units}  (モード別: {by_mode.to_dict()})")
        if skipped:
            print(f"  監視保留      : {skipped} 単位(先行機種カーブ不足など)")
        latest = (result.sort_values("経過月")
                  .groupby(unit_keys, dropna=False).tail(1))
        latest_alarm = latest[latest["総合アラート"]]
        print(f"  当月点灯中    : {len(latest_alarm)} 単位  ← Tableau worklist先頭群")
        if len(latest_alarm):
            show_cols = unit_keys + ["基準モード", "経過月", "月次使用数",
                                     "期待故障数", "CUSUM", "月次p値", "アラート種別"]
            show_cols = [c for c in show_cols if c in latest_alarm.columns]
            disp = latest_alarm[show_cols].copy()
            for c in ["期待故障数", "CUSUM", "月次p値"]:
                if c in disp.columns:
                    disp[c] = disp[c].round(3)
            print(disp.head(12).to_string(index=False))
    else:
        print("  監視対象なし")
    return result


def _count_leaders(curve: pd.DataFrame, df_unit: pd.DataFrame,
                   group_keys: list[str]) -> int:
    """この単位の集団に対応するカーブの先行機種数(最大n_models)を返す。"""
    gvals = {k: df_unit[k].iloc[0] for k in group_keys}
    c = curve
    for k, val in gvals.items():
        c = c[c[k] == val]
    return int(c["n_models"].max()) if len(c) else 0


if __name__ == "__main__":
    print("========== 主監視: 販社合算(機種×部番) ==========")
    run("agg")
    print("\n========== 副監視: 販社別(機種×部番×販社) ==========")
    run("sha")
