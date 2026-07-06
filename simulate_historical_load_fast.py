# -*- coding: utf-8 -*-
"""
simulate_historical_load_fast.py
================================
simulate_historical_load.py と同じ結果を、40機種規模でも現実的な時間で返す高速版。
インターフェースは元と同じ（compare_dispositions / simulate_monthly_load）。

なぜ速いか
----------
元の simulate_monthly_load は「月ごとに evaluate_units を丸ごと呼び直す」ため、
台帳に依存しない重い前処理を毎月×両処置ぶん作り直していた:
  - build_earlylife_curves(...)  … 先行機種カーブの経験ベイズ推定（最重量）
  - _pooled_lambda0(...)         … プール平均
  - units_df.groupby([...])      … パネル全体の機種分割
  - estimate_baseline / earlylife_lambda0 … 各機種の基準（自機種データのみ依存）
これらはパネルが固定ならシミュレーション中ずっと不変なので、ループの外で
一度だけ計算して使い回す。月ごとに変わるのは「台帳（リセット注入）」だけなので、
各月では unit_events → build_plan → replay_unit → resolve_state と、件数集計だけを回す。

出力の一致
----------
monthly の各列（件数/発火/再評価/保留継続/対象単位数）は元と一致する。
evaluate_units + build_inbox の判定ロジックをそのまま写しているため。
verify_fast.py で実データ/テストデータの差0を確認できる。

前提
----
本ツールが注入する処置は "ノイズ" / "対策中" のみ（元ツールと同じ）。どちらも
リセットは起こすが「新ベースライン差し替え」はしないので、各機種の base（lambda0）は
台帳に依存せず不変＝先に確定してよい。この2つ以外を assumed_disposition に渡すと
前提が崩れるためガードしている。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import state_logic_cusum as s
from simulate_historical_load import LEDGER_COLS  # 台帳列は元と共通にする


def _prep_static(units_df: pd.DataFrame, cfg: dict):
    """台帳に依存しない不変物を一度だけ用意する。"""
    pooled = s._pooled_lambda0(units_df, cfg)
    levels = s.build_earlylife_curves(units_df, cfg)
    unit_map: dict = {}
    base_map: dict = {}
    mode_map: dict = {}
    for key, u in units_df.groupby(["biz", "dev", "part"], sort=False):
        u = u.sort_values("ym").reset_index(drop=True)  # カーブ配列と行を対応
        unit_map[key] = u
        if s.classify_mode(u, cfg) == "安定期":
            base_map[key] = s.estimate_baseline(u, cfg)      # (lambda0, C, E)
            mode_map[key] = "安定期"
        else:
            lam0, mode_label, _, _ = s.earlylife_lambda0(u, levels, pooled, cfg)
            base_map[key] = (lam0, None, None)
            mode_map[key] = mode_label
    return unit_map, base_map, mode_map


def _eval_unit_at(key, unit, base, mode_label, ledger, cfg, asof, machine_end):
    """evaluate_units の1機種ぶんを、precompute した base/mode を使って再現する。"""
    biz, dev, part = key
    events = s.unit_events(ledger, biz, dev, part, asof)
    plan = s.build_plan(events, unit, base, cfg)
    rep = s.replay_unit(unit, base, mode_label, plan, cfg)
    if rep.empty:
        return None, events
    states = [s.resolve_state(events, bool(r.total_alert), int(r.ym),
                              machine_end.get((biz, dev)), plan)
              for r in rep.itertuples()]
    rep = rep.assign(state=[st["state"] for st in states],
                     reevaluation_due=[st["reevaluation_due"] for st in states])
    return rep, events


def simulate_monthly_load(units_df: pd.DataFrame, cfg: dict,
                          start_ym: int | None = None, end_ym: int | None = None,
                          assumed_disposition: str = "ノイズ",
                          reeval_months: int = 3,
                          verbose: bool = True,
                          collect_details: bool = False,
                          _static=None):
    """元 simulate_monthly_load の高速版。戻り値は (monthly, details, ledger)。

    collect_details=False（既定）のときは O(機種数)/月で件数だけを積むため details は
    空DataFrame。内訳行が要る場合だけ True にする（その分やや遅くなる）。
    _static を渡すと _prep_static の再計算を省ける（compare_dispositions が利用）。
    """
    if assumed_disposition not in ("ノイズ", "対策中"):
        raise ValueError("この高速版は assumed_disposition='ノイズ' または '対策中' 前提です。"
                         "他区分はbaseの不変前提が崩れるため元のsimulate_historical_loadを使ってください。")

    unit_map, base_map, mode_map = _static if _static is not None else _prep_static(units_df, cfg)
    machine_end: dict = {}  # 機種終了系は注入しないので常に空・不変

    ledger = pd.DataFrame(columns=LEDGER_COLS)
    yms = sorted(units_df["ym"].unique())
    if start_ym is not None:
        yms = [y for y in yms if y >= start_ym]
    if end_ym is not None:
        yms = [y for y in yms if y <= end_ym]

    monthly_rows, detail_frames = [], []
    for ym in yms:
        n_fire = n_reeval = n_hold = 0
        n_meta = 0
        fired_keys, detail_rows = [], []
        for key, unit in unit_map.items():
            rep, events = _eval_unit_at(key, unit, base_map[key], mode_map[key],
                                        ledger, cfg, ym, machine_end)
            if rep is None:
                continue
            n_meta += 1  # len(meta) 相当（rep非空の機種数）
            cur = rep[rep["ym"] == ym]
            if cur.empty:
                continue
            cur = cur.iloc[0]
            if str(cur["state"]).startswith("終了"):
                continue

            latest = events[-1] if events else None
            latest_beh = s.DISPOSITIONS.get(latest.get("処置区分"), {}) if latest else {}
            recorded_this_month = bool(latest) and int(latest["判定年月"]) == ym
            is_hold = latest_beh.get("hold", False)
            fired = bool(cur["total_alert"])
            reeval_due = bool(cur["reevaluation_due"])

            reason = None
            if fired and not (recorded_this_month and not is_hold):
                reason = "発火"
            elif is_hold:
                reason = "保留継続"
            elif reeval_due and not recorded_this_month:
                reason = "再評価"
            if reason is None:
                continue

            if reason == "発火":
                n_fire += 1
                fired_keys.append(key)
            elif reason == "再評価":
                n_reeval += 1
            else:
                n_hold += 1

            if collect_details:
                detail_rows.append(dict(
                    シミュレーション年月=ym, 事業コード=key[0], 開発コード=key[1], 部番=key[2],
                    載る理由=reason, 判定種別=("再" if events else "初回"),
                    S=round(float(cur["S"]), 3), 観測使用数=int(cur["use"]),
                    期待故障数=round(float(cur["mu0"]), 3), 経過月=cur["elapsed"]))

        n_total = n_fire + n_reeval + n_hold
        monthly_rows.append(dict(年月=ym, 件数=n_total, 発火=n_fire, 再評価=n_reeval,
                                 保留継続=n_hold, 対象単位数=n_meta))
        if collect_details and detail_rows:
            detail_frames.append(pd.DataFrame(detail_rows))

        # 「発火」だけその場で台帳に記録し、次月以降の再着火を抑える（元と同じ）
        if fired_keys:
            new_rows = []
            for (biz, dev, part) in fired_keys:
                row = dict.fromkeys(LEDGER_COLS, pd.NA)
                row.update(事業コード=biz, 開発コード=dev, 部番=part,
                           判定年月=ym, 記録日=ym, 処置区分=assumed_disposition)
                if assumed_disposition == "対策中":
                    row["再評価年月"] = s._add_months(ym, reeval_months)
                new_rows.append(row)
            ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)

    monthly = pd.DataFrame(monthly_rows)
    details = (pd.concat(detail_frames, ignore_index=True) if detail_frames
               else pd.DataFrame())
    if verbose and not monthly.empty:
        nz = monthly[monthly["件数"] > 0]
        print(f"[{assumed_disposition}前提] 対象月数={len(monthly)} "
              f"平均件数/月={monthly['件数'].mean():.2f} "
              f"最大件数/月={monthly['件数'].max()} "
              f"(0件だった月={len(monthly) - len(nz)}件)")
    return monthly, details, ledger


def compare_dispositions(units_df: pd.DataFrame, cfg: dict,
                         start_ym: int | None = None, end_ym: int | None = None,
                         reeval_months: int = 3) -> pd.DataFrame:
    """元 compare_dispositions の高速版。不変物は両処置で1回だけ用意して共有する。"""
    static = _prep_static(units_df, cfg)   # ← 2処置で使い回す（元は月×2で作り直していた）
    out = {}
    for disp in ("ノイズ", "対策中"):
        monthly, _, _ = simulate_monthly_load(
            units_df, cfg, start_ym, end_ym, assumed_disposition=disp,
            reeval_months=reeval_months, verbose=False, _static=static)
        out[disp] = monthly.set_index("年月")["件数"]
    cmp = pd.DataFrame(out)
    cmp.columns = [f"件数({c}前提)" for c in cmp.columns]
    print(cmp.describe().loc[["mean", "50%", "max"]])
    return cmp.reset_index()
