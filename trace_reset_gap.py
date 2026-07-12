# -*- coding: utf-8 -*-
"""
trace_reset_gap.py
==================
diagnose_detection_gap で「★リセット/運用起因」と出たラベルについて、
本番リプレイ（月送り）でそのラベル単位が各月どうなっていたかを1行=1月で並べ、
「なぜ発火がインボックス検知に結びつかなかったか」を確定する。

素の系列(A)では発火するのに本番(C)で消える経路は主に3つ:
  (1) 消し込み沈黙 : 先に発火→"対策中"記録→reeval_monthsまで沈黙。ラベル月が沈黙窓。
  (2) リセットでS低下: 発火→リセット→S=0再出発。緩ドリフトだと再点灯前にlookback窓終了。
  (3) 状態吸収      : 発火してるが resolve_state で「発火」以外(保留継続/再評価/終了)に分類。

トレース列:
  発火 : その月 total_alert が立ったか（素の検出そのもの）
  載る理由 : simulate 内の分類（発火/再評価/保留継続/載らず）
  状態 : resolve_state の状態
  沈黙原因 : 直近の台帳記録と reeval による沈黙か
  S / mu0 / use : 数値の裏取り
これを見れば (1)(2)(3) のどれで落ちたか一目で分かる。
"""
import numpy as np
import pandas as pd
import state_logic_cusum as s
from simulate_capped_triage import (prep_static, build_initial_caches,
                                    _replay_to_cache, month_diff)


def trace_labels(panel_path, labels_path, cfg_base, target_keys=None,
                 lookback_m=6, assumed_disposition="対策中", reeval_months=3,
                 window_before=8, window_after=4):
    """target_keys: [(biz,dev,part), ...]。Noneなら全ラベルをトレース。
    ラベルの報告月まわり [報告月-window_before, 報告月+window_after] を月送り再現し、
    対象単位の毎月の様子を出す。"""
    labels = pd.read_csv(labels_path, encoding="utf-8-sig")
    cfg = dict(cfg_base)
    panel = s._prepare_panel(pd.read_csv(panel_path), cfg)
    units = s.aggregate_units(panel)

    static = prep_static(units, cfg, verbose=False)
    unit_map, base_map, mode_map = static

    # 月送りシミュレーション（simulate_capped_triage と同じ挙動）を、
    # 対象単位のトレースを取りながら再現する。
    lab = labels.copy()
    lab["発生年月"] = lab["発生年月"].map(s.to_yyyymm)
    if target_keys is None:
        target_keys = [(r["事業コード"], r["開発コード"], r["部番"])
                       for _, r in lab.iterrows()]
    target_set = set(target_keys)
    onset_by_key = {(r["事業コード"], r["開発コード"], r["部番"]): int(r["発生年月"])
                    for _, r in lab.iterrows()}

    caches = build_initial_caches(unit_map, base_map, mode_map, cfg, verbose=False)
    events_by_key = {k: [] for k in unit_map}
    trace = {k: [] for k in target_set}

    yms = sorted(int(v) for v in units["ym"].unique())
    for ym in yms:
        fired_cands = []
        for key, cache in caches.items():
            if cache is None:
                continue
            idx = cache["ym_idx"].get(ym)
            if idx is None:
                continue
            events = events_by_key[key]
            st = s.resolve_state(events, bool(cache["total_alert"][idx]), ym,
                                 None, cache["plan"])
            latest = events[-1] if events else None
            latest_beh = s.DISPOSITIONS.get(latest.get("処置区分"), {}) if latest else {}
            recorded_this_month = bool(latest) and int(latest["判定年月"]) == ym
            is_hold = latest_beh.get("hold", False)
            fired = bool(cache["total_alert"][idx])
            reeval_due = bool(st["reevaluation_due"])

            reason = "載らず"
            if fired and not (recorded_this_month and not is_hold):
                reason = "発火"
            elif is_hold:
                reason = "保留継続"
            elif reeval_due and not recorded_this_month:
                reason = "再評価"

            if reason == "発火":
                attn = s.attention_score(cache["S"][idx], cache["h"][idx],
                                         cache["p_spike"][idx], cache["p_burst"][idx], cfg)
                fired_cands.append((attn, float(cache["S"][idx]), key, idx))

            # 対象単位ならトレース行を残す
            if key in target_set:
                onset = onset_by_key.get(key)
                if onset is None or (month_diff(ym, onset) >= -window_before
                                     and month_diff(ym, onset) <= window_after):
                    # 直近記録と沈黙原因
                    silence = ""
                    if recorded_this_month and not is_hold:
                        silence = "当月記録でリセット"
                    elif latest is not None:
                        rev = latest.get("再評価年月")
                        if pd.notna(rev) and int(latest["判定年月"]) < ym <= int(rev):
                            silence = f"対策中沈黙({latest['判定年月']}記録,{int(rev)}まで)"
                    trace[key].append(dict(
                        年月=ym, 報告差=month_diff(ym, onset) if onset else None,
                        発火=("○" if fired else ""),
                        種別=s._alert_kind(cache["alert_drift"][idx],
                                           cache["alert_spike"][idx],
                                           cache["alert_burst"][idx]),
                        載る理由=reason, 状態=st["state"],
                        沈黙原因=silence,
                        S=round(float(cache["S"][idx]), 2),
                        h=round(float(cache["h"][idx]), 1),
                        use=int(cache["use"][idx]),
                        mu0=round(float(cache["mu0"][idx]), 2)))

        # top_n=None 相当（全件処理）で台帳注入＋再リプレイ
        for attn, S_, key, idx in fired_cands:
            ev = dict.fromkeys(
                ["事業コード", "開発コード", "部番", "判定年月", "記録日", "処置区分",
                 "再評価年月", "上書きR", "上書きh", "新ベースライン値",
                 "ベースライン窓起点", "ベースライン窓長", "原因メモ", "確認者"], pd.NA)
            ev.update(事業コード=key[0], 開発コード=key[1], 部番=key[2],
                      判定年月=ym, 記録日=ym, 処置区分=assumed_disposition)
            if assumed_disposition == "対策中":
                ev["再評価年月"] = s._add_months(ym, reeval_months)
            events_by_key[key].append(ev)
            caches[key] = _replay_to_cache(key, unit_map[key], base_map[key],
                                           mode_map[key], events_by_key[key], cfg)

    # 出力
    for key in target_keys:
        onset = onset_by_key.get(key)
        floor = s._add_months(onset, -lookback_m) if onset else None
        print("=" * 96)
        print(f"{key}  販社報告月={onset}  検知探索の起点={floor}（報告{lookback_m}ヶ月前）")
        print("=" * 96)
        df = pd.DataFrame(trace[key])
        if df.empty:
            print("  （トレース行なし＝監視レンジ外の可能性）")
            continue
        # lookback窓内で「発火したのに載らなかった」月を強調
        df["窓内"] = df["年月"].map(
            lambda y: "◆" if (floor is not None and y >= floor
                              and (onset is None or y <= onset)) else "")
        with pd.option_context("display.width", 220, "display.max_columns", 20):
            print(df.to_string(index=False))
        # 診断コメント
        win = df[(df["窓内"] == "◆")]
        fired_in_win = win[win["発火"] == "○"]
        loaded_in_win = win[win["載る理由"] == "発火"]
        print(f"\n  窓内で素に発火した月: {len(fired_in_win)} / "
              f"うちインボックス検知に至った月: {len(loaded_in_win)}")
        if len(fired_in_win) and not len(loaded_in_win):
            reasons = win[win["発火"] == "○"]["沈黙原因"].value_counts()
            print("  → 発火はしたが載らなかった。沈黙原因内訳:")
            print("    " + reasons.to_string().replace("\n", "\n    "))
        print()


if __name__ == "__main__":
    cfg = dict(s.CONFIG)
    cfg.update(R=1.5, h=5.0, alpha_spike=0.005, min_count=3, burst_window=0)
    # デモ: 合成データの全ラベルのうち最初の3件をトレース
    labels = pd.read_csv("scale_labels.csv", encoding="utf-8-sig")
    keys = [(r["事業コード"], r["開発コード"], r["部番"])
            for _, r in labels.head(3).iterrows()]
    trace_labels("scale_panel.csv", "scale_labels.csv", cfg, target_keys=keys)
