# -*- coding: utf-8 -*-
"""
verify_capped.py — simulate_capped_triage の検証
 1. top_n=None（無制限）の月次内訳が simulate_historical_load_fast と完全一致するか
    （発火/再評価/保留継続/対象単位数。判定ロジックの忠実性の確認）
 2. 増分キャッシュ版と「毎月フルリプレイ」版で、月次・台帳・捕捉月が一致するか
    （キャッシュ最適化がロジックを変えていないことの確認）
両方 ノイズ / 対策中 の2前提で回す。
"""
import pandas as pd
import state_logic_cusum as s
from simulate_historical_load_fast import simulate_monthly_load as fast_sim
from simulate_capped_triage import (simulate_capped_triage, prep_static,
                                    _replay_to_cache, build_initial_caches)
import simulate_capped_triage as sc

cfg = dict(s.CONFIG)
cfg.update(R=1.5, h=5.0, alpha_spike=0.005, min_count=3, burst_window=0)

panel = s._prepare_panel(pd.read_csv("test_panel.csv"), cfg)
units = s.aggregate_units(panel)
labels = pd.read_csv("test_labels.csv", encoding="utf-8-sig")

ok_all = True

# ---------------------------------------------------------------- 検証1
print("=" * 72)
print("検証1: top_n=None が simulate_historical_load_fast と一致するか")
print("=" * 72)
for disp in ("ノイズ", "対策中"):
    m_fast, _, led_fast = fast_sim(units, cfg, assumed_disposition=disp, verbose=False)
    m_cap, _, _, led_cap = simulate_capped_triage(
        units, cfg, top_n=None, assumed_disposition=disp, verbose=False)

    a = m_fast[["年月", "件数", "発火", "再評価", "保留継続", "対象単位数"]].reset_index(drop=True)
    b = m_cap.rename(columns={"要対応計": "件数", "処理": "発火"})[
        ["年月", "件数", "発火", "再評価", "保留継続", "対象単位数"]].reset_index(drop=True)
    diff = (a != b).to_numpy().sum()
    def _norm(led):
        d = led[["事業コード", "開発コード", "部番", "判定年月", "処置区分"]].astype(str)
        return d.sort_values(["判定年月", "開発コード", "部番"]).reset_index(drop=True)
    led_diff = 0 if _norm(led_fast).equals(_norm(led_cap)) else 1
    ok = (diff == 0 and led_diff == 0)
    ok_all &= ok
    print(f"  [{disp}前提] 月次セル差分={diff} 台帳一致={'○' if led_diff==0 else '×'}"
          f" → {'OK' if ok else 'NG'}")

# ---------------------------------------------------------------- 検証2
print("\n" + "=" * 72)
print("検証2: 増分キャッシュ = 毎月フルリプレイ（参照実装）と一致するか")
print("=" * 72)


def reference_capped(units_df, cfg, top_n, disp, reeval_months=3, lookback_m=6):
    """毎月・全単位をフルリプレイする素朴版（遅いが確実）。"""
    static = prep_static(units_df, cfg, verbose=False)
    unit_map, base_map, mode_map = static
    events_by_key = {k: [] for k in unit_map}
    watch_m, inbox_m, triage_m = {}, {}, {}
    yms = sorted(int(v) for v in units_df["ym"].unique())
    monthly_rows, ledger_rows = [], []
    for ym in yms:
        cands, n_re, n_ho, n_wa, n_me = [], 0, 0, 0, 0
        for key, unit in unit_map.items():
            cache = _replay_to_cache(key, unit, base_map[key], mode_map[key],
                                     events_by_key[key], cfg)  # 毎回フル
            if cache is None:
                continue
            n_me += 1
            idx = cache["ym_idx"].get(ym)
            if idx is None:
                continue
            events = events_by_key[key]
            st = s.resolve_state(events, bool(cache["total_alert"][idx]), ym,
                                 None, cache["plan"])
            if str(st["state"]).startswith("終了"):
                continue
            latest = events[-1] if events else None
            beh = s.DISPOSITIONS.get(latest.get("処置区分"), {}) if latest else {}
            rec = bool(latest) and int(latest["判定年月"]) == ym
            hold = beh.get("hold", False)
            fired = bool(cache["total_alert"][idx])
            attn = s.attention_score(cache["S"][idx], cache["h"][idx],
                                     cache["p_spike"][idx], cache["p_burst"][idx], cfg)
            if fired and not (rec and not hold):
                cands.append((attn, float(cache["S"][idx]), key, idx))
                inbox_m.setdefault(key, []).append(ym)
            elif hold:
                n_ho += 1
            elif bool(st["reevaluation_due"]) and not rec:
                n_re += 1
            elif attn >= 0.5:
                n_wa += 1
                watch_m.setdefault(key, []).append(ym)
        cands.sort(key=lambda t: (-t[0], -t[1], t[2]))
        cut = len(cands) if top_n is None else min(top_n, len(cands))
        for attn, S_, key, idx in cands[:cut]:
            triage_m.setdefault(key, []).append(ym)
            ev = dict.fromkeys(sc.LEDGER_COLS, pd.NA)
            ev.update(事業コード=key[0], 開発コード=key[1], 部番=key[2],
                      判定年月=ym, 記録日=ym, 処置区分=disp)
            if disp == "対策中":
                ev["再評価年月"] = s._add_months(ym, reeval_months)
            events_by_key[key].append(ev)
            ledger_rows.append(ev)
        monthly_rows.append(dict(年月=ym, 発火候補=len(cands), 処理=cut,
                                 積み残し=len(cands) - cut, 再評価=n_re,
                                 保留継続=n_ho, 二軍水準=n_wa))
    return (pd.DataFrame(monthly_rows),
            pd.DataFrame(ledger_rows, columns=sc.LEDGER_COLS),
            watch_m, inbox_m, triage_m)


for disp in ("ノイズ", "対策中"):
    for N in (1, 2, None):
        m_ref, led_ref, w_ref, i_ref, t_ref = reference_capped(units, cfg, N, disp)
        m_cap, inc, _, led_cap = simulate_capped_triage(
            units, cfg, top_n=N, assumed_disposition=disp,
            labels=labels, verbose=False)
        cols = ["年月", "発火候補", "処理", "積み残し", "再評価", "保留継続", "二軍水準"]
        diff = (m_ref[cols].reset_index(drop=True)
                != m_cap[cols].reset_index(drop=True)).to_numpy().sum()
        la = led_ref[["開発コード", "部番", "判定年月"]].astype(str).reset_index(drop=True)
        lb = led_cap[["開発コード", "部番", "判定年月"]].astype(str).reset_index(drop=True)
        led_ok = la.equals(lb)
        ok = (diff == 0 and led_ok)
        ok_all &= ok
        tag = "無制限" if N is None else N
        print(f"  [{disp} / N={tag}] 月次セル差分={diff} "
              f"台帳一致={'○' if led_ok else '×'} → {'OK' if ok else 'NG'}")

# ---------------------------------------------------------------- 結果例
print("\n" + "=" * 72)
print("参考: インシデント別の捕捉（対策中前提, N=1）")
print("=" * 72)
_, inc, _, _ = simulate_capped_triage(units, cfg, top_n=1,
                                      assumed_disposition="対策中",
                                      labels=labels, verbose=False)
with pd.option_context("display.width", 220, "display.max_columns", 25):
    print(inc.to_string(index=False))

print("\n" + ("★ 全検証 OK" if ok_all else "★ NG があります — 上を確認"))
