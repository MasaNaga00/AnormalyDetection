# -*- coding: utf-8 -*-
"""
diagnose_detection_gap.py
=========================
「backtest の backtest_labeled では検知できているラベルが、
 compare_spike_onoff / simulate_capped_triage では未検知になる」
という食い違いの原因を、ラベル1件ずつ切り分ける。

両者は「検知」を別の条件で測っている。切り分けるべき軸は3つ:

  (1) リセット/台帳: backtest は台帳リセットなし・reset_after_alarm=True の
      素の系列で「一度でも発火したか」。simulate は台帳リセット注入・
      reset_after_alarm=False で本番運用を月送り再現（発火→処置でSが0から再出発）。
  (2) 消し込み: simulate は assumed_disposition="対策中" だと発火後 reeval まで沈黙。
  (3) スパイク: compare のオフ側は spike_test を切っている。ラベルが spike 先行なら
      オフで未検知になる（＝バグでなく、まさに知りたい答え）。

各ラベルについて、次の4条件で「検知できるか・最初の発火月・種別」を並べる:

  A. backtest相当   : リセットなし・reset_after_alarm=True・スパイクオン
                      （＝backtest_labeled と同じ土俵。ここが基準）
  B. 素+オフ         : リセットなし・reset_after_alarm=True・スパイクオフ
                      （A→Bで消えたら (3)スパイク起因）
  C. 本番相当+オン   : 台帳リセット注入・reset_after_alarm=False・スパイクオン
                      （A→Cで消えたら (1)リセット/運用起因）
  D. 本番相当+オフ   : 台帳リセット注入・reset_after_alarm=False・スパイクオフ
                      （＝compare_spike_onoff のオフ側と同じ土俵）

D で消えているラベルが、A→B→C→D のどこで落ちたかを見れば原因が確定する。
"""
import numpy as np
import pandas as pd
import state_logic_cusum as s
import cusum_monitor as cm
from backtest_cusum import prepare_basis, _drift_run, _spike_run
from simulate_capped_triage import simulate_capped_triage, month_diff


def _first_alarm_bare(b, cfg, R, h, spike_on, lookback_onset):
    """素の系列（リセットなし・reset_after_alarm=True）で最初の発火月と種別。
    lookback_onset: 探索開始index。"""
    S, ad, k = cm.poisson_cusum(b["use"], b["fleet"], b["lam"], R, h,
                                reset_after_alarm=True)
    alarm = np.asarray(ad, dtype=bool).copy()
    kinds = np.where(alarm, "drift", "")
    if spike_on:
        a_s, a_b = _spike_run(b["use"], b["fleet"], b["lam"], b["C"], b["E"],
                              cfg["alpha_spike"], cfg["min_count"],
                              cfg["burst_window"], None)
        sp = (a_s | a_b) if cfg["burst_window"] >= 2 else a_s
        for i in range(len(alarm)):
            if sp[i]:
                kinds[i] = (kinds[i] + "+spike") if kinds[i] else "spike"
        alarm = alarm | sp
    idx = np.flatnonzero(alarm[lookback_onset:])
    if idx.size == 0:
        return None, None
    at = lookback_onset + int(idx[0])
    return int(b["ym"][at]), kinds[at]


def diagnose(panel_path, labels_path, cfg_base, lookback_m=6,
             assumed_disposition="対策中"):
    labels = pd.read_csv(labels_path, encoding="utf-8-sig")
    cfg_on = dict(cfg_base)
    cfg_off = dict(cfg_base); cfg_off["alpha_spike"] = None
    R, h = cfg_on["R"], cfg_on["h"]

    panel = s._prepare_panel(pd.read_csv(panel_path), cfg_on)
    units = s.aggregate_units(panel)
    basis = prepare_basis(units, cfg_on)
    bmap = {b["key"]: b for b in basis}

    # C, D（本番相当・月送り）はシミュレータで一括取得
    _, inc_on, _, _ = simulate_capped_triage(
        units, cfg_on, top_n=None, assumed_disposition=assumed_disposition,
        labels=labels, lookback_m=lookback_m, verbose=False)
    _, inc_off, _, _ = simulate_capped_triage(
        units, cfg_off, top_n=None, assumed_disposition=assumed_disposition,
        labels=labels, lookback_m=lookback_m, verbose=False)

    def _ib_map(inc):
        m = {}
        for _, r in inc.iterrows():
            m[(r["事業コード"], r["開発コード"], r["部番"], int(r["販社報告月"]))] = \
                (r["検知"] == "あり", r["インボックス入り月"])
        return m
    C_map, D_map = _ib_map(inc_on), _ib_map(inc_off)

    rows = []
    for _, r in labels.iterrows():
        key = (r["事業コード"], r["開発コード"], r["部番"])
        onset_ym = s.to_yyyymm(r["発生年月"])
        b = bmap.get(key)
        rec = dict(開発コード=key[1], 部番=key[2], 販社報告月=onset_ym)
        if b is None:
            rec.update(A_backtest="対象外", B_素オフ="対象外",
                       C_本番オン="対象外", D_本番オフ="対象外", 原因="監視対象外")
            rows.append(rec); continue

        # onset の系列内index
        pos = np.flatnonzero(b["ym"] >= onset_ym)
        onset_idx = int(pos[0]) if pos.size else 0
        search_from = max(0, onset_idx - int(lookback_m))

        a_ym, a_kind = _first_alarm_bare(b, cfg_on, R, h, True, search_from)
        b_ym, b_kind = _first_alarm_bare(b, cfg_on, R, h, False, search_from)
        c_ok, c_ym = C_map.get((key[0], key[1], key[2], onset_ym), (False, None))
        d_ok, d_ym = D_map.get((key[0], key[1], key[2], onset_ym), (False, None))

        def fmt(ym, kind=None):
            if ym is None:
                return "×"
            return f"{ym}" + (f"({kind})" if kind else "")

        rec["A_backtest"] = fmt(a_ym, a_kind)
        rec["B_素オフ"] = fmt(b_ym, b_kind)
        rec["C_本番オン"] = fmt(c_ym if c_ok else None)
        rec["D_本番オフ"] = fmt(d_ym if d_ok else None)

        # 原因判定: A検知ありを基準に、どこで最初に落ちたか
        a_ok = a_ym is not None
        if not a_ok:
            cause = "backtestでも未検知(基準外)"
        elif a_ok and b_ym is None:
            cause = "★スパイク起因(素の系列でもオフで消える)"
        elif a_ok and not c_ok:
            cause = "★リセット/運用起因(本番リプレイで消える)"
        elif c_ok and not d_ok:
            cause = "★スパイク起因(本番リプレイ上でオフにすると消える)"
        elif d_ok:
            cause = "D でも検知（食い違い無し）"
        else:
            cause = "要確認"
        rec["原因"] = cause
        rows.append(rec)

    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print("=" * 100)
        print("検知食い違いの切り分け（A=backtest土俵 → D=compareオフ土俵）")
        print("列の値 = 最初の発火年月(種別)。× = 未検知")
        print("=" * 100)
        print(out.to_string(index=False))
    print("\n--- 原因の集計 ---")
    print(out["原因"].value_counts().to_string())
    return out


if __name__ == "__main__":
    cfg = dict(s.CONFIG)
    cfg.update(R=1.5, h=5.0, alpha_spike=0.005, min_count=3, burst_window=0)
    diagnose("scale_panel.csv", "scale_labels.csv", cfg)
