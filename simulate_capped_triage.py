# -*- coding: utf-8 -*-
"""
simulate_capped_triage.py
=========================
「厳しい閾値 ＋ 上位N件triage ＋ 二軍ウォッチリスト」を過去データで月送り再現し、
ラベル済みインシデントがどの段階（ウォッチ入り / インボックス入り / 実際のtriage）で
いつ捕捉できていたかを測るシミュレータ。

simulate_historical_load(_fast) の後継。次の3点が新しい:

 1. **上位N件キャップ**: 各月、発火候補を注目度降順に並べ上位N件だけ台帳処理する。
    残りは注入せず持ち越し（Sは積み上がったままなので翌月も注目度順に再競争する。
    reset_after_alarm=False の設計思想どおり、本物は勝ち上がってくる）。
 2. **二軍ウォッチリスト**: 注目度 >= watch_frac（既定0.5 = ドリフトはS/h半分、
    スパイクはlogスケールでalpha_spikeの半分）を「Tableauで目視可能」な線として
    月ごとに記録する。台帳処理はしない。
 3. **ラベル評価**: 各インシデント（発生年月=販社報告月）について、
    ウォッチ入り月 / インボックス入り月 / top-N処理月と、報告月に対する遅れ月
    （負=先回り成功）を backtest_cusum.measure_labeled と同じ lookback_m 流儀で出す。

―――――――――――――――――――――――――――――――――――――――――――――――
なぜ速いか（simulate_historical_load_fast がスケールしなかった理由）
―――――――――――――――――――――――――――――――――――――――――――――――
fast 版は静的前処理（カーブ推定・ベースライン）は月ループの外に出したが、
**replay_unit（CUSUM全履歴の再計算）を毎月×全単位でやり直していた**。
2000単位×36ヶ月 = 72,000回のフルリプレイで、これが支配項。

しかし replay の結果は「その単位の台帳イベント」が増えた時しか変わらない
（asof再現性で実証済みの性質）。本シミュレーションでは台帳を自分で注入する
ので、どの単位のイベントが増えたか常に分かる。よって:

  - 各単位の replay 結果を numpy 配列キャッシュとして保持
  - 月ループでは ym→行index の O(1) ルックアップ ＋ 当月だけの resolve_state
  - 再リプレイは「その月に台帳注入された単位」だけ（= アラーム件数回）

計算量は O(初期リプレイ 単位数) + O(月数×単位数の軽い判定) + O(注入件数×リプレイ)。
2000単位規模でも初期リプレイ1回分＋αで済む。

判定ロジック（inbox載り理由 / resolve_state / 注目度）は state_logic_cusum の
関数をそのまま呼ぶか忠実に写しており、top_n=None（無制限）にすると
simulate_historical_load_fast の monthly と完全一致する（verify_capped.py で確認）。

前提（fast 版と同じ）
--------------------
- 注入する処置は "ノイズ" / "対策中" のみ。どちらも新ベースライン差し替えを
  しないので base は不変。他区分はガードする。
- machine_end は注入しないので常に空。

使い方
------
    import pandas as pd, state_logic_cusum as s
    from simulate_capped_triage import run_sweep

    cfg = dict(s.CONFIG); cfg.update(R=1.5, h=5.0)
    panel = s._prepare_panel(pd.read_csv("panel.csv"), cfg)
    units = s.aggregate_units(panel)
    labels = pd.read_csv("labels.csv")

    monthly_all, incidents_all = run_sweep(
        units, cfg, n_list=[5, 10, 15, 20, None],
        labels=labels, assumed_disposition="対策中")
    # → capped_月次負荷.csv / capped_インシデント別.csv（Tableau用に縦積み）
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import state_logic_cusum as s

LEDGER_COLS = ["事業コード", "開発コード", "部番", "判定年月", "記録日", "処置区分",
               "再評価年月", "上書きR", "上書きh", "新ベースライン値",
               "ベースライン窓起点", "ベースライン窓長", "原因メモ", "確認者"]

_ALLOWED_DISP = ("ノイズ", "対策中")


# ============================================================================
# 年月ユーティリティ
# ============================================================================
def _ym_to_serial(ym: int) -> int:
    y, m = divmod(int(ym), 100)
    return y * 12 + (m - 1)


def month_diff(ym_a: int, ym_b: int) -> int:
    """ym_a - ym_b を月数で。負 = ym_a が早い。"""
    return _ym_to_serial(ym_a) - _ym_to_serial(ym_b)


def _sub_months(ym: int, n: int) -> int:
    return s._add_months(ym, -n)


# ============================================================================
# 静的前処理（台帳に依存しない不変物。fast版 _prep_static と同一ロジック）
# ============================================================================
def prep_static(units_df: pd.DataFrame, cfg: dict, verbose: bool = True):
    t0 = time.time()
    if verbose:
        print("[前処理] 先行機種カーブ推定・ベースライン確定中 ...", flush=True)
    pooled = s._pooled_lambda0(units_df, cfg)
    levels = s.build_earlylife_curves(units_df, cfg)
    unit_map, base_map, mode_map = {}, {}, {}
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
    if verbose:
        print(f"[前処理] 完了: {len(unit_map)}単位 ({time.time()-t0:.1f}秒)", flush=True)
    return unit_map, base_map, mode_map


# ============================================================================
# リプレイ結果 → numpy キャッシュ
# ============================================================================
def _replay_to_cache(key, unit, base, mode_label, events, cfg):
    """1単位を events でリプレイし、月ルックアップ用の軽量キャッシュにする。
    rep が空（監視レンジ内の月なし）なら None。"""
    plan = s.build_plan(events, unit, base, cfg)
    rep = s.replay_unit(unit, base, mode_label, plan, cfg)
    if rep.empty:
        return None
    ymv = rep["ym"].to_numpy(dtype=np.int64)
    return dict(
        plan=plan,
        ym_idx={int(v): i for i, v in enumerate(ymv)},
        total_alert=rep["total_alert"].to_numpy(dtype=bool),
        alert_drift=rep["alert_drift"].to_numpy(dtype=bool),
        alert_spike=rep["alert_spike"].to_numpy(dtype=bool),
        alert_burst=rep["alert_burst"].to_numpy(dtype=bool),
        S=rep["S"].to_numpy(dtype=float),
        h=rep["h"].to_numpy(dtype=float),
        p_spike=rep["p_spike"].to_numpy(dtype=float),
        p_burst=rep["p_burst"].to_numpy(dtype=float),
        use=rep["use"].to_numpy(dtype=float),
        mu0=rep["mu0"].to_numpy(dtype=float),
        elapsed=rep["elapsed"].to_numpy(),
    )


def build_initial_caches(unit_map, base_map, mode_map, cfg, verbose: bool = True):
    """台帳が空の状態の初期リプレイキャッシュ（全Nスイープで共有可）。"""
    t0 = time.time()
    caches, n = {}, len(unit_map)
    for i, (key, unit) in enumerate(unit_map.items(), 1):
        caches[key] = _replay_to_cache(key, unit, base_map[key], mode_map[key], [], cfg)
        if verbose and (i % 200 == 0 or i == n):
            el = time.time() - t0
            eta = el / i * (n - i)
            print(f"[初期リプレイ] {i}/{n} ({el:.0f}秒経過, 残り目安 {eta:.0f}秒)", flush=True)
    return caches


# ============================================================================
# 本体: キャップ付き月送りシミュレーション
# ============================================================================
def simulate_capped_triage(
    units_df: pd.DataFrame,
    cfg: dict,
    top_n: int | None = 10,
    watch_frac: float = 0.5,
    assumed_disposition: str = "対策中",
    reeval_months: int = 3,
    start_ym: int | None = None,
    end_ym: int | None = None,
    labels: pd.DataFrame | None = None,
    lookback_m: int = 6,
    collect_details: bool = False,
    verbose: bool = True,
    _static=None,
    _init_caches=None,
):
    """月送りで「上位N件triage＋二軍ウォッチ」を再現する。

    Parameters
    ----------
    top_n : 月間triage予算N。発火候補のうち注目度上位N件だけ台帳処理する。
            None なら無制限（= simulate_historical_load_fast と同じ挙動）。
            持ち越し分は注入しないだけ（Sが残るので翌月も注目度順に再競争）。
    watch_frac : 二軍の線。注目度 >= watch_frac をウォッチリスト入りとみなす。
            注目度 = max(S/h, log(p)/log(alpha_spike)) なので 0.5 は
            「ドリフトなら h の半分、スパイクなら log スケールで alpha の半分」。
    labels : 既知インシデント（列: 事業コード/開発コード/部番/発生年月）。
            発生年月 = 販社からの報告月。lookback_m ヶ月前から捕捉を探し、
            遅れ月 = 捕捉月 - 報告月（負 = 先回り成功）。
    collect_details : True で月×単位の内訳行も返す（少し遅くなる）。

    Returns
    -------
    (monthly, incidents, details, ledger_df)
      monthly   : 月次の負荷内訳（発火候補/処理/積み残し/再評価/保留継続/二軍水準数）
      incidents : ラベルごとの捕捉月・遅れ月（labels=None なら空）
      details   : collect_details=True のときの内訳行
      ledger_df : シミュレーションが注入した台帳（検証用）
    """
    if assumed_disposition not in _ALLOWED_DISP:
        raise ValueError(f"assumed_disposition は {_ALLOWED_DISP} のみ"
                         "（他区分は base 不変の前提が崩れる）")
    if top_n is not None and top_n < 0:
        raise ValueError("top_n は None（無制限）か 0 以上")

    unit_map, base_map, mode_map = (_static if _static is not None
                                    else prep_static(units_df, cfg, verbose=verbose))
    caches = dict(_init_caches) if _init_caches is not None else \
        build_initial_caches(unit_map, base_map, mode_map, cfg, verbose=verbose)

    events_by_key: dict[tuple, list[dict]] = {k: [] for k in unit_map}
    watch_months: dict[tuple, list[int]] = {}   # 注目度>=watch_frac だった月
    inbox_months: dict[tuple, list[int]] = {}   # 発火候補としてインボックスに載った月
    triage_months: dict[tuple, list[int]] = {}  # 実際に top-N 処理された月

    yms = sorted(int(v) for v in units_df["ym"].unique())
    if start_ym is not None:
        yms = [y for y in yms if y >= start_ym]
    if end_ym is not None:
        yms = [y for y in yms if y <= end_ym]

    monthly_rows, detail_frames, ledger_rows = [], [], []
    t0 = time.time()
    n_label = "無制限" if top_n is None else str(top_n)

    for mi, ym in enumerate(yms, 1):
        fired_cands = []   # (attention, S, key, cur行の情報)
        n_reeval = n_hold = n_watch = n_meta = 0
        detail_rows = []

        for key, cache in caches.items():
            if cache is None:
                continue
            idx = cache["ym_idx"].get(ym)
            if idx is None:
                # この単位はこの月に監視レンジ内の行が無い。ただし rep 非空なら
                # 元実装の len(meta) 相当（対象単位数）には数える。
                n_meta += 1
                continue
            n_meta += 1

            events = events_by_key[key]
            st = s.resolve_state(events, bool(cache["total_alert"][idx]), ym,
                                 None, cache["plan"])
            if str(st["state"]).startswith("終了"):
                continue

            latest = events[-1] if events else None
            latest_beh = s.DISPOSITIONS.get(latest.get("処置区分"), {}) if latest else {}
            recorded_this_month = bool(latest) and int(latest["判定年月"]) == ym
            is_hold = latest_beh.get("hold", False)
            fired = bool(cache["total_alert"][idx])
            reeval_due = bool(st["reevaluation_due"])

            attn = s.attention_score(cache["S"][idx], cache["h"][idx],
                                     cache["p_spike"][idx], cache["p_burst"][idx], cfg)

            reason = None
            if fired and not (recorded_this_month and not is_hold):
                reason = "発火"
            elif is_hold:
                reason = "保留継続"
            elif reeval_due and not recorded_this_month:
                reason = "再評価"

            if reason == "発火":
                fired_cands.append((attn, float(cache["S"][idx]), key, idx))
                inbox_months.setdefault(key, []).append(ym)
            elif reason == "保留継続":
                n_hold += 1
            elif reason == "再評価":
                n_reeval += 1
            else:
                # インボックスに載らない月。二軍（ウォッチ水準）かだけ見る。
                if attn >= watch_frac:
                    n_watch += 1
                    watch_months.setdefault(key, []).append(ym)

            if collect_details and reason is not None:
                detail_rows.append(dict(
                    シミュレーション年月=ym, 事業コード=key[0], 開発コード=key[1],
                    部番=key[2], 載る理由=reason,
                    判定種別=("再" if events else "初回"),
                    注目度=round(attn, 3), S=round(float(cache["S"][idx]), 3),
                    観測使用数=int(cache["use"][idx]),
                    期待故障数=round(float(cache["mu0"][idx]), 3),
                    経過月=cache["elapsed"][idx]))

        # --- 上位N件だけ処理（注目度降順、同点はS降順→キーで決定的に） ---
        fired_cands.sort(key=lambda t: (-t[0], -t[1], t[2]))
        cut = len(fired_cands) if top_n is None else min(top_n, len(fired_cands))
        processed = fired_cands[:cut]
        carried = fired_cands[cut:]

        for attn, S_, key, idx in processed:
            triage_months.setdefault(key, []).append(ym)
            ev = dict.fromkeys(LEDGER_COLS, pd.NA)
            ev.update(事業コード=key[0], 開発コード=key[1], 部番=key[2],
                      判定年月=ym, 記録日=ym, 処置区分=assumed_disposition)
            if assumed_disposition == "対策中":
                ev["再評価年月"] = s._add_months(ym, reeval_months)
            events_by_key[key].append(ev)
            ledger_rows.append(ev)
            # 台帳イベントが増えた単位だけ再リプレイ（ここが増分キャッシュの肝）
            caches[key] = _replay_to_cache(
                key, unit_map[key], base_map[key], mode_map[key],
                events_by_key[key], cfg)

        monthly_rows.append(dict(
            年月=ym, 発火候補=len(fired_cands), 処理=len(processed),
            積み残し=len(carried), 再評価=n_reeval, 保留継続=n_hold,
            要対応計=len(fired_cands) + n_reeval + n_hold,
            二軍水準=n_watch, 対象単位数=n_meta))
        if collect_details and detail_rows:
            detail_frames.append(pd.DataFrame(detail_rows))

        if verbose:
            el = time.time() - t0
            eta = el / mi * (len(yms) - mi)
            print(f"[N={n_label}] {mi}/{len(yms)}ヶ月 {ym}: "
                  f"発火{len(fired_cands)} 処理{len(processed)} 積み残し{len(carried)} "
                  f"({el:.0f}秒経過, 残り目安 {eta:.0f}秒)", flush=True)

    monthly = pd.DataFrame(monthly_rows)
    details = (pd.concat(detail_frames, ignore_index=True)
               if detail_frames else pd.DataFrame())
    ledger_df = pd.DataFrame(ledger_rows, columns=LEDGER_COLS)

    incidents = (_measure_incidents(labels, watch_months, inbox_months,
                                    triage_months, lookback_m, yms)
                 if labels is not None else pd.DataFrame())
    return monthly, incidents, details, ledger_df


# ============================================================================
# ラベル評価
# ============================================================================
def _first_at_or_after(months: list[int] | None, floor_ym: int) -> int | None:
    if not months:
        return None
    for m in months:  # 記録順 = 昇順
        if m >= floor_ym:
            return m
    return None


def _measure_incidents(labels, watch_months, inbox_months, triage_months,
                       lookback_m, yms):
    """各ラベルの捕捉段階と遅れ月。backtest_cusum.measure_labeled と同じ流儀:
    報告月の lookback_m ヶ月前から捕捉を探し、遅れ月 = 捕捉月 - 報告月（負=先回り）。"""
    lab = labels.copy()
    lab["発生年月"] = lab["発生年月"].map(s.to_yyyymm)
    rows = []
    for _, r in lab.iterrows():
        key = (r["事業コード"], r["開発コード"], r["部番"])
        rep_ym = int(r["発生年月"])
        floor_ym = _sub_months(rep_ym, lookback_m)

        w = _first_at_or_after(watch_months.get(key), floor_ym)
        ib = _first_at_or_after(inbox_months.get(key), floor_ym)
        tr = _first_at_or_after(triage_months.get(key), floor_ym)
        # ウォッチはインボックス入りより後なら意味が無いので前段だけ採用
        if w is not None and ib is not None and w > ib:
            w = ib

        rows.append({
            "事業コード": key[0], "開発コード": key[1], "部番": key[2],
            "販社報告月": rep_ym,
            "ウォッチ入り月": w, "インボックス入り月": ib, "top-N処理月": tr,
            "ウォッチ遅れ月": (month_diff(w, rep_ym) if w is not None else None),
            "インボックス遅れ月": (month_diff(ib, rep_ym) if ib is not None else None),
            "処理遅れ月": (month_diff(tr, rep_ym) if tr is not None else None),
            "キャップ起因の追加遅れ": (month_diff(tr, ib)
                                       if (tr is not None and ib is not None) else None),
            "検知": ("あり" if ib is not None else "なし"),
            "処理": ("あり" if tr is not None else "なし"),
            "販社より早い": ("○" if (tr is not None and month_diff(tr, rep_ym) < 0)
                             else "×"),
        })
    return pd.DataFrame(rows)


# ============================================================================
# N スイープ（静的物と初期リプレイを共有して回す）
# ============================================================================
def run_sweep(units_df: pd.DataFrame, cfg: dict, n_list: list,
              labels: pd.DataFrame | None = None,
              watch_frac: float = 0.5,
              assumed_disposition: str = "対策中",
              reeval_months: int = 3, lookback_m: int = 6,
              start_ym: int | None = None, end_ym: int | None = None,
              out_monthly: str = "capped_月次負荷.csv",
              out_incidents: str = "capped_インシデント別.csv",
              verbose: bool = True):
    """N を振って月次負荷とインシデント捕捉を縦積みで出す（Tableau 用）。

    n_list の None は無制限。静的前処理と初期リプレイは全Nで1回だけ計算して共有する
    （台帳注入で差が付くのは注入後の単位だけで、初期キャッシュは不変のため）。
    """
    static = prep_static(units_df, cfg, verbose=verbose)
    init_caches = build_initial_caches(*static, cfg, verbose=verbose)

    monthly_all, incident_all = [], []
    for n in n_list:
        tag = "無制限" if n is None else int(n)
        if verbose:
            print(f"\n===== N = {tag} =====", flush=True)
        monthly, incidents, _, _ = simulate_capped_triage(
            units_df, cfg, top_n=n, watch_frac=watch_frac,
            assumed_disposition=assumed_disposition, reeval_months=reeval_months,
            start_ym=start_ym, end_ym=end_ym, labels=labels, lookback_m=lookback_m,
            verbose=verbose, _static=static, _init_caches=init_caches)
        monthly.insert(0, "月間予算N", tag)
        monthly_all.append(monthly)
        if not incidents.empty:
            incidents.insert(0, "月間予算N", tag)
            incident_all.append(incidents)

    monthly_all = pd.concat(monthly_all, ignore_index=True)
    monthly_all.to_csv(out_monthly, index=False, encoding="utf-8-sig")
    incident_all = (pd.concat(incident_all, ignore_index=True)
                    if incident_all else pd.DataFrame())
    if not incident_all.empty:
        incident_all.to_csv(out_incidents, index=False, encoding="utf-8-sig")

    if verbose:
        print(f"\n→ {out_monthly} / {out_incidents} を出力")
        if not incident_all.empty:
            cols = ["月間予算N", "開発コード", "部番", "販社報告月",
                    "ウォッチ遅れ月", "インボックス遅れ月", "処理遅れ月",
                    "キャップ起因の追加遅れ", "販社より早い"]
            with pd.option_context("display.width", 200, "display.max_columns", 20):
                print(incident_all[cols].to_string(index=False))
    return monthly_all, incident_all


# ============================================================================
# デモ（テストデータ）
# ============================================================================
if __name__ == "__main__":
    cfg = dict(s.CONFIG)
    cfg.update(R=1.5, h=5.0, alpha_spike=0.005, min_count=3, burst_window=0)
    panel = s._prepare_panel(pd.read_csv("test_panel.csv"), cfg)
    units = s.aggregate_units(panel)
    labels = pd.read_csv("test_labels.csv", encoding="utf-8-sig")
    run_sweep(units, cfg, n_list=[1, 2, None], labels=labels,
              assumed_disposition="対策中")
