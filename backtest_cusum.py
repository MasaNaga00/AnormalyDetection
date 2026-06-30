# -*- coding: utf-8 -*-
"""
backtest_cusum.py — 操作点（R, h, alpha_spike, min_count, burst_window）決めのための
ローカル実行バックテスト・ハーネス。

役割
----
ドリフト（CUSUM）とスパイク（Shewhart）は独立した検出器なので、別々にグリッドサーチする。
各パラメータ点について、
  - 誤報側: 実データを「ほぼ正常」とみなし誤報率を測る（ラベル不要）
            ドリフト=ARL0（誤報までの平均“単位×月”）、艦隊全体の月間アラート件数換算。
            スパイク=正常単位月のうち発火した割合、月間件数換算。
  - 検出力側: (a) 既知の過去異常ラベルがあれば検出遅れを測る。
              (b) 無ければ実データの正常系列に合成異常を注入して検出力・遅れを測る。
を出して、トレードオフ表（CSV）にする。最後に「月間アラート予算」を与えると操作点を推薦する。

データはローカルから出さない前提。本ハーネスを手元で実データに当てて、出てきた表で操作点を選ぶ。
検知エンジンは state_logic_cusum と同じ cusum_monitor / earlylife_baseline に委譲するので、
ここで選んだ操作点は本番の挙動と一致する。

測定の約束
----------
ドリフトの ARL0/遅れは、各アラートを離散イベントとして数えるため reset_after_alarm=True で測る
（教科書的な ARL 測定の流儀）。本番のリセットは台帳由来だが、(R,h) の誤報特性づけはこの流儀で正しい。

使い方（ローカル）
------------------
    from backtest_cusum import run_backtest, BT
    cfg = dict(state_logic_cusum.CONFIG)          # 列名マッピング等は本番と同じ
    drift_tbl, spike_tbl = run_backtest("panel.xlsx", cfg=cfg, labels_path=None)
    # drift_tbl / spike_tbl を眺める。予算を決めたら:
    from backtest_cusum import recommend_drift, recommend_spike
    print(recommend_drift(drift_tbl, max_alarms_per_month=20))
    print(recommend_spike(spike_tbl, max_alarms_per_month=20))

依存: numpy / pandas / cusum_monitor / earlylife_baseline / state_logic_cusum。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import cusum_monitor as cm
import state_logic_cusum as s


# ============================================================================
# グリッド既定
# ============================================================================
BT = {
    # ドリフト
    "R_grid": [1.5, 2.0, 3.0],
    "h_grid": [3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
    "drift_factors": [1.5, 2.0, 3.0],   # 注入する真のレート倍率（検出力）
    # スパイク
    "alpha_grid": [0.01, 0.005, 0.001],
    "mincount_grid": [2, 3, 4],
    "burst_grid": [0, 3],
    "spike_mults": [3.0, 5.0],          # 注入スパイクの倍率（期待故障数の何倍を足すか）
    # 注入
    "onset_frac": 0.6,                  # 監視レンジのどこで異常を開始するか
    "inject_reps": 3,                   # 単位×倍率あたりの注入反復（乱数違い）
    "seed": 20240601,
}


# ============================================================================
# 基準（lambda0 の物差し）を単位ごとに用意（台帳・リセットは使わない素の検出）
# ============================================================================
def prepare_basis(units: pd.DataFrame, cfg: dict) -> list[dict]:
    """各監視単位について、検出に使う lambda0 の物差しと監視レンジの系列を用意する。
    安定化前(保留)は監視不能なので除外する。"""
    levels = s.build_earlylife_curves(units, cfg)
    pooled = s._pooled_lambda0(units, cfg)
    out = []
    for (biz, dev, part), unit in units.groupby(["biz", "dev", "part"], sort=False):
        unit = unit.sort_values("ym").reset_index(drop=True)
        mode = s.classify_mode(unit, cfg)
        if mode == "安定期":
            lam, C, E = s.estimate_baseline(unit, cfg)
            lo = cfg["stable_start_m"]
            mode_label = "安定期"
        else:
            lam, mode_label, _lvl, _cnt = s.earlylife_lambda0(unit, levels, pooled, cfg)
            C = E = None
            lo = 0
            if mode_label.endswith("(保留)"):
                continue  # 監視不能は対象外
        hi = cfg["monitor_end_m"]
        mask = (unit["elapsed"] >= lo) & (unit["elapsed"] <= hi)
        if not mask.any():
            continue
        idx = np.flatnonzero(mask.to_numpy())
        use = unit["use"].to_numpy(float)[idx]
        fleet = unit["fleet"].to_numpy(float)[idx]
        ym = unit["ym"].to_numpy()[idx]
        elapsed = unit["elapsed"].to_numpy()[idx]
        if isinstance(lam, np.ndarray):
            lam_mon = lam[idx]
        else:
            lam_mon = float(lam)
        out.append(dict(key=(biz, dev, part), mode=mode_label,
                        use=use, fleet=fleet, ym=ym, elapsed=elapsed,
                        lam=lam_mon, C=C, E=E, n_months=len(use)))
    return out


def _lam_array(lam, n):
    a = np.asarray(lam, dtype=float)
    return a if a.ndim == 1 else np.full(n, float(lam))


# ============================================================================
# 異常の注入
# ============================================================================
def _inject_drift(b: dict, factor: float, onset_frac: float, rng) -> tuple[np.ndarray, int]:
    """onset 以降の使用数を、真のレートを factor 倍にした Poisson で置き換える。"""
    use = b["use"].copy()
    n = len(use)
    onset = int(n * onset_frac)
    lam = _lam_array(b["lam"], n)
    mu = lam[onset:] * b["fleet"][onset:] * factor
    use[onset:] = rng.poisson(np.maximum(mu, 0.0))
    return use, onset


def _inject_spike(b: dict, size_mult: float, onset_frac: float, rng) -> tuple[np.ndarray, int]:
    """onset 月に、期待故障数の size_mult 倍ぶんの追加カウントを足す（単月スパイク）。"""
    use = b["use"].copy()
    n = len(use)
    onset = int(n * onset_frac)
    lam = _lam_array(b["lam"], n)
    mu_o = float(lam[onset] * b["fleet"][onset])
    add = rng.poisson(max(mu_o * size_mult, 1.0))
    use[onset] += add
    return use, onset


# ============================================================================
# ドリフトのバックテスト
# ============================================================================
def _drift_run(use, fleet, lam, R, h):
    # ARL/遅れ測定のため reset_after_alarm=True（各アラートを離散イベントに）
    S, alarm, k = cm.poisson_cusum(use, fleet, lam, R, h, reset_after_alarm=True)
    return np.asarray(alarm, dtype=bool)


def backtest_drift(basis: list[dict], bt: dict, cfg: dict, n_units_full: int | None = None):
    rng = np.random.default_rng(bt["seed"])
    n_units = n_units_full if n_units_full is not None else len(basis)
    rows = []
    for R in bt["R_grid"]:
        for h in bt["h_grid"]:
            # --- 誤報（実データ=ほぼ正常） ---
            fa_events = 0
            unit_months = 0
            for b in basis:
                lam = b["lam"]
                alarm = _drift_run(b["use"], b["fleet"], lam, R, h)
                fa_events += int(alarm.sum())
                unit_months += b["n_months"]
            far = fa_events / unit_months if unit_months else 0.0
            arl0 = (1.0 / far) if far > 0 else np.inf      # 単位×月
            alarms_per_month = far * n_units               # 艦隊全体の月間誤報件数

            # --- 検出力（合成注入） ---
            for factor in bt["drift_factors"]:
                detected = 0
                trials = 0
                delays = []
                for b in basis:
                    for _ in range(bt["inject_reps"]):
                        use_inj, onset = _inject_drift(b, factor, bt["onset_frac"], rng)
                        alarm = _drift_run(use_inj, b["fleet"], b["lam"], R, h)
                        post = np.flatnonzero(alarm[onset:])
                        trials += 1
                        if post.size:
                            detected += 1
                            delays.append(int(post[0]))
                rows.append(dict(
                    検出器="ドリフト", R=R, h=h, 真倍率=factor,
                    誤報率_単位月=round(far, 5),
                    ARL0_単位月=round(arl0, 1) if np.isfinite(arl0) else np.inf,
                    月間誤報件数=round(alarms_per_month, 2),
                    検出率=round(detected / trials, 3) if trials else np.nan,
                    遅れ月_中央値=(float(np.median(delays)) if delays else np.nan),
                ))
    return pd.DataFrame(rows)


# ============================================================================
# スパイクのバックテスト
# ============================================================================
def _spike_run(use, fleet, lam, C, E, alpha, min_count, burst_window, alpha_burst):
    p_s, a_s, p_b, a_b = cm.spike_test(
        use, fleet, lam, alpha, min_count=min_count,
        burst_window=burst_window, alpha_burst=alpha_burst,
        baseline_count=C, baseline_exposure=E)
    return np.asarray(a_s, bool), np.asarray(a_b, bool)


def backtest_spike(basis: list[dict], bt: dict, cfg: dict, n_units_full: int | None = None):
    rng = np.random.default_rng(bt["seed"] + 1)
    n_units = n_units_full if n_units_full is not None else len(basis)
    rows = []
    for alpha in bt["alpha_grid"]:
        for mc in bt["mincount_grid"]:
            for bw in bt["burst_grid"]:
                ab = None  # alpha_burst は alpha 流用
                # --- 誤報（実データ=ほぼ正常） ---
                fa = 0
                unit_months = 0
                for b in basis:
                    a_s, a_b = _spike_run(b["use"], b["fleet"], b["lam"], b["C"], b["E"],
                                          alpha, mc, bw, ab)
                    hit = a_s | a_b if bw >= 2 else a_s
                    fa += int(np.nansum(hit.astype(float)))
                    unit_months += b["n_months"]
                far = fa / unit_months if unit_months else 0.0
                alarms_per_month = far * n_units

                # --- 検出力（単月スパイク注入） ---
                for mult in bt["spike_mults"]:
                    detected = 0
                    trials = 0
                    for b in basis:
                        for _ in range(bt["inject_reps"]):
                            use_inj, onset = _inject_spike(b, mult, bt["onset_frac"], rng)
                            a_s, a_b = _spike_run(use_inj, b["fleet"], b["lam"], b["C"], b["E"],
                                                  alpha, mc, bw, ab)
                            hit = (a_s | a_b) if bw >= 2 else a_s
                            trials += 1
                            # onset 月（バーストは onset 以降数月で拾える）
                            window = hit[onset:onset + max(bw, 1)]
                            if np.any(np.nan_to_num(window.astype(float))):
                                detected += 1
                    rows.append(dict(
                        検出器="スパイク", alpha=alpha, min_count=mc, burst_window=bw,
                        注入倍率=mult,
                        誤報率_単位月=round(far, 5),
                        月間誤報件数=round(alarms_per_month, 2),
                        検出率=round(detected / trials, 3) if trials else np.nan,
                    ))
    return pd.DataFrame(rows)


# ============================================================================
# 既知ラベルでの検出遅れ（任意）
# ============================================================================
def measure_labeled(basis: list[dict], labels: pd.DataFrame, cfg: dict,
                    R: float, h: float, detector: str = "drift",
                    alpha: float = 0.005, min_count: int = 3, burst_window: int = 0) -> pd.DataFrame:
    """labels: 列 [事業コード, 開発コード, 部番, 発生年月]。
    各既知異常について、発生年月以降で最初に発火した月までの遅れ（月数）を返す。
    detector='drift' は CUSUM、'spike' は spike_test。未検出は遅れ NaN。"""
    bmap = {b["key"]: b for b in basis}
    rows = []
    for _, r in labels.iterrows():
        key = (r["事業コード"], r["開発コード"], r["部番"])
        onset_ym = s.to_yyyymm(r["発生年月"])
        b = bmap.get(key)
        if b is None or onset_ym is None:
            rows.append(dict(事業コード=key[0], 開発コード=key[1], 部番=key[2],
                             発生年月=onset_ym, 検出=False, 遅れ月=np.nan, 備考="監視対象外/保留"))
            continue
        pos = np.flatnonzero(b["ym"] >= onset_ym)
        if pos.size == 0:
            rows.append(dict(事業コード=key[0], 開発コード=key[1], 部番=key[2],
                             発生年月=onset_ym, 検出=False, 遅れ月=np.nan, 備考="発生月が監視レンジ外"))
            continue
        onset = int(pos[0])
        if detector == "drift":
            alarm = _drift_run(b["use"], b["fleet"], b["lam"], R, h)
        else:
            a_s, a_b = _spike_run(b["use"], b["fleet"], b["lam"], b["C"], b["E"],
                                  alpha, min_count, burst_window, None)
            alarm = (a_s | a_b) if burst_window >= 2 else a_s
        post = np.flatnonzero(np.nan_to_num(alarm[onset:].astype(float)))
        detected = post.size > 0
        rows.append(dict(事業コード=key[0], 開発コード=key[1], 部番=key[2],
                         発生年月=onset_ym, 検出=bool(detected),
                         遅れ月=(int(post[0]) if detected else np.nan), 備考=""))
    out = pd.DataFrame(rows)
    if not out.empty:
        det = out["検出"].mean()
        med = out.loc[out["検出"], "遅れ月"].median() if out["検出"].any() else np.nan
        print(f"[ラベル評価 {detector} R={R} h={h}] 検出率={det:.2f} 遅れ中央値={med}")
    return out


# ============================================================================
# 推薦
# ============================================================================
def recommend_drift(table: pd.DataFrame, max_alarms_per_month: float,
                    factor: float | None = None) -> pd.DataFrame:
    """月間誤報件数 ≤ 予算 の中から、検出率最大→遅れ最小→誤報最小で並べる。"""
    t = table.copy()
    if factor is not None:
        t = t[t["真倍率"] == factor]
    ok = t[t["月間誤報件数"] <= max_alarms_per_month]
    if ok.empty:
        return t.sort_values("月間誤報件数").head(5)
    return ok.sort_values(["検出率", "遅れ月_中央値", "月間誤報件数"],
                          ascending=[False, True, True]).head(10)


def recommend_spike(table: pd.DataFrame, max_alarms_per_month: float,
                    mult: float | None = None) -> pd.DataFrame:
    t = table.copy()
    if mult is not None:
        t = t[t["注入倍率"] == mult]
    ok = t[t["月間誤報件数"] <= max_alarms_per_month]
    if ok.empty:
        return t.sort_values("月間誤報件数").head(5)
    return ok.sort_values(["検出率", "月間誤報件数"], ascending=[False, True]).head(10)


# ============================================================================
# メイン
# ============================================================================
def run_backtest(panel_path: str, cfg: dict, labels_path: str | None = None,
                 bt: dict = BT, out_prefix: str = "backtest"):
    raw = (pd.read_excel(panel_path, sheet_name=cfg.get("panel_sheet", 0))
           if str(panel_path).endswith((".xlsx", ".xls")) else pd.read_csv(panel_path))
    panel = s._prepare_panel(raw, cfg)
    units = s.aggregate_units(panel)
    basis = prepare_basis(units, cfg)
    n_units = len(basis)
    drift_tbl = backtest_drift(basis, bt, cfg, n_units_full=n_units)
    spike_tbl = backtest_spike(basis, bt, cfg, n_units_full=n_units)
    drift_tbl.to_csv(f"{out_prefix}_drift.csv", index=False, encoding="utf-8-sig")
    spike_tbl.to_csv(f"{out_prefix}_spike.csv", index=False, encoding="utf-8-sig")

    if labels_path:
        labels = (pd.read_excel(labels_path) if str(labels_path).endswith((".xlsx", ".xls"))
                  else pd.read_csv(labels_path))
        # 既知異常は各 (R,h) で遅れを測れる。代表点で1枚出す（必要に応じてループ）。
        lab = measure_labeled(basis, labels, cfg, R=cfg["R"], h=cfg["h"], detector="drift")
        lab.to_csv(f"{out_prefix}_labeled.csv", index=False, encoding="utf-8-sig")
    return drift_tbl, spike_tbl


# ============================================================================
# デモ（合成データで動作確認）
# ============================================================================
def _demo_units(cfg, n_stable=40, n_early=12, seed=1):
    import math
    rng = np.random.default_rng(seed)
    rows = []

    def base_rate(t):
        return 0.0012 + 0.0050 * math.exp(-t / 3.0)

    # 安定期の正常機種（自己ベースライン用に24ヶ月）
    for u in range(n_stable):
        dev = f"S{u:02d}"
        lam = rng.uniform(0.0008, 0.0020)  # 機種ごとの平常レート
        F0 = rng.integers(600, 2000)
        for t in range(24):
            F = int(F0 + 70 * t)
            cnt = rng.poisson(lam * F)
            rows.append(dict(事業コード="E1", 開発コード=dev, 部番="P1", 販社="D0",
                             **{"SF-コード": "SF-100", "ランク": "B"},
                             年月=s._add_months(202201, t), 経過月=t,
                             月次使用数=int(cnt), 累積販売台数=F))
    # 安定化前の正常な新機種（9ヶ月）
    for u in range(n_early):
        dev = f"N{u:02d}"
        jit = rng.uniform(0.85, 1.15)
        F0 = rng.integers(200, 500)
        for t in range(9):
            F = int(F0 + 160 * t)
            cnt = rng.poisson(base_rate(t) * jit * F)
            rows.append(dict(事業コード="E1", 開発コード=dev, 部番="P1", 販社="D0",
                             **{"SF-コード": "SF-100", "ランク": "B"},
                             年月=s._add_months(202401, t), 経過月=t,
                             月次使用数=int(cnt), 累積販売台数=F))
    return s.aggregate_units(s._prepare_panel(pd.DataFrame(rows), cfg))


if __name__ == "__main__":
    cfg = dict(s.CONFIG)
    units = _demo_units(cfg)
    basis = prepare_basis(units, cfg)
    print(f"監視対象（保留除く）: {len(basis)} 単位  "
          f"（安定期 {sum(b['mode']=='安定期' for b in basis)} / "
          f"安定化前 {sum(b['mode'].startswith('安定化前') for b in basis)}）")

    bt = dict(BT)
    bt["inject_reps"] = 3
    drift_tbl = backtest_drift(basis, bt, cfg, n_units_full=len(basis))
    spike_tbl = backtest_spike(basis, bt, cfg, n_units_full=len(basis))

    print("\n=== ドリフト・トレードオフ（真倍率=2.0 で表示）===")
    d = drift_tbl[drift_tbl["真倍率"] == 2.0][
        ["R", "h", "ARL0_単位月", "月間誤報件数", "検出率", "遅れ月_中央値"]]
    with pd.option_context("display.width", 200):
        print(d.to_string(index=False))

    print("\n=== スパイク・トレードオフ（注入倍率=5.0, burst=0 で表示）===")
    sp = spike_tbl[(spike_tbl["注入倍率"] == 5.0) & (spike_tbl["burst_window"] == 0)][
        ["alpha", "min_count", "誤報率_単位月", "月間誤報件数", "検出率"]]
    with pd.option_context("display.width", 200):
        print(sp.to_string(index=False))

    print("\n=== 推薦（月間誤報予算=10件）===")
    print("[ドリフト, 真倍率2.0]")
    print(recommend_drift(drift_tbl, max_alarms_per_month=10, factor=2.0)[
        ["R", "h", "月間誤報件数", "検出率", "遅れ月_中央値"]].to_string(index=False))
    print("[スパイク, 注入倍率5.0]")
    print(recommend_spike(spike_tbl, max_alarms_per_month=10, mult=5.0)[
        ["alpha", "min_count", "burst_window", "月間誤報件数", "検出率"]].to_string(index=False))
