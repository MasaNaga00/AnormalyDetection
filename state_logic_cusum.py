# -*- coding: utf-8 -*-
"""
state_logic_cusum.py — CUSUM/Shewhart トラックの状態判定モジュール（フェーズ1結線版）

固定しきい値トラックの state_logic.py に対応する、CUSUM/Shewhart 異常検知のための
「台帳をマスタにした人間レビュー運用」の状態判定層。

検知の数式は cusum_monitor.py に委譲する（フェーズ1で結線済み）:
  - ドリフト: cusum_monitor.poisson_cusum（時変平均ポアソン上側CUSUM）
  - スパイク: cusum_monitor.spike_test（安定期=条件付き二項検定 / 安定化前=直接ポアソン）
本モジュールはその上に、台帳イベントによる **リセット注入** と **ベースライン差し替え**、
監視単位 × 月の **状態解決**、**要確認インボックス** と **Tableau 用監視テーブル** の生成を載せる。

安定化前モードの時変カーブ lambda0(t)（earlylife_baseline.py）への結線はフェーズ2。本版は
配列 lambda0 を受けられる構造まで用意し、安定化前単位は暫定フォールバックで動かす。

------------------------------------------------------------------------------
リセットの噛み合わせ（フェーズ0の結論）
------------------------------------------------------------------------------
cusum_monitor.poisson_cusum は系列一括計算で、既定では発火月に自動でSを0へ戻す
(reset_after_alarm=True)。本運用では **reset_after_alarm=False** にして自動リセットを切り、
リセットは台帳由来（reset-after-M）だけにする。台帳のリセット点でタイムラインをセグメントに
分割し、各セグメントを S=0 から poisson_cusum で計算することで reset-after-M を再現する。

  semantics(reset-after-M): 判定年月Mの当月Sは発火として残し（=Mはセグメント末尾に含める）、
  M の翌月から新しいセグメントを S=0 で開始する。

依存: numpy / pandas / cusum_monitor / （フェーズ2で earlylife_baseline）。Python 3.10+。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import cusum_monitor as cm  # 検知エンジン（ドリフト/スパイク）を委譲


# ============================================================================
# CONFIG（つまみ一覧）
# ============================================================================
CONFIG: dict = {
    # --- 入出力 ---
    "panel_path": None,
    "panel_sheet": 0,
    "ledger_path": None,
    "ledger_sheet": "台帳",
    "out_inbox": "要確認インボックス_cusum.csv",
    "out_tableau": "tableau_監視テーブル_cusum.csv",
    # --- 列名マッピング ---
    "cols": {
        "biz": "事業コード", "dev": "開発コード", "part": "部番", "dist": "販社",
        "ym": "年月", "elapsed": "経過月",
        "monthly_use": "月次使用数", "cum_sales": "累積販売台数",
    },
    # --- 検知パラメータ（cusum_monitor へ渡す。台帳の感度上書きで単位ごとに上書き可）---
    "R": 2.0,
    "h": 5.0,
    "alpha_spike": 0.005,
    "min_count": 3,
    "burst_window": 0,        # 2以上でバーストウィンドウ検定を有効化
    "alpha_burst": None,      # None なら alpha_spike を流用
    "reset_after_alarm": False,  # 自動リセットは切る（リセットは台帳由来のみ）
    # --- ベースライン / 監視レンジ（安定期モード）---
    "stable_start_m": 4,
    "baseline_len": 12,
    "monitor_end_m": 60,      # 妥当性ウィンドウ上限（経過月）。末期は監視対象外
    "lambda0_floor": 1e-6,
    "min_leaders": 3,
    # --- 前処理 / 判定 ---
    "fill_zero_months": True,
    "asof_ym": None,
    "machine_all_part_token": "機種全体",
}


# ============================================================================
# 処置区分 → 振る舞いの対応表
# ============================================================================
DISPOSITIONS: dict[str, dict] = {
    "対策中":      dict(reset=True,  rebaseline=False, watch=True,  hold=False, closes=False, override=False, machine=False),
    "新常態受容":  dict(reset=True,  rebaseline=True,  watch=False, hold=False, closes=False, override=False, machine=False),
    "ノイズ":      dict(reset=True,  rebaseline=False, watch=False, hold=False, closes=False, override=False, machine=False),
    "感度上書き":  dict(reset=True,  rebaseline=False, watch=False, hold=False, closes=False, override=True,  machine=False),
    "スパイク確認": dict(reset=False, rebaseline=False, watch=False, hold=False, closes=False, override=False, machine=False),
    "保留":        dict(reset=False, rebaseline=False, watch=False, hold=True,  closes=False, override=False, machine=False),
    "監視終了":    dict(reset=False, rebaseline=False, watch=False, hold=False, closes=True,  override=False, machine=False),
    "機種終了":    dict(reset=False, rebaseline=False, watch=False, hold=False, closes=True,  override=False, machine=True),
}


# ============================================================================
# 年月ユーティリティ
# ============================================================================
def to_yyyymm(v) -> int | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (int, np.integer)):
        return int(v)
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return None
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 6:
        return int(digits[:6])
    return int(digits) if digits else None


def next_month(ym: int) -> int:
    y, m = divmod(ym, 100)
    return (y + 1) * 100 + 1 if m == 12 else y * 100 + (m + 1)


def _add_months(ym: int, n: int) -> int:
    y, m = divmod(ym, 100)
    total = (y * 12 + (m - 1)) + n
    return (total // 12) * 100 + (total % 12) + 1


# ============================================================================
# 入力の読み込みと前処理
# ============================================================================
def _prepare_panel(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = cfg["cols"]
    df = df.rename(columns={
        c["biz"]: "biz", c["dev"]: "dev", c["part"]: "part", c["dist"]: "dist",
        c["ym"]: "ym", c["elapsed"]: "elapsed",
        c["monthly_use"]: "use", c["cum_sales"]: "fleet",
    }).copy()
    df["ym"] = df["ym"].map(to_yyyymm).astype("int64")
    for col in ("use", "fleet", "elapsed"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["biz", "dev", "part", "dist", "ym"]).reset_index(drop=True)
    if cfg["fill_zero_months"]:
        df = _fill_zero_months(df)
    return df


def _fill_zero_months(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    keys = ["biz", "dev", "part", "dist"]
    for key, g in df.groupby(keys, sort=False):
        g = g.sort_values("ym")
        months = list(g["ym"])
        full, m = [], months[0]
        while m <= months[-1]:
            full.append(m)
            m = next_month(m)
        g2 = g.set_index("ym").reindex(full)
        g2["use"] = g2["use"].fillna(0.0)
        g2["fleet"] = g2["fleet"].ffill()
        g2["elapsed"] = g2["elapsed"].ffill()
        for i, k in enumerate(keys):
            g2[k] = key[i] if isinstance(key, tuple) else key
        out.append(g2.reset_index().rename(columns={"index": "ym"}))
    return pd.concat(out, ignore_index=True)


def aggregate_units(panel: pd.DataFrame) -> pd.DataFrame:
    g = (panel.groupby(["biz", "dev", "part", "ym"], as_index=False)
               .agg(use=("use", "sum"), fleet=("fleet", "sum"), elapsed=("elapsed", "max")))
    return g.sort_values(["biz", "dev", "part", "ym"]).reset_index(drop=True)


def load_ledger(cfg: dict, df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        if cfg["ledger_path"] is None:
            return _demo_ledger()
        df = pd.read_excel(cfg["ledger_path"], sheet_name=cfg["ledger_sheet"])
    df = df.copy()
    for col in ("判定年月", "再評価年月", "ベースライン窓起点"):
        if col in df.columns:
            df[col] = df[col].map(to_yyyymm)
    for col in ("上書きR", "上書きh", "新ベースライン値", "ベースライン窓長"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "記録日" not in df.columns:
        df["記録日"] = pd.NaT
    return df


# ============================================================================
# ベースライン推定とモード分類
# ============================================================================
def estimate_baseline(unit: pd.DataFrame, cfg: dict):
    """安定期モードの自己ベースライン。
    Returns (lambda0, C, E): lambda0=平常レート, C=窓内使用数合計, E=窓内販売台数合計。
    C,E は spike_test の条件付き二項検定（baseline_count/baseline_exposure）に渡す。"""
    lo = cfg["stable_start_m"]
    hi = lo + cfg["baseline_len"]      # [lo, hi)
    w = unit[(unit["elapsed"] >= lo) & (unit["elapsed"] < hi)]
    C = float(w["use"].sum())
    E = float(w["fleet"].sum())
    lam = cm.estimate_lambda0(w["use"].to_numpy(), w["fleet"].to_numpy())
    return max(float(lam), cfg["lambda0_floor"]), C, E


def classify_mode(unit: pd.DataFrame, cfg: dict) -> str:
    max_elapsed = unit["elapsed"].max()
    threshold = cfg["stable_start_m"] + cfg["baseline_len"] - 1
    if pd.notna(max_elapsed) and max_elapsed >= threshold:
        return "安定期"
    return "安定化前"


def earlylife_lambda0(unit: pd.DataFrame, pooled_lambda0: float, cfg: dict) -> float:
    """安定化前モードの暫定ベースライン（フェーズ2で earlylife_baseline へ結線）。
    本来は earlylife_baseline.estimate_earlylife_curve + attach_curve_to_unit で
    配列 lambda0(t) を返す。本版はプール平均にフォールバックし「安定化前(暫定)」と明示する。"""
    return max(float(pooled_lambda0), cfg["lambda0_floor"])


# ============================================================================
# 台帳イベント → リプレイ計画
# ============================================================================
@dataclass
class ReplayPlan:
    reset_after: set = field(default_factory=set)        # {M: reset-after-M}
    baseline_from: list = field(default_factory=list)    # [(effective_month, lambda0, C, E)]
    override_from: list = field(default_factory=list)    # [(effective_month, R, h)]
    close_month: int | None = None

    def baseline_at(self, ym: int, base):
        """その月に有効な (lambda0, C, E)。base=(lambda0,C,E)。"""
        cur = base
        for m, lam, C, E in self.baseline_from:
            if m <= ym:
                cur = (lam, C, E)
            else:
                break
        return cur

    def override_at(self, ym: int, baseR: float, baseh: float):
        R, h = baseR, baseh
        for m, r, hh in self.override_from:
            if m <= ym:
                R, h = r, hh
            else:
                break
        return R, h


def unit_events(ledger: pd.DataFrame, biz, dev, part, asof: int) -> list[dict]:
    m = ((ledger["事業コード"] == biz) & (ledger["開発コード"] == dev) &
         (ledger["部番"] == part) & (ledger["判定年月"].notna()) &
         (ledger["判定年月"] <= asof))
    g = ledger[m].copy()
    if g.empty:
        return []
    g["_ord"] = range(len(g))
    g = g.sort_values(["判定年月", "記録日", "_ord"])
    return g.to_dict("records")


def machine_end_at(ledger: pd.DataFrame, cfg: dict) -> dict:
    tok = cfg["machine_all_part_token"]
    m = (ledger["処置区分"] == "機種終了") | (ledger["部番"] == tok)
    g = ledger[m & ledger["判定年月"].notna()]
    out: dict = {}
    for _, r in g.iterrows():
        key = (r["事業コード"], r["開発コード"])
        ym = int(r["判定年月"])
        out[key] = min(out.get(key, ym), ym)
    return out


def build_plan(events: list[dict], unit: pd.DataFrame, base, cfg: dict) -> ReplayPlan:
    """base=(lambda0,C,E)。"""
    plan = ReplayPlan()
    for ev in events:
        beh = DISPOSITIONS.get(ev.get("処置区分"))
        if beh is None:
            continue
        M = int(ev["判定年月"])
        if beh["closes"] and not beh["machine"]:
            plan.close_month = M if plan.close_month is None else min(plan.close_month, M)
        if beh["reset"]:
            plan.reset_after.add(M)
        if beh["rebaseline"]:
            lam, C, E = _rebaseline_value(ev, unit, base, cfg)
            plan.baseline_from.append((next_month(M), lam, C, E))
        if beh["override"]:
            R = ev.get("上書きR"); h = ev.get("上書きh")
            R = float(R) if pd.notna(R) else cfg["R"]
            h = float(h) if pd.notna(h) else cfg["h"]
            plan.override_from.append((next_month(M), R, h))
    plan.baseline_from.sort()
    plan.override_from.sort()
    return plan


def _rebaseline_value(ev: dict, unit: pd.DataFrame, base, cfg: dict):
    """新常態受容の新 (lambda0, C, E)。明示値→窓推定→現状維持の順。"""
    v = ev.get("新ベースライン値")
    if pd.notna(v) and float(v) > 0:
        # 直接指定。C,E は不明なので現ベースラインの露出を流用（条件付き二項の保守側に作用）
        return max(float(v), cfg["lambda0_floor"]), base[1], base[2]
    start = ev.get("ベースライン窓起点"); length = ev.get("ベースライン窓長")
    if pd.notna(start) and pd.notna(length):
        s, n = int(start), int(length)
        w = unit[(unit["ym"] >= s) & (unit["ym"] < _add_months(s, n))]
        C = float(w["use"].sum()); E = float(w["fleet"].sum())
        if E > 0:
            return max(C / E, cfg["lambda0_floor"]), C, E
    return base


# ============================================================================
# CUSUM リプレイ（リセット注入の本体）— セグメント分割して cusum_monitor へ委譲
# ============================================================================
def replay_unit(unit: pd.DataFrame, base, mode: str, plan: ReplayPlan, cfg: dict) -> pd.DataFrame:
    """base: 安定期は (lambda0:scalar, C, E)、安定化前は (lambda0:ndarray, None, None)。
    監視レンジに絞り、台帳リセット点でセグメント分割して各セグメントを S=0 から計算する。"""
    d = unit.sort_values("ym").reset_index(drop=True)
    lo = 0 if mode.startswith("安定化前") else cfg["stable_start_m"]
    hi = cfg["monitor_end_m"]
    mask = (d["elapsed"] >= lo) & (d["elapsed"] <= hi)
    mon = d[mask].reset_index(drop=True)
    if mon.empty:
        return pd.DataFrame()

    months = mon["ym"].tolist()
    reset_set = {m for m in plan.reset_after if m in set(months)}
    seg_bounds, seg_start = [], 0
    for i, ym in enumerate(months):
        if ym in reset_set:
            seg_bounds.append((seg_start, i))
            seg_start = i + 1
    if seg_start <= len(months) - 1:
        seg_bounds.append((seg_start, len(months) - 1))

    is_array = isinstance(base[0], np.ndarray)
    base_curve = base[0] if is_array else None
    if is_array:
        mon_idx = np.flatnonzero(mask.to_numpy())

    rows = []
    for a, b in seg_bounds:
        seg = mon.iloc[a:b + 1]
        usage = seg["use"].to_numpy(dtype=float)
        fleet = seg["fleet"].to_numpy(dtype=float)
        seg_start_ym = int(seg["ym"].iloc[0])
        R, h = plan.override_at(seg_start_ym, cfg["R"], cfg["h"])

        if is_array:
            lam = base_curve[mon_idx[a:b + 1]]
            C = E = None
        else:
            lam, C, E = plan.baseline_at(seg_start_ym, base)

        S, alarm_d, k = cm.poisson_cusum(
            usage, fleet, lam, R, h, reset_after_alarm=cfg["reset_after_alarm"])
        p_s, a_s, p_b, a_b = cm.spike_test(
            usage, fleet, lam, cfg["alpha_spike"], min_count=cfg["min_count"],
            burst_window=cfg["burst_window"], alpha_burst=cfg["alpha_burst"],
            baseline_count=C, baseline_exposure=E)

        lam_arr = np.asarray(lam, dtype=float)
        if lam_arr.ndim == 0:
            lam_arr = np.full(len(usage), float(lam))
        mu0 = lam_arr * fleet
        for j in range(len(usage)):
            ym = int(seg["ym"].iloc[j])
            rows.append(dict(
                biz=seg["biz"].iloc[j], dev=seg["dev"].iloc[j], part=seg["part"].iloc[j],
                ym=ym, elapsed=seg["elapsed"].iloc[j],
                use=usage[j], fleet=fleet[j], lambda0=lam_arr[j], mu0=mu0[j], k=k[j],
                S=S[j], h=h, R=R,
                p_spike=p_s[j], p_burst=p_b[j],
                alert_drift=bool(alarm_d[j]), alert_spike=bool(a_s[j]), alert_burst=bool(a_b[j]),
                total_alert=bool(alarm_d[j] or a_s[j] or a_b[j]),
                reset_here=(ym in reset_set), mode=mode,
            ))
    return pd.DataFrame(rows)


# ============================================================================
# 状態解決
# ============================================================================
def resolve_state(events: list[dict], rep: pd.DataFrame, ym: int,
                  machine_end: int | None, plan: ReplayPlan):
    row = rep[rep["ym"] == ym]
    ranged = bool(row["total_alert"].iloc[0]) if not row.empty else False

    if machine_end is not None and machine_end <= ym:
        return dict(state="終了(機種)", kind="-", reevaluation_due=False, ranged=ranged)
    if plan.close_month is not None and plan.close_month <= ym:
        return dict(state="終了(単位)", kind="-", reevaluation_due=False, ranged=ranged)

    ev_before = [e for e in events if int(e["判定年月"]) <= ym]
    if not ev_before:
        return dict(state=("要確認" if ranged else "正常"), kind="初回",
                    reevaluation_due=False, ranged=ranged)

    latest = ev_before[-1]
    beh = DISPOSITIONS.get(latest.get("処置区分"), {})
    reeval = latest.get("再評価年月")
    reeval_due = pd.notna(reeval) and int(reeval) <= ym

    if beh.get("hold"):
        state = "要確認保留"
    elif beh.get("watch"):
        state = "対策効果確認中"
    else:
        state = "監視中"
    if ranged:
        state = "要確認"
    return dict(state=state, kind="再", reevaluation_due=bool(reeval_due), ranged=ranged)


# ============================================================================
# 単位の一括評価
# ============================================================================
def evaluate_units(units_df: pd.DataFrame, ledger: pd.DataFrame, cfg: dict, asof: int):
    machine_end = machine_end_at(ledger, cfg)
    pooled = _pooled_lambda0(units_df, cfg)
    all_rows, meta = [], {}
    for (biz, dev, part), unit in units_df.groupby(["biz", "dev", "part"], sort=False):
        mode = classify_mode(unit, cfg)
        if mode == "安定期":
            base = estimate_baseline(unit, cfg)          # (lambda0, C, E)
            mode_label = "安定期"
        else:
            base = (earlylife_lambda0(unit, pooled, cfg), None, None)
            mode_label = "安定化前(暫定)"
        events = unit_events(ledger, biz, dev, part, asof)
        plan = build_plan(events, unit, base, cfg)
        rep = replay_unit(unit, base, mode_label, plan, cfg)
        if rep.empty:
            continue

        states = [resolve_state(events, rep, int(r.ym), machine_end.get((biz, dev)), plan)
                  for r in rep.itertuples()]
        rep = rep.assign(state=[s["state"] for s in states],
                         reevaluation_due=[s["reevaluation_due"] for s in states])
        judged = {int(e["判定年月"]) for e in events}
        rep["judged_point"] = rep["ym"].isin(judged)
        all_rows.append(rep)

        meta[(biz, dev, part)] = dict(
            mode=mode_label, base_lambda0=base[0],
            baseline_origin=(plan.baseline_from[-1][0] if plan.baseline_from
                             else int(rep["ym"].min())),
            reset_count=len(plan.reset_after),
            last_reset=max(plan.reset_after) if plan.reset_after else None,
            machine_end=machine_end.get((biz, dev)), close_month=plan.close_month,
            events=events, plan=plan, base=base,
        )
    table = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    return table, meta


def _pooled_lambda0(units_df: pd.DataFrame, cfg: dict) -> float:
    rates = []
    for _, unit in units_df.groupby(["biz", "dev", "part"], sort=False):
        if classify_mode(unit, cfg) == "安定期":
            rates.append(estimate_baseline(unit, cfg)[0])
    return float(np.mean(rates)) if rates else cfg["lambda0_floor"]


# ============================================================================
# 注目度
# ============================================================================
def attention_score(S: float, h: float, p_spike: float, p_burst: float, cfg: dict) -> float:
    drift = (S / h) if h > 0 else 0.0
    ps = [p for p in (p_spike, p_burst)
          if p is not None and not (isinstance(p, float) and math.isnan(p))]
    p_eff = max(min(ps), 1e-12) if ps else 1.0
    spike = (math.log(p_eff) / math.log(cfg["alpha_spike"])) if cfg["alpha_spike"] < 1 else 0.0
    return float(max(drift, spike))


def _alert_kind(d, s, b) -> str:
    parts = [name for name, flag in (("drift", d), ("spike", s), ("burst", b)) if flag]
    return "+".join(parts) if parts else "-"


# ============================================================================
# 要確認インボックス
# ============================================================================
def build_inbox(table: pd.DataFrame, meta: dict, cfg: dict, asof: int) -> pd.DataFrame:
    rows = []
    for key, m in meta.items():
        biz, dev, part = key
        cur = table[(table["biz"] == biz) & (table["dev"] == dev) &
                    (table["part"] == part) & (table["ym"] == asof)]
        if cur.empty:
            continue
        cur = cur.iloc[0]
        if str(cur["state"]).startswith("終了"):
            continue
        events = m["events"]
        latest = events[-1] if events else None
        latest_beh = DISPOSITIONS.get(latest.get("処置区分"), {}) if latest else {}
        recorded_this_month = bool(latest) and int(latest["判定年月"]) == asof
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

        atype = _alert_kind(cur["alert_drift"], cur["alert_spike"], cur["alert_burst"])
        p_s = None if pd.isna(cur["p_spike"]) else round(float(cur["p_spike"]), 5)
        rows.append({
            "事業コード": biz, "開発コード": dev, "部番": part, "判定年月": asof,
            "アラート種別": atype, "基準モード": m["mode"],
            "判定種別": ("再" if events else "初回"), "載る理由": reason,
            "S": round(float(cur["S"]), 3), "h": float(cur["h"]),
            "月次p値": p_s,
            "観測使用数": int(cur["use"]), "期待故障数": round(float(cur["mu0"]), 3),
            "経過月": cur["elapsed"],
            "注目度": round(attention_score(cur["S"], cur["h"], cur["p_spike"], cur["p_burst"], cfg), 3),
            "現ベースライン起点月": m["baseline_origin"], "リセット回数": m["reset_count"],
            "処置区分": "", "再評価年月": "", "上書きR": "", "上書きh": "",
            "新ベースライン値": "", "ベースライン窓起点": "", "ベースライン窓長": "",
            "原因メモ": "", "確認者": "", "記録日": "",
        })
    inbox = pd.DataFrame(rows)
    if not inbox.empty:
        inbox = inbox.sort_values("注目度", ascending=False).reset_index(drop=True)
    return inbox


# ============================================================================
# Tableau 用監視テーブル
# ============================================================================
def build_tableau(table: pd.DataFrame, meta: dict, cfg: dict) -> pd.DataFrame:
    if table.empty:
        return table
    t = table.copy()
    last_state, last_attn, last_mode = {}, {}, {}
    for key in meta:
        sub = t[(t["biz"] == key[0]) & (t["dev"] == key[1]) & (t["part"] == key[2])]
        if sub.empty:
            continue
        last = sub.sort_values("ym").iloc[-1]
        last_state[key] = last["state"]
        last_attn[key] = attention_score(last["S"], last["h"], last["p_spike"], last["p_burst"], cfg)
        last_mode[key] = meta[key]["mode"]

    def _k(r):
        return (r["biz"], r["dev"], r["part"])

    t["注目度"] = t.apply(lambda r: attention_score(r["S"], r["h"], r["p_spike"], r["p_burst"], cfg), axis=1)
    t["最新状態"] = t.apply(lambda r: last_state.get(_k(r)), axis=1)
    t["単位注目度"] = t.apply(lambda r: round(last_attn.get(_k(r), float("nan")), 3), axis=1)
    t["基準モード"] = t.apply(lambda r: last_mode.get(_k(r), r["mode"]), axis=1)
    t["現ベースライン起点月"] = t.apply(lambda r: meta[_k(r)]["baseline_origin"], axis=1)
    t["リセット回数"] = t.apply(lambda r: meta[_k(r)]["reset_count"], axis=1)
    t["直近リセット月"] = t.apply(lambda r: meta[_k(r)]["last_reset"], axis=1)
    t["アラート種別"] = t.apply(lambda r: _alert_kind(r["alert_drift"], r["alert_spike"], r["alert_burst"]), axis=1)

    cols = {
        "biz": "事業コード", "dev": "開発コード", "part": "部番", "ym": "年月",
        "elapsed": "経過月", "use": "月次使用数", "fleet": "累積販売台数",
        "lambda0": "lambda0", "mu0": "期待故障数", "k": "参照値k", "S": "CUSUM",
        "h": "しきい値h", "R": "R", "p_spike": "月次p値", "p_burst": "バーストp値",
        "alert_drift": "ドリフトアラート", "alert_spike": "スパイクアラート",
        "alert_burst": "バーストアラート", "total_alert": "総合アラート",
        "reset_here": "リセット点", "judged_point": "判定点", "state": "状態",
    }
    t = t.rename(columns=cols)
    ordered = list(cols.values()) + ["アラート種別", "注目度", "最新状態", "単位注目度",
                                     "基準モード", "現ベースライン起点月", "リセット回数", "直近リセット月"]
    return t[ordered].sort_values(["事業コード", "開発コード", "部番", "年月"]).reset_index(drop=True)


# ============================================================================
# 実行
# ============================================================================
def run(cfg: dict = CONFIG):
    panel_raw = (pd.read_excel(cfg["panel_path"], sheet_name=cfg["panel_sheet"])
                 if str(cfg["panel_path"]).endswith((".xlsx", ".xls"))
                 else pd.read_csv(cfg["panel_path"]))
    panel = _prepare_panel(panel_raw, cfg)
    units = aggregate_units(panel)
    ledger = load_ledger(cfg)
    asof = cfg["asof_ym"] or int(units["ym"].max())
    table, meta = evaluate_units(units, ledger, cfg, asof)
    inbox = build_inbox(table, meta, cfg, asof)
    tableau = build_tableau(table, meta, cfg)
    inbox.to_csv(cfg["out_inbox"], index=False, encoding="utf-8-sig")
    tableau.to_csv(cfg["out_tableau"], index=False, encoding="utf-8-sig")
    return inbox, tableau, meta


# ============================================================================
# デモ
# ============================================================================
_DEMO_MONTHS = [202401]
while len(_DEMO_MONTHS) < 24:
    _DEMO_MONTHS.append(next_month(_DEMO_MONTHS[-1]))

_DEMO_COUNTS = {
    "PART-A": [2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 5, 7, 8, 9, 2, 3, 2, 2],
    "PART-B": [2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3],
    "PART-C": [2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 6, 9, 11, 13],
}


def _demo_panel(cfg: dict) -> pd.DataFrame:
    rows = []
    biz, dev = "E1", "DEVX"
    for part, counts in _DEMO_COUNTS.items():
        for i, ym in enumerate(_DEMO_MONTHS):
            F = 1200 + 60 * i
            for d, (frac, cnt) in enumerate([(0.6, counts[i]), (0.4, 0)]):
                rows.append(dict(事業コード=biz, 開発コード=dev, 部番=part, 販社=f"D{d}",
                                 年月=ym, 経過月=i, 月次使用数=int(cnt),
                                 累積販売台数=int(round(F * frac))))
    return pd.DataFrame(rows)


def _demo_ledger() -> pd.DataFrame:
    return pd.DataFrame([
        dict(事業コード="E1", 開発コード="DEVX", 部番="PART-A", 判定年月=202508,
             記録日=202509, 処置区分="対策中", 再評価年月=202511,
             上書きR=None, 上書きh=None, 新ベースライン値=None,
             ベースライン窓起点=None, ベースライン窓長=None,
             原因メモ="部品ロット起因と判明。対策ロット投入", 確認者="demo"),
    ])


def run_demo(cfg: dict = CONFIG):
    panel = _prepare_panel(_demo_panel(cfg), cfg)
    units = aggregate_units(panel)
    ledger = load_ledger(cfg, df=_demo_ledger())
    asof = int(units["ym"].max())
    table, meta = evaluate_units(units, ledger, cfg, asof)
    inbox = build_inbox(table, meta, cfg, asof)
    tableau = build_tableau(table, meta, cfg)
    return inbox, tableau, table, meta, units


if __name__ == "__main__":
    cfg = dict(CONFIG)
    cfg["h"] = 4.0
    cfg["burst_window"] = 3  # バースト検定も有効化して結線を確認

    inbox, tableau, table, meta, units = run_demo(cfg)

    a_unit = units[(units["part"] == "PART-A")]
    base = estimate_baseline(a_unit, cfg)
    real_plan = meta[("E1", "DEVX", "PART-A")]["plan"]
    rep_with = replay_unit(a_unit, base, "安定期", real_plan, cfg)
    rep_without = replay_unit(a_unit, base, "安定期", ReplayPlan(), cfg)
    j = rep_with.set_index("ym")
    j2 = rep_without.set_index("ym")
    cmp = pd.DataFrame({
        "年月": j.index, "経過月": j["elapsed"].astype(int).values,
        "使用数": j["use"].astype(int).values, "期待故障数": j["mu0"].round(2).values,
        "CUSUM(リセットあり)": j["S"].round(2).values, "発火": j["alert_drift"].values,
        "CUSUM(リセットなし)": j2["S"].round(2).values, "発火_なし": j2["alert_drift"].values,
    })
    print("=" * 96)
    print("PART-A：リセット注入の効果（cusum_monitor.poisson_cusum を reset_after_alarm=False でセグメント計算）")
    print("=" * 96)
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        print(cmp.to_string(index=False))
    print("\n→ 202508でCUSUM=13.1まで上昇し発火。『対策中』記録のreset-after-202508で202509から0再開。")
    print("  リセットなしなら鳴り続ける（対比）。検知エンジンは cusum_monitor 本体に委譲済み。")

    print("\n" + "=" * 96)
    print("各部品の最新状態（基準月 {}）".format(int(table["ym"].max())))
    print("=" * 96)
    latest = (tableau.sort_values("年月").groupby(["事業コード", "開発コード", "部番"])
              .tail(1)[["部番", "基準モード", "最新状態", "CUSUM", "しきい値h",
                        "単位注目度", "リセット回数"]])
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        print(latest.to_string(index=False))

    print("\n" + "=" * 96)
    print(f"要確認インボックス（基準月 {int(table['ym'].max())}）")
    print("=" * 96)
    if inbox.empty:
        print("（該当なし）")
    else:
        show = inbox[["部番", "アラート種別", "基準モード", "判定種別", "載る理由",
                      "S", "h", "月次p値", "観測使用数", "注目度", "リセット回数"]]
        with pd.option_context("display.width", 240, "display.max_columns", 30):
            print(show.to_string(index=False))
