# -*- coding: utf-8 -*-
"""
state_logic_cusum.py — CUSUM/Shewhart トラックの状態判定モジュール（骨格）

固定しきい値トラックの state_logic.py に対応する、CUSUM/Shewhart 異常検知のための
「台帳をマスタにした人間レビュー運用」の状態判定層。検知の数式そのもの（CUSUM 再帰・
スパイク検定・ベースライン推定）は cusum_monitor.py / earlylife_baseline.py が持つ。
本モジュールはその上に、

  - 台帳イベントによる CUSUM の **リセット注入** と **ベースライン差し替え**
  - 監視単位 × 月の **状態解決**（state machine）
  - **要確認インボックス** と **Tableau 用監視テーブル** の生成

を載せる。state_logic.py と同じ原則を保つ:
  * 台帳が唯一のマスタ。状態は保存せず毎回パネル＋台帳から再計算する。
  * asof（基準月）を指定すれば過去月の判定を完全再現できる（台帳は追記専用）。

依存: numpy / pandas / 標準ライブラリ（math）のみ。scipy 不要。Python 3.10+。

------------------------------------------------------------------------------
リセット注入の意味（最重要）
------------------------------------------------------------------------------
CUSUM は累積統計量なので、一度 h を超えたら放置すれば毎月鳴り続ける。固定しきい値で
「閾値を Y に上げて鳴り止ませる」のに相当する操作が、CUSUM では「S_t を 0 に戻す」。

  semantics: リセットを判定年月 M で記録すると、月 M の S_M は通常どおり計算して
  アラートを残す（監査用に発火を見せる）。そのうえで M の翌月へ繰り越す値を 0 にする。
  → S_{M+1} = max(0, 0 + 使用数_{M+1} - k_{M+1})。これを reset-after-M と呼ぶ。

ベースライン再推定（新常態の受容）も同じ仕組みで、M の翌月から新しい lambda0 を使い、
M で繰り越しを 0 にする（リセットを伴う）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ============================================================================
# CONFIG（つまみ一覧）
# ============================================================================
CONFIG: dict = {
    # --- 入出力 ---
    "panel_path": None,      # datamart_A エクスポート。None ならデモ実行
    "panel_sheet": 0,
    "ledger_path": None,     # 台帳 Excel。None ならデモ用の組み込み台帳
    "ledger_sheet": "台帳",
    "out_inbox": "要確認インボックス_cusum.csv",
    "out_tableau": "tableau_監視テーブル_cusum.csv",
    # --- 列名マッピング（実データの列名に合わせる）---
    "cols": {
        "biz": "事業コード",
        "dev": "開発コード",
        "part": "部番",
        "dist": "販社",
        "ym": "年月",
        "elapsed": "経過月",
        "monthly_use": "月次使用数",
        "cum_sales": "累積販売台数",
    },
    # --- 検知パラメータ（全体既定。台帳の感度上書きで単位ごとに上書き可）---
    "R": 2.0,                # CUSUM 悪化倍率
    "h": 5.0,                # CUSUM しきい値
    "alpha_spike": 0.005,    # スパイク p 値しきい値
    "min_count": 3,          # スパイク発火に要する最小カウント（期待値極小での誤報ガード）
    # --- ベースライン（安定期モード）---
    "stable_start_m": 4,     # ベースライン窓の開始経過月
    "baseline_len": 12,      # ベースライン窓長（月）
    "lambda0_floor": 1e-6,   # lambda0 の下限（mu0=0 による p 異常を防ぐ）
    "min_leaders": 3,        # 安定化前モードで必要な先行機種数（骨格では分類のみ）
    # --- 前処理 ---
    "fill_zero_months": True,
    # --- 判定 ---
    "asof_ym": None,         # 基準月。None ならパネル最新月
    "machine_all_part_token": "機種全体",
}


# ============================================================================
# 処置区分 → 振る舞いの対応表
#   reset      : reset-after-M で繰り越しを 0 にするか
#   rebaseline : lambda0 を再推定/差し替えするか（reset も伴う）
#   watch      : 結果状態を「対策効果確認中」にするか
#   hold       : 毎月インボックスに残し続けるか
#   closes     : 監視終了/機種終了か
#   override   : 感度（R/h）を単位ごとに上書きするか
#   machine    : 機種全体に効く終了か
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
# 年月（YYYYMM 整数）のユーティリティ
# ============================================================================
def to_yyyymm(v) -> int | None:
    """YYYYMM 整数へ正規化。整数/文字列/日付様式いずれも可。空は None。"""
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
    """各系列の初出月〜最終月の抜けを 0 行で埋める安全網。
    使用数=0、累積販売台数・経過月は前方補完。"""
    out = []
    keys = ["biz", "dev", "part", "dist"]
    for key, g in df.groupby(keys, sort=False):
        g = g.sort_values("ym")
        months = list(g["ym"])
        full = []
        m = months[0]
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
    """主監視単位（機種×部番、販社合算）の月次系列を作る。
    使用数・累積販売台数とも販社合計。経過月は最大（同一部品なら系列共通の想定）。"""
    g = (panel.groupby(["biz", "dev", "part", "ym"], as_index=False)
               .agg(use=("use", "sum"), fleet=("fleet", "sum"), elapsed=("elapsed", "max")))
    return g.sort_values(["biz", "dev", "part", "ym"]).reset_index(drop=True)


def load_ledger(cfg: dict, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """台帳の読み込み・正規化。df を渡せばそれを使う（デモ/テスト用）。"""
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
def estimate_baseline(unit: pd.DataFrame, cfg: dict) -> float:
    """安定期モードの自己ベースライン lambda0（1台あたり月次故障ペース）。
    ベースライン窓（経過月 [stable_start_m, +baseline_len-1]）の使用数合計 / 販売台数合計。"""
    lo = cfg["stable_start_m"]
    hi = lo + cfg["baseline_len"] - 1
    w = unit[(unit["elapsed"] >= lo) & (unit["elapsed"] <= hi)]
    use = w["use"].sum()
    fleet = w["fleet"].sum()
    lam = (use / fleet) if fleet > 0 else 0.0
    return max(float(lam), cfg["lambda0_floor"])


def classify_mode(unit: pd.DataFrame, cfg: dict) -> str:
    """安定期 / 安定化前 / 監視保留 の自動振り分け（handoff §10）。
    骨格では安定期を完全実装。安定化前は分類のみで lambda0 はプール平均にフォールバック。"""
    max_elapsed = unit["elapsed"].max()
    threshold = cfg["stable_start_m"] + cfg["baseline_len"] - 1
    if pd.notna(max_elapsed) and max_elapsed >= threshold:
        return "安定期"
    return "安定化前"  # 骨格: 先行機種カーブは earlylife_baseline.py の責務（下記フォールバック参照）


def earlylife_lambda0(unit: pd.DataFrame, pooled_lambda0: float, cfg: dict) -> float:
    """安定化前モードの暫定ベースライン。
    本来は earlylife_baseline.estimate_curve(...) が先行機種から時変カーブ lambda0(t) を返す。
    骨格ではプール平均にフォールバック（基準モード列に「安定化前(暫定)」と明示する）。"""
    # TODO: cusum_monitor / earlylife_baseline と接続し、配列 lambda0(t) を返すよう拡張
    return max(float(pooled_lambda0), cfg["lambda0_floor"])


# ============================================================================
# 検知プリミティブ（単一セグメント）— 本来は cusum_monitor.py に委譲する箇所
# ============================================================================
def poisson_sf(count: int, mu: float) -> float:
    """上側裾確率 P(X >= count), X~Poisson(mu)。scipy 不使用の素朴実装（低頻度向け）。"""
    if count <= 0:
        return 1.0
    if mu <= 0:
        return 0.0
    # k=0..count-1 の pmf を漸化式で累積し 1 から引く
    term = math.exp(-mu)   # P(X=0)
    cdf = term
    for k in range(1, count):
        term *= mu / k
        cdf += term
    return float(max(0.0, min(1.0, 1.0 - cdf)))


def cusum_step(s_prev: float, count: float, lam: float, fleet: float,
               R: float, h: float):
    """時変平均ポアソン上側 CUSUM の 1 ステップ。
    mu0 = lam*fleet,  k = (R-1)*lam*fleet/ln R,  S = max(0, S_prev + count - k)。
    返り値: (S, mu0, k, alert_drift)。本来は cusum_monitor.cusum_drift に委譲してよい。"""
    mu0 = lam * fleet
    k = (R - 1.0) * lam * fleet / math.log(R)
    S = max(0.0, s_prev + count - k)
    return S, mu0, k, (S >= h)


# ============================================================================
# 台帳イベント → リプレイ計画（リセット点・ベースライン差し替え・感度上書き）
# ============================================================================
@dataclass
class ReplayPlan:
    reset_after: set = field(default_factory=set)        # {M: reset-after-M}
    lambda0_from: list = field(default_factory=list)     # [(effective_month, lambda0)]（昇順）
    override_from: list = field(default_factory=list)    # [(effective_month, R, h)]（昇順）
    close_month: int | None = None                       # 監視終了の発効月（その月以降は判定しない）

    def lambda0_at(self, ym: int, base: float) -> float:
        lam = base
        for m, v in self.lambda0_from:
            if m <= ym:
                lam = v
            else:
                break
        return lam

    def override_at(self, ym: int, baseR: float, baseh: float):
        R, h = baseR, baseh
        for m, r, hh in self.override_from:
            if m <= ym:
                R, h = r, hh
            else:
                break
        return R, h


def unit_events(ledger: pd.DataFrame, biz, dev, part, asof: int) -> list[dict]:
    """単位 (biz,dev,part) の、判定年月 <= asof のイベントを時系列順に返す。
    同一判定年月は記録日の昇順（記録日が無ければ元順）。機種終了行は別途解決。"""
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
    """(biz,dev) -> 機種終了の発効月（最も早い判定年月）。
    処置区分=機種終了 または 部番=機種全体トークン の行を集める。"""
    tok = cfg["machine_all_part_token"]
    m = (ledger["処置区分"] == "機種終了") | (ledger["部番"] == tok)
    g = ledger[m & ledger["判定年月"].notna()]
    out: dict = {}
    for _, r in g.iterrows():
        key = (r["事業コード"], r["開発コード"])
        ym = int(r["判定年月"])
        out[key] = min(out.get(key, ym), ym)
    return out


def build_plan(events: list[dict], unit: pd.DataFrame, base_lambda0: float, cfg: dict) -> ReplayPlan:
    """単位のイベント列から ReplayPlan（リセット点・ベースライン・感度・終了月）を組む。"""
    plan = ReplayPlan()
    for ev in events:
        disp = ev.get("処置区分")
        beh = DISPOSITIONS.get(disp)
        if beh is None:
            continue
        M = int(ev["判定年月"])
        if beh["closes"] and not beh["machine"]:
            plan.close_month = M if plan.close_month is None else min(plan.close_month, M)
        if beh["reset"]:
            plan.reset_after.add(M)            # reset-after-M: M の翌月から S を 0 で再開
        if beh["rebaseline"]:
            new_lam = _rebaseline_value(ev, unit, base_lambda0, cfg)
            plan.lambda0_from.append((next_month(M), new_lam))   # 翌月から新ベースライン
        if beh["override"]:
            R = ev.get("上書きR")
            h = ev.get("上書きh")
            R = float(R) if pd.notna(R) else cfg["R"]
            h = float(h) if pd.notna(h) else cfg["h"]
            plan.override_from.append((next_month(M), R, h))     # 翌月から新感度
    plan.lambda0_from.sort()
    plan.override_from.sort()
    return plan


def _rebaseline_value(ev: dict, unit: pd.DataFrame, base_lambda0: float, cfg: dict) -> float:
    """新常態受容の新 lambda0。明示値があればそれ、無ければ窓から推定、無ければ現状維持。"""
    v = ev.get("新ベースライン値")
    if pd.notna(v) and float(v) > 0:
        return max(float(v), cfg["lambda0_floor"])
    start = ev.get("ベースライン窓起点")
    length = ev.get("ベースライン窓長")
    if pd.notna(start) and pd.notna(length):
        s, n = int(start), int(length)
        w = unit[(unit["ym"] >= s) & (unit["ym"] < _add_months(s, n))]
        use, fleet = w["use"].sum(), w["fleet"].sum()
        if fleet > 0:
            return max(float(use / fleet), cfg["lambda0_floor"])
    return base_lambda0


def _add_months(ym: int, n: int) -> int:
    y, m = divmod(ym, 100)
    total = (y * 12 + (m - 1)) + n
    return (total // 12) * 100 + (total % 12) + 1


# ============================================================================
# CUSUM リプレイ（リセット注入の本体）
# ============================================================================
def replay_unit(unit: pd.DataFrame, base_lambda0: float, mode: str,
                plan: ReplayPlan, cfg: dict) -> pd.DataFrame:
    """単位の月次系列を、リセット注入・ベースライン差し替え・感度上書きを織り込んで
    リプレイし、月次の S/mu0/p 値/各アラートを返す。"""
    rows = []
    s_prev = 0.0
    for _, r in unit.sort_values("ym").iterrows():
        ym = int(r["ym"])
        count = float(r["use"])
        fleet = float(r["fleet"])
        lam = plan.lambda0_at(ym, base_lambda0)
        R, h = plan.override_at(ym, cfg["R"], cfg["h"])
        S, mu0, k, alert_drift = cusum_step(s_prev, count, lam, fleet, R, h)
        # スパイク検定（単月・上側ポアソン裾。安定期は条件付き二項に置換するのが本筋）
        p = poisson_sf(int(round(count)), mu0)
        alert_spike = (p <= cfg["alpha_spike"]) and (count >= cfg["min_count"])
        reset_here = ym in plan.reset_after
        rows.append(dict(
            biz=r["biz"], dev=r["dev"], part=r["part"], ym=ym, elapsed=r["elapsed"],
            use=count, fleet=fleet, lambda0=lam, mu0=mu0, k=k, S=S, h=h, R=R,
            p_spike=p, alert_drift=bool(alert_drift), alert_spike=bool(alert_spike),
            total_alert=bool(alert_drift or alert_spike),
            reset_here=bool(reset_here), mode=mode,
        ))
        # reset-after-M: 当月 S はアラート可視化のため残し、翌月への繰り越しのみ 0 に
        s_prev = 0.0 if reset_here else S
    return pd.DataFrame(rows)


# ============================================================================
# 状態解決（state machine）
# ============================================================================
def resolve_state(events: list[dict], rep: pd.DataFrame, ym: int,
                  machine_end: int | None, plan: ReplayPlan):
    """単位 × 月 ym の状態・判定種別・再評価到来を返す。
    返り値 dict: state, kind(初回|再), reevaluation_due, ranged(当月発火)。"""
    row = rep[rep["ym"] == ym]
    ranged = bool(row["total_alert"].iloc[0]) if not row.empty else False

    # 1. 機種終了
    if machine_end is not None and machine_end <= ym:
        return dict(state="終了(機種)", kind="-", reevaluation_due=False, ranged=ranged)
    # 1'. 監視終了（単位）
    if plan.close_month is not None and plan.close_month <= ym:
        return dict(state="終了(単位)", kind="-", reevaluation_due=False, ranged=ranged)

    ev_before = [e for e in events if int(e["判定年月"]) <= ym]
    if not ev_before:
        state = "要確認" if ranged else "正常"
        return dict(state=state, kind="初回", reevaluation_due=False, ranged=ranged)

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

    # 当月発火 or 再評価到来は「要確認」に引き上げ
    if ranged:
        state = "要確認"
    return dict(state=state, kind="再", reevaluation_due=bool(reeval_due), ranged=ranged)


# ============================================================================
# 単位の一括評価（リプレイ + 各月の状態）
# ============================================================================
def evaluate_units(units_df: pd.DataFrame, ledger: pd.DataFrame, cfg: dict, asof: int):
    """全単位をリプレイし、月次の検知量＋状態を付与した縦長テーブルと、
    単位ごとのメタ（基準モード・現ベースライン起点・リセット回数等）を返す。"""
    machine_end = machine_end_at(ledger, cfg)
    pooled = _pooled_lambda0(units_df, cfg)
    all_rows, meta = [], {}
    for (biz, dev, part), unit in units_df.groupby(["biz", "dev", "part"], sort=False):
        mode = classify_mode(unit, cfg)
        if mode == "安定期":
            base_lam = estimate_baseline(unit, cfg)
            mode_label = "安定期"
        else:
            base_lam = earlylife_lambda0(unit, pooled, cfg)
            mode_label = "安定化前(暫定)"
        events = unit_events(ledger, biz, dev, part, asof)
        plan = build_plan(events, unit, base_lam, cfg)
        rep = replay_unit(unit, base_lam, mode_label, plan, cfg)

        states = [resolve_state(events, rep, int(r.ym), machine_end.get((biz, dev)), plan)
                  for r in rep.itertuples()]
        rep = rep.assign(
            state=[s["state"] for s in states],
            reevaluation_due=[s["reevaluation_due"] for s in states],
        )
        # 判定点（その月に台帳記録があるか）
        judged = {int(e["判定年月"]) for e in events}
        rep["judged_point"] = rep["ym"].isin(judged)
        all_rows.append(rep)

        meta[(biz, dev, part)] = dict(
            mode=mode_label,
            base_lambda0=base_lam,
            baseline_origin=plan.lambda0_from[-1][0] if plan.lambda0_from else int(unit["ym"].min()),
            reset_count=len(plan.reset_after),
            last_reset=max(plan.reset_after) if plan.reset_after else None,
            machine_end=machine_end.get((biz, dev)),
            close_month=plan.close_month,
            events=events,
            plan=plan,
        )
    table = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    return table, meta


def _pooled_lambda0(units_df: pd.DataFrame, cfg: dict) -> float:
    """安定化前フォールバック用の、安定期単位からのプール平均レート。"""
    rates = []
    for _, unit in units_df.groupby(["biz", "dev", "part"], sort=False):
        if classify_mode(unit, cfg) == "安定期":
            rates.append(estimate_baseline(unit, cfg))
    return float(np.mean(rates)) if rates else cfg["lambda0_floor"]


# ============================================================================
# 注目度（worklist ソート用）
# ============================================================================
def attention_score(S: float, h: float, p: float, cfg: dict) -> float:
    """ドリフト比 S/h とスパイク比 log p / log alpha の最大。どちらも >=1 で発火。"""
    drift = (S / h) if h > 0 else 0.0
    p_eff = max(p, 1e-12)
    spike = (math.log(p_eff) / math.log(cfg["alpha_spike"])) if cfg["alpha_spike"] < 1 else 0.0
    return float(max(drift, spike))


# ============================================================================
# 要確認インボックス
# ============================================================================
def build_inbox(table: pd.DataFrame, meta: dict, cfg: dict, asof: int) -> pd.DataFrame:
    """基準月 asof で人がレビューすべき単位だけを一覧化。載る条件:
       (1) 当月発火（drift or spike）かつ当月未記録   … 初回/再
       (2) 最新行が保留                              … 毎月継続
       (3) 再評価到来（再評価年月 <= asof）           … 率が動かなくても時間で
    除外: 終了(単位/機種)、当月記録済みで保留でない（冪等性）。"""
    rows = []
    for key, m in meta.items():
        biz, dev, part = key
        cur = table[(table["biz"] == biz) & (table["dev"] == dev) &
                    (table["part"] == part) & (table["ym"] == asof)]
        if cur.empty:
            continue
        cur = cur.iloc[0]
        state = cur["state"]
        if state.startswith("終了"):
            continue
        events = m["events"]
        latest = events[-1] if events else None
        latest_beh = DISPOSITIONS.get(latest.get("処置区分"), {}) if latest else {}
        recorded_this_month = bool(latest) and int(latest["判定年月"]) == asof
        is_hold = latest_beh.get("hold", False)

        fired = bool(cur["total_alert"])
        reeval_due = bool(cur["reevaluation_due"])

        # 載せる理由の判定
        reason = None
        if fired and not (recorded_this_month and not is_hold):
            reason = "発火"
        elif is_hold:
            reason = "保留継続"
        elif reeval_due and not recorded_this_month:
            reason = "再評価"
        if reason is None:
            continue

        kind = "再" if events else "初回"
        atype = ("drift+spike" if cur["alert_drift"] and cur["alert_spike"]
                 else "drift" if cur["alert_drift"]
                 else "spike" if cur["alert_spike"] else "-")
        rows.append({
            "事業コード": biz, "開発コード": dev, "部番": part, "判定年月": asof,
            "アラート種別": atype, "基準モード": m["mode"], "判定種別": kind,
            "載る理由": reason,
            "S": round(float(cur["S"]), 3), "h": float(cur["h"]),
            "月次p値": round(float(cur["p_spike"]), 5),
            "観測使用数": int(cur["use"]), "期待故障数": round(float(cur["mu0"]), 3),
            "経過月": cur["elapsed"],
            "注目度": round(attention_score(cur["S"], cur["h"], cur["p_spike"], cfg), 3),
            "現ベースライン起点月": m["baseline_origin"], "リセット回数": m["reset_count"],
            # --- 以下、人間記入列（台帳へコピーして埋める）---
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
    """単位 × 月の全行に注釈を付けた縦長テーブル。最新状態・注目度は単位レベルで付与。"""
    if table.empty:
        return table
    t = table.copy()
    # 単位レベルの最新月メタ
    last_state, last_attn, last_mode = {}, {}, {}
    for key, m in meta.items():
        sub = t[(t["biz"] == key[0]) & (t["dev"] == key[1]) & (t["part"] == key[2])]
        if sub.empty:
            continue
        last = sub.sort_values("ym").iloc[-1]
        last_state[key] = last["state"]
        last_attn[key] = attention_score(last["S"], last["h"], last["p_spike"], cfg)
        last_mode[key] = m["mode"]

    def _key(r):
        return (r["biz"], r["dev"], r["part"])

    t["注目度"] = t.apply(lambda r: attention_score(r["S"], r["h"], r["p_spike"], cfg), axis=1)
    t["最新状態"] = t.apply(lambda r: last_state.get(_key(r)), axis=1)
    t["単位注目度"] = t.apply(lambda r: round(last_attn.get(_key(r), float("nan")), 3), axis=1)
    t["基準モード"] = t.apply(lambda r: last_mode.get(_key(r), r["mode"]), axis=1)
    t["現ベースライン起点月"] = t.apply(lambda r: meta[_key(r)]["baseline_origin"], axis=1)
    t["リセット回数"] = t.apply(lambda r: meta[_key(r)]["reset_count"], axis=1)
    t["直近リセット月"] = t.apply(lambda r: meta[_key(r)]["last_reset"], axis=1)

    cols = {
        "biz": "事業コード", "dev": "開発コード", "part": "部番", "ym": "年月",
        "elapsed": "経過月", "use": "月次使用数", "fleet": "累積販売台数",
        "lambda0": "lambda0", "mu0": "期待故障数", "k": "k", "S": "CUSUM",
        "h": "しきい値h", "R": "R", "p_spike": "月次p値",
        "alert_drift": "ドリフトアラート", "alert_spike": "スパイクアラート",
        "total_alert": "総合アラート", "reset_here": "リセット点",
        "judged_point": "判定点", "state": "状態",
    }
    t = t.rename(columns=cols)
    ordered = list(cols.values()) + ["注目度", "最新状態", "単位注目度", "基準モード",
                                     "現ベースライン起点月", "リセット回数", "直近リセット月"]
    return t[ordered].sort_values(["事業コード", "開発コード", "部番", "年月"]).reset_index(drop=True)


# ============================================================================
# 実行
# ============================================================================
def run(cfg: dict = CONFIG):
    panel_raw = pd.read_excel(cfg["panel_path"], sheet_name=cfg["panel_sheet"]) \
        if str(cfg["panel_path"]).endswith((".xlsx", ".xls")) else pd.read_csv(cfg["panel_path"])
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
# デモ（実データなし）— リセット注入が見えるように作ってある
# ============================================================================
# 決定論的なデモ系列（経過月 0..23 = 202401..202512）
_DEMO_MONTHS = [202401 + 0]
while len(_DEMO_MONTHS) < 24:
    _DEMO_MONTHS.append(next_month(_DEMO_MONTHS[-1]))

#                  経過月 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
_DEMO_COUNTS = {
    # 平常→中盤でドリフト(16-19)→対策後に平常へ復帰(20-23)。台帳で対策中リセットを入れる対象
    "PART-A": [2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 5, 7, 8, 9, 2, 3, 2, 2],
    # ずっと平常の対照（鳴らない）
    "PART-B": [2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3],
    # 直近でドリフト開始（現在進行で発火中・台帳履歴なし＝初回）
    "PART-C": [2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 6, 9, 11, 13],
}


def _demo_panel(cfg: dict) -> pd.DataFrame:
    """決定論的な縦長パネル。各部品を2販社(D0/D1)に分け、使用数はD0へ、
    累積販売台数はF=1200+60*経過月 を 0.6/0.4 で分配（合算で元のF・使用数に戻る）。"""
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
    """PART-A: ドリフト発火を調査し『対策中』を判定年月202508で記録。
    reset-after-202508 で翌月202509からCUSUMが0で再開し、対策が効いて静かなままになる
    （＝対策効果確認中）。再評価年月202511を入れ、静かでも時間でインボックスに載ることも示す。"""
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
    cfg["h"] = 4.0  # デモが分かりやすいようしきい値を少し下げる

    inbox, tableau, table, meta, units = run_demo(cfg)

    # --- リセット注入の効果を「あり/なし」並べて見せる ---
    a_unit = units[(units["part"] == "PART-A")]
    base = estimate_baseline(a_unit, cfg)
    real_plan = meta[("E1", "DEVX", "PART-A")]["plan"]
    rep_with = replay_unit(a_unit, base, "安定期", real_plan, cfg)
    rep_without = replay_unit(a_unit, base, "安定期", ReplayPlan(), cfg)  # 台帳なし＝リセットなし
    cmp = pd.DataFrame({
        "年月": rep_with["ym"], "経過月": rep_with["elapsed"].astype(int),
        "使用数": rep_with["use"].astype(int),
        "期待故障数": rep_with["mu0"].round(2),
        "CUSUM(リセットあり)": rep_with["S"].round(2),
        "発火": rep_with["alert_drift"],
        "CUSUM(リセットなし)": rep_without["S"].round(2),
        "発火_なし": rep_without["alert_drift"],
    })

    print("=" * 92)
    print("PART-A：リセット注入の効果（台帳『対策中』判定年月202508 → reset-after で202509から0再開）")
    print("=" * 92)
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        print(cmp.to_string(index=False))
    print("\n→ 202508でCUSUM=13.1まで上昇し発火。リセットなしなら以降も鳴り続けるが、")
    print("  『対策中』記録によりreset-after-202508が効き、202509から0で再開。対策が効いて静かなまま。")

    # --- 各部品の最新状態 ---
    print("\n" + "=" * 92)
    print("各部品の最新状態（基準月 {}）".format(int(table["ym"].max())))
    print("=" * 92)
    latest = (tableau.sort_values("年月").groupby(["事業コード", "開発コード", "部番"])
              .tail(1)[["部番", "基準モード", "最新状態", "CUSUM", "しきい値h",
                        "単位注目度", "リセット回数", "現ベースライン起点月"]])
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        print(latest.to_string(index=False))

    print("\n" + "=" * 92)
    print(f"要確認インボックス（基準月 {int(table['ym'].max())}）")
    print("=" * 92)
    if inbox.empty:
        print("（該当なし）")
    else:
        show = inbox[["部番", "アラート種別", "基準モード", "判定種別", "載る理由",
                      "S", "h", "月次p値", "観測使用数", "期待故障数", "注目度", "リセット回数"]]
        with pd.option_context("display.width", 240, "display.max_columns", 30):
            print(show.to_string(index=False))
    print("\n（インボックスの人間記入列: 処置区分/再評価年月/上書きR・h/新ベースライン値/"
          "ベースライン窓起点・窓長/原因メモ/確認者/記録日 を埋めて台帳へコピー）")
