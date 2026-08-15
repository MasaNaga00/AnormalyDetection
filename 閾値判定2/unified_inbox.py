# -*- coding: utf-8 -*-
"""
unified_inbox.py — 閾値 / 信号B / 信号C を1本のインボックスに統合する

役割
----
3つの検出器がそれぞれ「当月の発火候補」を出し、共通の台帳フィルタで
抑制・終了を適用し、統合注目度で並べて上位N件を人がレビューする。
台帳は CUSUM 版のスキーマ（処置区分 + 上書き列）に統一する。

台帳スキーマ（CUSUM版 + 3列追加）
--------------------------------
既存: 事業コード 開発コード 部番 判定年月 記録日 処置区分 再評価年月
      上書きR 上書きh 新ベースライン値 ベースライン窓起点 ベースライン窓長
      原因メモ 確認者
追加: 検出器      … 閾値 / 信号B / 信号C / CUSUM（複数なら "/" 区切り）
      上書き閾値  … 閾値検出器の新しい基準X(%)。新常態受容で使う
      対象販社    … 信号C で特定販社の話のとき。空欄=全販社

抑制モデル（3検出器共通・部品単位）
----------------------------------
CUSUM だけは S を積むためセグメント分割という固有の機構が要るが、
閾値・信号B・信号C は状態を持たないので、台帳の役割は次の3つだけになる。

  1. 終了     : 監視終了 / 機種終了 → 以後ずっと出さない
  2. 抑制     : 再評価年月が未到来なら出さない（対策中などの消し込み）
  3. 基準上書き: 新常態受容なら閾値を上げる（上書き閾値）

「保留」だけは例外で、率に関係なく毎月出し続ける（放置の検知漏れを防ぐ）。
抑制は**部品単位**で効く。閾値で記録した部品は信号Bでも出さない
（1件=1部品に集約する既存の設計思想に合わせる）。
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

import state_logic as sl
import state_logic_cusum as sc
import signal_b_peer as sb
import signal_c_dist as sd

DETECTORS = ("閾値", "信号B", "信号C", "CUSUM")


def _diff_ym(a: int, b: int) -> int:
    ya, ma = divmod(int(a), 100)
    yb, mb = divmod(int(b), 100)
    return (ya * 12 + ma) - (yb * 12 + mb)

LEDGER_COLS = [
    "記録日", "事業コード", "開発コード", "部番", "検出器", "対象販社",
    "判定年月", "処置区分", "再評価年月",
    "上書き閾値", "上書きR", "上書きh",
    "新ベースライン値", "ベースライン窓起点", "ベースライン窓長",
    "原因メモ", "確認者",
]

CONFIG = {
    "base_threshold_pct": 2.0,
    "threshold_overrides": {},
    "margin_pct": 0.5,
    "min_denominator": 0,
    # 信号B
    "b_elapsed_cap": 36, "b_min_peers": 2, "b_alpha": 0.005, "b_min_count": 3,
    "b_min_oe": 1.5,
    # 信号C
    "c_base_len": 12, "c_alpha": 0.005, "c_min_count": 3,
    "c_min_base_months": 6, "c_min_base_count": 3.0,
    # 検出器ごとの既定抑制期間（月）。台帳の再評価年月が空欄のとき適用。
    # 信号B は経過月0〜cap の累積 O/E なので月次ではほぼ動かない。
    # 短く設定すると同じ機種が毎月鳴り続けるため、既定を長めに取る。
    "suppress_months": {"閾値": 6, "信号B": 12, "信号C": 3, "CUSUM": 1},
    # 統合
    "multi_bonus": 0.5,      # 検出器が1つ増えるごとの加点
    "score_cap": 3.0,
    "top_n": 15,
    "all_token": "ALL",
    "machine_all_part_token": "機種全体",
    # 販社の報告遅れ対応（reporting_horizon.py）
    "use_horizon": False,           # True で有効化。既定Falseなので従来と同挙動
    "horizon_margin_months": 0,
    "horizon_margin_overrides": {},
    "horizon_fixed": {},
    "horizon_auto_margin": True,
    "horizon_thin_ratio": 0.7,
    "c_revisit_months": 0,          # 信号Cが毎回さかのぼる月数。最大遅れ以上に
    "b_truncate_to_horizon": True,  # 信号Bを global horizon で打ち切る
}


# ============================================================================
# 台帳
# ============================================================================
def empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=LEDGER_COLS)


def load_ledger(path: str, sheet: str = "台帳") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    for c in LEDGER_COLS:
        if c not in df.columns:
            df[c] = np.nan
    for c in ("判定年月", "再評価年月", "ベースライン窓起点"):
        df[c] = df[c].map(sc.to_yyyymm)
    for c in ("上書き閾値", "上書きR", "上書きh", "新ベースライン値", "ベースライン窓長"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("事業コード", "開発コード", "部番", "処置区分"):
        df[c] = df[c].astype(str).replace({"nan": ""})
    return df


class LedgerView:
    """当月Tの時点で、各部品がどう扱われるかを解決する。"""

    def __init__(self, ledger: pd.DataFrame, T: int, cfg: dict):
        self.T = T
        self.cfg = cfg
        tok = cfg["machine_all_part_token"]
        led = ledger[ledger["判定年月"].notna()].copy()
        led = led[led["判定年月"] <= T]

        m = led[(led["処置区分"] == "機種終了") | (led["部番"] == tok)]
        self.machine_end = set(zip(m["事業コード"], m["開発コード"]))

        self.last = {}
        u = led[led["部番"] != tok]
        for key, g in u.groupby(["事業コード", "開発コード", "部番"]):
            g = g.sort_values(["判定年月", "記録日"], na_position="first")
            self.last[key] = g

    def status(self, biz, dev, part, detector: str | None = None,
               event_ym: int | None = None) -> dict:
        """月Tの時点で、この部品を(指定の検出器で)出すべきかを解決する。

        detector=None なら「どの検出器でも共通の判定」（終了・保留）だけを見る。
        検出器名を渡すと、その検出器に対する抑制まで含めて判定する。

        抑制は**検出器ごとに独立**。信号Bを12ヶ月抑制しても、その間の
        販社急増（信号C）や累積率超過（閾値）は出続ける。
        終了（監視終了/機種終了）と保留だけが部品単位で全検出器に効く。

        event_ym : **発火した月**（None なら T）。販社の報告遅れにより過去月が
                   後から発火するようになったため、抑制は「run月Tが窓内か」でなく
                   「発火月が窓内か」で判定する必要がある。
                   判定年月 ≤ event_ym < 再評価年月 を満たす台帳記録があれば抑制。
                   event_ym=T のときは従来と完全に同じ結果になる。
        """
        ev = self.T if event_ym is None else int(event_ym)
        if (biz, dev) in self.machine_end:
            return dict(state="終了機種", thr_ov=None, suppressed=True,
                        carry=False, until=None)
        g = self.last.get((biz, dev, part))
        if g is None:
            return dict(state="未記録", thr_ov=None, suppressed=False,
                        carry=False, until=None)

        # --- 部品単位で効くもの（検出器を問わない）---
        if (g["処置区分"] == "監視終了").any():
            return dict(state="終了単位", thr_ov=None, suppressed=True,
                        carry=False, until=None)
        carry = str(g.iloc[-1]["処置区分"]) == "保留"

        ov = g[g["処置区分"] == "新常態受容"]["上書き閾値"].dropna()
        thr_ov = float(ov.iloc[-1]) if len(ov) else None

        if carry:
            return dict(state="要確認保留", thr_ov=thr_ov, suppressed=False,
                        carry=True, until=None)
        if detector is None:
            return dict(state="監視中", thr_ov=thr_ov, suppressed=False,
                        carry=False, until=None)

        # --- 検出器ごとの抑制 ---
        rows = g[[self._covers(v, detector) for v in g["検出器"]]]
        rows = rows[rows["判定年月"] <= ev]        # 発火月より後の記録は効かせない
        if rows.empty:
            return dict(state="監視中", thr_ov=thr_ov, suppressed=False,
                        carry=False, until=None)
        last = rows.iloc[-1]
        M = int(last["判定年月"])
        rev = last["再評価年月"]
        if rev is not None and not pd.isna(rev):
            until = int(rev)                       # 人が書いた再評価年月が最優先
        else:
            n = self.cfg["suppress_months"].get(detector, 1)
            until = sc._add_months(M, int(n))      # 空欄なら検出器ごとの既定
        return dict(state="監視中", thr_ov=thr_ov, suppressed=(ev < until),
                    carry=False, until=until)

    @staticmethod
    def _covers(value, detector: str) -> bool:
        """台帳の検出器欄がこの検出器を含むか。空欄は「全検出器」とみなす。"""
        s = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
        s = s.strip()
        if s == "" or s == "nan":
            return True
        return detector in [x.strip() for x in s.split("/")]


# ============================================================================
# 検出器1: 閾値（累積使用率）
# ============================================================================
def candidates_threshold(rates: pd.DataFrame, lv: LedgerView, cfg: dict) -> pd.DataFrame:
    T = lv.T
    sub = rates[(rates["ym"] == T) & rates["eligible"] & rates["rate_pct"].notna()]
    rows = []
    for (biz, dev, part), g in sub.groupby(["biz", "dev", "part"]):
        st = lv.status(biz, dev, part, "閾値")
        if st["suppressed"] and not st["carry"]:
            continue
        thr = st["thr_ov"] if st["thr_ov"] is not None \
            else sl.base_threshold(cfg, biz, dev, part)
        pick = g.loc[g["rate_pct"].idxmax()]
        obs = float(pick["rate_pct"])
        if obs < thr and not st["carry"]:
            continue
        rows.append(dict(事業コード=biz, 開発コード=dev, 部番=part, 対象販社=pick["series"],
                         検出器="閾値", 判定年月=T,
                         指標=f"累積率{obs:.2f}% / 閾値{thr:.2f}%",
                         比=obs / thr if thr > 0 else np.nan,
                         生スコア=min(obs / thr, cfg["score_cap"]) if thr > 0 else 0.0,
                         観測率=round(obs, 2), 当月閾値=thr,
                         提案Y下限=round(obs + cfg["margin_pct"], 2)))
    return pd.DataFrame(rows)


# ============================================================================
# 検出器2/3: 信号B・信号C を統合形式に変換
# ============================================================================
def candidates_signal_b(res_b: pd.DataFrame, panel_b: pd.DataFrame,
                        lv: LedgerView, cfg: dict) -> pd.DataFrame:
    """res_b: signal_b_peer.run_signal_b の出力（機種×SF単位）。

    台帳のキーは部番なので、発火した (機種, SF) をその中の**実部番**に展開してから
    統合する。こうしないと閾値/信号Cの行とキーが揃わず、同じ部品が
    別行で二重に出たり、台帳に記録しても片方だけ抑制されない。
    """
    if res_b is None or res_b.empty:
        return pd.DataFrame()
    a = res_b[res_b["alert_peer"]]
    if a.empty:
        return pd.DataFrame()

    cap = cfg["b_elapsed_cap"]
    d = panel_b[panel_b["elapsed"] <= cap]
    pu = (d.groupby(["biz", "dev", "sf", "part"], as_index=False)["use"].sum()
            .rename(columns={"use": "部番使用数"}))
    m = a.merge(pu, on=["biz", "dev", "sf"], how="left")
    m = m[m["部番使用数"].fillna(0) > 0]      # 実績のある部番だけに展開

    rows = []
    for r in m.itertuples():
        rows.append(dict(事業コード=r.biz, 開発コード=r.dev, 部番=r.part,
                         対象販社="", 検出器="信号B", 判定年月=lv.T,
                         指標=(f"SF={r.sf} O/E={r.O_E:.2f} "
                             f"(ピア{r.n_peers}機種, nb={r.nb}, 部番{int(r.部番使用数)}件)"),
                         比=r.O_E, 生スコア=float(r.注目度),
                         観測率=np.nan, 当月閾値=np.nan, 提案Y下限=np.nan))
    return pd.DataFrame(rows)


def expand_signal_b_to_parts(res_b: pd.DataFrame, panel: pd.DataFrame,
                             elapsed_cap: int) -> pd.DataFrame:
    """信号Bの発火(機種×SF)を、その中の部番に展開する。
    台帳は部番単位なので、レビュー対象を具体化するために使う。"""
    if res_b is None or res_b.empty:
        return pd.DataFrame()
    a = res_b[res_b["alert_peer"]][["biz", "sf", "dev", "O_E", "p", "注目度", "nb", "n_peers"]]
    d = panel[panel["elapsed"] <= elapsed_cap]
    pu = (d.groupby(["biz", "dev", "sf", "part"], as_index=False)["use"].sum()
            .rename(columns={"use": "部番使用数"}))
    m = a.merge(pu, on=["biz", "dev", "sf"], how="left")
    return m.sort_values(["注目度", "部番使用数"], ascending=False)


def candidates_signal_c(res_c: pd.DataFrame, lv: LedgerView, cfg: dict) -> pd.DataFrame:
    if res_c is None or res_c.empty:
        return pd.DataFrame()
    # 販社の報告遅れがあるため、発火月は lv.T とは限らない（過去月が後から届く）。
    # 発火月ごとに1行立て、抑制判定もその月で行う。
    a = res_c[(res_c["alert_dist"]) & (res_c["ym"] <= lv.T)]
    if a.empty:
        return pd.DataFrame()
    g = a.groupby(["biz", "dev", "part", "ym"], as_index=False).agg(
        販社=("dist", lambda s: "/".join(sorted(set(s)))),
        O_E=("O_E", "max"), p=("p", "min"), 注目度=("注目度", "max"),
        use=("use", "sum"))
    rows = []
    for r in g.itertuples():
        M = int(r.ym)
        lag = _diff_ym(lv.T, M)
        tag = f"[{M}の月・{lag}ヶ月遅れ] " if lag > 0 else ""
        rows.append(dict(事業コード=r.biz, 開発コード=r.dev, 部番=r.part,
                         対象販社=r.販社, 検出器="信号C", 判定年月=M,
                         指標=(f"{tag}O/E={r.O_E:.2f} 使用{int(r.use)}件 "
                             f"p={r.p:.1e}"),
                         比=r.O_E, 生スコア=float(r.注目度),
                         観測率=np.nan, 当月閾値=np.nan, 提案Y下限=np.nan))
    return pd.DataFrame(rows)


# ============================================================================
# 統合
# ============================================================================
def merge_candidates(parts: list[pd.DataFrame], lv: LedgerView, cfg: dict) -> pd.DataFrame:
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame(columns=["事業コード", "開発コード", "部番", "検出器"])
    c = pd.concat(parts, ignore_index=True)

    # 台帳による抑制（信号B/Cはここで初めて適用）
    # 抑制は run月Tでなく**発火月**で判定する（遅れて届いた過去月があるため）。
    keep = []
    for r in c.itertuples():
        st = lv.status(r.事業コード, r.開発コード, r.部番, r.検出器,
                       event_ym=r.判定年月)
        keep.append(not st["suppressed"] or st["carry"])
    c = c[np.array(keep)].reset_index(drop=True)
    if c.empty:
        return c

    key = ["事業コード", "開発コード", "部番"]
    g = c.groupby(key, as_index=False).agg(
        検出器=("検出器", lambda s: "/".join(sorted(set(s), key=DETECTORS.index))),
        検出器数=("検出器", "nunique"),
        対象販社=("対象販社", lambda s: "/".join(sorted({x for x in s if x}))),
        指標=("指標", lambda s: " ｜ ".join(s)),
        最大スコア=("生スコア", "max"),
        # 台帳に書く判定年月は**最も古い発火月**にする。run月にすると、次回runで
        # 同じ過去月が再判定されたとき「判定年月 > 発火月」で抑制が効かず二重に出る。
        判定年月=("判定年月", "min"),
        対象月内訳=("判定年月", lambda s: "/".join(str(x) for x in sorted(set(s)))),
        観測率=("観測率", "max"), 当月閾値=("当月閾値", "max"),
        提案Y下限=("提案Y下限", "max"))
    g["遅延月"] = [_diff_ym(lv.T, int(m)) for m in g["判定年月"]]

    g["統合注目度"] = (g["最大スコア"]
                   + cfg["multi_bonus"] * (g["検出器数"] - 1)).round(3)
    g["状態"] = [lv.status(r.事業コード, r.開発コード, r.部番)["state"]
               for r in g.itertuples()]
    g["抑制解除月"] = [lv.status(r.事業コード, r.開発コード, r.部番,
                            r.検出器.split("/")[0],
                            event_ym=r.判定年月)["until"] for r in g.itertuples()]
    g["run年月"] = lv.T
    g["処置区分"] = ""
    g["再評価年月"] = ""
    g["原因メモ"] = ""
    g["確認者"] = ""
    return g.sort_values("統合注目度", ascending=False).reset_index(drop=True)


def build_unified_inbox(panel_all: pd.DataFrame, panel_dist: pd.DataFrame,
                        ledger: pd.DataFrame, cfg: dict,
                        cols: dict, asof_ym: int | None = None) -> dict:
    """3検出器を回して統合インボックスを作る。

    panel_all  : ALL行のみのパネル（生の列名）。閾値・信号Bで使う
    panel_dist : 販社別行のパネル（生の列名）。信号Cで使う
    Returns: dict(inbox, top, b_raw, c_raw, b_parts, horizon, horizon_report)

    販社の報告遅れへの対応（cfg["use_horizon"]=True のとき）
    ------------------------------------------------------
    販社ごとに送付頻度が違うため、パネルの最新月は販社ごとに揃っていない。
    検出器の時間特性に応じて扱いを変える:

    閾値  … そのまま最新月Tで判定する。分子だけ欠けるので率は**過小**に出るが、
            誤報でなく見逃し方向なので危険はなく、翌月データが届けば自動で是正される。
            どの販社が欠けているかは戻り値の completeness に出す。
    信号B … 全販社が揃っている月（global horizon）までで打ち切る。累積O/Eの
            機種間比較なので、機種ごとの販社構成の違いで歪むと不公平になる。
            12ヶ月抑制の遅い検出器なので数ヶ月遅れても実害がない。
    信号C … 販社ごとの horizon まで系列を切り、直近 c_revisit_months ヶ月を
            毎回さかのぼって再判定する。**これをやらないと遅れている販社の月は
            永久に一度も検定されない。**
    """
    import reporting_horizon as rhz

    use_hz = bool(cfg.get("use_horizon", False))
    horizon, hz_report, hz_global = None, None, None
    if use_hz:
        horizon = rhz.estimate_horizon(
            panel_dist, cols, all_token=cfg["all_token"],
            margin_months=cfg.get("horizon_margin_months", 0),
            margin_overrides=cfg.get("horizon_margin_overrides") or {},
            fixed=cfg.get("horizon_fixed") or {},
            auto_margin=cfg.get("horizon_auto_margin", True),
            thin_ratio=cfg.get("horizon_thin_ratio", 0.7))
        hz_report = rhz.report(
            panel_dist, cols, all_token=cfg["all_token"],
            margin_months=cfg.get("horizon_margin_months", 0),
            margin_overrides=cfg.get("horizon_margin_overrides") or {},
            fixed=cfg.get("horizon_fixed") or {},
            auto_margin=cfg.get("horizon_auto_margin", True),
            thin_ratio=cfg.get("horizon_thin_ratio", 0.7),
            revisit_months=cfg.get("c_revisit_months", 0))
        hz_global = rhz.global_horizon(horizon)

    # --- 閾値用の率テーブル ---
    sl_cfg = dict(sl.CONFIG)
    sl_cfg["cols"] = {**sl.CONFIG["cols"], **{k: v for k, v in cols.items()}}
    sl_cfg.update({k: cfg[k] for k in
                   ("base_threshold_pct", "threshold_overrides", "min_denominator")})
    ren = {v: k for k, v in sl_cfg["cols"].items() if v is not None}
    p_all = sl._prepare_panel(panel_all.rename(columns=ren).copy(), sl_cfg)
    rates = sl.series_rates(p_all, sl_cfg)

    T = int(asof_ym) if asof_ym is not None else int(rates["ym"].max())
    lv = LedgerView(ledger, T, cfg)

    # --- 信号B用のパネル（CUSUM側の前処理を使う: elapsed/sf が要る）---
    sc_cfg = dict(sc.CONFIG)
    sc_cfg["cols"] = {**sc.CONFIG["cols"], **{k: v for k, v in cols.items()}}
    p_b = sc._prepare_panel(panel_all.copy(), sc_cfg)
    if hz_global is not None and cfg.get("b_truncate_to_horizon", True):
        # 全販社が揃っている月までで打ち切る。累積O/Eの機種間比較なので、
        # 直近の欠測が機種ごとに違う量だけ効くと比較が不公平になる。
        p_b = p_b[p_b["ym"] <= int(hz_global)]
    res_b = sb.run_signal_b(p_b, elapsed_cap=cfg["b_elapsed_cap"],
                            min_peers=cfg["b_min_peers"], alpha_peer=cfg["b_alpha"],
                            min_count=cfg["b_min_count"],
                            min_oe=cfg.get("b_min_oe", 1.5))

    # --- 信号C ---
    res_c = sd.run_signal_c(panel_dist, cols, base_len=cfg["c_base_len"],
                            alpha=cfg["c_alpha"], min_count=cfg["c_min_count"],
                            min_base_months=cfg["c_min_base_months"],
                            min_base_count=cfg["c_min_base_count"],
                            asof_ym=T, months_back=0, all_token=cfg["all_token"],
                            horizon=horizon,
                            # 再評価窓も use_horizon で一括して切る。ここを独立させると
                            # USE_HORIZON=False でも過去月が判定対象に入り、
                            # 「遅延対応前と同一挙動」というマスタスイッチが成立しない。
                            revisit_months=(cfg.get("c_revisit_months", 0)
                                            if use_hz else 0))

    cand = [candidates_threshold(rates, lv, cfg),
            candidates_signal_b(res_b, p_b, lv, cfg),
            candidates_signal_c(res_c, lv, cfg)]
    inbox = merge_candidates(cand, lv, cfg)

    completeness = (rhz.missing_note(panel_dist, cols, horizon, T,
                                     all_token=cfg["all_token"])
                    if horizon else None)
    return dict(inbox=inbox, top=inbox.head(cfg["top_n"]),
                b_raw=res_b, c_raw=res_c,
                b_parts=expand_signal_b_to_parts(res_b, p_b, cfg["b_elapsed_cap"]),
                rates=rates, asof=T,
                horizon=horizon, horizon_report=hz_report,
                horizon_global=hz_global, completeness=completeness)
