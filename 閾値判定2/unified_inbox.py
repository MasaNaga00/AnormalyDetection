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
import reporting_horizon as rh

DETECTORS = ("閾値", "信号B", "信号C", "CUSUM")

LEDGER_COLS = [
    "記録日", "事業コード", "開発コード", "部番", "部品名", "検出器", "対象販社",
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
    "b_min_oe": 1.5, "b_min_peer_count": 20, "b_expand_min_share": 0.1,
    # 信号C
    "c_base_len": 12, "c_alpha": 0.005, "c_min_count": 3,
    "c_min_oe": 3.0, "c_min_excess": 0.0,
    "c_exceed_hist": 0.0, "c_exclude": set(),
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
    # 販社の報告遅れ（reporting_horizon.py）。use_horizon=False なら
    # horizon・再評価窓・信号Bの打ち切りがすべて無効＝遅延対応前と同一挙動。
    "use_horizon": False,
    "horizon_margin_months": 0, "horizon_auto_margin": True,
    "horizon_thin_ratio": 0.7, "horizon_margin_overrides": {},
    "horizon_fixed": {},
    "c_revisit_months": 6,
    "b_truncate_to_horizon": True,
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

        event_ym: 発火した対象月。None なら T。信号Cの再評価窓では過去月が
        毎回再判定されるため、抑制は **run月Tでなく発火月** で判定する
        （run月で判定すると、記録済みの過去月が抑制期間明けに再登場する）。

        detector=None なら「どの検出器でも共通の判定」（終了・保留）だけを見る。
        検出器名を渡すと、その検出器に対する抑制まで含めて判定する。

        抑制は**検出器ごとに独立**。信号Bを12ヶ月抑制しても、その間の
        販社急増（信号C）や累積率超過（閾値）は出続ける。
        終了（監視終了/機種終了）と保留だけが部品単位で全検出器に効く。
        """
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
        e = self.T if event_ym is None or pd.isna(event_ym) else int(event_ym)
        return dict(state="監視中", thr_ov=thr_ov, suppressed=(e < until),
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
    m = m[m["部番使用数"].fillna(0) > 0]

    # SF内でのシェア。判定はSF単位なので、件数のごく少ない部番まで展開すると
    # 累積1件の部番がレビューに並ぶ。主役だけ出し、端数は指標欄に併記する。
    tot = m.groupby(["biz", "dev", "sf"])["部番使用数"].transform("sum")
    m["シェア"] = m["部番使用数"] / tot.replace(0, np.nan)
    thr = cfg.get("b_expand_min_share", 0.1)
    keep = m["シェア"] >= thr
    # どのSFでも最低1部番は残す（全部が閾値未満なら最大のものを採用）
    top = m.groupby(["biz", "dev", "sf"])["部番使用数"].transform("max")
    keep = keep | (m["部番使用数"] == top)
    drop = m[~keep].groupby(["biz", "dev", "sf"]).agg(
        他部番数=("part", "size"), 他件数=("部番使用数", "sum"))

    rows = []
    for r in m[keep].itertuples():
        k = (r.biz, r.dev, r.sf)
        extra = ""
        if k in drop.index:
            e = drop.loc[k]
            extra = f" ＋他{int(e.他部番数)}部番({int(e.他件数)}件)"
        rows.append(dict(事業コード=r.biz, 開発コード=r.dev, 部番=r.part,
                         対象販社="", 検出器="信号B", 判定年月=lv.T,
                         指標=(f"SF={r.sf} O/E={r.O_E:.2f} "
                             f"(ピア{r.n_peers}機種, nb={r.nb}, "
                             f"部番{int(r.部番使用数)}件/{r.シェア*100:.0f}%{extra})"),
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
    """信号Cの発火を部品単位にまとめる。

    再評価窓（c_revisit_months）があると、遅れて届いた過去月も発火しうる。
    台帳の抑制は**発火月ごと**に先に当ててから部品単位に畳む。
    判定年月 = 最も古い発火月（台帳にはこれを書く。run月にすると次回runで
    「判定年月 > 発火月」となり抑制が効かず二重に出る）。
    """
    if res_c is None or res_c.empty:
        return pd.DataFrame()
    a = res_c[(res_c["alert_dist"]) & (res_c["ym"] <= lv.T)]
    if a.empty:
        return pd.DataFrame()
    keep = []
    for r in a.itertuples():
        st = lv.status(r.biz, r.dev, r.part, "信号C", event_ym=int(r.ym))
        keep.append(not st["suppressed"] or st["carry"])
    a = a[np.array(keep)]
    if a.empty:
        return pd.DataFrame()

    rows = []
    for (biz, dev, part), g in a.groupby(["biz", "dev", "part"]):
        months = sorted(set(int(x) for x in g["ym"]))
        m0 = months[0]
        lag = sc_diff(lv.T, m0)
        o = g.loc[g["注目度"].idxmax()]
        tag = f"[{m0}の月・{lag}ヶ月遅れ] " if lag > 0 else ""
        rows.append(dict(事業コード=biz, 開発コード=dev, 部番=part,
                         対象販社="/".join(sorted(set(g["dist"].astype(str)))),
                         検出器="信号C", 判定年月=m0,
                         対象月内訳="/".join(str(m) for m in months),
                         遅延月=lag,
                         指標=(f"{tag}O/E={g['O_E'].max():.2f} "
                             f"使用{int(g['use'].sum())}件 p={g['p'].min():.1e}"),
                         比=float(g["O_E"].max()), 生スコア=float(o["注目度"]),
                         観測率=np.nan, 当月閾値=np.nan, 提案Y下限=np.nan))
    return pd.DataFrame(rows)


def sc_diff(a: int, b: int) -> int:
    return sd._diff_ym(a, b)


# ============================================================================
# 統合
# ============================================================================
def merge_candidates(parts: list[pd.DataFrame], lv: LedgerView, cfg: dict) -> pd.DataFrame:
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame(columns=["事業コード", "開発コード", "部番", "検出器"])
    c = pd.concat(parts, ignore_index=True)

    # 台帳による抑制（信号B/Cはここで初めて適用）
    for col, default in (("対象月内訳", ""), ("遅延月", 0)):
        if col not in c.columns:
            c[col] = default
    c["対象月内訳"] = [x if isinstance(x, str) and x else str(int(m))
                    for x, m in zip(c["対象月内訳"].fillna(""), c["判定年月"])]
    c["遅延月"] = pd.to_numeric(c["遅延月"], errors="coerce").fillna(0).astype(int)
    keep = []
    for r in c.itertuples():
        st = lv.status(r.事業コード, r.開発コード, r.部番, r.検出器,
                       event_ym=int(r.判定年月))
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
        判定年月=("判定年月", "min"),   # 最も古い発火月（台帳にはこれを書く）
        対象月内訳=("対象月内訳", lambda s: "/".join(
            sorted({m for x in s for m in str(x).split("/") if m}))),
        観測率=("観測率", "max"), 当月閾値=("当月閾値", "max"),
        提案Y下限=("提案Y下限", "max"),
        遅延月=("遅延月", "max"))

    g["統合注目度"] = (g["最大スコア"]
                   + cfg["multi_bonus"] * (g["検出器数"] - 1)).round(3)
    g["状態"] = [lv.status(r.事業コード, r.開発コード, r.部番)["state"]
               for r in g.itertuples()]
    g["抑制解除月"] = [lv.status(r.事業コード, r.開発コード, r.部番,
                            r.検出器.split("/")[0])["until"] for r in g.itertuples()]
    g["run年月"] = lv.T
    g["処置区分"] = ""
    g["再評価年月"] = ""
    g["原因メモ"] = ""
    g["確認者"] = ""
    return g.sort_values("統合注目度", ascending=False).reset_index(drop=True)


# ============================================================================
# 部品名（表示用）
# ============================================================================
def part_name_map(panels: list, cols: dict) -> pd.DataFrame:
    """機種×部番 → 部品名。パネルに部品名の列が無ければ空の表。

    機種によって同じ部番でも名前が違うことがあるので、キーは (事業, 機種, 部番)。
    月によって表記が変わる場合は最新月の値を使う。
    """
    out_cols = ["事業コード", "開発コード", "部番", "部品名"]
    col = cols.get("part_name")
    ps = [p for p in panels if p is not None and col and col in p.columns]
    if not ps:
        return pd.DataFrame(columns=out_cols)
    k = [cols["biz"], cols["dev"], cols["part"]]
    d = pd.concat([p[k + [cols["ym"], col]] for p in ps], ignore_index=True)
    d = d[d[col].notna() & (d[col].astype(str).str.strip() != "")]
    if d.empty:
        return pd.DataFrame(columns=out_cols)
    d["_ym"] = d[cols["ym"]].map(sd._to_ym)
    d = d.sort_values("_ym").groupby(k, as_index=False).last()
    d = d[k + [col]]
    d.columns = out_cols
    for c in out_cols[:3]:
        d[c] = d[c].astype(str)
    return d


def add_part_name(df: pd.DataFrame, names: pd.DataFrame,
                  keys=("事業コード", "開発コード", "部番")) -> pd.DataFrame:
    """df の部番の右に部品名列を入れる（見つからなければ空欄）。keys は df 側の列名。"""
    if df is None or df.empty or names is None:
        return df
    k = list(keys)
    m = names.rename(columns=dict(zip(["事業コード", "開発コード", "部番"], k)))
    left = df.drop(columns=["部品名"], errors="ignore")
    tmp = left[k].astype(str).merge(m, on=k, how="left")
    out = left.copy()
    pos = list(out.columns).index(k[2]) + 1
    out.insert(pos, "部品名", tmp["部品名"].fillna("").to_numpy())
    return out


def build_unified_inbox(panel_all: pd.DataFrame, panel_dist: pd.DataFrame,
                        ledger: pd.DataFrame, cfg: dict,
                        cols: dict, asof_ym: int | None = None) -> dict:
    """3検出器を回して統合インボックスを作る。

    panel_all  : ALL行のみのパネル（生の列名）。閾値・信号Bで使う
    panel_dist : 販社別行のパネル（生の列名）。信号Cで使う
    Returns: dict(inbox, top, b_raw, c_raw, b_parts, rates, asof,
                  horizon, horizon_report, completeness, part_names)
    cols["part_name"] があれば inbox / c_raw / b_parts の部番の右に部品名を付ける（表示用）。

    cfg["use_horizon"] が真のマスタースイッチ。False なら horizon・再評価窓・
    信号Bの打ち切りはすべて無効（遅延対応前と同一挙動）。
    """
    names = part_name_map([panel_all, panel_dist], cols)
    cols = {k: v for k, v in cols.items() if k != "part_name"}   # 検出器には渡さない

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

    # --- 販社の報告遅れ（horizon）---
    hz, hz_global, hz_report, completeness = None, None, None, None
    revisit = 0
    if cfg.get("use_horizon", False):
        ymc = cols["ym"]
        pd_T = panel_dist[panel_dist[ymc].map(sd._to_ym) <= T]
        hkw = dict(all_token=cfg["all_token"],
                   margin_months=cfg.get("horizon_margin_months", 0),
                   margin_overrides=cfg.get("horizon_margin_overrides") or {},
                   fixed=cfg.get("horizon_fixed") or {},
                   auto_margin=cfg.get("horizon_auto_margin", True),
                   thin_ratio=cfg.get("horizon_thin_ratio", 0.7))
        hz = rh.estimate_horizon(pd_T, cols, **hkw)
        hz_global = rh.global_horizon(hz)
        revisit = int(cfg.get("c_revisit_months", 0))
        hz_report = rh.report(pd_T, cols, revisit_months=revisit, **hkw)
        completeness = rh.missing_note(pd_T, cols, hz, T, all_token=cfg["all_token"])

    # --- 信号B用のパネル（CUSUM側の前処理を使う: elapsed/sf が要る）---
    sc_cfg = dict(sc.CONFIG)
    sc_cfg["cols"] = {**sc.CONFIG["cols"], **{k: v for k, v in cols.items()}}
    p_b = sc._prepare_panel(panel_all.copy(), sc_cfg)
    # 判定基準月 T より後ろは使わない（asof_ym で過去を再現するとき未来のデータが
    # 累積に混ざらないように。通常の月次実行では T=最新月なので何も落ちない）
    p_b = p_b[p_b["ym"] <= T]
    if hz_global is not None and cfg.get("b_truncate_to_horizon", True):
        # 全販社が揃っている月で打ち切る（分子だけ欠けた月で機種間比較が歪むため）
        p_b = p_b[p_b["ym"] <= min(T, hz_global)]
    res_b = sb.run_signal_b(p_b, elapsed_cap=cfg["b_elapsed_cap"],
                            min_peers=cfg["b_min_peers"], alpha_peer=cfg["b_alpha"],
                            min_count=cfg["b_min_count"],
                            min_peer_count=cfg.get("b_min_peer_count", 20),
                            min_oe=cfg.get("b_min_oe", 1.5))

    # --- 信号C ---
    res_c = sd.run_signal_c(panel_dist, cols, base_len=cfg["c_base_len"],
                            alpha=cfg["c_alpha"], min_count=cfg["c_min_count"],
                            min_base_months=cfg["c_min_base_months"],
                            min_base_count=cfg["c_min_base_count"],
                            min_oe=cfg.get("c_min_oe", 3.0),
                            min_excess=cfg.get("c_min_excess", 0.0),
                            exceed_hist=cfg.get("c_exceed_hist", 0.0),
                            exclude=cfg.get("c_exclude") or None,
                            asof_ym=T, months_back=0, all_token=cfg["all_token"],
                            horizon=hz, revisit_months=revisit)

    cand = [candidates_threshold(rates, lv, cfg),
            candidates_signal_b(res_b, p_b, lv, cfg),
            candidates_signal_c(res_c, lv, cfg)]
    inbox = add_part_name(merge_candidates(cand, lv, cfg), names)
    res_c = add_part_name(res_c, names, keys=("biz", "dev", "part"))
    b_parts = add_part_name(expand_signal_b_to_parts(res_b, p_b, cfg["b_elapsed_cap"]),
                            names, keys=("biz", "dev", "part"))
    return dict(inbox=inbox, top=inbox.head(cfg["top_n"]),
                b_raw=res_b, c_raw=res_c, b_parts=b_parts, part_names=names,
                rates=rates, asof=T,
                horizon=hz, horizon_global=hz_global,
                horizon_report=hz_report, completeness=completeness)
