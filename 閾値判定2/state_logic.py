# -*- coding: utf-8 -*-
"""部品出庫 異常検知 — 累積使用率の固定しきい値運用：状態判定ロジック.

台帳(Excel)をマスタに、各部品(事業×開発×部番)の「現在の有効な再アラート条件」を
最新行から決め、当月の累積使用率と突き合わせて状態を判定する。

出力は2つ:
  1) 要確認インボックス … その月に人がレビューすべき部品の一覧(台帳の自動列に対応)
  2) Tableau用 縦長テーブル … 系列×月ごとの累積率・当月閾値・閾値超・状態・判定点

判定の粒度:
  - 全販社合算の累積率と、販社別の累積率の「両方」を閾値と比較し、両方をTableauに出す。
  - ただしアラートと判定は部品単位(機種×部番)に集約する。どれかの系列が越えたら要確認を1件。
  - 人は1部品につき1本のY(再アラート閾値)を決め、以降は販社を問わずどれかがYを越えたら再アラート。
  - 「判定元」列はどの系列が越えたかの診断用で、台帳のキーではない。
累積率の性質:
  - 累積比なので対策後も率は下がらず横ばい。改善確認には使えない(=要注目ゲートとして使う)。
"""

import numpy as np
import pandas as pd

CONFIG = {
    # --- 入力 ---
    "ledger_path": "部品出庫_異常検知_台帳テンプレート.xlsx",
    "ledger_sheet": "台帳",
    "panel_path": None,            # datamart_A のエクスポート(.csv/.xlsx)。None ならデモ実行。
    "panel_sheet": 0,
    # 列マッピング: 内部名 -> 実データの列名(自分の環境に合わせて変更)
    "cols": {
        "biz": "事業コード",
        "dev": "開発コード",
        "part": "部番",
        "dist": "販社",
        "ym": "年月",            # YYYYMM
        "monthly_use": "月次使用数",
        "cum_sales": "累積販売台数",
        "cum_use": None,          # 累積使用数の列があれば列名、無ければ None で月次使用数から算出
    },
    # --- パラメータ ---
    "base_threshold_pct": 2.0,    # 既定の基準閾値 X(%)
    "threshold_overrides": {},    # 例: {("E10",): 1.5} 事業ごとに X を上書き
    "margin_pct": 0.5,            # インボックスの「提案Y下限」= 観測率 + margin（実Yは人が台帳に記入）
    "min_denominator": 0,         # 販社系列がトリガ資格を持つ累積販売台数の下限(0=オフ、合算は常に資格)
    "fill_zero_months": True,      # 月次0補完(各系列の初出〜最終月で抜けた年月を0行で埋める)
    "fill_to_global_max": False,   # Trueでパネル全体の最新月まで延ばす(既定は各系列の最終月まで)
    "asof_ym": None,              # 判定の基準月 YYYYMM。None ならパネル最新月
    # --- 出力 ---
    "out_inbox": "要確認インボックス.csv",
    "out_tableau": "tableau_監視テーブル.csv",
    "machine_all_part_token": "機種全体",  # 機種終了行の部番に入る特別値
}

COMBINED = "合算"


# --------------------------------------------------------------------------- #
# 入力読み込み
# --------------------------------------------------------------------------- #
def _to_ym(x):
    if pd.isna(x) or x == "":
        return None
    return int(float(str(x).replace("-", "").replace("/", "")[:6]))


def load_panel(cfg):
    """datamart_A を読み、内部列名の縦長パネルに整形する。"""
    c = cfg["cols"]
    path = cfg["panel_path"]
    if str(path).lower().endswith((".xlsx", ".xlsm")):
        df = pd.read_excel(path, sheet_name=cfg["panel_sheet"])
    else:
        df = pd.read_csv(path)
    ren = {v: k for k, v in c.items() if v is not None}
    df = df.rename(columns=ren)
    return _prepare_panel(df, cfg)


def _ym_to_period(ym):
    ym = int(ym)
    return pd.Period(year=ym // 100, month=ym % 100, freq="M")


def _period_to_ym(p):
    return p.year * 100 + p.month


def _fill_zero_months(df, cfg):
    """各系列(事業×開発×部番×販社)の初出〜最終月で抜けた年月を0行で補完する。

    月次使用数=0、累積販売台数は前方補完(0使用の月は分母を直前月で据え置き)。
    累積使用数の列がある場合も前方補完で据え置く(=月次0と整合)。
    """
    keys = ["biz", "dev", "part", "dist"]
    has_cum_use = cfg["cols"]["cum_use"] is not None
    gmax = _ym_to_period(df["ym"].max()) if cfg.get("fill_to_global_max") else None
    out = []
    for key, g in df.groupby(keys, sort=False):
        g = g.sort_values("ym")
        p1 = gmax if gmax is not None else _ym_to_period(g["ym"].max())
        full = [_period_to_ym(p) for p in pd.period_range(_ym_to_period(g["ym"].min()), p1, freq="M")]
        g = g.set_index("ym").reindex(full)
        for i, k in enumerate(keys):
            g[k] = key[i]
        if "monthly_use" in g.columns:
            g["monthly_use"] = g["monthly_use"].fillna(0)
        g["cum_sales"] = g["cum_sales"].ffill()
        if has_cum_use:
            g["cum_use"] = g["cum_use"].ffill()
        g = g.reset_index().rename(columns={"index": "ym"})
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _prepare_panel(df, cfg):
    for k in ("biz", "dev", "part", "dist"):
        df[k] = df[k].astype(str)
    df["ym"] = df["ym"].map(_to_ym)
    df = df[df["ym"].notna()].copy()
    df["ym"] = df["ym"].astype(int)
    df["cum_sales"] = pd.to_numeric(df["cum_sales"], errors="coerce")
    if cfg["cols"]["cum_use"] is None:
        df["monthly_use"] = pd.to_numeric(df["monthly_use"], errors="coerce").fillna(0)
    else:
        df["cum_use"] = pd.to_numeric(df["cum_use"], errors="coerce")
    if cfg.get("fill_zero_months", True):
        df = _fill_zero_months(df, cfg)
    if cfg["cols"]["cum_use"] is None:
        df = df.sort_values(["biz", "dev", "part", "dist", "ym"])
        df["cum_use"] = df.groupby(["biz", "dev", "part", "dist"])["monthly_use"].cumsum()
    return df


def series_rates(panel, cfg):
    """系列(合算/販社別)×月の累積率テーブルを作る。"""
    floor = cfg["min_denominator"]
    keys = ["biz", "dev", "part"]

    dist = panel.copy()
    dist["series"] = dist["dist"]
    dist = dist[keys + ["series", "ym", "cum_use", "cum_sales"]]

    comb = (panel.groupby(keys + ["ym"], as_index=False)[["cum_use", "cum_sales"]].sum())
    comb["series"] = COMBINED

    out = pd.concat([dist, comb[keys + ["series", "ym", "cum_use", "cum_sales"]]], ignore_index=True)
    out["rate_pct"] = np.where(out["cum_sales"] > 0, 100.0 * out["cum_use"] / out["cum_sales"], np.nan)
    out["eligible"] = (out["cum_sales"] > 0) & ((out["series"] == COMBINED) | (out["cum_sales"] >= floor))
    return out.sort_values(keys + ["series", "ym"]).reset_index(drop=True)


def load_ledger(cfg, df=None):
    """台帳(Excel)を読み、判定に使う形に整える。df を渡せばそれを使う(デモ用)。"""
    if df is None:
        df = pd.read_excel(cfg["ledger_path"], sheet_name=cfg["ledger_sheet"],
                           dtype={"判定年月": str, "再評価年月": str})
    df = df.copy()
    for k in ("事業コード", "開発コード", "部番", "再アラート条件タイプ", "処置区分"):
        if k in df:
            df[k] = df[k].astype(str).replace({"nan": ""})
    df["判定年月"] = df["判定年月"].map(_to_ym)
    df["再評価年月"] = df["再評価年月"].map(_to_ym)
    df["Y"] = pd.to_numeric(df["条件パラメータY"], errors="coerce")
    df["記録日"] = df.get("記録日", "")
    return df[df["事業コード"] != ""].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 状態の解決
# --------------------------------------------------------------------------- #
def machine_end_set(ledger, cfg):
    """機種終了((事業,開発) -> 発効年月)。同一機種の全部品を監視対象外にする。"""
    tok = cfg["machine_all_part_token"]
    m = ledger[(ledger["再アラート条件タイプ"] == "機種終了") | (ledger["部番"] == tok)]
    out = {}
    for _, r in m.iterrows():
        key = (r["事業コード"], r["開発コード"])
        ym = r["判定年月"]
        if ym is not None and (key not in out or ym < out[key]):
            out[key] = ym
    return out


def unit_timelines(ledger, cfg):
    """部品単位((事業,開発,部番) -> 判定年月昇順のDataFrame)。機種終了行は除く。"""
    tok = cfg["machine_all_part_token"]
    u = ledger[ledger["部番"] != tok]
    tl = {}
    for key, g in u.groupby(["事業コード", "開発コード", "部番"]):
        g = g.sort_values(["判定年月", "記録日"], na_position="first")
        tl[key] = g
    return tl


def base_threshold(cfg, biz, dev, part):
    ov = cfg["threshold_overrides"]
    for k in ((biz, dev, part), (biz, dev), (biz,)):
        if k in ov:
            return ov[k]
    return cfg["base_threshold_pct"]


def resolve(cfg, mend, timelines, biz, dev, part, t):
    """月 t 時点での部品の状態・当月閾値・次アラート種別・再評価到来を返す。"""
    base = base_threshold(cfg, biz, dev, part)
    mk = (biz, dev)
    if mk in mend and mend[mk] is not None and mend[mk] <= t:
        return dict(state="終了機種", threshold=np.nan, kind="再", review_due=False, last_ym=None)

    g = timelines.get((biz, dev, part))
    if g is None:
        return dict(state="未記録", threshold=base, kind="初回", review_due=False, last_ym=None)
    past = g[g["判定年月"].apply(lambda x: x is not None and x <= t)]
    if past.empty:
        return dict(state="未記録", threshold=base, kind="初回", review_due=False, last_ym=None)

    last = past.iloc[-1]
    ctype = last["再アラート条件タイプ"]
    if ctype == "監視終了":
        return dict(state="終了単位", threshold=np.nan, kind="再", review_due=False, last_ym=last["判定年月"])
    if ctype == "保留":
        prior = past[past["再アラート条件タイプ"] == "率超過"]
        thr = float(prior.iloc[-1]["Y"]) if len(prior) else base
        state = "要確認保留"
    elif ctype == "率超過":
        thr = float(last["Y"]) if pd.notna(last["Y"]) else base
        state = "監視中"
    else:
        thr = base
        state = "監視中"
    rev = last["再評価年月"]
    review_due = rev is not None and rev <= t
    return dict(state=state, threshold=thr, kind="再", review_due=review_due, last_ym=last["判定年月"])


# --------------------------------------------------------------------------- #
# 出力1: 要確認インボックス
# --------------------------------------------------------------------------- #
def build_inbox(rates, ledger, cfg, asof=None):
    mend = machine_end_set(ledger, cfg)
    tl = unit_timelines(ledger, cfg)
    T = asof if asof is not None else int(rates["ym"].max())
    rows = []
    units = rates[["biz", "dev", "part"]].drop_duplicates().itertuples(index=False)
    for biz, dev, part in units:
        res = resolve(cfg, mend, tl, biz, dev, part, T)
        if res["state"] in ("終了単位", "終了機種"):
            continue
        sub = rates[(rates.biz == biz) & (rates.dev == dev) & (rates.part == part) & (rates.ym == T)]
        elig = sub[sub.eligible & sub.rate_pct.notna()]
        thr = res["threshold"]
        over = elig[elig.rate_pct >= thr] if pd.notna(thr) else elig.iloc[0:0]
        fired = len(over) > 0
        carry = res["state"] == "要確認保留"
        resolved_now = (res["last_ym"] == T) and not carry
        if resolved_now:
            continue
        if not (fired or res["review_due"] or carry):
            continue

        if fired:
            pick = over.loc[over.rate_pct.idxmax()]
            obs, src = pick.rate_pct, pick.series
        elif len(elig):
            pick = elig.loc[elig.rate_pct.idxmax()]
            obs, src = pick.rate_pct, pick.series
        else:
            obs, src = np.nan, ""

        kind = "初回" if res["state"] == "未記録" else "再"
        if carry:
            reason = "保留継続"
        elif fired and res["state"] == "未記録":
            reason = "初回"
        elif fired:
            reason = "再"
        else:
            reason = "再評価"

        rows.append({
            "事業コード": biz, "開発コード": dev, "部番": part,
            "判定元": src, "判定年月": T, "判定種別": kind,
            "観測率": round(obs, 1) if pd.notna(obs) else np.nan,
            "当月閾値": thr, "理由": reason,
            "提案Y下限": round(obs + cfg["margin_pct"], 1) if pd.notna(obs) else np.nan,
        })
    cols = ["事業コード", "開発コード", "部番", "判定元", "判定年月", "判定種別",
            "観測率", "当月閾値", "理由", "提案Y下限"]
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# 出力2: Tableau用 縦長テーブル
# --------------------------------------------------------------------------- #
def build_tableau(rates, ledger, cfg):
    mend = machine_end_set(ledger, cfg)
    tl = unit_timelines(ledger, cfg)
    judge_pts = {(r["事業コード"], r["開発コード"], r["部番"], r["判定年月"])
                 for _, r in ledger.iterrows() if r["判定年月"] is not None}

    um = rates[["biz", "dev", "part", "ym"]].drop_duplicates()
    recs = {}
    for biz, dev, part, ym in um.itertuples(index=False):
        res = resolve(cfg, mend, tl, biz, dev, part, int(ym))
        thr = res["threshold"]
        sub = rates[(rates.biz == biz) & (rates.dev == dev) & (rates.part == part) & (rates.ym == ym)]
        elig = sub[sub.eligible & sub.rate_pct.notna()]
        fired = pd.notna(thr) and (elig.rate_pct >= thr).any()
        is_pt = (biz, dev, part, int(ym)) in judge_pts
        if res["state"] in ("終了単位", "終了機種"):
            st = "終了"
        elif is_pt:
            st = "要確認"
        elif res["state"] == "未記録" and not fired:
            st = "正常"
        elif fired:
            st = "要確認"
        else:
            st = "監視中"
        recs[(biz, dev, part, int(ym))] = (thr, st, is_pt)

    def _thr(r): return recs[(r.biz, r.dev, r.part, int(r.ym))][0]
    def _st(r):  return recs[(r.biz, r.dev, r.part, int(r.ym))][1]
    def _pt(r):  return recs[(r.biz, r.dev, r.part, int(r.ym))][2]

    out = rates.copy()
    out["当月閾値"] = out.apply(_thr, axis=1)
    out["状態"] = out.apply(_st, axis=1)
    out["判定点"] = out.apply(_pt, axis=1)
    out["閾値超"] = out.eligible & out.rate_pct.notna() & out.当月閾値.notna() & (out.rate_pct >= out.当月閾値)
    out["累積率"] = out.rate_pct.round(2)

    # --- 部品レベルの注釈列(全行に同じ値が付く。Tableauの行フィルタ・ソート用) ---
    pkeys = ["biz", "dev", "part"]
    latest = out.loc[out.groupby(pkeys)["ym"].idxmax(), pkeys + ["ym"]] \
                .rename(columns={"ym": "_latest_ym"})
    out = out.merge(latest, on=pkeys, how="left")

    # 台帳記録あり: 部品固有の行がある、または機種終了の機種に属する
    def _has_record(r):
        return ((r.biz, r.dev, r.part) in tl) or ((r.biz, r.dev) in mend)
    rec_flag = {k: _has_record(pd.Series(dict(zip(pkeys, k)))) for k in
                out[pkeys].drop_duplicates().itertuples(index=False, name=None)}
    out["台帳記録あり"] = out.apply(lambda r: rec_flag[(r.biz, r.dev, r.part)], axis=1)

    # 最新状態: 最新月の状態(部品単位)
    lstate = {(r.biz, r.dev, r.part): recs[(r.biz, r.dev, r.part, int(r.ym))][1]
              for r in latest.rename(columns={"_latest_ym": "ym"}).itertuples(index=False)}
    out["最新状態"] = out.apply(lambda r: lstate[(r.biz, r.dev, r.part)], axis=1)

    # 注目度: 最新月における 累積率÷当月閾値 の最大(資格あり系列のみ)。終了部品は NaN
    last_rows = out[out.ym == out._latest_ym]
    cand = last_rows[last_rows.eligible & last_rows.rate_pct.notna() & last_rows.当月閾値.notna()].copy()
    cand["_ratio"] = cand.rate_pct / cand.当月閾値
    att = cand.groupby(pkeys)["_ratio"].max().round(3)
    out["注目度"] = out.set_index(pkeys).index.map(att)
    out = out.drop(columns=["_latest_ym"])

    out = out.rename(columns={"biz": "事業コード", "dev": "開発コード", "part": "部番",
                              "series": "系列", "ym": "年月", "eligible": "トリガ資格"})
    return out[["事業コード", "開発コード", "部番", "系列", "年月", "累積率",
                "cum_use", "cum_sales", "トリガ資格", "当月閾値", "閾値超", "状態", "判定点",
                "台帳記録あり", "最新状態", "注目度"]] \
        .rename(columns={"cum_use": "累積使用数", "cum_sales": "累積販売台数"})


# --------------------------------------------------------------------------- #
# 実行
# --------------------------------------------------------------------------- #
def run(cfg):
    panel = load_panel(cfg)
    ledger = load_ledger(cfg)
    rates = series_rates(panel, cfg)
    inbox = build_inbox(rates, ledger, cfg, cfg["asof_ym"])
    table = build_tableau(rates, ledger, cfg)
    inbox.to_csv(cfg["out_inbox"], index=False, encoding="utf-8-sig")
    table.to_csv(cfg["out_tableau"], index=False, encoding="utf-8-sig")
    return inbox, table


# --------------------------------------------------------------------------- #
# デモ(実データ無しで仕組みを確認)
# --------------------------------------------------------------------------- #
def _demo_panel():
    months = [202601, 202602, 202603, 202604, 202605, 202606, 202607, 202608]
    sales = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700]   # 合算の累積販売台数
    # 合算の累積使用数(これを 東70%/西30%、販売は50/50 に分配)
    use_1001 = [16, 23, 28, 33, 37, 42, 51, 71]   # 202607で合算3.2%(再)、202608で4.18% > Y=4.0(再)
    use_1002 = [10, 12, 15, 18, 21, 24, 27, 41]   # 台帳行なし。202608で合算2.41% > X=2.0 → 初回
    use_2003 = [20, 30, 42, 55, 68, 80, 92, 105]  # DEV-B。機種終了で対象外(高くても出ない)
    rows = []

    def emit(part, dev, comb_use, split_use, dist, half_sales):
        prev = 0
        for ym, cu in zip(months, comb_use):
            du = round(split_use * cu)
            mu = du - prev
            prev = du
            rows.append(dict(事業コード="E10", 開発コード=dev, 部番=part, 販社=dist,
                             年月=ym, 月次使用数=mu, 累積販売台数=half_sales[months.index(ym)]))

    half = [s // 2 for s in sales]
    emit("PN-1001", "DEV-A", use_1001, 0.55, "東販社", half)
    emit("PN-1001", "DEV-A", use_1001, 0.45, "西販社", half)
    emit("PN-1002", "DEV-A", use_1002, 0.50, "東販社", half)
    emit("PN-1002", "DEV-A", use_1002, 0.50, "西販社", half)
    emit("PN-2003", "DEV-B", use_2003, 1.00, "東販社", sales)
    return pd.DataFrame(rows)


def _demo_ledger():
    return pd.DataFrame([
        dict(記録日="2026-02-15", 事業コード="E10", 開発コード="DEV-A", 部番="PN-1001",
             判定元="合算", 判定年月="202602", 判定種別="初回", 観測率=2.1,
             処置区分="既知・経過観察", 再アラート条件タイプ="率超過", 条件パラメータY=3.0,
             再評価年月="", 確認者="山田", メモ="初回越え"),
        dict(記録日="2026-07-12", 事業コード="E10", 開発コード="DEV-A", 部番="PN-1001",
             判定元="東販社", 判定年月="202607", 判定種別="再", 観測率=3.2,
             処置区分="対策投入済み", 再アラート条件タイプ="率超過", 条件パラメータY=4.0,
             再評価年月="", 確認者="山田", メモ="対策ロット投入。再加速のみ監視"),
        dict(記録日="2026-05-20", 事業コード="E10", 開発コード="DEV-B", 部番="機種全体",
             判定元="合算", 判定年月="202605", 判定種別="初回", 観測率=5.0,
             処置区分="機種ごと対象外", 再アラート条件タイプ="機種終了", 条件パラメータY="",
             再評価年月="", 確認者="佐藤", メモ="十分古い既出。機種全体を監視終了。引き金 PN-2003"),
    ])


def run_demo(cfg):
    raw = _demo_panel()
    ren = {v: k for k, v in cfg["cols"].items() if v is not None}
    panel = _prepare_panel(raw.rename(columns=ren), cfg)
    ledger = load_ledger(cfg, df=_demo_ledger())
    rates = series_rates(panel, cfg)
    inbox = build_inbox(rates, ledger, cfg)
    table = build_tableau(rates, ledger, cfg)
    return inbox, table, rates


if __name__ == "__main__":
    if CONFIG["panel_path"]:
        inbox, table = run(CONFIG)
        print("[実行] インボックスと監視テーブルを出力しました。")
    else:
        inbox, table, rates = run_demo(CONFIG)
        pd.set_option("display.unicode.east_asian_width", True)
        pd.set_option("display.width", 200)
        print("=== 当月の各系列の累積率(基準月 202608) ===")
        latest = table[table.年月 == 202608][["事業コード", "開発コード", "部番", "系列",
                                              "累積率", "当月閾値", "閾値超", "状態"]]
        print(latest.to_string(index=False))
        print("\n=== 要確認インボックス(人がレビューして台帳に追記する一覧) ===")
        if inbox.empty:
            print("(なし)")
        else:
            print(inbox.to_string(index=False))
        print("\n注: DEV-B/PN-2003 は機種終了のためインボックスに出ない(累積率は高いが対象外)。")
