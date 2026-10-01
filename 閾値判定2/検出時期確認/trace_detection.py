# -*- coding: utf-8 -*-
"""
trace_detection.py — 機種×部番を指定して「どの月に、どの検出器で検出できたか」を時系列で追う

過去に異常が見つかった部品について、閾値・信号B・信号Cがそれぞれ何年何月に
初めて鳴ったか（鳴らなかった月はその理由）を月ごとに再現する。

使い方（閾値判定2 フォルダで実行）
------------------------------------
    python 検出時期確認/trace_detection.py M02 101-0 --cat レンズ
    python 検出時期確認/trace_detection.py M02 101-0 --from 202301 --to 202412 --cat レンズ
    python 検出時期確認/trace_detection.py M02 101-0 --biz E1 --panel panel.csv
    python 検出時期確認/trace_detection.py --list 対象一覧.csv --cat レンズ

    --list の CSV: 開発コード, 部番 の列（列名は settings.COLS）。事業コード・発生年月
    （YYYYMM）があれば使う。発生年月があると「何ヶ月前に検出できたか」も出す。
    パネルは --panel（省略時はカテゴリ設定の PANEL_PATH）。

出力（出力/<カテゴリ>/検出時期確認/）
--------------------------------------
    検出時期_<機種>_<部番>.csv        … 月ごとの推移（Tableau で折れ線にできる）
    検出時期_<機種>_<部番>_信号C販社別.csv … 信号Cの販社×月の生データ
    検出時期_要約.csv                  … --list のとき、部品ごとの初検出月

再現の仕方と前提
----------------
- 各月 m について「m の時点で run していたら鳴ったか」を検出器ごとに計算する。
  閾値・信号C は部品自身の系列、信号B は同じSF群の機種だけで計算する
  （検出器は単位ごとに独立なので、全部品を回した本番と同じ結果になる）。
- 設定（閾値・min_oe など）は本番と同じ settings / カテゴリ設定を使う。
- **台帳は使わない**（消し込み無しで「検出できたか」を見る）。top_n も考えない。
- **今のパネルで過去を再計算する**。過去の月は当時より修理データが出そろっているので、
  実運用より早めに検出されて見えることがある。
- **販社の報告遅れ（horizon）は考慮しない**。データが揃っていた場合の最速の検出月。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # 閾値判定2
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

import config_loader as cl
cl.activate_from_argv()          # --cat X でカテゴリ設定に切り替え
import settings as st
import state_logic as sl
import state_logic_cusum as slc
import signal_b_peer as sb
import signal_c_dist as sd


# ============================================================================
# 読み込み
# ============================================================================
def load_panel(panel_path: str | None = None) -> pd.DataFrame:
    """パネルを読み、run_month と同じ前処理（年月の数字化・累積台数の cummax）を通す。"""
    path = panel_path or cl.panel_arg([], 0)
    c = st.COLS
    raw = pd.read_csv(path, encoding="utf-8-sig")
    key = [c["biz"], c["dev"], c["part"], c["dist"]]
    raw[c["ym"]] = raw[c["ym"]].astype(str).str.replace(r"\D", "", regex=True).astype(int)
    raw = raw.sort_values(key + [c["ym"]])
    raw[c["cum_sales"]] = raw.groupby(key)[c["cum_sales"]].cummax()
    for k in ("biz", "dev", "part"):
        raw[c[k]] = raw[c[k]].astype(str)
    raw.attrs["path"] = path
    return raw


def _resolve_biz(raw, dev, part, biz=None) -> str:
    c = st.COLS
    m = raw[(raw[c["dev"]] == str(dev)) & (raw[c["part"]] == str(part))]
    if m.empty:
        raise SystemExit(f"[エラー] パネルに {dev} / {part} がありません（綴り・全角半角を確認）")
    bizs = sorted(m[c["biz"]].unique())
    if biz is not None:
        if str(biz) not in bizs:
            raise SystemExit(f"[エラー] {dev} / {part} の事業コードは {bizs} です（指定={biz}）")
        return str(biz)
    if len(bizs) > 1:
        raise SystemExit(f"[エラー] {dev} / {part} が複数の事業コードにあります: {bizs}。--biz で指定")
    return bizs[0]


# ============================================================================
# 検出器ごとの時系列
# ============================================================================
def _threshold_series(p_all_unit: pd.DataFrame, cfg: dict, biz, dev, part) -> pd.DataFrame:
    """閾値: 月ごとの累積率と当月閾値。build_unified_inbox と同じ前処理。"""
    sl_cfg = dict(sl.CONFIG)
    sl_cfg["cols"] = {**sl.CONFIG["cols"],
                      **{k: v for k, v in st.COLS.items() if k != "part_name"}}
    sl_cfg.update({k: cfg[k] for k in
                   ("base_threshold_pct", "threshold_overrides", "min_denominator")})
    ren = {v: k for k, v in sl_cfg["cols"].items() if v is not None}
    p = sl._prepare_panel(p_all_unit.rename(columns=ren).copy(), sl_cfg)
    r = sl.series_rates(p, sl_cfg)
    r = r[r["eligible"] & r["rate_pct"].notna()]
    g = r.groupby("ym", as_index=False).agg(累積率=("rate_pct", "max"),
                                            累積使用数=("cum_use", "max"),
                                            累積台数=("cum_sales", "max"))
    thr = sl.base_threshold(cfg, biz, dev, part)
    g["閾値"] = thr
    g["閾値判定"] = g["累積率"] >= thr
    g["閾値理由"] = np.where(g["閾値判定"], "★発火",
                         [f"累積率{x:.2f}%<{thr:.2f}%" for x in g["累積率"]])
    return g


def _b_group_panel(raw: pd.DataFrame, biz: str, sf) -> pd.DataFrame:
    """信号B: 同じ (事業, SF) の全機種の ALL 行を CUSUM 側前処理に通したもの。"""
    c = st.COLS
    g = raw[(raw[c["biz"]] == biz) & (raw[c["sf"]].astype(str) == str(sf))
            & (raw[c["dist"]].astype(str) == st.ALL_TOKEN)]
    sc_cfg = dict(slc.CONFIG)
    sc_cfg["cols"] = {**slc.CONFIG["cols"], **st.COLS}
    return slc._prepare_panel(g.copy(), sc_cfg)


def _signal_b_at(pb_group: pd.DataFrame, m: int, dev: str, part: str, cfg: dict) -> dict:
    """月 m 時点の信号B（本番と同じ2パス）。対象機種×SFの行と、鳴らない理由。"""
    cap, mp = cfg["b_elapsed_cap"], cfg["b_min_peers"]
    d = pb_group[pb_group["ym"] <= m]
    out = dict(信号B_O_E=np.nan, 信号B_p=np.nan, 信号B_C=np.nan, 信号B_ピア数=np.nan,
               信号B_nb=np.nan, 信号B判定=False, 信号B理由="")
    if d.empty:
        out["信号B理由"] = "データなし"
        return out
    u_all = sb.build_peer_units(d, elapsed_cap=cap, require_full=False)
    me = u_all[u_all["dev"] == dev]
    if me.empty:
        out["信号B理由"] = "判定対象外（SFの実績なし）"
        return out
    me = me.iloc[0]
    out["信号B_nb"] = int(me["nb"])
    if me["cover"] < cap:
        out["信号B理由"] = f"窓未完走（経過月{int(me['cover'])}<{cap}）"
        return out
    res = sb.run_signal_b(d, elapsed_cap=cap, min_peers=mp, alpha_peer=cfg["b_alpha"],
                          min_count=cfg["b_min_count"],
                          min_peer_count=cfg.get("b_min_peer_count", 20),
                          min_oe=cfg.get("b_min_oe", 1.5))
    row = res[res["dev"] == dev] if len(res) else res
    if row is None or len(row) == 0:
        u_full = u_all[u_all["cover"] >= cap]
        same = int((u_full["nb"] == me["nb"]).sum()) - 1
        out["信号B理由"] = f"ピア不足（同nb={int(me['nb'])}の完走機種{same}<{mp}）"
        return out
    r = row.iloc[0]
    out.update(信号B_O_E=round(float(r["O_E"]), 3), 信号B_p=float(r["p"]),
               信号B_C=float(r["C"]), 信号B_ピア数=int(r["n_peers"]))
    if bool(r["alert_peer"]):
        # 本番は部番に展開する: SF内シェアが b_expand_min_share 未満の部番は出ない
        dd = d[(d["dev"] == dev) & (d["elapsed"] <= cap)]
        pu = dd.groupby("part")["use"].sum()
        share = float(pu.get(part, 0.0)) / max(float(pu.sum()), 1e-9)
        thr = cfg.get("b_expand_min_share", 0.1)
        if pu.get(part, 0.0) <= 0 or (share < thr and pu.get(part, 0.0) < pu.max()):
            out["信号B理由"] = f"SFは発火・この部番はシェア{share*100:.0f}%<{thr*100:.0f}%で非表示"
            return out
        out["信号B判定"] = True
        out["信号B理由"] = "★発火"
        return out
    why = []
    if r["O_E"] < cfg.get("b_min_oe", 1.5):
        why.append(f"O/E{r['O_E']:.2f}<{cfg.get('b_min_oe', 1.5)}")
    if r["C"] < cfg["b_min_count"]:
        why.append(f"件数{int(r['C'])}<{cfg['b_min_count']}")
    if r["C_peer"] < cfg.get("b_min_peer_count", 20):
        why.append(f"ピア件数{int(r['C_peer'])}<{cfg.get('b_min_peer_count', 20)}")
    if r["p"] > cfg["b_alpha"]:
        why.append(f"p={r['p']:.1e}>α")
    out["信号B理由"] = " / ".join(why) or "非発火"
    return out


def _signal_c_series(p_dist_unit: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """信号C: 販社×月の判定。各月はそれ以前のデータだけで判定されるので、
    全月を一度に計算すれば「その月に run していたら」と同じになる。"""
    if p_dist_unit.empty:
        return pd.DataFrame()
    kw = dict(base_len=cfg["c_base_len"], alpha=cfg["c_alpha"], min_count=cfg["c_min_count"],
              min_base_months=cfg["c_min_base_months"], min_base_count=cfg["c_min_base_count"],
              min_oe=cfg.get("c_min_oe", 3.0), min_excess=cfg.get("c_min_excess", 0.0),
              exceed_hist=cfg.get("c_exceed_hist", 0.0))
    ex = cfg.get("c_exclude") or None
    res = sd.run_signal_c(p_dist_unit, st.COLS, exclude=ex, months_back=100000,
                          all_token=st.ALL_TOKEN, **kw)
    if res is None or res.empty:
        return pd.DataFrame()
    # 鳴らない理由（signal_c_dist.inspect と同じ判定順）
    res = res.sort_values(["dist", "ym"]).reset_index(drop=True)
    res["月番号"] = res.groupby("dist").cumcount()
    why = []
    for r in res.itertuples():
        if r.alert_dist:
            w = "★発火"
        elif r.月番号 < kw["min_base_months"]:
            w = "ベースライン月数不足"
        elif np.isnan(r.O_E):
            w = "判定対象外"
        elif r.use < kw["min_count"]:
            w = f"件数{int(r.use)}<{kw['min_count']}"
        elif r.O_E < kw["min_oe"]:
            w = f"O/E{r.O_E:.1f}<{kw['min_oe']}"
        elif np.isnan(r.p) and kw["exceed_hist"] > 0:
            w = f"自己履歴超えず(O/E{r.O_E:.1f})"
        elif np.isnan(r.p):
            w = "増加量不足"
        else:
            w = f"p={r.p:.1e}>α"
        why.append(w)
    res["理由"] = why
    return res.drop(columns=["月番号", "run_ym", "遅延月"], errors="ignore")


def _c_month_summary(cres: pd.DataFrame) -> pd.DataFrame:
    if cres is None or cres.empty:
        return pd.DataFrame(columns=["ym"])
    rows = []
    for ym, g in cres.groupby("ym"):
        fired = g[g["alert_dist"]]
        pick = (fired if len(fired) else g).sort_values("O_E", ascending=False,
                                                        na_position="last").iloc[0]
        rows.append(dict(ym=int(ym),
                         信号C_最大O_E=round(float(pick["O_E"]), 3) if pick["O_E"] == pick["O_E"] else np.nan,
                         信号C_販社=pick["dist"], 信号C_使用数=float(pick["use"]),
                         信号C判定=bool(len(fired)),
                         信号C発火販社="/".join(sorted(fired["dist"].astype(str))),
                         信号C理由="★発火" if len(fired) else f"最大{pick['dist']}: {pick['理由']}"))
    return pd.DataFrame(rows)


# ============================================================================
# 本体
# ============================================================================
def trace(dev, part, biz=None, raw: pd.DataFrame | None = None,
          ym_from: int | None = None, ym_to: int | None = None,
          save: bool = True, quiet: bool = False) -> dict:
    """1部品の検出時期を追う。Returns dict(timeline, c_detail, summary)。"""
    raw = raw if raw is not None else load_panel()
    c = st.COLS
    cfg = st.build_cfg()
    dev, part = str(dev), str(part)
    biz = _resolve_biz(raw, dev, part, biz)

    unit = raw[(raw[c["biz"]] == biz) & (raw[c["dev"]] == dev) & (raw[c["part"]] == part)]
    is_all = unit[c["dist"]].astype(str) == st.ALL_TOKEN
    u_all, u_dist = unit[is_all], unit[~is_all]
    if u_all.empty:
        raise SystemExit(f"[エラー] {dev} / {part} に {st.ALL_TOKEN} 行がありません")

    # --- 閾値 ---
    tl = _threshold_series(u_all, cfg, biz, dev, part)
    base = (u_all[[c["ym"], c["elapsed"], c["monthly_use"]]]
            .rename(columns={c["ym"]: "ym", c["elapsed"]: "経過月", c["monthly_use"]: "月次使用数"}))
    tl = base.merge(tl, on="ym", how="left")
    data_start = int(tl["ym"].min())
    lo = ym_from or data_start
    hi = ym_to or int(tl["ym"].max())
    tl = tl[(tl["ym"] >= lo) & (tl["ym"] <= hi)].reset_index(drop=True)

    # --- 信号B ---
    sfs = u_all[c["sf"]].dropna().astype(str).unique()
    if len(sfs):
        sf = sfs[0]
        pb = _b_group_panel(raw, biz, sf)
        b = pd.DataFrame([dict(ym=int(m), **_signal_b_at(pb, int(m), dev, part, cfg))
                          for m in tl["ym"]])
        b.insert(1, "信号B_SF", sf)
    else:
        b = pd.DataFrame(dict(ym=tl["ym"], 信号B_SF="", 信号B判定=False, 信号B理由="SF-CODEなし"))
    tl = tl.merge(b, on="ym", how="left")

    # --- 信号C ---
    cres = _signal_c_series(u_dist, cfg)
    tl = tl.merge(_c_month_summary(cres), on="ym", how="left")
    tl["信号C判定"] = tl["信号C判定"].eq(True)        # 欠損（販社データなし）は False
    tl["信号C理由"] = tl["信号C理由"].fillna("販社別データなし")

    det = [("閾値", "閾値判定"), ("信号B", "信号B判定"), ("信号C", "信号C判定")]
    tl["鳴った検出器"] = ["/".join(n for n, col in det if bool(r[col])) for _, r in tl.iterrows()]
    import unified_inbox as ui
    names = ui.part_name_map([unit], c)
    pname = names["部品名"].iloc[0] if len(names) else ""
    tl.insert(0, "部品名", pname)
    tl.insert(0, "部番", part); tl.insert(0, "開発コード", dev); tl.insert(0, "事業コード", biz)
    tl = tl.rename(columns={"ym": "年月"})
    if cl.category():
        tl["カテゴリ"] = cl.category()

    # --- 要約 ---
    summ = []
    for name, col in det:
        hit = tl[tl[col].astype(bool)]
        if len(hit):
            h = hit.iloc[0]
            if name == "閾値":
                val = f"累積率{h['累積率']:.2f}% / 閾値{h['閾値']:.2f}%"
            elif name == "信号B":
                val = f"{h['信号B_SF']} O/E={h['信号B_O_E']:.2f}（ピア{int(h['信号B_ピア数'])}機種）"
            else:
                val = f"販社{h['信号C発火販社']} O/E={h['信号C_最大O_E']:.2f} 使用{int(h['信号C_使用数'])}件"
            if int(h["年月"]) == lo and lo > data_start and \
                    bool(tl.iloc[0][col]):
                val += "（期間の開始時点ですでに発火中。--from を外すと本当の初検出月が出る）"
            summ.append(dict(検出器=name, 初検出=int(h["年月"]), 経過月=int(h["経過月"]),
                             鳴った月数=len(hit), そのときの値=val))
        else:
            last = tl.iloc[-1][col.replace("判定", "理由")]
            summ.append(dict(検出器=name, 初検出=None, 経過月=None, 鳴った月数=0,
                             そのときの値=f"期間内に検出なし（最終月: {last}）"))
    summ = pd.DataFrame(summ)
    for k in ("初検出", "経過月"):
        summ[k] = pd.to_numeric(summ[k], errors="coerce").astype("Int64")
    first = summ.dropna(subset=["初検出"])
    earliest = (int(first["初検出"].min()),
                "/".join(first[first["初検出"] == first["初検出"].min()]["検出器"])) if len(first) else (None, "")

    if not quiet:
        print(f"\n=== {dev} / {part}" + (f" {pname}" if pname else "") + f"（{biz}）  期間 {lo}〜{hi}"
              + (f"  カテゴリ={cl.category()}" if cl.category() else "") + " ===")
        print(summ.to_string(index=False))
        print(f"→ 最も早い検出: {earliest[0]}（{earliest[1]}）" if earliest[0]
              else "→ 期間内にどの検出器でも検出されていない")

    paths = {}
    if save:
        d = os.path.join(cl.base_dir(), "検出時期確認")
        os.makedirs(d, exist_ok=True)
        stem = f"検出時期_{dev}_{part}".replace("/", "_")
        paths["timeline"] = os.path.join(d, f"{stem}.csv")
        tl.to_csv(paths["timeline"], index=False, encoding="utf-8-sig")
        if cres is not None and len(cres):
            cc = cres[(cres["ym"] >= lo) & (cres["ym"] <= hi)]
            paths["c_detail"] = os.path.join(d, f"{stem}_信号C販社別.csv")
            cc.to_csv(paths["c_detail"], index=False, encoding="utf-8-sig")
        if not quiet:
            print(f"推移: {paths['timeline']}")
    return dict(timeline=tl, c_detail=cres, summary=summ, earliest=earliest,
                biz=biz, paths=paths)


def trace_list(list_path: str, raw: pd.DataFrame | None = None,
               ym_from=None, ym_to=None) -> pd.DataFrame:
    """一覧CSVの部品をまとめて追い、部品ごとの初検出月を要約する。"""
    raw = raw if raw is not None else load_panel()
    c, L = st.COLS, st.LABEL_COLS
    lab = pd.read_csv(list_path, encoding="utf-8-sig", dtype=str)
    col_dev = c["dev"] if c["dev"] in lab.columns else L["dev"]
    col_part = c["part"] if c["part"] in lab.columns else L["part"]
    col_biz = c["biz"] if c["biz"] in lab.columns else (L["biz"] if L["biz"] in lab.columns else None)
    col_ym = L["ym"] if L["ym"] in lab.columns else None
    rows = []
    for r in lab.itertuples(index=False):
        rr = r._asdict()
        dev, part = rr[col_dev], rr[col_part]
        biz = rr[col_biz] if col_biz else None
        try:
            res = trace(dev, part, biz, raw=raw, ym_from=ym_from, ym_to=ym_to)
        except SystemExit as e:
            print(e)
            rows.append(dict(開発コード=dev, 部番=part, 結果=str(e)))
            continue
        s = res["summary"].set_index("検出器")["初検出"]
        row = dict(事業コード=res["biz"], 開発コード=dev, 部番=part,
                   閾値=s.get("閾値"), 信号B=s.get("信号B"), 信号C=s.get("信号C"),
                   最早=res["earliest"][0], 最早の検出器=res["earliest"][1])
        row = {**{k: row[k] for k in ("事業コード", "開発コード", "部番")},
               "部品名": res["timeline"]["部品名"].iloc[0] if len(res["timeline"]) else "",
               **{k: v for k, v in row.items() if k not in ("事業コード", "開発コード", "部番")}}
        if col_ym and rr.get(col_ym):
            rep = int("".join(ch for ch in str(rr[col_ym]) if ch.isdigit())[:6])
            row["発生年月"] = rep
            if row["最早"]:
                row["先行月数"] = slc_diff(rep, int(row["最早"]))
        row["結果"] = "検出あり" if row["最早"] else "検出なし"
        rows.append(row)
    out = pd.DataFrame(rows)
    for k in ("閾値", "信号B", "信号C", "最早", "発生年月", "先行月数"):
        if k in out.columns:
            out[k] = pd.to_numeric(out[k], errors="coerce").astype("Int64")
    d = os.path.join(cl.base_dir(), "検出時期確認")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "検出時期_要約.csv")
    if cl.category():
        out["カテゴリ"] = cl.category()
    out.to_csv(p, index=False, encoding="utf-8-sig")
    print("\n=== 要約 ===")
    print(out.to_string(index=False))
    if "先行月数" in out.columns:
        print("  先行月数 = 発生年月 − 最早の検出月（正なら発生年月より前に検出できた）")
    print(f"要約: {p}")
    return out


def slc_diff(a: int, b: int) -> int:
    """a − b を月数で。"""
    ya, ma = divmod(int(a), 100)
    yb, mb = divmod(int(b), 100)
    return (ya * 12 + ma) - (yb * 12 + mb)


def _opt(args, name, cast=str):
    if name in args:
        i = args.index(name)
        v = args[i + 1]
        del args[i:i + 2]
        return cast(v)
    return None


if __name__ == "__main__":
    args = sys.argv[1:]
    ym_from = _opt(args, "--from", int)
    ym_to = _opt(args, "--to", int)
    biz = _opt(args, "--biz")
    panel = _opt(args, "--panel")
    lst = _opt(args, "--list")
    # 利用者が渡した相対パスを先に絶対パス化し、実行フォルダを 閾値判定2 に揃える
    # （カテゴリ設定のパス・出力先は 閾値判定2 からの相対のため）
    panel = os.path.abspath(panel) if panel else None
    lst = os.path.abspath(lst) if lst else None
    os.chdir(ROOT)
    if not lst and len(args) < 2:
        print(__doc__)
        sys.exit(1)
    raw = load_panel(cl.resolve_path(panel) if panel else None)
    print(f"パネル: {raw.attrs['path']}")
    if lst:
        trace_list(lst, raw=raw, ym_from=ym_from, ym_to=ym_to)
    else:
        trace(args[0], args[1], biz, raw=raw, ym_from=ym_from, ym_to=ym_to)
