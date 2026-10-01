# -*- coding: utf-8 -*-
"""
explain_signal_b.py — 信号Bで「なぜこの部番が検出されたのか」を分解して示す

「経過月×累積使用率で同じSFの部品を並べると、他機種より低く見えるのに信号Bで鳴る」
ときに、信号Bの計算のどこがプロットと違うのかを切り分ける。

使い方（閾値判定2 フォルダで実行）
------------------------------------
    python 検出時期確認/explain_signal_b.py M02 101-0 --cat レンズ
    python 検出時期確認/explain_signal_b.py M02 101-0 --asof 202406 --cat レンズ   # 過去の月で
    python 検出時期確認/explain_signal_b.py M02 101-0 --biz E1 --panel panel.csv

出力
----
画面: ① 比較の内訳（どの機種が比較相手で、誰がなぜ外れたか）
      ② 対象のSF内の部番内訳
      ③ 見方を変えたときのO/E（どの違いが効いて鳴ったか）
CSV（出力/<カテゴリ>/検出時期確認/）:
      信号B内訳_<機種>_<部番>.csv       … ①の表
      信号B曲線_<機種>_<部番>.csv       … 機種×部番×経過月の3つの物差し（Tableau 用）

プロットと信号Bの違い（③で1つずつ外して効き目を見る）
----------------------------------------------------
  1. 部番単位 ↔ 信号Bは SF内の全部番を合算して判定し、シェア10%以上の部番に割り当てる
  2. 分母が累積販売台数 ↔ 信号Bは「台数×月」の合計（市場に置かれていた量）
     → 販売が窓の後半に偏った機種は、累積率が低くても台数×月あたりでは高くなる
  3. 同SFの全機種 ↔ 信号Bは 同じnb（部番数）かつ 経過月 cap まで完走した機種だけ
  4. 2パス: 1回目で鳴った機種を比較相手から外して基準を下げる
  5. 各機種を同じ重みで見る ↔ 信号Bの基準は 件数合計÷露出合計（台数の多い機種が支配）
  6. 経過月 cap より先も見る ↔ 信号Bは 経過月0〜cap だけ
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

import config_loader as cl
cl.activate_from_argv()
import settings as st
import signal_b_peer as sb
import reporting_horizon as rh
import unified_inbox as ui
import trace_detection as td


# ============================================================================
def _truncate_month(raw: pd.DataFrame, T: int, cfg: dict) -> tuple[int, str]:
    """本番と同じ打ち切り月。USE_HORIZON かつ B_TRUNCATE_TO_HORIZON なら global horizon。"""
    c = st.COLS
    if cfg.get("use_horizon") and cfg.get("b_truncate_to_horizon", True):
        d = raw[(raw[c["dist"]].astype(str) != st.ALL_TOKEN) & (raw[c["ym"]] <= T)]
        if len(d):
            hz = rh.estimate_horizon(
                d, c, all_token=st.ALL_TOKEN,
                margin_months=cfg.get("horizon_margin_months", 0),
                margin_overrides=cfg.get("horizon_margin_overrides") or {},
                fixed=cfg.get("horizon_fixed") or {},
                auto_margin=cfg.get("horizon_auto_margin", True),
                thin_ratio=cfg.get("horizon_thin_ratio", 0.7))
            g = rh.global_horizon(hz)
            if g is not None and g < T:
                return int(g), f"（報告遅れのため global horizon {g} で打ち切り）"
    return int(T), ""


def _fleet_at(d: pd.DataFrame, cap: int) -> pd.Series:
    """機種ごとの『経過月 cap 時点の累積販売台数』（部番をまたいで max）。"""
    dd = d[d["elapsed"] <= cap]
    last = dd.groupby("dev")["elapsed"].transform("max")
    return dd[dd["elapsed"] == last].groupby("dev")["fleet"].max()


def explain(dev, part, biz=None, raw=None, asof: int | None = None,
            save: bool = True, quiet: bool = False) -> dict:
    raw = raw if raw is not None else td.load_panel()
    c = st.COLS
    cfg = st.build_cfg()
    dev, part = str(dev), str(part)
    biz = td._resolve_biz(raw, dev, part, biz)
    cap, mp = cfg["b_elapsed_cap"], cfg["b_min_peers"]
    min_oe, alpha = cfg.get("b_min_oe", 1.5), cfg["b_alpha"]
    mc, mpc = cfg["b_min_count"], cfg.get("b_min_peer_count", 20)

    unit_all = raw[(raw[c["biz"]] == biz) & (raw[c["dev"]] == dev) & (raw[c["part"]] == part)
                   & (raw[c["dist"]].astype(str) == st.ALL_TOKEN)]
    sfs = unit_all[c["sf"]].dropna().astype(str).unique()
    if not len(sfs):
        raise SystemExit(f"[エラー] {dev} / {part} に SF-CODE がありません（信号Bの対象外）")
    sf = sfs[0]
    T0 = int(asof) if asof else int(raw[c["ym"]].max())
    T, note = _truncate_month(raw, T0, cfg)

    pb = td._b_group_panel(raw, biz, sf)
    d = pb[pb["ym"] <= T]
    names = ui.part_name_map([raw[raw[c["dist"]].astype(str) == st.ALL_TOKEN]], c)
    nm = {(r.開発コード, r.部番): r.部品名 for r in names.itertuples()} if len(names) else {}

    # ---------------- 単位（機種×SF）と比較の扱い ----------------
    u = sb.build_peer_units(d, elapsed_cap=cap, require_full=False)
    if dev not in set(u["dev"]):
        raise SystemExit(f"[エラー] {dev} は {sf} の経過月0〜{cap}に実績がありません")
    S = _fleet_at(d, cap)
    u["販売台数@cap"] = u["dev"].map(S)
    u["累積使用率@cap"] = 100.0 * u["C"] / u["販売台数@cap"]
    u["台数×月あたり率"] = 100.0 * u["C"] / u["E"]
    u["平均在籍月数"] = u["E"] / u["販売台数@cap"]
    me = u[u["dev"] == dev].iloc[0]
    full = u[u["cover"] >= cap]
    key = ["biz", "sf", "nb"] if cfg.get("b_nb_strat", True) else ["biz", "sf"]

    # 本番と同じ2パスを再現して、誰が比較相手かを確定する
    p1 = sb._peer_compare(full, key, mp)
    bad = ((p1["p"] <= alpha) & (p1["C"] >= mc) & (p1["C_peer"] >= mpc) & (p1["O_E"] >= min_oe)) \
        if len(p1) else pd.Series(dtype=bool)
    flagged = set(p1.loc[bad, "dev"]) if len(p1) else set()
    final = sb.run_signal_b(d, elapsed_cap=cap, min_peers=mp, alpha_peer=alpha, min_count=mc,
                            min_peer_count=mpc, min_oe=min_oe,
                            nb_strat=cfg.get("b_nb_strat", True), two_pass=True)
    row = final[final["dev"] == dev]
    used_pass2 = False
    if len(row) and flagged and cfg.get("b_two_pass", True):
        excl = full["dev"].isin(flagged).to_numpy()
        p2 = sb._peer_compare(full, key, mp, exclude=excl)
        used_pass2 = dev in set(p2["dev"]) if len(p2) else False

    same_nb = (u["nb"] == me["nb"]) if cfg.get("b_nb_strat", True) else True
    status = []
    for r in u.itertuples():
        if r.dev == dev:
            status.append("対象")
        elif r.cover < cap:
            status.append(f"×窓未完走（経過月{int(r.cover)}<{cap}）")
        elif cfg.get("b_nb_strat", True) and r.nb != me["nb"]:
            status.append(f"×nb違い（{int(r.nb)}≠{int(me['nb'])}）")
        elif used_pass2 and r.dev in flagged:
            status.append("×1回目で発火→比較から除外")
        else:
            status.append("○比較相手")
    u["比較の扱い"] = status
    peers = u[u["比較の扱い"] == "○比較相手"]
    Ep = float(peers["E"].sum())
    u["基準への重み"] = np.where(u["比較の扱い"] == "○比較相手", u["E"] / Ep if Ep else np.nan, np.nan)

    # ---------------- 対象SF内の部番 ----------------
    dt = d[(d["dev"] == dev) & (d["elapsed"] <= cap)]
    pu = dt.groupby("part")["use"].sum().sort_values(ascending=False)
    share = pu / max(float(pu.sum()), 1e-9)
    thr = cfg.get("b_expand_min_share", 0.1)
    parts = pd.DataFrame(dict(部番=pu.index, 件数=pu.values.astype(int),
                              シェア=(100 * share).round(1).values))
    parts["部品名"] = [nm.get((dev, p), "") for p in parts["部番"]]
    parts["累積使用率@cap"] = (100 * parts["件数"] / me["販売台数@cap"]).round(3)
    parts["レビューに出るか"] = ["○" if (s >= thr or i == 0) else "×(シェア不足)"
                          for i, s in enumerate(share.values)]

    # ---------------- 見方を変えたときの O/E ----------------
    alt = []
    def add(name, oe, what):
        alt.append(dict(見方=name, O_E=round(float(oe), 3) if oe == oe else np.nan,
                        判定=("鳴る" if oe == oe and oe >= min_oe else "鳴らない"), 内容=what))
    oe_final = float(row["O_E"].iloc[0]) if len(row) else np.nan
    fired = bool(row["alert_peer"].iloc[0]) if len(row) else False
    add("実際の信号B", oe_final, f"SF合計・台数×月・同nb完走機種・{'2パス' if used_pass2 else '1パス'}")
    if len(peers):
        rate_p = peers["C"].sum() / Ep
        if used_pass2:
            pp = u[u["比較の扱い"].isin(["○比較相手", "×1回目で発火→比較から除外"])]
            add("4を外す: 2パス目の除外なし", me["C"] / me["E"] / (pp["C"].sum() / pp["E"].sum()),
                "1回目に鳴った機種も比較相手に戻す")
        add("2を外す: 分母を累積販売台数に",
            (me["C"] / me["販売台数@cap"]) / (peers["C"].sum() / peers["販売台数@cap"].sum()),
            "プロットと同じ分母（市場に置かれた期間を考えない）")
        add("5を外す: 機種を同じ重みに", (me["C"] / me["E"]) / peers["台数×月あたり率"].div(100).mean(),
            "比較相手の率を単純平均")
        add("2と5を外す: 累積使用率の単純平均と比較",
            me["累積使用率@cap"] / peers["累積使用率@cap"].mean(), "プロットを目で見る比較に最も近い")
        top = parts["件数"].iloc[0] if len(parts) else 0
        p_share = float(share.get(part, 0.0))
        add("1を外す: この部番だけで比べる", oe_final * p_share if oe_final == oe_final else np.nan,
            f"SF合計のうちこの部番の分（{p_share*100:.0f}%）だけ")
    others = full[(full["dev"] != dev) & ~(full["dev"].isin(flagged) if used_pass2 else False)]
    if len(others):
        add("3を外す: 完走した同SF全機種と比較（nb問わず）",
            (me["C"] / me["E"]) / (others["C"].sum() / others["E"].sum()),
            "nb層別をしない（2パスの除外は実際と同じ）")

    alt = pd.DataFrame(alt)

    # ---------------- 表示 ----------------
    if not quiet:
        pn = nm.get((dev, part), "")
        print(f"\n=== 信号Bの内訳: {dev} / {part}{(' ' + pn) if pn else ''}（{biz}） SF={sf} ===")
        print(f"判定月 {T}{note}  比較に使う区間: 経過月0〜{cap}  "
              f"条件: O/E≥{min_oe}, 件数≥{mc}, ピア件数≥{mpc}, p≤{alpha}")
        if me["cover"] < cap:
            print(f"\n★ この機種はまだ経過月{int(me['cover'])}で、{cap}まで完走していないため"
                  "信号Bの判定対象外です。")
        elif not len(row):
            print(f"\n★ 同じnb={int(me['nb'])}で完走した比較相手が{mp}機種未満のため判定対象外です。")
        else:
            r0 = row.iloc[0]
            print(f"→ {'★発火' if fired else '沈黙'}  O/E={r0.O_E:.2f} "
                  f"(対象 {me['台数×月あたり率']:.4f}% / 基準 {100*r0.peer_rate:.4f}%)  "
                  f"件数{int(r0.C)}  ピア{int(r0.n_peers)}機種 件数{int(r0.C_peer)}  p={r0.p:.1e}")

        show = u.sort_values(["比較の扱い", "dev"]).copy()
        show = pd.concat([show[show["比較の扱い"] == "対象"], show[show["比較の扱い"] != "対象"]])
        print("\n① 比較の内訳（累積使用率はプロットの物差し、台数×月あたり率が信号Bの物差し）")
        print(show[["dev", "nb", "cover", "比較の扱い", "C", "販売台数@cap", "累積使用率@cap",
                    "台数×月あたり率", "平均在籍月数", "基準への重み"]]
              .rename(columns={"dev": "機種", "cover": "経過月", "C": "件数"})
              .round({"累積使用率@cap": 3, "台数×月あたり率": 4, "平均在籍月数": 1, "基準への重み": 2})
              .to_string(index=False))
        print("   平均在籍月数 = 台数×月 ÷ 販売台数。小さいほど販売が窓の後半に偏っている")
        print("   基準への重み = 比較相手の基準率に占める露出の割合（大きい機種が基準をほぼ決める）")

        print(f"\n② {dev} の {sf} 内の部番（信号Bはこれを合算して判定）")
        print(parts.to_string(index=False))

        print("\n③ 見方を変えたときの O/E（『鳴らない』になる見方が、鳴った理由）")
        print(alt.to_string(index=False))
        if fired:
            why = alt[(alt["見方"] != "実際の信号B") & (alt["判定"] == "鳴らない")]
            if len(why):
                print("→ 次の違いを外すと鳴らなくなる: " + " / ".join(why["見方"].str.split(":").str[0]))
            else:
                print("→ どの見方でも鳴る。プロットとの差は経過月の範囲（6）か表示の重なりの可能性")

    # ---------------- 曲線CSV（Tableau 用）----------------
    curves = _curves(pb[pb["ym"] <= T], u.set_index("dev")["比較の扱い"].to_dict(), cap, nm)
    out = dict(units=u, parts=parts, alternatives=alt, curves=curves, fired=fired,
               oe=oe_final, sf=sf, T=T)
    if save:
        dd = os.path.join(cl.base_dir(), "検出時期確認")
        os.makedirs(dd, exist_ok=True)
        stem = f"{dev}_{part}".replace("/", "_")
        p1_ = os.path.join(dd, f"信号B内訳_{stem}.csv")
        p2_ = os.path.join(dd, f"信号B曲線_{stem}.csv")
        u.assign(対象部番=part, 判定月=T).to_csv(p1_, index=False, encoding="utf-8-sig")
        curves.to_csv(p2_, index=False, encoding="utf-8-sig")
        if not quiet:
            print(f"\n内訳: {p1_}\n曲線: {p2_}")
    return out


def _curves(d: pd.DataFrame, status: dict, cap: int, nm: dict) -> pd.DataFrame:
    """機種×（部番 / SF合計）×経過月 の3つの物差し。"""
    fm = d.groupby(["dev", "elapsed"], as_index=False)["fleet"].max()       # 部番で水増ししない
    fm = fm.sort_values(["dev", "elapsed"])
    fm["累積露出(台数×月)"] = fm.groupby("dev")["fleet"].cumsum()
    rows = []
    byp = d.groupby(["dev", "part", "elapsed"], as_index=False)["use"].sum()
    sft = d.groupby(["dev", "elapsed"], as_index=False)["use"].sum().assign(part="SF合計")
    for g in (byp, sft):
        g = g.sort_values(["dev", "part", "elapsed"]).copy()
        g["累積使用数"] = g.groupby(["dev", "part"])["use"].cumsum()
        g = g.merge(fm, on=["dev", "elapsed"], how="left")
        rows.append(g)
    cv = pd.concat(rows, ignore_index=True)
    cv["累積使用率%"] = 100 * cv["累積使用数"] / cv["fleet"]
    cv["台数×月あたり率%"] = 100 * cv["累積使用数"] / cv["累積露出(台数×月)"]
    cv["窓内"] = cv["elapsed"] <= cap
    cv["比較の扱い"] = cv["dev"].map(status).fillna("")
    cv["部品名"] = [nm.get((a, b), "") for a, b in zip(cv["dev"], cv["part"])]
    cv = cv.rename(columns={"dev": "機種", "part": "部番", "elapsed": "経過月",
                            "use": "月次使用数", "fleet": "累積販売台数"})
    return cv[["機種", "部番", "部品名", "比較の扱い", "経過月", "窓内", "月次使用数", "累積使用数",
               "累積販売台数", "累積露出(台数×月)", "累積使用率%", "台数×月あたり率%"]]


if __name__ == "__main__":
    args = sys.argv[1:]
    asof = td._opt(args, "--asof", int)
    biz = td._opt(args, "--biz")
    panel = td._opt(args, "--panel")
    panel = os.path.abspath(panel) if panel else None
    os.chdir(ROOT)
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)
    raw = td.load_panel(cl.resolve_path(panel) if panel else None)
    print(f"パネル: {raw.attrs['path']}")
    explain(args[0], args[1], biz, raw=raw, asof=asof)
