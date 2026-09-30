# -*- coding: utf-8 -*-
"""
tune_b_minoe.py — 信号B の min_oe / alpha を実データで決める

信号Cとの違い（決め方が構造的に違う理由）
----------------------------------------
- 信号Cは「単位×販社×月」を毎月検定する。alpha が月次の誤報件数に直結する。
- 信号Bは「機種×SF」を1回判定するだけで、値は月次でほぼ動かない。
  発火すると12ヶ月抑制されるので、効いてくるのは
  **立ち上げ初回の総件数**と、その後の細い流入（新機種が窓を完走した分）。

さらに、件数Cが数百になると p値は桁で飛ぶ（O/E=1.27でp=6e-4、O/E=2.48でp=1e-58）。
機種は正当な理由（設計世代・市場構成・使用環境）でも差が出る＝過分散があるため、
統計的有意性だけで切ると「有意だが実務的に無意味」な発火が大量に混ざる。
**主レバーは効果量 min_oe、alpha は件数の少ない群のガード**という役割分担にする。

使い方
------
    python tune_b_minoe.py panel.csv                # 件数の分布
    python tune_b_minoe.py panel.csv labels.csv     # 既知例の検知も評価
    python tune_b_minoe.py --cat レンズ              # カテゴリ設定の PANEL_PATH を使う

Jupyter:
    tb.top_list(s, min_oe=2.0, alpha=0.005)     # scan 結果で境界を目視
    tb.confirm("panel.csv", min_oe=2.0)         # 本番と同じ2パスで確認
    tb.nb_profile("panel.csv")                  # nb層別でピア不足の群
    （パネルを省略するとカテゴリ設定の PANEL_PATH）
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

import state_logic_cusum as sc
import signal_b_peer as sb
import config_loader as cl
cl.activate_from_argv()          # --cat X でカテゴリ設定に切り替え
import settings as st

MIN_OES = [1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]
ALPHAS = [0.05, 0.005, 0.001]


def scan(panel_path: str, elapsed_cap: int | None = None) -> pd.DataFrame:
    """全 (機種, SF) の O/E と p を1回だけ計算する。"""
    cap = elapsed_cap if elapsed_cap is not None else st.B_ELAPSED_CAP
    pb = _load_pb(panel_path)

    # 判定対象の内訳（どこで単位が減るか）
    n_all = len(sb.build_peer_units(pb, elapsed_cap=cap, require_full=False))
    n_full = len(sb.build_peer_units(pb, elapsed_cap=cap, require_full=True))

    # min_oe=0, alpha=0.9999 で全件の O/E と p を残す
    r = sb.run_signal_b(pb, elapsed_cap=cap, min_peers=st.B_MIN_PEERS,
                        alpha_peer=0.9999, min_count=st.B_MIN_COUNT,
                        min_oe=0.0, two_pass=False)
    r = r[r["p"].notna()].copy()
    out = cl.work_path("scan_b.csv")
    r.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"機種×SF の組合せ {n_all}  → require_full(cover>={cap}) {n_full}"
          f"  → min_peers>={st.B_MIN_PEERS} {len(r)} = 判定対象")
    if n_all > n_full:
        print(f"  require_full で落ちた {n_all - n_full} 単位は窓を完走していない若い機種")
    print(f"判定対象 (機種×SF): {len(r)}  中央C={int(r['C'].median())}  "
          f"→ {out} に保存")
    print(f"O/E の分布: "
          f"中央={r['O_E'].median():.2f}  P90={r['O_E'].quantile(0.9):.2f}  "
          f"P95={r['O_E'].quantile(0.95):.2f}  最大={r['O_E'].max():.2f}")
    print("  ※ scan は1パス。本番は2パスなので正常機種のO/Eは1.0へ是正され分布は上にずれる。")
    print("     P95 から min_oe を決めるときは confirm() で本番の値を確認すること。")
    return r


def _load_pb(panel_path: str | None = None) -> pd.DataFrame:
    """ALL行を CUSUM 側の前処理に通したパネル（信号Bの入力）。"""
    panel_path = panel_path or cl.panel_arg([], 0)
    raw = pd.read_csv(panel_path, encoding="utf-8-sig")
    raw[st.COLS["ym"]] = raw[st.COLS["ym"]].astype(str).str.replace(r"\D", "", regex=True).astype(int)
    p_all = raw[raw[st.COLS["dist"]].astype(str) == st.ALL_TOKEN]
    cfg = dict(sc.CONFIG)
    cfg["cols"] = {**sc.CONFIG["cols"], **st.COLS}
    pb = sc._prepare_panel(p_all.copy(), cfg)
    pb.attrs["raw"] = raw
    return pb


def sweep(s: pd.DataFrame, min_oes=MIN_OES, alphas=ALPHAS) -> pd.DataFrame:
    """min_oe × alpha で立ち上げ初回の発火件数を出す。"""
    rows = []
    for oe in min_oes:
        for a in alphas:
            hit = s[(s["O_E"] >= oe) & (s["p"] <= a) & (s["C"] >= st.B_MIN_COUNT)]
            rows.append(dict(min_oe=oe, alpha=a, 発火数=len(hit),
                             対象比=round(len(hit) / max(len(s), 1), 3),
                             最小O_E=round(hit["O_E"].min(), 2) if len(hit) else None,
                             機種数=hit["dev"].nunique()))
    t = pd.DataFrame(rows)
    print("\n=== min_oe × alpha（立ち上げ初回の発火件数、機種×SF単位）===")
    print("  発火するとその単位は12ヶ月抑制されるので、以後はこの数より大幅に減る")
    print(t.to_string(index=False))
    return t


def check_labels(s: pd.DataFrame, labels_path: str,
                 min_oes=MIN_OES, alphas=ALPHAS) -> pd.DataFrame:
    """既知例（機種×部番で与える）が拾えるかを操作点ごとに見る。

    信号Bの単位は機種×SFなので、ラベルの部番が属するSFに読み替えて判定する。
    """
    lab = pd.read_csv(labels_path, encoding="utf-8-sig")
    L = st.LABEL_COLS
    # 部番→SF の対応を scan から作れないので、パネル側で引く必要がある場合は
    # ラベルに SF 列を持たせてもよい。ここでは機種単位で当たりを見る。
    key = "SF" if "SF" in lab.columns else None

    rows = []
    for oe in min_oes:
        for a in alphas:
            hit = s[(s["O_E"] >= oe) & (s["p"] <= a) & (s["C"] >= st.B_MIN_COUNT)]
            n = 0
            for r in lab.itertuples():
                dev = getattr(r, L["dev"])
                m = hit[hit["dev"] == dev]
                if key:
                    m = m[m["sf"] == getattr(r, key)]
                if len(m):
                    n += 1
            rows.append(dict(min_oe=oe, alpha=a, 検知=f"{n}/{len(lab)}",
                             検知率=round(n / max(len(lab), 1), 2), 発火数=len(hit)))
    t = pd.DataFrame(rows)
    print(f"\n=== 既知例の検知（ラベル{len(lab)}件）===")
    if not key:
        print("  ※ ラベルに SF 列が無いので機種単位で判定（同機種の別SFでも○になる）")
    print(t.to_string(index=False))
    return t


def top_list(s: pd.DataFrame, min_oe: float, alpha: float, n: int = 30):
    """選んだ操作点で実際に何が出るかを見る。"""
    hit = s[(s["O_E"] >= min_oe) & (s["p"] <= alpha) & (s["C"] >= st.B_MIN_COUNT)]
    hit = hit.sort_values("O_E", ascending=False)
    print(f"\n=== min_oe={min_oe}, alpha={alpha} での発火 {len(hit)}件（上位{n}）===")
    print(hit.head(n)[["dev", "sf", "nb", "n_peers", "C", "O_E", "p"]].to_string(index=False))
    return hit


def oe_by_count(s: pd.DataFrame) -> pd.DataFrame:
    """件数帯別の O/E 分布。**件数下限を決めるための主要な表。**

    薄い単位は異常がなくても O/E が跳ねる。件数帯ごとの O/E の上側
    （P95・最大）を見て、ノイズで到達しうる水準を実データで確かめる。
    """
    d = s.copy()
    d["件数帯"] = pd.cut(d["C"], [0, 5, 10, 20, 50, 100, 1e9],
                      labels=["〜5", "6-10", "11-20", "21-50", "51-100", "100〜"])
    t = d.groupby("件数帯", observed=True).agg(
        単位数=("O_E", "size"), O_E中央=("O_E", "median"),
        P90=("O_E", lambda x: x.quantile(0.9)),
        P95=("O_E", lambda x: x.quantile(0.95)),
        最大=("O_E", "max"))
    print("\n=== 件数帯別の O/E 分布 ===")
    print("  薄い帯の P95・最大が大きいほど、その帯はノイズで上位に来る")
    print("  → その帯が消える件数を B_MIN_COUNT にする")
    print(t.round(2).to_string())
    return t


def count_floor_sweep(s: pd.DataFrame, floors=(3, 10, 20, 30, 50),
                      min_oe: float = 1.5, alpha: float = 0.005) -> pd.DataFrame:
    """件数下限を振って、発火件数と最小件数がどう動くかを見る。"""
    rows = []
    for mc in floors:
        hit = s[(s["C"] >= mc) & (s.get("C_peer", s["C"]) >= mc)
                & (s["O_E"] >= min_oe) & (s["p"] <= alpha)]
        rows.append(dict(下限=mc, 判定対象=int(((s["C"] >= mc)).sum()),
                         発火数=len(hit),
                         発火の最小件数=int(hit["C"].min()) if len(hit) else None,
                         発火の最大O_E=round(hit["O_E"].max(), 2) if len(hit) else None))
    t = pd.DataFrame(rows)
    print(f"\n=== 件数下限スイープ（min_oe={min_oe}, alpha={alpha}）===")
    print(t.to_string(index=False))
    return t


def confirm(panel_path: str | None = None, min_oe: float | None = None,
            alpha: float | None = None, n: int = 30) -> pd.DataFrame:
    """選んだ操作点を**本番と同じ2パス**で再計算して、実際に何が発火するか確認する。

    scan() は1パス（two_pass=False）なので、scan_b.csv の O/E は本番の値と
    わずかにずれる。2パス目で「発火した機種をピアプールから外す」と、
    同じ群の正常機種の O/E が 1.0 へ是正され、分布全体が上にずれるため。
    同一SF群に複数の異常があると、本番の方が多く発火することがある。
    USE_HORIZON かつ B_TRUNCATE_TO_HORIZON なら、本番と同じく global horizon で打ち切る。
    """
    min_oe = st.B_MIN_OE if min_oe is None else min_oe
    alpha = st.B_ALPHA if alpha is None else alpha
    pb = _load_pb(panel_path)
    note = ""
    if getattr(st, "USE_HORIZON", False) and getattr(st, "B_TRUNCATE_TO_HORIZON", True):
        import reporting_horizon as rh
        raw = pb.attrs["raw"]
        dist = raw[raw[st.COLS["dist"]].astype(str) != st.ALL_TOKEN]
        hz = rh.estimate_horizon(
            dist, st.COLS, all_token=st.ALL_TOKEN,
            margin_months=st.HORIZON_MARGIN_MONTHS,
            margin_overrides=st.HORIZON_MARGIN_OVERRIDES, fixed=st.HORIZON_FIXED,
            auto_margin=st.HORIZON_AUTO_MARGIN, thin_ratio=st.HORIZON_THIN_RATIO)
        g = rh.global_horizon(hz)
        if g is not None:
            pb = pb[pb["ym"] <= g]
            note = f"（global horizon {g} で打ち切り）"
    r = sb.run_signal_b(pb, elapsed_cap=st.B_ELAPSED_CAP, min_peers=st.B_MIN_PEERS,
                        alpha_peer=alpha, min_count=st.B_MIN_COUNT,
                        min_peer_count=st.B_MIN_PEER_COUNT, min_oe=min_oe,
                        nb_strat=st.B_NB_STRAT, two_pass=True)
    hit = r[r["alert_peer"]] if len(r) else r
    print(f"\n=== 本番（2パス）: min_oe={min_oe}, alpha={alpha} → 発火 {len(hit)}件 ==={note}")
    if len(r):
        print(f"  O/E の分布: 中央={r['O_E'].median():.2f}  P90={r['O_E'].quantile(0.9):.2f}  "
              f"P95={r['O_E'].quantile(0.95):.2f}")
    if len(hit):
        print(hit.head(n)[["dev", "sf", "nb", "n_peers", "C", "O_E", "p", "注目度"]]
              .to_string(index=False))
    else:
        print("  （発火なし）")
    return r


def nb_profile(panel_path: str | None = None, sf: str | None = None) -> pd.DataFrame:
    """SF群ごとの nb 構成を見る。nb層別でピアが足りず沈黙する群を特定する。

    「幅0」（全機種のnbが同じ）の群は信号Bがそのまま効く。
    nb がばらつく群は nb ごとに割れるので、単独nbの機種は判定対象から外れる。
    sf を渡すとその群の機種ごとの内訳を出す。
    Returns: ピア不足で沈黙する (機種, SF) の一覧
    """
    pb = _load_pb(panel_path)
    u = sb.build_peer_units(pb, elapsed_cap=st.B_ELAPSED_CAP, require_full=True)
    if u.empty:
        print("判定対象の単位がありません（require_full で全滅していないか）")
        return u
    if sf is not None:
        d = u[u["sf"].astype(str) == str(sf)].sort_values(["nb", "dev"])
        print(f"\n=== {sf} の nb 構成 ===")
        print(d[["biz", "dev", "nb", "C", "cover"]].to_string(index=False))
    g = u.groupby(["biz", "sf"]).agg(機種数=("dev", "nunique"),
                                     nb最小=("nb", "min"), nb最大=("nb", "max"))
    g["幅"] = g["nb最大"] - g["nb最小"]
    multi = g[g["機種数"] >= st.B_MIN_PEERS + 1]
    print(f"\n=== SF群の nb 構成（require_full 後、機種{st.B_MIN_PEERS + 1}以上の群 {len(multi)}）===")
    print(f"  幅0（そのまま比較可）: {int((multi['幅'] == 0).sum())}")
    print(f"  幅1                  : {int((multi['幅'] == 1).sum())}")
    print(f"  幅2以上              : {int((multi['幅'] >= 2).sum())}")

    u["同nb機種数"] = u.groupby(["biz", "sf", "nb"])["dev"].transform("nunique")
    lonely = u[u["同nb機種数"] - 1 < st.B_MIN_PEERS].copy()
    lonely = lonely.merge(g[["機種数", "幅"]].reset_index(), on=["biz", "sf"], how="left")
    lonely = lonely.sort_values("C", ascending=False)
    print(f"\n=== ピア不足で沈黙する (機種, SF): {len(lonely)} / {len(u)} 単位 ===")
    print("  同nbのピアが B_MIN_PEERS 未満。**信号Bでは監視できていない**（閾値側で拾う）")
    if len(lonely):
        print(lonely.head(30)[["biz", "sf", "dev", "nb", "同nb機種数", "機種数", "幅", "C"]]
              .to_string(index=False))
    return lonely.reset_index(drop=True)


if __name__ == "__main__":
    if len(sys.argv) < 2 and not cl.category():
        print(__doc__)
        sys.exit(1)
    s = scan(cl.panel_arg(sys.argv[1:], 0))
    oe_by_count(s)
    sweep(s)
    if len(sys.argv) > 2:
        check_labels(s, sys.argv[2])
    print("\n--- 次にやること ---")
    print(f"  import pandas as pd, tune_b_minoe as tb")
    print(f"  s = pd.read_csv({cl.work_path('scan_b.csv')!r}, encoding='utf-8-sig')")
    print("  tb.top_list(s, min_oe=2.0, alpha=0.005)   # 境界に何が来るかを目視")
    print("  tb.confirm(min_oe=2.0, alpha=0.005)       # 本番（2パス）で確認")
    print("  tb.nb_profile()                           # 沈黙する群の特定")
