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
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

import state_logic_cusum as sc
import signal_b_peer as sb
import settings as st

MIN_OES = [1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]
ALPHAS = [0.05, 0.005, 0.001]


def scan(panel_path: str, elapsed_cap: int | None = None) -> pd.DataFrame:
    """全 (機種, SF) の O/E と p を1回だけ計算する。"""
    cap = elapsed_cap if elapsed_cap is not None else st.B_ELAPSED_CAP
    raw = pd.read_csv(panel_path, encoding="utf-8-sig")
    # 列名は settings.COLS を参照する（実データで触るのは settings.py だけ、の原則）
    c_ym = st.COLS["ym"]
    raw[c_ym] = raw[c_ym].astype(str).str.replace(r"\D", "", regex=True).astype(int)
    # run_month.py と同じ前処理（累積販売台数の逆転を cummax で補正）
    keys = [st.COLS["biz"], st.COLS["dev"], st.COLS["part"], st.COLS["dist"]]
    raw = raw.sort_values(keys + [c_ym])
    raw[st.COLS["cum_sales"]] = raw.groupby(keys)[st.COLS["cum_sales"]].cummax()
    p_all = raw[raw[st.COLS["dist"]].astype(str) == st.ALL_TOKEN]

    cfg = dict(sc.CONFIG)
    cfg["cols"] = {**sc.CONFIG["cols"], **st.COLS}
    pb = sc._prepare_panel(p_all.copy(), cfg)

    # 判定対象がどこで減ったかの内訳（実データで B_ELAPSED_CAP を決めるのに要る）
    uf = sb.build_peer_units(pb, elapsed_cap=cap, require_full=False)
    ur = sb.build_peer_units(pb, elapsed_cap=cap, require_full=True)
    dropped = sorted(set(uf["dev"]) - set(ur["dev"]))

    # min_oe=0, alpha=0.9999 で全件の O/E と p を残す。
    # two_pass=False は意図的: 2パス目の「発火機種をピアから外す」判定は
    # min_oe/alpha に依存するので、1回のスキャンで全操作点を表せない。
    # → 操作点を選んだら confirm() で2パス（本番と同じ）を必ず確認すること。
    r = sb.run_signal_b(pb, elapsed_cap=cap, min_peers=st.B_MIN_PEERS,
                        alpha_peer=0.9999, min_count=st.B_MIN_COUNT,
                        min_oe=0.0, two_pass=False)
    r = r[r["p"].notna()].copy()
    r.to_csv("scan_b.csv", index=False, encoding="utf-8-sig")
    print(f"機種×SF の組合せ: {len(uf)}  → require_full(cover>={cap}): {len(ur)}"
          f"  → min_peers>={st.B_MIN_PEERS}: {len(r)} = 判定対象")
    if dropped:
        print(f"  require_full で落ちた機種: {dropped}（窓を完走していない若い機種）")
    print(f"中央C={int(r['C'].median())}  → scan_b.csv に保存")
    print(f"O/E の分布(1パス): "
          f"中央={r['O_E'].median():.2f}  P90={r['O_E'].quantile(0.9):.2f}  "
          f"P95={r['O_E'].quantile(0.95):.2f}  最大={r['O_E'].max():.2f}")
    print("  ※ 本番は2パスなので分布はこれより上にずれる（正常機種のO/Eが1.0へ是正される）。"
          "\n     P95 から min_oe を決めるときは、confirm() で本番の値を確認すること。")
    return r


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


def confirm(panel_path: str, min_oe: float, alpha: float,
            elapsed_cap: int | None = None) -> pd.DataFrame:
    """選んだ操作点を**本番と同じ2パス**で再計算して、実際に何が発火するか確認する。

    scan() は 1パス（two_pass=False）なので、scan_b.csv の O/E は本番の値と
    わずかにずれる。2パス目で「発火した機種をピアプールから外す」と、
    同じ群の正常機種の O/E が 1.0 へ是正され、分布全体が上にずれるため。
    同一SF群に複数の異常があると、本番の方が多く発火することがある。
    """
    cap = elapsed_cap if elapsed_cap is not None else st.B_ELAPSED_CAP
    raw = pd.read_csv(panel_path, encoding="utf-8-sig")
    c_ym = st.COLS["ym"]
    raw[c_ym] = raw[c_ym].astype(str).str.replace(r"\D", "", regex=True).astype(int)
    p_all = raw[raw[st.COLS["dist"]].astype(str) == st.ALL_TOKEN]
    cfg = dict(sc.CONFIG)
    cfg["cols"] = {**sc.CONFIG["cols"], **st.COLS}
    pb = sc._prepare_panel(p_all.copy(), cfg)

    r = sb.run_signal_b(pb, elapsed_cap=cap, min_peers=st.B_MIN_PEERS,
                        alpha_peer=alpha, min_count=st.B_MIN_COUNT,
                        min_oe=min_oe, two_pass=True)
    a = r[r["alert_peer"]]
    print(f"\n=== 本番(2パス) min_oe={min_oe}, alpha={alpha} → 発火 {len(a)}件 ===")
    print(f"O/E の分布(2パス): 中央={r['O_E'].median():.2f}  "
          f"P90={r['O_E'].quantile(0.9):.2f}  P95={r['O_E'].quantile(0.95):.2f}")
    if len(a):
        print(a[["dev", "sf", "nb", "n_peers", "C", "O_E", "p", "注目度"]]
              .to_string(index=False))
    else:
        print("  （発火なし）")
    return a


def nb_profile(panel_path: str, sf: str | None = None,
               elapsed_cap: int | None = None) -> pd.DataFrame:
    """SF群ごとの nb 構成を見る。nb層別でピアが足りず沈黙する群を特定する。

    「幅0」（全機種のnbが同じ）の群は信号Bがそのまま効く。
    nb がばらつく群は nb ごとに割れるので、単独nbの機種は判定対象から外れる。
    """
    cap = elapsed_cap if elapsed_cap is not None else st.B_ELAPSED_CAP
    raw = pd.read_csv(panel_path, encoding="utf-8-sig")
    c_ym = st.COLS["ym"]
    raw[c_ym] = raw[c_ym].astype(str).str.replace(r"\D", "", regex=True).astype(int)
    p_all = raw[raw[st.COLS["dist"]].astype(str) == st.ALL_TOKEN]
    cfg = dict(sc.CONFIG)
    cfg["cols"] = {**sc.CONFIG["cols"], **st.COLS}
    pb = sc._prepare_panel(p_all.copy(), cfg)
    u = sb.build_peer_units(pb, elapsed_cap=cap, require_full=True)
    if sf is not None:
        print(f"\n=== {sf} の nb 構成 ===")
        print(u[u["sf"] == sf][["dev", "nb", "C", "E"]]
              .sort_values("nb").to_string(index=False))
        return u[u["sf"] == sf]
    g = u.groupby(["biz", "sf"]).agg(
        機種数=("dev", "nunique"), nb最小=("nb", "min"), nb最大=("nb", "max"))
    g["幅"] = g["nb最大"] - g["nb最小"]
    # nb ごとの機種数が min_peers+1 未満なら、その nb の機種は沈黙する
    cnt = u.groupby(["biz", "sf", "nb"])["dev"].nunique().rename("同nb機種数")
    lonely = cnt[cnt < st.B_MIN_PEERS + 1].reset_index()
    print("\n=== SF群の nb 構成 ===")
    print(g.sort_values("幅", ascending=False).to_string())
    if len(lonely):
        print(f"\n=== ピア不足で沈黙する (SF, nb) : {len(lonely)}件 ===")
        print(lonely.to_string(index=False))
    return g


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    s = scan(sys.argv[1])
    sweep(s)
    if len(sys.argv) > 2:
        check_labels(s, sys.argv[2])
    print("\n--- 次にやること ---")
    print("  import pandas as pd, tune_b_minoe as tb")
    print("  s = pd.read_csv('scan_b.csv', encoding='utf-8-sig')")
    print("  tb.top_list(s, min_oe=2.0, alpha=0.005)   # 境界に何が来るかを目視")
    print(f"  tb.confirm('{sys.argv[1]}', min_oe=2.0, alpha=0.005)  # 本番(2パス)で確認")
    print(f"  tb.nb_profile('{sys.argv[1]}')            # 沈黙する群の特定")
