# -*- coding: utf-8 -*-
"""diag_b_row.py — 信号Bのインボックス行が、どう計算されて出てきたかを辿る。

インボックスの行は「機種×SF」の判定を「部番」に展開したもの。
展開元のSF判定と、そのSF内の部番内訳を並べて表示する。

確認方法
python diag_b_row.py panel.csv M02          # 機種の全SFを辿る
python diag_b_row.py panel.csv M02 P-1234   # その部番が属するSFだけ
"""
import pandas as pd, numpy as np
import state_logic_cusum as sc, signal_b_peer as sb, settings as st


def trace(panel_path, dev, part=None, elapsed_cap=None):
    cap = elapsed_cap or st.B_ELAPSED_CAP
    raw = pd.read_csv(panel_path, encoding="utf-8-sig")
    raw["年月"] = raw["年月"].astype(str).str.replace(r"\D", "", regex=True).astype(int)
    cfg = dict(sc.CONFIG); cfg["cols"] = {**sc.CONFIG["cols"], **st.COLS}
    pb = sc._prepare_panel(raw[raw[st.COLS["dist"]].astype(str) == st.ALL_TOKEN].copy(), cfg)

    r = sb.run_signal_b(pb, elapsed_cap=cap, min_peers=st.B_MIN_PEERS,
                        alpha_peer=st.B_ALPHA, min_count=st.B_MIN_COUNT,
                        min_peer_count=st.B_MIN_PEER_COUNT, min_oe=st.B_MIN_OE)
    d = pb[pb["elapsed"] <= cap]
    pu = (d.groupby(["biz", "dev", "sf", "part"], as_index=False)["use"].sum()
            .rename(columns={"use": "部番使用数"}))

    sub = r[r["dev"].astype(str) == str(dev)]
    if part is not None:
        sfs = pu[(pu.dev.astype(str) == str(dev)) &
                 (pu.part.astype(str) == str(part))]["sf"].unique()
        sub = sub[sub["sf"].isin(sfs)]
    if sub.empty:
        print(f"{dev} の判定行がありません（判定対象外の可能性）"); return

    for row in sub.itertuples():
        print(f"■ {row.dev} / SF={row.sf}  →  {'★発火' if row.alert_peer else '沈黙'}")
        print(f"   対象:  C={int(row.C):>6}  E={int(row.E):>9}  "
              f"レート={row.C/row.E:.6f}")
        print(f"   ピア:  C={int(row.C_peer):>6}  (ピア{int(row.n_peers)}機種, nb={int(row.nb)})  "
              f"レート={row.peer_rate:.6f}")
        print(f"   期待={row.expected:.1f}  O/E={row.O_E:.2f}  p={row.p:.2e}")
        chk = [("O/E>=min_oe", row.O_E >= st.B_MIN_OE, f"{row.O_E:.2f}>={st.B_MIN_OE}"),
               ("p<=alpha", row.p <= st.B_ALPHA, f"{row.p:.1e}<={st.B_ALPHA}"),
               ("C>=min_count", row.C >= st.B_MIN_COUNT, f"{int(row.C)}>={st.B_MIN_COUNT}"),
               ("C_peer>=min_peer", row.C_peer >= st.B_MIN_PEER_COUNT,
                f"{int(row.C_peer)}>={st.B_MIN_PEER_COUNT}")]
        for name, ok, txt in chk:
            print(f"     {'○' if ok else '×'} {name:18s} {txt}")

        parts = pu[(pu.biz == row.biz) & (pu.dev == row.dev) & (pu.sf == row.sf)]
        parts = parts.sort_values("部番使用数", ascending=False)
        print(f"\n   このSFに紐づく部番（合計{int(parts['部番使用数'].sum())}件）:")
        for q in parts.itertuples():
            share = q.部番使用数 / max(parts["部番使用数"].sum(), 1) * 100
            mark = "  ← インボックスに出る" if row.alert_peer else ""
            print(f"     {q.part:<12} {int(q.部番使用数):>6}件 ({share:5.1f}%){mark}")
        print()
    return sub


if __name__ == "__main__":
    import sys
    trace(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
