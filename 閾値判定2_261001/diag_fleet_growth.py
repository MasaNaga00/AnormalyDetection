# -*- coding: utf-8 -*-
"""signal_c_dist.py の分母が効いているかを実データで測る。

販売台数が横ばいなら「率ベース」と「件数のみ」は数学的に同一（pi=1/13）。
差が出るのは台数が伸びている系列だけ。実データでどれだけあるかを数える。
"""
import pandas as pd, numpy as np, signal_c_dist as sd, settings as st

L = st.C_BASE_LEN


def diagnose(panel_path, months_back=24):
    raw = pd.read_csv(panel_path, encoding="utf-8-sig")
    raw["年月"] = raw["年月"].astype(str).str.replace(r"\D", "", regex=True).astype(int)
    d = sd.prepare_dist_panel(raw[raw[st.COLS["dist"]] != st.ALL_TOKEN], st.COLS)

    rows = []
    for key, g in d.groupby(["biz", "dev", "part", "dist"], sort=False):
        g = g.sort_values("ym")
        f = g["fleet"].to_numpy(float)
        ym = g["ym"].to_numpy()
        for t in range(L, len(g)):
            Eb = f[t - L:t].sum()
            if Eb <= 0:
                continue
            rows.append(dict(ym=int(ym[t]), 比=f[t] / (Eb / L)))
    r = pd.DataFrame(rows)
    r = r[r["ym"] >= sd._shift_ym(int(d["ym"].max()), -months_back)]

    print(f"判定機会 {len(r)} 件の『当月台数 ÷ 直近{L}ヶ月平均』")
    print(r["比"].describe(percentiles=[.5, .9, .95, .99]).round(3).to_string())
    for lo, hi, tag in [(0, 1.05, "横ばい（分母は無意味）"),
                        (1.05, 1.20, "微増"),
                        (1.20, 1.50, "成長中（分母が効く）"),
                        (1.50, 99, "急成長（件数のみだと誤報）")]:
        n = ((r["比"] >= lo) & (r["比"] < hi)).sum()
        print(f"  {tag:28s}: {n:6d} ({n/len(r)*100:5.1f}%)")
    return r


if __name__ == "__main__":
    import sys
    diagnose(sys.argv[1] if len(sys.argv) > 1 else "panel_初回.csv")
