# -*- coding: utf-8 -*-
"""gen_dist_panel.py — 信号C検証用。販社別行＋ALL行を持つ合成パネル。

仕込み:
  D0P0 : 販社C で 202305 に単月4倍の急増（拾えるべき）
  D0P1 : 全販社で緩やかに上昇（信号Cは沈黙すべき＝スパイクではない）
  D1P0 : 異常なし
  販社E は途中から行が立つ（左側打ち切りの再現）
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(11)
MONTHS = 36
START = 202201
DISTS = {"A": 0.35, "B": 0.25, "C": 0.20, "D": 0.15, "E": 0.05}
SPIKE = ("D0", "P0", "C", 202305, 4.0)


def _ym(k):
    y, m = divmod(START, 100)
    i = y * 12 + (m - 1) + k
    return (i // 12) * 100 + (i % 12) + 1


def build():
    rows = []
    for dev in ("D0", "D1"):
        for part in ("P0", "P1"):
            lam = 0.0012 if part == "P0" else 0.0009
            for t in range(MONTHS):
                ym = _ym(t)
                F = 4000 + 150 * t
                tot_u = tot_f = 0
                for dist, frac in DISTS.items():
                    if dist == "E" and t < 10:      # 左側打ち切りの再現
                        continue
                    Fd = int(round(F * frac))
                    l = lam
                    if part == "P1":                 # 全販社で緩やかに上昇
                        l = lam * (1.0 + 0.04 * t)
                    sd, sp, sdist, sym, mult = SPIKE
                    if dev == sd and part == sp and dist == sdist and ym == sym:
                        l = l * mult
                    cnt = int(rng.poisson(max(l * Fd, 0.0)))
                    rows.append(dict(事業コード="E1", 開発コード=dev, 部番=part, 販社=dist,
                                     年月=ym, 月次使用数=cnt, 累積販売台数=Fd))
                    tot_u += cnt
                    tot_f += Fd
                rows.append(dict(事業コード="E1", 開発コード=dev, 部番=part, 販社="ALL",
                                 年月=ym, 月次使用数=tot_u, 累積販売台数=F))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(build().shape)
