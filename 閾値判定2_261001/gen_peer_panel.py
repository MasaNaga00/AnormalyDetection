# -*- coding: utf-8 -*-
"""gen_peer_panel.py — 信号B検証用の合成パネル。

複数SF群・機種ごとに異なる nb・既知の異常機種を仕込む。
販社は "ALL" のみ（本線と同じ前提）。
"""
import numpy as np
import pandas as pd
import state_logic_cusum as s

rng = np.random.default_rng(7)
MONTHS = 40
START = 202101

# (sf, 機種数, 基準レート, 各機種のnb, 異常機種index→倍率)
SPEC = [
    ("SF-100", 5, 0.0012, [1, 1, 1, 1, 1], {3: 3.0}),   # nb揃い。dev3が3倍
    ("SF-200", 4, 0.0008, [2, 2, 2, 2], {2: 2.5}),      # nb=2で揃い。dev2が2.5倍
    ("SF-300", 5, 0.0015, [1, 1, 2, 2, 1], {}),         # nb混在・異常なし
    ("SF-400", 2, 0.0010, [1, 1], {1: 4.0}),            # ピア不足→沈黙するはず
]


def build() -> pd.DataFrame:
    rows = []
    for sf, ndev, lam, nbs, anom in SPEC:
        for i in range(ndev):
            dev = f"{sf[-3:]}D{i}"
            factor = anom.get(i, 1.0)
            nb = nbs[i]
            # 併用型を模す: nb 個の部番に、それぞれ独立に同じレートで発生
            for t in range(MONTHS):
                ym = s._add_months(START, t)
                F = 2000 + 80 * t
                for j in range(nb):
                    cnt = int(rng.poisson(max(lam * factor * F, 0.0)))
                    rows.append({
                        "事業コード": "E1", "開発コード": dev, "部番": f"{dev}-P{j}",
                        "販社": "ALL", "SF-コード": sf, "ランク": "B",
                        "年月": ym, "経過月": t,
                        "月次使用数": cnt, "累積販売台数": F,
                    })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = build()
    print(p.shape, p["開発コード"].nunique())
