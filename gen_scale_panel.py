# -*- coding: utf-8 -*-
"""実データ規模（40機種×50部品=2000単位×36ヶ月）の合成パネルを作る。
先行機種にも初期不良カーブ（初期高→減衰）を持たせる（過去の落とし穴対策）。"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

N_DEV, N_PART, N_M = 40, 50, 36
YMS = []
ym = 202301
for _ in range(N_M):
    YMS.append(ym)
    y, m = divmod(ym, 100)
    ym = (y + 1) * 100 + 1 if m == 12 else ym + 1

rows = []
labels = []
for di in range(N_DEV):
    dev = f"S{di:02d}"
    sf = f"SF-{di % 8}"
    rank = ["A", "B", "C"][di % 3]
    start = di % 12                      # 機種ごとに発売時期をずらす（安定化前も混ざる）
    for pi in range(N_PART):
        part = f"P{pi:02d}"
        lam_st = rng.uniform(0.0008, 0.003)   # 安定期レート
        # 異常注入: 2%にドリフト, 1%にスパイク
        drift_at = spike_at = None
        u = rng.uniform()
        if u < 0.02:
            drift_at = rng.integers(20, 30)
        elif u < 0.03:
            spike_at = rng.integers(18, 32)
        fleet = 0
        for t, ymv in enumerate(YMS[start:], 0):
            fleet += int(rng.integers(300, 800))
            lam = lam_st * (1 + 3.0 * np.exp(-t / 2.5))   # 初期不良カーブ
            if drift_at is not None and t >= drift_at:
                lam *= 2.5
            mu = lam * fleet
            if spike_at is not None and t == spike_at:
                mu *= 6
            use = rng.poisson(mu)
            rows.append((f"E1", dev, part, "D0", sf, rank, ymv, t, int(use), fleet))
        if drift_at is not None:
            # 販社報告 = 変化点の3ヶ月後、という想定
            rep_idx = min(start + int(drift_at) + 3, len(YMS) - 1)
            labels.append(("E1", dev, part, YMS[rep_idx]))
        if spike_at is not None:
            rep_idx = min(start + int(spike_at) + 1, len(YMS) - 1)
            labels.append(("E1", dev, part, YMS[rep_idx]))

pd.DataFrame(rows, columns=["事業コード", "開発コード", "部番", "販社", "SF-コード",
                            "ランク", "年月", "経過月", "月次使用数", "累積販売台数"]) \
    .to_csv("scale_panel.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(labels, columns=["事業コード", "開発コード", "部番", "発生年月"]) \
    .to_csv("scale_labels.csv", index=False, encoding="utf-8-sig")
print(f"単位数={N_DEV*N_PART}, ラベル数={len(labels)}")
