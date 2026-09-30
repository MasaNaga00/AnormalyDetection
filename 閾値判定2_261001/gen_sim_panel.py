# -*- coding: utf-8 -*-
"""
gen_sim_panel.py — 実運用シミュレーション用のパネル生成

作るもの
--------
  panel_初回.csv   … 立ち上げ時点（202101〜202412、48ヶ月）
  panel_翌月.csv   … 1ヶ月分を継ぎ足したもの（〜202501）
  台帳_空.xlsx     … 0行の台帳（統合版スキーマ）
  仕込み一覧.csv   … 何をどこに仕込んだかの答え合わせ用（実運用には存在しない）

データ構造
----------
SF-CODE は「部品の種類」を表す。1機種は複数のSF（部品種類）を持ち、
同一SFに複数の部番が紐づく機種もある（nb>1）。機種どうしは各SF内でピアになる。

  13機種（発売時期をずらす） × 6つのSF × 各1〜3部番 × 販社5社(A〜E) + ALL行

  SF-100〜SF-102 : 全機種 nb=1（幅0。信号Bがそのまま効く）
  SF-103         : 全機種 nb=2（幅0だが併用型。nb層別で比較できる）
  SF-104         : 機種で nb がばらつく（幅≥1。nb層別でピア不足になる群）
  SF-105         : 全機種 nb=1 だが件数が少ない（薄い群の挙動）

仕込んだ異常
------------
  X1 : M02 / SF-101 … 恒常的に4.5倍         → 閾値＋信号B
  X2 : M06 / SF-102 … 恒常的に2.6倍         → 信号B のみ（閾値には届かない）
  X3 : M01 / SF-100 … 販社C で最終月に急増   → 信号C（初回runで出る）
  X4 : M10 / SF-103 … 販社A で「翌月」に急増 → 信号C（2回目runで初めて出る）
  X5 : M04 / SF-104 … 恒常的に3.0倍・nb混在群 → 検知されにくい例（構造的な穴の体感用）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

rng = np.random.default_rng(2024)

START = 202101
N_MONTHS = 48
DISTS = {"A": 0.34, "B": 0.24, "C": 0.20, "D": 0.15, "E": 0.07}

MACHINES = [("M01", 0, "A"), ("M02", 2, "A"), ("M03", 3, "A"), ("M04", 5, "A"),
            ("M05", 6, "B"), ("M06", 7, "B"), ("M07", 9, "B"), ("M08", 10, "B"),
            ("M09", 11, "A"), ("M10", 12, "A"), ("M11", 14, "B"), ("M12", 16, "B"),
            ("M13", 18, "B")]

SF_SPEC = {
    "SF-100": (0.0012, 1),
    "SF-101": (0.0009, 1),
    "SF-102": (0.0011, 1),
    "SF-103": (0.0007, 2),
    "SF-104": (0.0010, {"M01": 1, "M02": 2, "M03": 1, "M04": 3, "M05": 2,
                        "M06": 1, "M07": 2, "M08": 1, "M09": 2, "M10": 1,
                        "M11": 3, "M12": 1, "M13": 2}),
    "SF-105": (0.0003, 1),
}

CHRONIC = {
    ("M02", "SF-101"): 4.5,   # X1
    ("M06", "SF-102"): 2.6,   # X2
    ("M04", "SF-104"): 3.0,   # X5
}
SPIKES = {
    ("M01", "SF-100", "C", N_MONTHS - 1): 7.0,   # X3 初回runの最終月
    ("M10", "SF-103", "A", N_MONTHS):     9.0,   # X4 翌月
}


def _ym(k: int) -> int:
    y, m = divmod(START, 100)
    i = y * 12 + (m - 1) + k
    return (i // 12) * 100 + (i % 12) + 1


def _nb(sf: str, dev: str) -> int:
    spec = SF_SPEC[sf][1]
    return spec if isinstance(spec, int) else spec.get(dev, 1)


def _build(n_months: int) -> pd.DataFrame:
    rows = []
    for dev, delay, rank in MACHINES:
        for sf, (lam, _) in SF_SPEC.items():
            nb = _nb(sf, dev)
            fac = CHRONIC.get((dev, sf), 1.0)
            for t in range(delay, n_months):
                e = t - delay
                Y = _ym(t)
                F = int(2500 + 130 * e)
                for j in range(nb):
                    part = f"{sf[-3:]}-{j}"
                    tot = 0
                    for dist, frac in DISTS.items():
                        if dist == "E" and e < 8:
                            continue
                        Fd = int(round(F * frac))
                        l = lam * fac
                        mult = SPIKES.get((dev, sf, dist, t))
                        if mult:
                            l = l * mult
                        c = int(rng.poisson(max(l * Fd, 0.0)))
                        if c == 0 and e < 2:
                            continue
                        rows.append(dict(
                            事業コード="E1", 開発コード=dev, 部番=part, 販社=dist,
                            **{"SF-CODE": sf, "ランク": rank},
                            年月=Y, 経過月=e, 月次使用数=c, 累積販売台数=Fd))
                        tot += c
                    rows.append(dict(
                        事業コード="E1", 開発コード=dev, 部番=part, 販社="ALL",
                        **{"SF-CODE": sf, "ランク": rank},
                        年月=Y, 経過月=e, 月次使用数=tot, 累積販売台数=F))
    return pd.DataFrame(rows)


def build_answer_key() -> pd.DataFrame:
    return pd.DataFrame([
        dict(ID="X1", 機種="M02", SF="SF-101", 販社="", 内容="恒常的に4.5倍",
             期待検出器="閾値+信号B", 出る回="初回"),
        dict(ID="X2", 機種="M06", SF="SF-102", 販社="", 内容="恒常的に2.6倍",
             期待検出器="信号B", 出る回="初回"),
        dict(ID="X3", 機種="M01", SF="SF-100", 販社="C", 内容="最終月に7倍の急増",
             期待検出器="信号C", 出る回="初回"),
        dict(ID="X4", 機種="M10", SF="SF-103", 販社="A", 内容="翌月に9倍の急増",
             期待検出器="信号C", 出る回="2回目"),
        dict(ID="X5", 機種="M04", SF="SF-104", 販社="", 内容="恒常的に3.0倍・nb混在群",
             期待検出器="検知されにくい", 出る回="—"),
    ])


if __name__ == "__main__":
    import unified_inbox as ui

    p1 = _build(N_MONTHS)
    p2 = _build(N_MONTHS + 1)
    last_ym = _ym(N_MONTHS)
    p2 = pd.concat([p1, p2[p2["年月"] == last_ym]], ignore_index=True)

    p1.to_csv("panel_初回.csv", index=False, encoding="utf-8-sig")
    p2.to_csv("panel_翌月.csv", index=False, encoding="utf-8-sig")
    build_answer_key().to_csv("仕込み一覧.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter("台帳_空.xlsx", engine="openpyxl") as w:
        ui.empty_ledger().to_excel(w, sheet_name="台帳", index=False)

    print(f"panel_初回.csv : {len(p1):>6} 行  最新={int(p1['年月'].max())}")
    print(f"panel_翌月.csv : {len(p2):>6} 行  最新={int(p2['年月'].max())}")
    print(f"機種={p1['開発コード'].nunique()}  SF={p1['SF-CODE'].nunique()}  "
          f"監視単位(機種×部番)={p1.groupby(['開発コード','部番']).ngroups}")
