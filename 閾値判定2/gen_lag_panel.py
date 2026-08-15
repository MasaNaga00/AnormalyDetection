# -*- coding: utf-8 -*-
"""
gen_lag_panel.py — 完全なパネルから「販社の報告遅れがある2時点の抽出」を作る

なぜ要るか
----------
既存の `gen_sim_panel.py` が作るパネルは全販社のデータが揃っている。
実データは受付月ベースで、修理完了→販社送付を経て届くため、直近数ヶ月は
販社ごとに違う量だけ欠けている。**その状態を再現しないとリハーサルにならない。**

やること
--------
1. 真のパネルの最新月を Tmax とし、2つの抽出時点を作る
     run1 = Tmax-1 時点の抽出
     run2 = Tmax   時点の抽出
2. 販社ごとの熟成カーブ f(k)（k=抽出月から何ヶ月前か）で月次使用数を間引く。
   f(k)=0 の月は**行ごと落とす**（月次一括送付の販社がまだ送っていない状態）。
3. ALL行の月次使用数を、間引いた販社別行の合計で作り直す
   （累積販売台数＝分母は社内システム由来なので間引かない。ここが非対称の肝）。
4. 遅れている販社の過去月に異常を仕込む。
   → run1 では打ち切られて見えず、run2 ではじめて見える、という状況を作る。

使い方
------
    python gen_lag_panel.py panel_翌月.csv 遅延
      → 遅延_run1.csv / 遅延_run2.csv / 仕込み一覧_遅延.csv

    python run_month.py 遅延_run1.csv 台帳_空.xlsx
    （レビュー→台帳記入）
    python run_month.py 遅延_run2.csv 台帳_記入.xlsx
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

import settings as st

COLS = st.COLS

# ---------------------------------------------------------------------------
# 熟成カーブ: f[k] = 抽出月から k ヶ月前の受付月が、何割まで出そろっているか
# 販社名がここに無ければ DEFAULT を使う。実データの実測値が出たら差し替える。
# ---------------------------------------------------------------------------
MATURITY = {
    "A": [0.85, 1.00],                    # 毎日送付・修理も速い
    "B": [0.60, 0.92, 1.00],              # 週次送付
    "C": [0.00, 0.55, 0.85, 0.97, 1.00],  # 月次送付（当月はまだ1行も来ない）
    "D": [0.00, 0.00, 0.40, 0.75, 0.95, 1.00],  # 月次送付＋部品待ちで完了が遅い
}
DEFAULT_MATURITY = [0.70, 0.95, 1.00]

SPIKE_FACTOR = 8.0     # 仕込む急増の倍率
SPIKE_LAG = 3          # run2 の最新月から何ヶ月前に仕込むか


def _shift(ym: int, k: int) -> int:
    y, m = divmod(int(ym), 100)
    i = y * 12 + (m - 1) + k
    return (i // 12) * 100 + (i % 12) + 1


def _diff(a: int, b: int) -> int:
    ya, ma = divmod(int(a), 100)
    yb, mb = divmod(int(b), 100)
    return (ya * 12 + ma) - (yb * 12 + mb)


def _f(dist: str, k: int) -> float:
    curve = MATURITY.get(str(dist), DEFAULT_MATURITY)
    return float(curve[k]) if k < len(curve) else 1.0


def censor(true_panel: pd.DataFrame, extract_ym: int) -> pd.DataFrame:
    """extract_ym 時点の抽出を再現する。"""
    c_ym, c_dist, c_use = COLS["ym"], COLS["dist"], COLS["monthly_use"]
    d = true_panel[true_panel[c_ym] <= extract_ym].copy()

    dist = d[d[c_dist].astype(str) != st.ALL_TOKEN].copy()
    allr = d[d[c_dist].astype(str) == st.ALL_TOKEN].copy()

    lag = [_diff(extract_ym, int(m)) for m in dist[c_ym]]
    ratio = np.array([_f(s, k) for s, k in zip(dist[c_dist].astype(str), lag)])
    dist[c_use] = np.floor(pd.to_numeric(dist[c_use]).to_numpy() * ratio + 0.5)
    dist = dist[ratio > 0]          # まだ1件も届いていない月は行ごと消える

    # ALL行の分子だけ作り直す（分母＝累積販売台数はそのまま）
    if not allr.empty:
        key = [COLS["biz"], COLS["dev"], COLS["part"], c_ym]
        agg = dist.groupby(key, as_index=False)[c_use].sum()
        allr = allr.drop(columns=[c_use]).merge(agg, on=key, how="left")
        allr[c_use] = allr[c_use].fillna(0.0)

    out = pd.concat([allr, dist], ignore_index=True)
    return out.sort_values([COLS["biz"], COLS["dev"], COLS["part"],
                            c_dist, c_ym]).reset_index(drop=True)


def pick_spike_target(true_panel: pd.DataFrame, ym: int) -> tuple | None:
    """遅れの大きい販社の中から、ベースラインが十分ある単位を選ぶ。"""
    c_ym, c_dist, c_use = COLS["ym"], COLS["dist"], COLS["monthly_use"]
    d = true_panel[true_panel[c_dist].astype(str) != st.ALL_TOKEN]
    # 遅れの大きい販社の順に探す
    order = sorted({str(x) for x in d[c_dist]},
                   key=lambda s: -len(MATURITY.get(s, DEFAULT_MATURITY)))
    lo = _shift(ym, -st.C_BASE_LEN)
    for dist in order:
        g = d[(d[c_dist].astype(str) == dist) & (d[c_ym] >= lo) & (d[c_ym] < ym)]
        if g.empty:
            continue
        base = g.groupby([COLS["biz"], COLS["dev"], COLS["part"]])[c_use].agg(
            ["sum", "size"])
        base = base[(base["sum"] >= max(st.C_MIN_BASE_COUNT, 6)) &
                    (base["size"] >= st.C_MIN_BASE_MONTHS)]
        if base.empty:
            continue
        biz, dev, part = base["sum"].idxmax()
        return (biz, dev, part, dist)
    return None


def main(src: str, prefix: str = "遅延"):
    raw = pd.read_csv(src, encoding="utf-8-sig")
    c_ym, c_dist, c_use = COLS["ym"], COLS["dist"], COLS["monthly_use"]
    raw[c_ym] = raw[c_ym].astype(str).str.replace(r"\D", "", regex=True).astype(int)

    T2 = int(raw[c_ym].max())
    T1 = _shift(T2, -1)
    spike_ym = _shift(T2, -SPIKE_LAG)

    # --- 整合チェック: ALL行 == 販社別合計 か ---
    a = raw[raw[c_dist].astype(str) == st.ALL_TOKEN]
    b = raw[raw[c_dist].astype(str) != st.ALL_TOKEN]
    if not a.empty:
        key = [COLS["biz"], COLS["dev"], COLS["part"], c_ym]
        m = (a.groupby(key)[c_use].sum()
              .rename("all").to_frame()
              .join(b.groupby(key)[c_use].sum().rename("sum"), how="inner"))
        gap = int((m["all"] != m["sum"]).sum())
        print(f"[確認] ALL行と販社別合計の不一致: {gap} / {len(m)} 行"
              + ("（一致。以降 ALL は合計で作り直す）" if gap == 0 else
                 "  ← ALLは別ソースの可能性。実データでも必ず確認すること"))

    # --- 異常の仕込み ---
    tgt = pick_spike_target(raw, spike_ym)
    rec = []
    if tgt:
        biz, dev, part, dist = tgt
        sel = ((raw[COLS["biz"]] == biz) & (raw[COLS["dev"]] == dev) &
               (raw[COLS["part"]] == part) & (raw[c_dist].astype(str) == dist) &
               (raw[c_ym] == spike_ym))
        if sel.any():
            before = float(raw.loc[sel, c_use].iloc[0])
            after = max(before * SPIKE_FACTOR, st.C_MIN_COUNT + 3)
            raw.loc[sel, c_use] = after
            rec.append(dict(ID="L1", 事業コード=biz, 開発コード=dev, 部番=part,
                            販社=dist, 仕込み年月=spike_ym,
                            変更前=before, 変更後=after,
                            run1で見えるか=("×" if _f(dist, _diff(T1, spike_ym)) == 0
                                       else f"一部({_f(dist, _diff(T1, spike_ym)):.0%})"),
                            run2で見えるか=f"{_f(dist, _diff(T2, spike_ym)):.0%}"))
            print(f"[仕込み] {dev}/{part}/販社{dist}/{spike_ym}: "
                  f"{before:.0f} → {after:.0f} 件")
    else:
        print("[仕込み] 条件を満たす単位が見つからず。異常なしで生成します。")

    for T, tag in ((T1, "run1"), (T2, "run2")):
        out = censor(raw, T)
        path = f"{prefix}_{tag}.csv"
        out.to_csv(path, index=False, encoding="utf-8-sig")
        nd = out[out[c_dist].astype(str) != st.ALL_TOKEN]
        cov = nd.groupby(nd[c_dist].astype(str))[c_ym].max().to_dict()
        print(f"{path}: {len(out)}行  抽出={T}  販社別の最終月={cov}")

    if rec:
        pd.DataFrame(rec).to_csv(f"仕込み一覧_{prefix}.csv", index=False,
                                 encoding="utf-8-sig")
        print(f"仕込み一覧_{prefix}.csv （答え合わせ用。先に開かないこと）")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "遅延")
