# -*- coding: utf-8 -*-
"""
signal_c_dist.py — 信号C: 販社（国）別の月次急増検知

「特定の国で突然その部品の使用数が増えた」を拾う。
累積使用率の閾値検知は分母が積み上がるため単月の急増でほとんど動かず、
構造的にスパイクを検知できない。そこを埋めるのが本検出器。

CUSUM トラックの spike_test との違い
------------------------------------
- 監視単位が **機種 × 部番 × 販社**（販社合算しない）。
  1国だけ3倍になっても他4国が平常なら合算では埋もれるため。
- ベースラインが **直近 N ヶ月のローリング窓**（既存は経過月4〜15の固定窓）。
  「過去1年の交換率と比べて突然増えた」という定義に合わせる。

検定
----
条件付き二項（cusum_monitor.binom_sf をそのまま流用）:
  H0: 当月のレート = 直近N ヶ月のレート
  n = x_t + C を固定すると X ~ Binomial(n, e_t / (e_t + E))
  p = P(X >= x_t)
ベースラインが薄い（Cが小さい）ほど自動的に保守側に開く。

データ上の注意
--------------
販社別の行は「その販社で修理が1件でも入るまで生成されない」ため、
露出（累積販売台数）が左側で打ち切られている。初回行が立った直後は
露出が不安定なので min_base_months でガードする。
ALL 行は必ず除外すること（合算は本線トラックの担当）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import cusum_monitor as cm

ALL_TOKEN = "ALL"


def _to_ym(x) -> int:
    s = "".join(ch for ch in str(x) if ch.isdigit())
    return int(s[:6]) if len(s) >= 6 else int(s)


def prepare_dist_panel(df: pd.DataFrame, cols: dict,
                       all_token: str = ALL_TOKEN) -> pd.DataFrame:
    """販社別パネルを内部名に正規化する。ALL 行は除外し、月次0補完を行う。"""
    ren = {cols["biz"]: "biz", cols["dev"]: "dev", cols["part"]: "part",
           cols["dist"]: "dist", cols["ym"]: "ym",
           cols["monthly_use"]: "use", cols["cum_sales"]: "fleet"}
    d = df.rename(columns=ren).copy()
    d = d[d["dist"].astype(str) != all_token]
    d["ym"] = d["ym"].map(_to_ym).astype("int64")
    for c in ("use", "fleet"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["fleet"]).sort_values(["biz", "dev", "part", "dist", "ym"])

    # 月次0補完（抜けた月は使用0・台数は前方補完）
    out = []
    for key, g in d.groupby(["biz", "dev", "part", "dist"], sort=False):
        g = g.sort_values("ym")
        months, m = [], int(g["ym"].iloc[0])
        last = int(g["ym"].iloc[-1])
        while m <= last:
            months.append(m)
            y, mo = divmod(m, 100)
            m = (y + 1) * 100 + 1 if mo == 12 else m + 1
        g2 = g.set_index("ym").reindex(months)
        g2["use"] = g2["use"].fillna(0.0)
        g2["fleet"] = g2["fleet"].ffill()
        for i, k in enumerate(["biz", "dev", "part", "dist"]):
            g2[k] = key[i]
        out.append(g2.reset_index().rename(columns={"index": "ym"}))
    return pd.concat(out, ignore_index=True)


def _rolling_test(use: np.ndarray, fleet: np.ndarray, base_len: int,
                  alpha: float, min_count: int, min_base_months: int,
                  min_base_count: float):
    """1系列のローリング条件付き二項検定。Returns (p, alert, O_E, base_rate)."""
    n = len(use)
    p = np.full(n, np.nan)
    oe = np.full(n, np.nan)
    br = np.full(n, np.nan)
    alert = np.zeros(n, dtype=bool)

    cu = np.concatenate([[0.0], np.cumsum(use)])
    cf = np.concatenate([[0.0], np.cumsum(fleet)])

    for t in range(n):
        a = max(0, t - base_len)
        if t - a < min_base_months:
            continue
        C = cu[t] - cu[a]
        E = cf[t] - cf[a]
        if E <= 0 or C < min_base_count:
            continue
        rate = C / E
        br[t] = rate
        exp = rate * fleet[t]
        if exp <= 0:
            continue
        oe[t] = use[t] / exp
        # 高速化: 上振れかつ最低件数を満たす月だけ正確検定にかける
        if use[t] < min_count or use[t] <= exp:
            continue
        pv = cm.binom_sf(int(round(use[t])),
                         int(round(use[t] + C)),
                         fleet[t] / (fleet[t] + E))
        p[t] = pv
        alert[t] = pv <= alpha
    return p, alert, oe, br


def run_signal_c(panel: pd.DataFrame, cols: dict, base_len: int = 12,
                 alpha: float = 0.005, min_count: int = 3,
                 min_base_months: int = 6, min_base_count: float = 3.0,
                 asof_ym: int | None = None, months_back: int = 0,
                 all_token: str = ALL_TOKEN,
                 horizon: dict | None = None,
                 revisit_months: int = 0) -> pd.DataFrame:
    """信号Cの本体。

    asof_ym     : 判定基準月。None ならパネル最新月。
    months_back : 0 なら基準月のみ判定（本番運用）。>0 なら過去にさかのぼって
                  判定し、過去例の再現確認・バックテストに使う。
    min_base_months : ベースラインに必要な最低月数（左側打ち切り対策）
    min_base_count  : ベースライン窓の最低使用数（薄すぎる窓での判定を避ける）

    ▼ 販社別の報告遅れへの対応（reporting_horizon.py と対で使う）
    horizon        : {販社: 完全と見なせる最終年月}。指定すると
                     (a) その月より後ろを系列から**切り落としてから**検定する
                         （未確定月がローリングベースラインを下振れさせるのを防ぐ）
                     (b) 判定対象月もその販社の horizon までに制限する
    revisit_months : 直近この月数を毎回さかのぼって再判定する。
                     **0 のままだと遅れている販社の月は永久に一度も検定されない。**
                     販社の最大遅れ月数以上にすること（reporting_horizon.check_revisit）。

    horizon=None かつ revisit_months=0 なら従来と完全に同じ挙動になる。
    """
    d = prepare_dist_panel(panel, cols, all_token=all_token)
    if d.empty:
        return d
    T = int(asof_ym) if asof_ym is not None else int(d["ym"].max())

    lo = _shift_ym(T, -(int(months_back) + int(revisit_months)))

    rows = []
    for (biz, dev, part, dist), g in d.groupby(["biz", "dev", "part", "dist"], sort=False):
        g = g.sort_values("ym").reset_index(drop=True)

        # --- 販社ごとの完全月で系列を切る ---------------------------------
        # 切ってから検定するのが要点。未確定（部分的にしか届いていない）月を
        # ローリングベースラインに残すとレートが下振れし、翌月が鳴りやすくなる。
        Td = T if horizon is None else min(T, int(horizon.get(dist, T)))
        g = g[g["ym"] <= Td]
        if g.empty:
            continue

        use = g["use"].to_numpy(dtype=float)
        fleet = g["fleet"].to_numpy(dtype=float)
        p, al, oe, br = _rolling_test(use, fleet, base_len, alpha, min_count,
                                      min_base_months, min_base_count)
        ym = g["ym"].to_numpy()
        sel = np.flatnonzero((ym <= Td) & (ym >= lo))
        for t in sel:
            rows.append(dict(
                biz=biz, dev=dev, part=part, dist=dist, ym=int(ym[t]),
                use=use[t], fleet=fleet[t], base_rate=br[t],
                expected=br[t] * fleet[t] if not np.isnan(br[t]) else np.nan,
                O_E=oe[t], p=p[t], alert_dist=bool(al[t]),
                run_ym=T, 遅延月=_diff_ym(T, int(ym[t])),
            ))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["注目度"] = np.where(
        out["p"].notna() & (out["p"] > 0),
        np.minimum(np.log(out["p"].clip(lower=1e-300)) / np.log(alpha), 3.0), 0.0)
    out.loc[out["p"].isna(), "注目度"] = 0.0
    return out.sort_values(["alert_dist", "注目度"], ascending=[False, False]).reset_index(drop=True)


def _shift_ym(ym: int, k: int) -> int:
    y, m = divmod(int(ym), 100)
    idx = y * 12 + (m - 1) + k
    return (idx // 12) * 100 + (idx % 12) + 1


def _diff_ym(a: int, b: int) -> int:
    ya, ma = divmod(int(a), 100)
    yb, mb = divmod(int(b), 100)
    return (ya * 12 + ma) - (yb * 12 + mb)


def summarize_by_unit(res: pd.DataFrame) -> pd.DataFrame:
    """販社別の発火を 機種×部番 に畳んで、本線インボックスと突き合わせやすくする。"""
    a = res[res["alert_dist"]]
    if a.empty:
        return a
    g = a.groupby(["biz", "dev", "part", "ym"], as_index=False).agg(
        発火販社数=("dist", "nunique"),
        販社=("dist", lambda s: "/".join(sorted(set(s)))),
        最大O_E=("O_E", "max"), 最小p=("p", "min"), 注目度=("注目度", "max"),
        使用数計=("use", "sum"))
    return g.sort_values("注目度", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import gen_dist_panel as gd
    raw = gd.build()
    cols = dict(biz="事業コード", dev="開発コード", part="部番", dist="販社",
                ym="年月", monthly_use="月次使用数", cum_sales="累積販売台数")
    res = run_signal_c(raw, cols, months_back=24)
    print("=== 信号C 発火:", int(res["alert_dist"].sum()), "／判定行:", len(res), "===")
    print(res[res.alert_dist].head(10).to_string(index=False))
    print("\n=== 機種×部番 に畳んだもの ===")
    print(summarize_by_unit(res).head(10).to_string(index=False))
