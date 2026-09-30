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
                  min_base_count: float, min_oe: float = 1.0,
                  min_excess: float = 0.0, exceed_hist: float = 0.0):
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
        br[t] = C / E

        # --- 自己履歴による下限 ---
        # その系列が過去に見せた最大の振れ幅を超えないと発火させない。
        # 「部品の性質上もともと出入りが大きい」系列を自動的に鈍感にする。
        # まとめ発注・季節性など、倍率では止まらない振れに効く。
        hist_floor = 0.0
        if exceed_hist > 0:
            wu, wf = use[a:t], fleet[a:t]
            ok = wf > 0
            if ok.sum() >= 2 and C > 0:
                ratios = (wu[ok] / wf[ok]) / (C / E)
                hist_floor = float(np.max(ratios)) * exceed_hist

        # O/E の分母。ベースラインが 0 件でも比が定義できるよう、
        # 「0.5件はあったかもしれない」という下駄を履かせる（Jeffreys 相当）。
        # p値の計算には実測の C をそのまま使うので、検定の厳密さは損なわれない。
        exp_eff = (max(C, 0.5) / E) * fleet[t]
        if exp_eff <= 0:
            continue
        oe[t] = use[t] / exp_eff

        # 効果量で先に切る。p値だけだと件数の多い系列ほど小さい変化で鳴るため
        # （月50件の系列は1.4倍で有意になるが、実務的には変動の範囲）。
        if use[t] < min_count or oe[t] < max(min_oe, hist_floor):
            continue
        if (use[t] - exp_eff) < min_excess:   # 増加の絶対量が小さすぎるもの
            continue

        # 条件付き二項は C=0 でも定義できる（n = 当月件数、p = pi^x）。
        pv = cm.binom_sf(int(round(use[t])),
                         int(round(use[t] + C)),
                         fleet[t] / (fleet[t] + E))
        p[t] = pv
        alert[t] = pv <= alpha
    return p, alert, oe, br


def run_signal_c(panel: pd.DataFrame, cols: dict, base_len: int = 12,
                 alpha: float = 0.005, min_count: int = 3,
                 min_base_months: int = 6, min_base_count: float = 0.0,
                 min_oe: float = 3.0, min_excess: float = 0.0,
                 exceed_hist: float = 0.0, exclude: set | None = None,
                 asof_ym: int | None = None, months_back: int = 0,
                 all_token: str = ALL_TOKEN) -> pd.DataFrame:
    """信号Cの本体。

    asof_ym     : 判定基準月。None ならパネル最新月。
    months_back : 0 なら基準月のみ判定（本番運用）。>0 なら過去にさかのぼって
                  判定し、過去例の再現確認・バックテストに使う。
    min_base_months : ベースラインに必要な最低月数（左側打ち切り対策）
    min_base_count  : ベースライン窓の最低使用数。**既定0＝制限なし**。
                      条件付き二項はベースラインが薄いほど自動で保守側に開くので、
                      ここで切る必要は本来ない。むしろ「ほぼゼロ→突然多数」という
                      最も拾いたいパターンを消してしまう。
    min_oe          : **主レバー**。ベースラインの何倍以上で発火させるか。
                      p値だけで切ると件数の多い系列ほど小さい変化で鳴る
                      （月50件なら1.42倍で有意）。倍率で足切りする。
    min_excess      : 増加の絶対量の下限（件）。0なら無効。
                      「1件→4件」のような小さい絶対増を落としたいときに使う。
    exceed_hist     : **もともと振れの大きい系列を鈍感にするレバー。** 0で無効。
                      1.0 なら「その系列が過去12ヶ月に見せた最大の振れ幅を
                      超えたときだけ発火」。1.2 ならその1.2倍を要求する。
                      まとめ発注・季節性など、倍率の下限では止まらない
                      構造的な振れに効く。min_oe とは OR でなく max で併用。
    exclude         : 除外する部番の集合。{"P-1234"} または {("M01","P-1234")}。
                      消耗品など、性質上つねに出入りが多いと分かっているもの。
    """
    d = prepare_dist_panel(panel, cols, all_token=all_token)
    if d.empty:
        return d
    T = int(asof_ym) if asof_ym is not None else int(d["ym"].max())

    rows = []
    for (biz, dev, part, dist), g in d.groupby(["biz", "dev", "part", "dist"], sort=False):
        g = g.sort_values("ym").reset_index(drop=True)
        use = g["use"].to_numpy(dtype=float)
        fleet = g["fleet"].to_numpy(dtype=float)
        if exclude and (part in exclude or (dev, part) in exclude
                        or (biz, dev, part) in exclude):
            continue
        p, al, oe, br = _rolling_test(use, fleet, base_len, alpha, min_count,
                                      min_base_months, min_base_count,
                                      min_oe, min_excess, exceed_hist)
        ym = g["ym"].to_numpy()
        sel = np.flatnonzero((ym <= T) & (ym >= _shift_ym(T, -months_back)))
        for t in sel:
            rows.append(dict(
                biz=biz, dev=dev, part=part, dist=dist, ym=int(ym[t]),
                use=use[t], fleet=fleet[t], base_rate=br[t],
                expected=br[t] * fleet[t] if not np.isnan(br[t]) else np.nan,
                O_E=oe[t], p=p[t], alert_dist=bool(al[t]),
            ))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # 注目度は O/E ベース。p値は件数で桁が飛ぶので並び順の物差しにならない。
    base = max(min_oe, 1.0)
    out["注目度"] = np.where(
        out["alert_dist"],
        np.minimum(1.0 + 2.0 * (out["O_E"] - base) / max(2.0 * base, 1e-9), 3.0),
        0.0).round(3)
    return out.sort_values(["alert_dist", "注目度"], ascending=[False, False]).reset_index(drop=True)


def _shift_ym(ym: int, k: int) -> int:
    y, m = divmod(int(ym), 100)
    idx = y * 12 + (m - 1) + k
    return (idx // 12) * 100 + (idx % 12) + 1


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


def inspect(panel: pd.DataFrame, cols: dict, dev, part, dist=None,
            base_len: int = 12, alpha: float = 0.005, min_count: int = 3,
            min_base_months: int = 6, min_base_count: float = 0.0,
            min_oe: float = 3.0, min_excess: float = 0.0,
            exceed_hist: float = 0.0,
            all_token: str = ALL_TOKEN, months: int = 24) -> pd.DataFrame:
    """特定の部番の月次推移を、判定の中身つきで表示する（Jupyter用）。

    なぜ鳴った / なぜ鳴らなかったかが1行ずつ分かる。
        期待   … ベースラインから予測される件数
        O/E    … 実測 ÷ 期待（倍率）
        p      … 条件付き二項のp値。NaN は検定に到達しなかったことを意味する
        判定   … 発火なら ★、そうでなければ沈黙の理由
    """
    d = prepare_dist_panel(panel, cols, all_token=all_token)
    d = d[d["dev"].astype(str) == str(dev)]
    d = d[d["part"].astype(str) == str(part)]
    if dist is not None:
        d = d[d["dist"].astype(str) == str(dist)]
    if d.empty:
        print(f"該当データなし: dev={dev} part={part} dist={dist}")
        return d

    out = []
    for (b, dv, pt, ds), g in d.groupby(["biz", "dev", "part", "dist"], sort=False):
        g = g.sort_values("ym").reset_index(drop=True)
        u = g["use"].to_numpy(float); f = g["fleet"].to_numpy(float)
        pv, al, oe, br = _rolling_test(u, f, base_len, alpha, min_count,
                                       min_base_months, min_base_count,
                                       min_oe, min_excess, exceed_hist)
        ym = g["ym"].to_numpy()
        for t in range(len(g)):
            exp = br[t] * f[t] if not np.isnan(br[t]) else np.nan
            exp_eff = np.nan if np.isnan(oe[t]) or oe[t] == 0 else u[t] / oe[t]
            if al[t]:
                why = "★発火"
            elif t < min_base_months:
                why = "窓不足"
            elif np.isnan(oe[t]):
                why = "判定対象外"
            elif u[t] < min_count:
                why = f"件数{int(u[t])}<{min_count}"
            elif oe[t] < min_oe:
                why = f"O/E{oe[t]:.1f}<{min_oe}"
            elif np.isnan(pv[t]) and exceed_hist > 0:
                why = f"自己履歴超えず(O/E{oe[t]:.1f})"
            elif np.isnan(pv[t]):
                why = "増加量不足"
            else:
                why = f"p={pv[t]:.1e}>α"
            out.append(dict(販社=ds, 年月=int(ym[t]), 使用数=int(u[t]),
                            台数=int(f[t]),
                            期待=round(exp_eff, 2) if not np.isnan(exp_eff) else None,
                            O_E=round(oe[t], 2) if not np.isnan(oe[t]) else None,
                            p=f"{pv[t]:.2e}" if not np.isnan(pv[t]) else "",
                            判定=why))
    r = pd.DataFrame(out)
    r = r[r["年月"] >= _shift_ym(int(r["年月"].max()), -months)]
    print(f"=== {dev} / {part}"
          + (f" / 販社{dist}" if dist else "")
          + f"（直近{months}ヶ月, min_oe={min_oe}, α={alpha}）===")
    n = int((r["判定"] == "★発火").sum())
    print(f"発火 {n} 件\n")
    print(r.to_string(index=False))
    return r
