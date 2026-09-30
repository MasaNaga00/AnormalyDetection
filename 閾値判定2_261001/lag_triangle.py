# -*- coding: utf-8 -*-
"""
lag_triangle.py — 修理受付日ベースの打ち切りを、1回の抽出だけから測る

前提となるデータ構造
--------------------
- 監視の日付軸 = **修理受付日**（販社が修理品を預かった日）＝実イベント日
- レコードがデータに現れるのは **修理完了日以降**（客先返送後）
- さらに販社ごとの送付頻度（毎日/週次/月次）で本社到着が遅れる

したがって受付月Mの件数は、抽出時点では
    「M月に受け付け、かつ 修理完了 → 販社送付 まで済んだぶん」
しか見えていない。時間が経つほど後から積み上がる（保険のIBNRと同じ形）。

なぜスナップショットが要らないか
--------------------------------
各レコードが受付日と完了日を**両方**持っているので、1回の抽出から
「受付月 × 遅れ月数」のディベロップメント三角形が遡及的に作れる。
過去の抽出履歴を残していなくても、遅れの分布はいま測れる。

最重要の注意
------------
**この打ち切りは異常と相関する。** 大ロット不良が起きると部品が枯渇し、
修理完了までの日数が伸びる。つまり**最も早く知りたい月ほど、最も多く
打ち切られる**。単なる遅れではなく、見逃し方向の系統バイアス。
→ horizon は保守的に（完成度95〜97%）取るべきで、「だいたい揃った」では甘い。

使い方
------
    import lag_triangle as lt
    prof = lt.lag_profile(rec, C)               # 遅れ分布 f(k)
    hz, K = lt.recommend_horizon(rec, C, target=0.95)
    # hz を settings.HORIZON_FIXED に、K を settings.C_REVISIT_MONTHS に
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def to_ym(x) -> int:
    s = "".join(ch for ch in str(x) if ch.isdigit())
    return int(s[:6]) if len(s) >= 6 else int(s)


def shift_ym(ym: int, k: int) -> int:
    y, m = divmod(int(ym), 100)
    i = y * 12 + (m - 1) + k
    return (i // 12) * 100 + (i % 12) + 1


def diff_ym(a: int, b: int) -> int:
    ya, ma = divmod(int(a), 100)
    yb, mb = divmod(int(b), 100)
    return (ya * 12 + ma) - (yb * 12 + mb)


# ============================================================================
# 前処理
# ============================================================================
def prepare(rec: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """修理レコードを内部名に正規化する。

    cols は最低限 recv / comp / dist を持つこと。part / dev は任意。
        cols = dict(recv="修理受付日", comp="修理完了日", dist="販社",
                    part="部番", dev="開発コード")
    """
    ren = {cols["recv"]: "recv", cols["comp"]: "comp", cols["dist"]: "dist"}
    for k in ("part", "dev", "biz"):
        if cols.get(k) and cols[k] in rec.columns:
            ren[cols[k]] = k
    d = rec.rename(columns=ren).copy()
    d["recv_ym"] = d["recv"].map(to_ym).astype("int64")
    d["comp_ym"] = d["comp"].map(to_ym).astype("int64")
    d["dist"] = d["dist"].astype(str)
    d["lag"] = [diff_ym(c, r) for c, r in zip(d["comp_ym"], d["recv_ym"])]
    bad = int((d["lag"] < 0).sum())
    if bad:
        print(f"[警告] 完了日 < 受付日 のレコードが {bad} 件。除外します。")
        d = d[d["lag"] >= 0]
    return d


# ============================================================================
# 完了側の horizon（販社ごとに「完了ベースでどこまで届いているか」）
# ============================================================================
def completion_horizon(d: pd.DataFrame, thin_ratio: float = 0.7,
                       ref_months: int = 6) -> dict:
    """販社ごとの「完了月ベースの受領済み最終月」。

    完了月の分布は打ち切りを受けない（届いたものは必ず完了済み）ので、
    ここは素直に最大値でよい。末尾が薄ければ月の途中と見なして1ヶ月落とす。
    """
    out = {}
    for dist, g in d.groupby("dist"):
        cnt = g.groupby("comp_ym").size().sort_index()
        last = int(cnt.index[-1])
        ref = cnt.iloc[max(0, len(cnt) - 1 - ref_months): len(cnt) - 1]
        med = float(np.median(ref)) if len(ref) else np.nan
        thin = (med and med > 0 and float(cnt.iloc[-1]) < thin_ratio * med)
        out[dist] = shift_ym(last, -1) if thin else last
    return out


# ============================================================================
# ディベロップメント（遅れ分布）
# ============================================================================
def lag_profile(d: pd.DataFrame, max_lag: int = 12,
                comp_hz: dict | None = None,
                by: str | None = None) -> pd.DataFrame:
    """遅れ月数 k までに何%が出そろうかの累積分布 f(k) を返す。

    by=None なら全体、"dist" なら販社別、"part" なら部番別。

    成熟コホートだけを使う
    ----------------------
    受付月Mのコホートは、完了側horizon まで max_lag ヶ月以上ある月でないと
    それ自体が打ち切られていて f(k) を歪める。そこだけを母集団にする。
    """
    comp_hz = comp_hz or completion_horizon(d)
    d = d.copy()
    d["comp_hz"] = d["dist"].map(comp_hz)
    d = d.dropna(subset=["comp_hz"])
    # 見えているレコードだけ（完了が販社horizonを越えるものは本来まだ届かない）
    d = d[d["comp_ym"] <= d["comp_hz"]]
    # 成熟コホート
    d["room"] = [diff_ym(int(h), int(r)) for h, r in zip(d["comp_hz"], d["recv_ym"])]
    mat = d[d["room"] >= max_lag]
    if mat.empty:
        raise ValueError("成熟コホートが無い。max_lag を小さくするか履歴を伸ばす。")

    keys = [] if by is None else [by]
    rows = []
    for key, g in (mat.groupby(keys) if keys else [((), mat)]):
        n = len(g)
        h = g["lag"].clip(upper=max_lag).value_counts().sort_index()
        cum = h.reindex(range(0, max_lag + 1), fill_value=0).cumsum() / n
        r = dict(件数=n)
        if keys:
            r[by] = key if not isinstance(key, tuple) else key[0]
        for k in range(0, max_lag + 1):
            r[f"f{k}"] = round(float(cum.loc[k]), 4)
        rows.append(r)
    out = pd.DataFrame(rows)
    front = ([by] if keys else []) + ["件数"]
    return out[front + [c for c in out.columns if c not in front]]


def required_lag(prof_row: pd.Series, target: float = 0.95,
                 max_lag: int = 12) -> int:
    """完成度 target に達する最小の遅れ月数 k を返す。"""
    for k in range(0, max_lag + 1):
        if float(prof_row.get(f"f{k}", 0.0)) >= target:
            return k
    return max_lag


# ============================================================================
# 推奨値
# ============================================================================
def recommend_horizon(d: pd.DataFrame, target: float = 0.95,
                      max_lag: int = 12, per_dist: bool = True) -> tuple:
    """(受付月ベースの horizon dict, 推奨 C_REVISIT_MONTHS) を返す。

    受付月 M が「完成度 target 以上」と言えるのは
        完了側horizon − M >= 必要遅れ月数 k*
    のとき。したがって horizon_recv[d] = comp_hz[d] − k*。
    """
    comp_hz = completion_horizon(d)
    if per_dist:
        try:
            prof = lag_profile(d, max_lag, comp_hz, by="dist").set_index("dist")
        except ValueError:
            per_dist = False
    if not per_dist:
        prof = None
    g_prof = lag_profile(d, max_lag, comp_hz).iloc[0]
    g_k = required_lag(g_prof, target, max_lag)

    hz, ks = {}, []
    T = int(d["recv_ym"].max())
    for dist, ch in comp_hz.items():
        if prof is not None and dist in prof.index and prof.loc[dist, "件数"] >= 100:
            k = required_lag(prof.loc[dist], target, max_lag)
        else:
            k = g_k
        hz[dist] = shift_ym(int(ch), -int(k))
        ks.append(diff_ym(T, hz[dist]))
    return hz, int(max(ks)) if ks else 0


def summary(d: pd.DataFrame, target: float = 0.95, max_lag: int = 12) -> pd.DataFrame:
    """販社別のまとめ。実データでまずこれを見る。"""
    comp_hz = completion_horizon(d)
    hz, K = recommend_horizon(d, target, max_lag)
    try:
        prof = lag_profile(d, max_lag, comp_hz, by="dist").set_index("dist")
    except ValueError:
        prof = None
    T = int(d["recv_ym"].max())
    rows = []
    for dist in sorted(comp_hz):
        p = prof.loc[dist] if (prof is not None and dist in prof.index) else None
        rows.append(dict(
            販社=dist,
            受付月最大=int(d[d.dist == dist]["recv_ym"].max()),
            完了月horizon=comp_hz[dist],
            中央遅れ月=(required_lag(p, 0.5, max_lag) if p is not None else np.nan),
            必要遅れ月=(required_lag(p, target, max_lag) if p is not None else np.nan),
            受付horizon=hz[dist],
            遅れ月数=diff_ym(T, hz[dist]),
            成熟件数=(int(p["件数"]) if p is not None else 0)))
    r = pd.DataFrame(rows).sort_values("遅れ月数", ascending=False).reset_index(drop=True)
    print(f"[推奨] C_REVISIT_MONTHS = {K}   （完成度目標 {target:.0%}）")
    return r


def slow_parts(d: pd.DataFrame, target: float = 0.95, max_lag: int = 12,
               min_count: int = 50, top: int = 20) -> pd.DataFrame:
    """遅れが構造的に長い部番を洗い出す（慢性的な部品供給難のサイン）。

    全体の必要遅れ月数より大きい部番は、その部番だけ horizon を余分に
    取る必要がある。同時に「その部番は普段から部品が足りていない」という
    独立した情報でもある。
    """
    if "part" not in d.columns:
        raise ValueError("部番列が無い。cols に part を渡すこと。")
    comp_hz = completion_horizon(d)
    prof = lag_profile(d, max_lag, comp_hz, by="part")
    prof = prof[prof["件数"] >= min_count].copy()
    prof["中央遅れ月"] = [required_lag(r, 0.5, max_lag) for _, r in prof.iterrows()]
    prof["必要遅れ月"] = [required_lag(r, target, max_lag) for _, r in prof.iterrows()]
    g = required_lag(lag_profile(d, max_lag, comp_hz).iloc[0], target, max_lag)
    prof["全体比"] = prof["必要遅れ月"] - g
    return (prof[["part", "件数", "中央遅れ月", "必要遅れ月", "全体比"]]
            .sort_values(["必要遅れ月", "件数"], ascending=[False, False])
            .head(top).reset_index(drop=True))


# ============================================================================
# 打ち切りが異常と相関することの確認
# ============================================================================
def turnaround_trend(d: pd.DataFrame, months: int = 24,
                     by: str | None = None) -> pd.DataFrame:
    """完了月ごとの平均・中央 遅れ月数の推移。

    遅れが伸びている＝部品が足りていない＝需要が跳ねている、の間接指標。
    件数が打ち切られて見えない局面でも、**完了したぶんの遅れ**は観測できる。
    by="part" にすれば部番別に見られる（診断用。検出器ではない）。
    """
    keys = ["comp_ym"] + ([by] if by else [])
    g = d.groupby(keys).agg(件数=("lag", "size"),
                            平均遅れ=("lag", "mean"),
                            中央遅れ=("lag", "median")).reset_index()
    last = int(g["comp_ym"].max())
    return g[g["comp_ym"] >= shift_ym(last, -(months - 1))].reset_index(drop=True)


# ============================================================================
if __name__ == "__main__":
    # --- 自己テスト: 遅れ分布と打ち切りを既知の値で作って復元できるか ---
    rng = np.random.default_rng(3)
    C = dict(recv="修理受付日", comp="修理完了日", dist="販社", part="部番")
    rows = []
    # 販社A: 遅れ0-1ヶ月が中心 / 販社B: 遅れが長い(0-4) / 送付は月次で1ヶ月遅い
    lag_p = {"A": [0.55, 0.30, 0.10, 0.03, 0.02],
             "B": [0.20, 0.25, 0.25, 0.20, 0.10]}
    for dist, pv in lag_p.items():
        m = 202301
        while m <= 202503:
            for _ in range(rng.poisson(120)):
                lg = int(rng.choice(len(pv), p=pv))
                rows.append({C["recv"]: m, C["comp"]: shift_ym(m, lg),
                             C["dist"]: dist,
                             C["part"]: f"P{rng.integers(1, 4)}"})
            m = shift_ym(m, 1)
    rec = pd.DataFrame(rows)
    # 抽出時点の打ち切り: 完了が 202503 を超えるものはまだ届いていない
    #                     さらに販社Bは月次送付で1ヶ月遅い
    cut = {"A": 202503, "B": 202502}
    rec = rec[[to_ym(c) <= cut[dd] for c, dd in zip(rec[C["comp"]], rec[C["dist"]])]]

    d = prepare(rec, C)
    print("=== 遅れ分布 f(k) 販社別 ===")
    print(lag_profile(d, max_lag=6, by="dist").to_string(index=False))
    print("\n=== まとめ ===")
    print(summary(d, target=0.95, max_lag=6).to_string(index=False))
    print("\n期待: Aは必要遅れ月2前後、Bは4前後。真の分布 A累積 .55/.85/.95, B .20/.45/.70/.90/1.0")
    print("\n=== 完了月ごとの遅れ推移（末尾は打ち切りで短く見える点に注意）===")
    print(turnaround_trend(d, months=6).to_string(index=False))
