"""
earlylife_baseline.py
=====================
安定化前(初期不良期)の期待故障カーブ lambda0(t) を、先行機種群から推定する。

なぜ別物か
----------
安定期の監視は「その機種自身の過去」を物差しにできた。だが安定化前は
自分の履歴がまだ無いので、期待値を外(=同じSF-コードの先行機種)から借りる
必要がある。故障率は経過月とともに「初期高→減衰→安定」と下がるので、
固定のlambda0ではなく経過月の関数 lambda0(t) を推定する。

少数・ランク差への対処: 階層ベイズ・ポアソン(部分プーリング)
--------------------------------------------------------------
SF-コードあたりの先行機種は2〜7と少なく、ランク差もある。少数のノイジーな
低カウント系列から各経過月のレートを素直に平均すると、機種ごとのばらつきに
振り回される。そこで各経過月 t について、機種横断のレートを

    lambda_{model,t} ~ Gamma 事前(集団のレート分布)
    使用数_{model,t} ~ Poisson(lambda_{model,t} * 台数_{model,t})

という階層で捉え、Gamma-Poisson共役による経験ベイズで集団分布(事前)を推定し、
各機種の推定を集団平均へ縮約する(partial pooling)。データの薄い機種ほど
強く集団平均へ寄せられるので、少数・低カウントに頑健になる。
MCMCは使わず、共役性とモーメント法だけで閉じる(numpy/scipyのみ)。

ランク差の扱い
--------------
ランク(価格帯)で基準レートの水準が変わる。最も素直な対処は「ランクごとに
別の集団としてカーブを推定する」こと。本モジュールは group_keys で
プーリング集団を指定できる(例: ["SF-コード"] あるいは ["SF-コード","ランク"])。
ランクを集団に含めれば、ランク内でプールされ階層差が物差しを歪めない。
ランクを跨いでプールしたい場合はランクを共変量にした対数線形モデルが要るが、
ここでは「集団を分ける」素直な方法に留める(設計の段階的拡張方針に沿う)。

スケールの固定(重要)
--------------------
推定したカーブは、監視対象の新機種【自身】の初期データでスケールを合わせ直さ
ないこと。合わせると検出したい異常そのものをカーブが吸収してしまう
(移動平均lambda0がドリフトを吸う問題と同根)。スケールは先行機種側で固定する。
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def estimate_earlylife_curve(
    df_leaders: pd.DataFrame,
    group_keys: list[str],
    max_keizoku: int,
    col_keizoku: str = "経過月",
    col_usage: str = "月次使用数",
    col_fleet: str = "累積販売台数",
    col_model: str = "開発コード",
    prior_strength_floor: float = 1.0,
    smooth_window: int = 3,
    rate_floor_frac: float = 0.1,
) -> pd.DataFrame:
    """先行機種群から、経過月ごとの期待レート lambda0(t) を経験ベイズで推定する。

    Parameters
    ----------
    df_leaders : 先行機種(履歴の揃った同SF-コードの機種)の縦長パネル。
    group_keys : プーリング集団を決めるキー。例 ["事業コード","SF-コード"]。
                 ランク差を分けたいなら ["事業コード","SF-コード","ランク"]。
    max_keizoku: カーブを推定する経過月の上限(安定化するあたりまで)。
    smooth_window: 経過月方向の移動平均窓(奇数)。早期は台数が小さくゼロ故障の
                 月が多いため、単月推定はガタつく。隣接月で均して安定させる。
                 1 なら平滑化なし。
    rate_floor_frac: ゼロ下限。各集団のカーブ中央値 * この割合 を下限とし、
                 推定が0の月でも極小の期待値を持たせる。これが無いと
                 「期待0の月にカウントが出ると常に p=0 で鳴る」病理が起きる。

    方法(各 group × 経過月 t について)
    -----------------------------------
    1. 機種ごとの当月レート r_i = usage_i / fleet_i を、その t に居る機種から集める。
    2. それらに Gamma(a, b) を当てはめる(モーメント法)。これが集団事前。
    3. 集団全体の縮約レートを lambda0(t) = (Σusage + a) / (Σfleet + b) で出す
       (Gamma-Poisson の事後平均=プールド)。データが薄い t ほど事前へ寄る。
    4. 経過月方向に移動平均で平滑化し、ゼロ下限を課す。

    Returns
    -------
    DataFrame: group_keys + [経過月, lambda0_hat, lambda0_raw, n_models,
                            total_usage, total_fleet]
    """
    rows = []
    for gkey, g in df_leaders.groupby(group_keys, dropna=False):
        gkey = gkey if isinstance(gkey, tuple) else (gkey,)
        recs = []
        for t in range(0, max_keizoku + 1):
            sub = g[g[col_keizoku] == t]
            sub = sub[sub[col_fleet] > 0]
            if len(sub) == 0:
                continue
            usage = sub[col_usage].to_numpy(dtype=float)
            fleet = sub[col_fleet].to_numpy(dtype=float)
            rates = usage / fleet
            n_models = len(rates)

            m = float(np.mean(rates))
            v = float(np.var(rates, ddof=1)) if n_models >= 2 else 0.0
            if m <= 0:
                lam_raw = 0.0
            else:
                if v > 0:
                    a = m * m / v
                    b = m / v
                else:
                    a = prior_strength_floor
                    b = prior_strength_floor / m
                a = max(a, prior_strength_floor * m)
                b = max(b, prior_strength_floor)
                lam_raw = (usage.sum() + a) / (fleet.sum() + b)

            recs.append({
                **{k: v_ for k, v_ in zip(group_keys, gkey)},
                col_keizoku: t,
                "lambda0_raw": lam_raw,
                "n_models": n_models,
                "total_usage": float(usage.sum()),
                "total_fleet": float(fleet.sum()),
            })

        if not recs:
            continue
        cg = pd.DataFrame(recs).sort_values(col_keizoku).reset_index(drop=True)

        # --- 経過月方向の平滑化(中央寄せ移動平均) ---
        raw = cg["lambda0_raw"].to_numpy(dtype=float)
        if smooth_window and smooth_window > 1:
            w = int(smooth_window)
            half = w // 2
            sm = np.empty_like(raw)
            for i in range(len(raw)):
                lo = max(0, i - half)
                hi = min(len(raw), i + half + 1)
                sm[i] = raw[lo:hi].mean()
        else:
            sm = raw.copy()

        # --- ゼロ下限: カーブ中央値の一定割合を最低レートにする ---
        pos = sm[sm > 0]
        floor = (np.median(pos) * rate_floor_frac) if len(pos) else 0.0
        sm = np.where(sm < floor, floor, sm)

        cg["lambda0_hat"] = sm
        rows.append(cg)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    cols = group_keys + [col_keizoku, "lambda0_hat", "lambda0_raw",
                         "n_models", "total_usage", "total_fleet"]
    return out[cols]


def attach_curve_to_unit(
    df_unit: pd.DataFrame,
    curve: pd.DataFrame,
    group_keys: list[str],
    col_keizoku: str = "経過月",
) -> np.ndarray:
    """監視対象1単位の各月に、推定カーブから lambda0_t を引き当てて配列で返す。

    単位の group_keys 値に対応するカーブを経過月でマージする。カーブに無い
    経過月(推定範囲外)は、最も近い推定済み経過月の値で前方/後方埋めする。
    """
    gvals = {k: df_unit[k].iloc[0] for k in group_keys}
    c = curve.copy()
    for k, val in gvals.items():
        c = c[c[k] == val]
    if len(c) == 0:
        # 対応する集団のカーブが無い → 全月0(監視不能)を返す
        return np.zeros(len(df_unit))

    c = c.sort_values(col_keizoku)
    lam_by_m = dict(zip(c[col_keizoku], c["lambda0_hat"]))
    ms = c[col_keizoku].to_numpy()
    lam_min_m, lam_max_m = ms.min(), ms.max()

    out = []
    for m in df_unit[col_keizoku].to_numpy():
        if m in lam_by_m:
            out.append(lam_by_m[m])
        elif m < lam_min_m:
            out.append(lam_by_m[lam_min_m])
        else:
            out.append(lam_by_m[lam_max_m])
    return np.asarray(out, dtype=float)
