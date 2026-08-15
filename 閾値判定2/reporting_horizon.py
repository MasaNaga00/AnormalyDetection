# -*- coding: utf-8 -*-
"""
reporting_horizon.py — 販社ごとの「データが揃っている最終年月」を決める

背景
----
販社によって修理データの送付頻度が違う（毎日／週次／月次）。そのため
ある時点でパネルを作ると、販社ごとに「どこまでのデータが入っているか」が
バラバラになる。3月上旬に集計すると、月次送付の販社は12月分までしか
入っていない、ということが起こる。

これが何を壊すか
----------------
1. **信号C（販社別の月次急増）** — 最も深刻。
   `run_signal_c` は既定で「基準月Tの1ヶ月だけ」を判定する。遅れている販社は
   その月の行が存在しないので**判定対象にすらならず**、翌月以降も T しか見ない
   ため、その販社の12月・1月は**永久に一度も検定されない**。
   → 直近 K ヶ月を毎回さかのぼって再評価する必要がある（revisit_months）。

2. **未確定月がベースラインを汚す** — 部分的にしか届いていない月を
   ローリングベースラインに入れるとレートが下振れし、翌月以降が鳴りやすくなる。
   → 販社ごとに horizon より後ろを**系列から切り落としてから**検定する。

3. **閾値・信号B（販社合算）** — 分母（累積販売台数）は社内システム由来で
   遅れないのに、分子（修理件数）だけが欠ける。累積使用率は**過小**に出る。
   誤報方向ではなく**見逃し方向**なので危険度は低いが、機種ごとに販社構成が
   違うため信号Bの機種間比較は不公平に歪む。
   → 信号Bは全販社が揃っている月（global horizon）までで打ち切る。

horizon の決め方
----------------
外部の受領管理表があるならそれが正解。無くてもパネルだけから決められる:
監視単位が3086件もあるので、ある販社がその月に1件も修理を報告しないことは
実質起こらない。したがって

    その販社の行が存在する最大の年月 = 受領済みの最終月

ただし**実行日が月の途中**なら最終月は部分的にしか入っていない。これを
落とすのが `margin_months`（既定1）。月末締めの後に必ず実行する運用なら
`margin_overrides={"A": 0}` のように販社単位で0にできる。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================================
# 年月ユーティリティ
# ============================================================================
def to_ym(x) -> int:
    s = "".join(ch for ch in str(x) if ch.isdigit())
    return int(s[:6]) if len(s) >= 6 else int(s)


def shift_ym(ym: int, k: int) -> int:
    y, m = divmod(int(ym), 100)
    idx = y * 12 + (m - 1) + k
    return (idx // 12) * 100 + (idx % 12) + 1


def diff_ym(a: int, b: int) -> int:
    """a - b を月数で返す。"""
    ya, ma = divmod(int(a), 100)
    yb, mb = divmod(int(b), 100)
    return (ya * 12 + ma) - (yb * 12 + mb)


# ============================================================================
# 集計
# ============================================================================
def _dist_month_stats(panel: pd.DataFrame, cols: dict,
                      all_token: str = "ALL") -> pd.DataFrame:
    """販社 × 年月 の 行数 と 使用数計。ALL行は除外する。

    「その月がどれだけ届いているか」の代理指標。月の途中で抽出した販社は
    行数も使用数も薄くなる。使用数の方が経過日数に素直に比例するので主指標に使う。
    """
    use_col = cols.get("monthly_use")
    keep = [cols["dist"], cols["ym"]] + ([use_col] if use_col in panel.columns else [])
    d = panel[keep].copy()
    d.columns = ["dist", "ym"] + (["use"] if len(keep) == 3 else [])
    d["dist"] = d["dist"].astype(str)
    d = d[d["dist"] != str(all_token)]
    d["ym"] = d["ym"].map(to_ym).astype("int64")
    if "use" not in d.columns:
        d["use"] = 1.0
    d["use"] = pd.to_numeric(d["use"], errors="coerce").fillna(0.0)
    g = d.groupby(["dist", "ym"], as_index=False).agg(
        行数=("use", "size"), 使用数計=("use", "sum"))
    return g


def lag_matrix(panel: pd.DataFrame, cols: dict, months: int = 15,
               all_token: str = "ALL", value: str = "使用数計") -> pd.DataFrame:
    """販社 × 直近Nヶ月 のマトリクス。目視用。

    右端が急に細る／途中で切れる販社が、遅れている販社。
    value に "行数" を渡せば行数で見られる。
    """
    cnt = _dist_month_stats(panel, cols, all_token)
    if cnt.empty:
        return cnt
    last = int(cnt["ym"].max())
    lo = shift_ym(last, -(months - 1))
    cnt = cnt[cnt["ym"] >= lo]
    return (cnt.pivot(index="dist", columns="ym", values=value)
               .fillna(0).round(0).astype(int).sort_index())


def _tail_ratio(g: pd.DataFrame, ref_months: int) -> float:
    """最終月の厚みが、その前 ref_months ヶ月の中央値の何倍か。"""
    ref = g["使用数計"].iloc[max(0, len(g) - 1 - ref_months): len(g) - 1]
    if not len(ref):
        return np.nan
    med = float(np.median(ref))
    if med <= 0:
        return np.nan
    return float(g["使用数計"].iloc[-1]) / med


# ============================================================================
# horizon の推定
# ============================================================================
def estimate_horizon(panel: pd.DataFrame, cols: dict,
                     all_token: str = "ALL",
                     margin_months: int = 0,
                     margin_overrides: dict | None = None,
                     fixed: dict | None = None,
                     auto_margin: bool = True,
                     thin_ratio: float = 0.7,
                     ref_months: int = 6) -> dict:
    """販社 -> 完全と見なせる最終年月(YYYYMM) の dict を返す。

    margin_months     : 全販社共通で末尾から落とす月数（既定0）
    auto_margin       : True なら、最終月の厚みが直前 ref_months の中央値の
                        thin_ratio 未満のとき margin を +1 する。
                        月次一括送付の販社は最終月が**完結している**ので margin 0、
                        毎日/週次送付の販社は月の途中で切れて薄いので margin 1、
                        が自動的に付く。
    margin_overrides  : {販社: 月数} で個別上書き（受領実態が分かっているとき）
    fixed             : {販社: YYYYMM} で完全に固定（受領管理表があるならこれが正解）
    thin_ratio        : 薄いと見なす閾値。0.7 なら「平常の7割未満なら未確定」

    保守側に外す（margin を多く取る）のは安全。落とした月は次回の再評価窓で
    拾い直されるため、遅れるだけで消えることはない。逆に未確定月を入れて
    しまうとベースラインが下振れして誤報になるので、迷ったら margin は多めに。
    """
    margin_overrides = margin_overrides or {}
    fixed = fixed or {}
    cnt = _dist_month_stats(panel, cols, all_token)
    out = {}
    for dist, g in cnt.groupby("dist"):
        if dist in fixed:
            out[dist] = int(to_ym(fixed[dist]))
            continue
        g = g.sort_values("ym")
        last = int(g["ym"].iloc[-1])
        if dist in margin_overrides:
            mg = int(margin_overrides[dist])
        else:
            mg = int(margin_months)
            if auto_margin:
                r = _tail_ratio(g, ref_months)
                if not np.isnan(r) and r < thin_ratio:
                    mg += 1
        out[dist] = shift_ym(last, -mg)
    return out


def global_horizon(horizon: dict) -> int | None:
    """全販社が揃っている最終年月。閾値・信号Bの打ち切りに使う。"""
    return min(horizon.values()) if horizon else None


# ============================================================================
# 診断レポート
# ============================================================================
def report(panel: pd.DataFrame, cols: dict, all_token: str = "ALL",
           margin_months: int = 0, margin_overrides: dict | None = None,
           fixed: dict | None = None, auto_margin: bool = True,
           thin_ratio: float = 0.7, ref_months: int = 6,
           revisit_months: int | None = None) -> pd.DataFrame:
    """販社別の受領状況をまとめる。実データ投入前にまずこれを見る。

    列
    --
    最終年月   : その販社の行がある最大の年月
    末尾使用数 : 最終年月の使用数計
    参照中央値 : その前 ref_months ヶ月の使用数計の中央値
    痩せ比     : 末尾使用数 / 参照中央値。thin_ratio 未満なら最終月は未確定と判断
    採用margin : 末尾から落とす月数
    horizon    : 完全と見なす最終年月
    遅れ月数   : パネル最新月 - horizon。**revisit_months はこの最大値以上必要**
    判定       : OK / 要確認 / 再評価窓不足
    """
    cnt = _dist_month_stats(panel, cols, all_token)
    if cnt.empty:
        return cnt
    T = int(cnt["ym"].max())
    hz = estimate_horizon(panel, cols, all_token, margin_months,
                          margin_overrides, fixed, auto_margin,
                          thin_ratio, ref_months)
    rows = []
    for dist, g in cnt.groupby("dist"):
        g = g.sort_values("ym")
        last = int(g["ym"].iloc[-1])
        ref = g["使用数計"].iloc[max(0, len(g) - 1 - ref_months): len(g) - 1]
        med = float(np.median(ref)) if len(ref) else np.nan
        lag = diff_ym(T, hz[dist])
        if revisit_months is not None and lag > revisit_months:
            judge = "再評価窓不足"
        elif lag >= 2:
            judge = "要確認"
        else:
            judge = "OK"
        rows.append(dict(販社=dist, 最終年月=last,
                         末尾使用数=round(float(g["使用数計"].iloc[-1]), 1),
                         参照中央値=round(med, 1) if med == med else np.nan,
                         痩せ比=round(_tail_ratio(g, ref_months), 2),
                         採用margin=diff_ym(last, hz[dist]),
                         horizon=hz[dist], 遅れ月数=lag, 判定=judge))
    r = pd.DataFrame(rows).sort_values("遅れ月数", ascending=False)
    return r.reset_index(drop=True)


def check_revisit(horizon: dict, T: int, revisit_months: int) -> list:
    """再評価窓に収まらない販社（＝月が永久に未検定になる販社）を返す。"""
    return sorted([d for d, h in horizon.items()
                   if diff_ym(int(T), int(h)) > int(revisit_months)])


def missing_note(panel: pd.DataFrame, cols: dict, horizon: dict, T: int,
                 all_token: str = "ALL") -> pd.DataFrame:
    """機種×部番ごとに「分子が欠けている販社」を出す。

    閾値検出器の累積使用率は分母だけ完全なので**過小**に出る。
    レビュー時に「この率は D社が3ヶ月分欠けた状態の数字」と分かるようにする。
    """
    d = panel[[cols["biz"], cols["dev"], cols["part"], cols["dist"]]].copy()
    d.columns = ["biz", "dev", "part", "dist"]
    d["dist"] = d["dist"].astype(str)
    d = d[d["dist"] != str(all_token)].drop_duplicates()
    lag = {k: diff_ym(int(T), int(v)) for k, v in horizon.items()}
    d["欠測月数"] = d["dist"].map(lag).fillna(0).astype(int)
    d = d[d["欠測月数"] > 0]
    if d.empty:
        return pd.DataFrame(columns=["biz", "dev", "part", "欠測販社", "最大欠測月数"])
    g = d.groupby(["biz", "dev", "part"], as_index=False).agg(
        欠測販社=("dist", lambda s: "/".join(sorted(set(s)))),
        最大欠測月数=("欠測月数", "max"))
    return g.sort_values("最大欠測月数", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    # 自己テスト用の小さな合成パネル（販社Bが2ヶ月遅れ）
    cols = dict(biz="事業コード", dev="開発コード", part="部番", dist="販社",
                ym="年月", monthly_use="月次使用数", cum_sales="累積販売台数")
    rows = []
    for dist, last in (("A", 202503), ("B", 202501), ("C", 202502)):
        m = 202401
        while m <= last:
            rows.append({cols["biz"]: "E1", cols["dev"]: "M01", cols["part"]: "P1",
                         cols["dist"]: dist, cols["ym"]: m,
                         cols["monthly_use"]: 5, cols["cum_sales"]: 1000})
            m = shift_ym(m, 1)
    panel = pd.DataFrame(rows)
    print("=== 行数マトリクス ===")
    print(lag_matrix(panel, cols, months=6, value="行数").to_string())
    print("\n=== 受領状況 ===")
    print(report(panel, cols, revisit_months=3).to_string(index=False))
    hz = estimate_horizon(panel, cols)
    print("\nhorizon:", hz, " global:", global_horizon(hz))
    print("窓不足:", check_revisit(hz, 202503, 1))
