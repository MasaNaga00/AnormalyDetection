# -*- coding: utf-8 -*-
"""
panel_maturity.py — 集計パネルだけから「受付月の熟成度」を測る

前提
----
手元にあるのは受付月で集計済みのパネルだけ（修理レコードには当面アクセス不可）。
レコードがデータに現れるのは修理完了後なので、受付月Mの件数は抽出時点では
まだ出そろっていない。時間が経つと後から積み上がる。

つまり **同じ受付月の値が、抽出のたびに増える**。
これを測る方法は2つある。

  方法1: 熟成カーブの推定（いま使える・近似）
      直近数ヶ月のレートが、成熟した過去月のレートに対して何割まで
      積み上がっているかを見る。ρ(k) が目標（95%など）に達する k で切る。
      → maturity_curve() / recommend_horizon()

  方法2: スナップショット差分（正解・ただし数ヶ月かかる）
      毎月のパネルを抽出年月つきで保存し、同じ受付月の値が
      抽出のたびにどう増えたかを直接測る。保険の発展表そのもの。
      **今日から保存を始めれば数ヶ月後に方法1を較正できる。コストはゼロ。**
      → save_snapshot() / build_triangle()

方法1の限界（承知して使うこと）
-------------------------------
「本当に修理が減った」と「まだ届いていない」を原理的に区別できない。
販社の全単位をプールして見るので個別の増減は均されるが、フリート全体の
トレンドや季節性は交絡する。seasonal=True で前年同月を基準にすると
季節性は落とせる（履歴が2年以上あるとき）。
**数値を鵜呑みにせず、必ずカーブの形を目視すること。**
ρ(k) が k とともに単調に1へ近づいていれば打ち切り。
ガタガタなら推定が効いていない。
"""
from __future__ import annotations

import os
import glob
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
# 共通: 販社×受付月の集計レート
# ============================================================================
def dist_monthly(panel: pd.DataFrame, cols: dict,
                 all_token: str = "ALL") -> pd.DataFrame:
    """販社 × 受付月 の 使用数計 / 台数計 / レート。

    台数計は部番をまたいで重複計上されるが、**月ごとの比を見るだけ**なので
    定数倍として相殺される。ここでは絶対水準を使わない。
    """
    d = panel[[cols["dist"], cols["ym"], cols["monthly_use"],
               cols["cum_sales"]]].copy()
    d.columns = ["dist", "ym", "use", "fleet"]
    d["dist"] = d["dist"].astype(str)
    d = d[d["dist"] != str(all_token)]
    d["ym"] = d["ym"].map(to_ym).astype("int64")
    for c in ("use", "fleet"):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    g = d.groupby(["dist", "ym"], as_index=False).agg(
        使用数計=("use", "sum"), 台数計=("fleet", "sum"))
    g["rate"] = np.where(g["台数計"] > 0, g["使用数計"] / g["台数計"], np.nan)
    return g.sort_values(["dist", "ym"]).reset_index(drop=True)


# ============================================================================
# 方法1: 熟成カーブの推定（単一パネルから）
# ============================================================================
def maturity_curve(panel: pd.DataFrame, cols: dict, max_lag: int = 8,
                   ref_lo: int = 9, ref_hi: int = 30, seasonal: bool = True,
                   all_token: str = "ALL") -> pd.DataFrame:
    """販社ごとに ρ(k)=「最新月からk ヶ月前の熟成度」を返す。

    ref_lo / ref_hi : 基準に使う成熟月の範囲（最新月から何ヶ月前〜何ヶ月前か）。
                      既定は 9〜30ヶ月前。max_lag より十分外側にすること。
    seasonal        : True かつ履歴が足りるなら前年同月を基準にする。
                      季節性のある品目で ρ が歪むのを防ぐ。

    返り値は 行=販社, 列=k0..k{max_lag} の ρ。1.0 に近いほど出そろっている。
    """
    g = dist_monthly(panel, cols, all_token)
    T = int(g["ym"].max())
    rows = []
    for dist, s in g.groupby("dist"):
        s = s.set_index("ym")["rate"]
        # 基準: 成熟月の中央値（seasonal なら前年同月の中央値）
        mature = [shift_ym(T, -k) for k in range(ref_lo, ref_hi + 1)]
        mature = [m for m in mature if m in s.index and s[m] == s[m]]
        if len(mature) < 6:
            rows.append(dict(販社=dist, 有効=False))
            continue
        base_med = float(np.median([s[m] for m in mature]))
        r = dict(販社=dist, 有効=True, 基準レート=round(base_med, 6),
                 参照月数=len(mature))
        for k in range(0, max_lag + 1):
            m = shift_ym(T, -k)
            if m not in s.index or s[m] != s[m]:
                r[f"k{k}"] = np.nan
                continue
            ref = base_med
            if seasonal:
                prev = [shift_ym(m, -12 * j) for j in (1, 2, 3)]
                prev = [p for p in prev if p in s.index and s[p] == s[p]
                        and diff_ym(T, p) >= ref_lo]
                if len(prev) >= 1:
                    ref = float(np.median([s[p] for p in prev]))
            r[f"k{k}"] = round(float(s[m]) / ref, 3) if ref > 0 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def recommend_horizon(panel: pd.DataFrame, cols: dict, target: float = 0.95,
                      max_lag: int = 8, all_token: str = "ALL",
                      safety_margin: int = 1, **kw) -> tuple:
    """(受付月horizon dict, 推奨 C_REVISIT_MONTHS) を返す。

    判定規則（ノイズ耐性つき）
    --------------------------
    ρ(k) は「k ヶ月前の受付月が何割出そろっているか」だが、**成熟した領域では
    単に月ごとの実変動を測っているだけ**になる。ある月に本当に多く修理が
    入っていれば ρ=2、静かな月なら ρ=0.5 になる。これは打ち切りではない。

    そこで「ρ(k) 以降がすべて target 以上」ではなく
    **「ρ(k) 以降の中央値が target 以上」** を採る。1点の落ち込みで
    推奨値が max_lag に跳ね上がるのを防ぐ。

    safety_margin : そのうえで k をさらに何ヶ月伸ばすか（既定1）。
        浅く見積もる誤り（未確定月を判定に入れる＝誤報方向）は危険、
        深く見積もる誤りは再評価窓で拾い直されるだけ。非対称なので1ヶ月積む。
    """
    cur = maturity_curve(panel, cols, max_lag, all_token=all_token, **kw)
    g = dist_monthly(panel, cols, all_token)
    T = int(g["ym"].max())
    hz, lags = {}, []
    for r in cur.itertuples():
        if not getattr(r, "有効", False):
            k = max_lag
        else:
            vals = [getattr(r, f"k{i}") for i in range(0, max_lag + 1)]
            k = max_lag
            for i in range(0, max_lag + 1):
                tail = [v for v in vals[i:] if v is not None and v == v]
                if not tail:
                    continue
                if float(np.median(tail)) >= target and vals[i] == vals[i] \
                        and vals[i] >= target * 0.9:
                    k = i
                    break
        k = min(int(k) + int(safety_margin), max_lag)
        hz[r.販社] = shift_ym(T, -k)
        lags.append(k)
    return hz, int(max(lags)) if lags else 0


def suggest_settings(panel: pd.DataFrame, cols: dict, target: float = 0.95,
                     max_lag: int = 8, all_token: str = "ALL",
                     safety_margin: int = 1, **kw) -> dict:
    """settings.py にそのまま貼れる形で推奨値を出す。

    2つの書き方があるので両方出す。

    HORIZON_MARGIN_OVERRIDES（推奨）
        {販社: 月数}。**その販社自身の最終月から何ヶ月落とすか**の相対指定。
        毎月の更新が要らない。基本はこちらを使う。

    HORIZON_FIXED（一時的な固定用）
        {販社: YYYYMM}。**絶対値**なので、翌月には古くなる。
        毎月書き換える前提でないと使えない。受領管理表の実測値を
        一時的に入れるときや、特定月だけ手当てするときに使う。
    """
    hz, K = recommend_horizon(panel, cols, target, max_lag, all_token,
                              safety_margin, **kw)
    g = dist_monthly(panel, cols, all_token)
    T = int(g["ym"].max())
    last = g.groupby("dist")["ym"].max().astype(int).to_dict()

    ov = {}
    for dist, h in hz.items():
        # margin は「その販社の最終月」からの相対。負なら0でよい。
        ov[dist] = max(0, diff_ym(int(last[dist]), int(h)))

    print("=" * 66)
    print(f"settings.py に貼る（最新月 T={T} / 完成度目標 {target:.0%}）")
    print("=" * 66)
    print(f"C_REVISIT_MONTHS = {K}")
    print("HORIZON_MARGIN_OVERRIDES = {")
    for d in sorted(ov):
        print(f'    "{d}": {ov[d]},'
              f'   # 最終月{last[d]} → horizon {hz[d]}'
              f'（Tから{diff_ym(T, hz[d])}ヶ月前）')
    print("}")
    print("\n# 絶対値で固定したい場合のみ（毎月書き換えが必要）")
    print("HORIZON_FIXED = {")
    for d in sorted(hz):
        print(f'    "{d}": {hz[d]},')
    print("}")
    print("=" * 66)
    print("※ 基本は HORIZON_MARGIN_OVERRIDES を使う。HORIZON_FIXED は絶対値なので")
    print("   翌月には古くなり、そのまま放置すると horizon が進まなくなる。")
    return dict(C_REVISIT_MONTHS=K, HORIZON_MARGIN_OVERRIDES=ov,
                HORIZON_FIXED=hz, T=T)


def explain(panel: pd.DataFrame, cols: dict, target: float = 0.95,
            max_lag: int = 8, all_token: str = "ALL",
            safety_margin: int = 1, **kw) -> pd.DataFrame:
    """カーブと推奨値をまとめて表示する。実データでまずこれを見る。"""
    cur = maturity_curve(panel, cols, max_lag, all_token=all_token, **kw)
    hz, K = recommend_horizon(panel, cols, target, max_lag, all_token,
                              safety_margin, **kw)
    cur["horizon"] = cur["販社"].map(hz)
    g = dist_monthly(panel, cols, all_token)
    T = int(g["ym"].max())
    cur["遅れ月数"] = [diff_ym(T, v) for v in cur["horizon"]]
    # 成熟領域（後半）のばらつき＝カーブの読みやすさ
    tail_cols = [f"k{i}" for i in range(max(1, max_lag - 3), max_lag + 1)]
    tail_cols = [c for c in tail_cols if c in cur.columns]
    if tail_cols:
        cur["成熟域中央"] = cur[tail_cols].median(axis=1).round(2)
        cur["成熟域の幅"] = (cur[tail_cols].max(axis=1)
                        - cur[tail_cols].min(axis=1)).round(2)
    print(f"[推奨] C_REVISIT_MONTHS = {K}   目標完成度 {target:.0%}   最新月 {T}")
    print("[読み方] 見るのは**左端（k=0,1,2）が低くて右へ上がるか**だけ。")
    print("         成熟域(右側)の 0.5 や 2.0 は打ち切りではなく月ごとの実変動。")
    print("         『成熟域の幅』が 0.5 を超える販社はカーブが読めていないので、")
    print("         その販社は保守的に max_lag で切るか実態を確認すること。")
    return cur


# ============================================================================
# 方法2: スナップショット差分（正解。今日から貯める）
# ============================================================================
def save_snapshot(panel: pd.DataFrame, cols: dict, extract_ym: int,
                  outdir: str = "スナップショット",
                  all_token: str = "ALL",
                  archive: bool = True) -> str:
    """抽出時点つきで販社×受付月の集計だけを保存する（軽量）。

    生パネルを丸ごと保存する必要はない。あとで発展表を組むのに要るのは
    販社×受付月の使用数計だけ。1回あたり数KB。

    同月に複数回実行した場合
    ------------------------
    ファイル名は受付月ベースの `snap_YYYYMM.csv` なので **上書き**される。
    これは意図的:

      発展表の遅れ軸は「月」単位なので、同じ月の複数時点を両方残すと、
      違う熟成度の観測が同じ遅れ k に混ざって ρ(k) が歪む。
      最後の1本（＝最も出そろっている）だけを残すのが正しい。

    ただし上書きで失われると困る場合に備えて、既存ファイルは
    `outdir/履歴/` に日時つきで退避する（archive=True）。
    履歴はサブフォルダなので build_triangle の glob には拾われない。

    あわせて `snapshot_log.csv` に1行追記する（追記のみ・上書きしない）。
    いつ・どのパネルで保存したかの監査証跡になる。
    """
    import datetime as _dt
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"snap_{int(extract_ym)}.csv")
    now = _dt.datetime.now()

    # --- 既存があれば履歴へ退避 ---
    replaced = None
    if archive and os.path.exists(path):
        hist = os.path.join(outdir, "履歴")
        os.makedirs(hist, exist_ok=True)
        old_ts = _dt.datetime.fromtimestamp(os.path.getmtime(path))
        dst = os.path.join(hist, f"snap_{int(extract_ym)}_{old_ts:%Y%m%d_%H%M%S}.csv")
        try:
            os.replace(path, dst)
            replaced = dst
        except OSError:
            pass

    g = dist_monthly(panel, cols, all_token)
    g.insert(0, "抽出年月", int(extract_ym))
    g.insert(1, "保存日時", now.strftime("%Y-%m-%d %H:%M:%S"))
    g.to_csv(path, index=False, encoding="utf-8-sig")

    # --- 監査ログ（追記のみ）---
    log = os.path.join(outdir, "snapshot_log.csv")
    cov = (g.groupby("dist")["ym"].max().astype(int).to_dict()
           if len(g) else {})
    rec = pd.DataFrame([dict(
        保存日時=now.strftime("%Y-%m-%d %H:%M:%S"),
        抽出年月=int(extract_ym),
        販社数=int(g["dist"].nunique()) if len(g) else 0,
        行数=int(len(g)),
        使用数計=float(g["使用数計"].sum()) if len(g) else 0.0,
        販社別最終月=str(cov),
        上書き退避先=(os.path.basename(replaced) if replaced else ""))])
    rec.to_csv(log, mode="a", index=False, header=not os.path.exists(log),
               encoding="utf-8-sig")
    return path


def build_triangle(outdir: str = "スナップショット") -> pd.DataFrame:
    """保存済みスナップショットから 受付月 × 遅れ月数 の発展表を組む。

    値は「抽出時点で見えていた使用数計」。同じ受付月の行が抽出のたびに
    どう増えるかが、そのまま熟成カーブになる。
    """
    files = sorted(glob.glob(os.path.join(outdir, "snap_*.csv")))
    if len(files) < 2:
        raise ValueError(f"スナップショットが {len(files)} 件しかない。2件以上必要。")
    df = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files],
                   ignore_index=True)
    # 念のための重複ガード（同じ抽出年月のファイルが何らかの理由で
    # 複数混ざった場合、同じ遅れ k に違う熟成度が混ざって ρ(k) が歪む）
    key = ["抽出年月", "dist", "ym"]
    n0 = len(df)
    if "保存日時" in df.columns:
        df = df.sort_values("保存日時")
    df = df.drop_duplicates(subset=key, keep="last")
    if len(df) < n0:
        print(f"[注意] 重複を {n0 - len(df)} 行落としました"
              "（同じ抽出年月の観測が複数ありました。最後のものを採用）")
    df["遅れ"] = [diff_ym(int(e), int(m))
                for e, m in zip(df["抽出年月"], df["ym"])]
    df = df[df["遅れ"] >= 0]
    return df.rename(columns={"ym": "受付月"})


def triangle_completeness(tri: pd.DataFrame, by_dist: bool = True) -> pd.DataFrame:
    """発展表から ρ(k)=（遅れk時点の値 / 最終的に見えている値）を出す。

    最も熟成したスナップショットの値を暫定の「最終値」とみなす。
    スナップショットの本数が増えるほど精度が上がる。
    """
    keys = (["dist"] if by_dist else []) + ["受付月"]
    fin = (tri.sort_values("遅れ").groupby(keys, as_index=False)
              .agg(最終値=("使用数計", "last"), 最大遅れ=("遅れ", "max")))
    m = tri.merge(fin, on=keys, how="left")
    m = m[m["最終値"] > 0]
    m["ρ"] = m["使用数計"] / m["最終値"]
    gk = (["dist"] if by_dist else []) + ["遅れ"]
    out = m.groupby(gk, as_index=False).agg(
        ρ中央=("ρ", "median"), ρ平均=("ρ", "mean"), 本数=("ρ", "size"))
    return out.sort_values(gk).reset_index(drop=True)


# ============================================================================
if __name__ == "__main__":
    # 既知の熟成カーブを仕込んで復元できるかの自己テスト
    rng = np.random.default_rng(11)
    C = dict(dist="販社", ym="年月", monthly_use="月次使用数",
             cum_sales="累積販売台数")
    true_f = {"A": [0.40, 0.78, 0.93, 0.98, 1.0],      # 完了が早い
              "B": [0.15, 0.45, 0.72, 0.90, 0.97, 1.0]}  # 完了が遅い
    T = 202503
    rows = []
    for dist, f in true_f.items():
        m = 202101
        while m <= T:
            k = diff_ym(T, m)
            comp = f[k] if k < len(f) else 1.0
            # 季節性: 夏に1.2倍
            seas = 1.2 if (m % 100) in (7, 8) else 1.0
            for u in range(30):   # 30単位ぶん
                base = rng.poisson(8 * seas)
                rows.append({C["dist"]: dist, C["ym"]: m,
                             C["monthly_use"]: round(base * comp),
                             C["cum_sales"]: 1000})
            m = shift_ym(m, 1)
    panel = pd.DataFrame(rows)

    print("=== 熟成カーブ ρ(k)（単一パネルからの推定）===")
    print(explain(panel, C, target=0.95, max_lag=6).to_string(index=False))
    print("\n真値 A: 0.40/0.78/0.93/0.98/1.0 → 95%到達は k=3")
    print("真値 B: 0.15/0.45/0.72/0.90/0.97/1.0 → 95%到達は k=4")
