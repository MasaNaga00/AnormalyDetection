# -*- coding: utf-8 -*-
"""
generate_test_data.py — 一連の流れ確認用テストデータ生成

作るもの
--------
1. test_panel.csv      : 縦長パネル（機種×部番×販社×年月）。SF-100/ランクB、1グループのみ。
2. test_labels.csv     : 過去の確定異常ラベル（バックテストの検出力測定・遅れ測定用）。
3. 台帳_初回_空.xlsx    : 台帳（まだ何も記録していない状態）。
4. 台帳_記入後.xlsx     : 1回目の run() で拾った要確認を人が判断・記録した後の台帳。

登場する機種（すべて 事業コード=E1, 部番=P1, SF-コード=SF-100, ランク=B）
----------------------------------------------------------------------
[安定期]（24ヶ月, 202301〜202412, 経過月0〜23）
  S01, S06, S07 : 素の正常（ベースライン/先行機種の頭数を稼ぐ）
  S02           : 経過月15から出庫レート2.5倍のドリフト。以後fixせず継続＝現在進行中で未対応。
  S03           : 経過月23（最終月＝asof月）に単月スパイク。現在進行中で未対応。
  S04           : 経過月8〜13だけレート3倍（対策済みの過去インシデント）。以後は正常に復帰。
                  → バックテスト用ラベルに搭載（報告は経過月12＝onsetの4ヶ月後、早期検知確認用）。
  S05           : 経過月10に単月スパイク（対策済みの過去インシデント、即日報告）。
                  → バックテスト用ラベルに搭載（報告は経過月10＝ズレ無し）。

[安定化前]（8ヶ月, 202405〜202412, 経過月0〜7）
  N01           : 素の正常な新機種。
  N02           : 立ち上がりからレート2倍の初期不良。現在進行中で未対応。

台帳の設計
----------
- 台帳_初回_空.xlsx  : 空（0行）。1回目の run() は「現在検出されているものを拾う」デモに使う。
- 台帳_記入後.xlsx   : S02(対策中)・S03(スパイク確認)・N02(対策中) の3件を、
                       1回目の run() で拾った月に記録した想定。2回目の run() で
                       リセット・状態遷移がかかることを確認するのに使う。
  S04・S05 は台帳には入れない（データ生成上すでに fix 済みで再点灯しないため、
  台帳運用ではなく「バックテストの既知ラベル」としてのみ使う）。
"""
import numpy as np
import pandas as pd
import state_logic_cusum as s

SEED = 42
rng = np.random.default_rng(SEED)

BIZ, PART, SF, RANK = "E1", "P1", "SF-100", "B"
STABLE_MONTHS = 24
EARLY_MONTHS = 8
STABLE_START_YM = 202301
EARLY_START_YM = 202405
DISTS = [("D0", 0.6), ("D1", 0.4)]


def _series_stable(dev, base_lam, anomaly=None):
    """anomaly: None、または (onset_elapsed, end_elapsed_or_None, factor)。
    end が None なら onset 以降ずっと継続（未対応の進行中ドリフト）。"""
    rows = []
    for t in range(STABLE_MONTHS):
        ym = s._add_months(STABLE_START_YM, t)
        F = 1500 + 55 * t
        lam = base_lam
        if anomaly is not None:
            onset, end, factor = anomaly
            if t >= onset and (end is None or t < end):
                lam = base_lam * factor
        for dist, frac in DISTS:
            Fd = int(round(F * frac))
            cnt = int(rng.poisson(max(lam * Fd, 0.0)))
            rows.append(dict(事業コード=BIZ, 開発コード=dev, 部番=PART, 販社=dist,
                             **{"SF-コード": SF, "ランク": RANK},
                             年月=ym, 経過月=t, 月次使用数=cnt, 累積販売台数=Fd))
    return rows


def _series_stable_spike(dev, base_lam, spike_elapsed, mult):
    rows = _series_stable(dev, base_lam, anomaly=None)
    df = pd.DataFrame(rows)
    F_total = df[df["経過月"] == spike_elapsed]["累積販売台数"].sum()
    add = int(rng.poisson(max(base_lam * F_total * mult, 1.0)))
    # 主販社(D0)側に追加カウントを乗せる
    m = (df["開発コード"] == dev) & (df["経過月"] == spike_elapsed) & (df["販社"] == "D0")
    df.loc[m, "月次使用数"] = df.loc[m, "月次使用数"] + add
    return df.to_dict("records")


def _series_early(dev, base_lam, factor=1.0):
    rows = []
    for t in range(EARLY_MONTHS):
        ym = s._add_months(EARLY_START_YM, t)
        F = 300 + 90 * t
        lam = base_lam * (2.5 * np.exp(-t / 2.5) + 1.0) * 0.001  # 初期高→減衰の共通カーブ形状
        lam = lam * factor
        for dist, frac in DISTS:
            Fd = int(round(F * frac))
            cnt = int(rng.poisson(max(lam * Fd, 0.0)))
            rows.append(dict(事業コード=BIZ, 開発コード=dev, 部番=PART, 販社=dist,
                             **{"SF-コード": SF, "ランク": RANK},
                             年月=ym, 経過月=t, 月次使用数=cnt, 累積販売台数=Fd))
    return rows


def build_panel():
    rows = []
    rows += _series_stable("S01", 0.0012)
    rows += _series_stable("S06", 0.0015)
    rows += _series_stable("S07", 0.0010)
    rows += _series_stable("S02", 0.0013, anomaly=(15, None, 2.5))   # 進行中・未対応
    rows += _series_stable_spike("S03", 0.0014, spike_elapsed=23, mult=6.0)  # 進行中・未対応
    rows += _series_stable("S04", 0.0012, anomaly=(8, 14, 3.0))      # 過去・対策済み
    rows += _series_stable_spike("S05", 0.0013, spike_elapsed=10, mult=5.0)  # 過去・対策済み
    rows += _series_early("N01", 1.0, factor=1.0)
    rows += _series_early("N02", 1.0, factor=2.0)                    # 進行中・未対応
    return pd.DataFrame(rows)


def build_labels(panel: pd.DataFrame):
    def ym_at(dev, elapsed):
        r = panel[(panel["開発コード"] == dev) & (panel["経過月"] == elapsed)]
        return int(r["年月"].iloc[0])
    rows = [
        dict(事業コード=BIZ, 開発コード="S04", 部番=PART,
             発生年月=ym_at("S04", 12), 備考="真の変化点は経過月8。販社報告は4ヶ月後"),
        dict(事業コード=BIZ, 開発コード="S05", 部番=PART,
             発生年月=ym_at("S05", 10), 備考="スパイク発生月と同月に報告"),
    ]
    return pd.DataFrame(rows)


def build_ledger_empty():
    cols = ["事業コード", "開発コード", "部番", "判定年月", "記録日", "処置区分",
            "再評価年月", "上書きR", "上書きh", "新ベースライン値",
            "ベースライン窓起点", "ベースライン窓長", "原因メモ", "確認者"]
    return pd.DataFrame(columns=cols)


def extend_panel(panel: pd.DataFrame, n_months: int = 3,
                 continuing_anomaly: dict | None = None) -> pd.DataFrame:
    """台帳記録後、翌月以降のデータが来た想定で panel を継ぎ足す。

    continuing_anomaly: {dev: factor} で、その機種の直近レートに対する倍率を指定。
      1.0=正常化（対策が効いた）、指定なしは直前と同じレート（対策未実施/効果不明）で継続。
    S03（過去の単月スパイク）は継ぎ足し月では正常に戻す前提。
    """
    continuing_anomaly = continuing_anomaly or {}
    out = [panel]
    for dev, g in panel.groupby("開発コード"):
        g = g.sort_values("年月")
        last = g[g["販社"] == "D0"].iloc[-1]  # 経過月・累積台数の基準
        is_early = dev.startswith("N")
        # 直近の「地のレート」を D0+D1 合算の使用数/台数から逆算（スパイク月は除く）
        hist = g.groupby("年月").agg(使用数=("月次使用数", "sum"), 台数=("累積販売台数", "sum"))
        hist_use = hist["使用数"].to_numpy(dtype=float)
        hist_fleet = hist["台数"].to_numpy(dtype=float)
        # 直近5ヶ月（スパイクの影響を避けるため中央値ベース）でレートを推定
        recent_rate = np.median((hist_use[-5:] / np.maximum(hist_fleet[-5:], 1.0)))
        factor = continuing_anomaly.get(dev, 1.0)
        lam = recent_rate * factor

        last_elapsed = int(last["経過月"])
        last_ym = int(last["年月"])
        last_F = {d: int(g[(g["年月"] == last_ym) & (g["販社"] == d)]["累積販売台数"].iloc[0])
                  for d, _ in DISTS}
        for i in range(1, n_months + 1):
            ym = s._add_months(last_ym, i)
            elapsed = last_elapsed + i
            for dist, frac in DISTS:
                growth = (55 if not is_early else 90)
                Fd = last_F[dist] + int(round(growth * frac * i))
                cnt = int(rng.poisson(max(lam * Fd, 0.0)))
                out.append(pd.DataFrame([dict(
                    事業コード=BIZ, 開発コード=dev, 部番=PART, 販社=dist,
                    **{"SF-コード": SF, "ランク": RANK},
                    年月=ym, 経過月=elapsed, 月次使用数=cnt, 累積販売台数=Fd)]))
    return pd.concat(out, ignore_index=True).sort_values(
        ["開発コード", "販社", "年月"]).reset_index(drop=True)



    panel = build_panel()
    labels = build_labels(panel)
    ledger_empty = build_ledger_empty()

    panel.to_csv("test_panel.csv", index=False, encoding="utf-8-sig")
    labels.drop(columns=["備考"]).to_csv("test_labels.csv", index=False, encoding="utf-8-sig")
    labels.to_csv("test_labels_備考つき.csv", index=False, encoding="utf-8-sig")  # 内容確認用
    with pd.ExcelWriter("台帳_初回_空.xlsx", engine="openpyxl") as w:
        ledger_empty.to_excel(w, sheet_name="台帳", index=False)

    print("パネル行数:", len(panel), " 機種数:", panel["開発コード"].nunique())
    print("ラベル件数:", len(labels))
    print(labels.drop(columns=["備考"]).to_string(index=False))
