# -*- coding: utf-8 -*-
"""
simulate_historical_load.py
============================
過去データに対して「毎月チェックして、拾われたものはその場で台帳に記録し
再着火しないようにした」と仮定した場合、実際には毎月何件チェックが必要
だったかを、現在のCONFIG（R, h, alpha_spike, min_count, burst_window等）で
再現するシミュレーター。

バックテスト(backtest_cusum.py)との違い
----------------------------------------
- backtest_cusum: 正常データに合成注入して「理論上の」誤報率・検出力を測る。
- こちら: 実際の過去パネルをそのまま月送りで再生し、「本当にその設定で
  毎月何件浮かび上がったか」を実測する。パラメータの最終健全性チェックや、
  運用開始前の「初月に一気に積み残しが出るのでは」という不安の解消に使う。

使い方1
------
    import state_logic_cusum as s
    from simulate_historical_load import simulate_monthly_load

    cfg = dict(s.CONFIG)
    cfg.update(R=..., h=..., alpha_spike=..., min_count=..., burst_window=...)

    raw = pd.read_csv("実データpanel_補正済み.csv")
    panel = s._prepare_panel(raw, cfg)
    units = s.aggregate_units(panel)

    monthly, details, ledger = simulate_monthly_load(
        units, cfg, assumed_disposition="ノイズ")   # まずはニュートラルに消し込む想定
    print(monthly)                                   # 年月ごとの件数
    monthly.to_csv("月次件数_過去再生.csv", index=False, encoding="utf-8-sig")
    details.to_csv("月次件数_詳細_過去再生.csv", index=False, encoding="utf-8-sig")

    import pandas as pd
    import state_logic_cusum as s
    from simulate_historical_load import simulate_monthly_load, compare_dispositions

使い方2
------
    cfg = dict(s.CONFIG)
    cfg.update(R=..., h=..., alpha_spike=..., min_count=..., burst_window=...)  # 決めた操作点

    raw = pd.read_csv("実データpanel_補正済み.csv")
    panel = s._prepare_panel(raw, cfg)
    units = s.aggregate_units(panel)

    # まず幅を見る（楽観/現実的な両方）
    cmp = compare_dispositions(units, cfg)
    cmp.to_csv("月次件数_比較.csv", index=False, encoding="utf-8-sig")

    # 本命の想定（実運用に近い処置区分を選ぶ）で詳細を見る
    monthly, details, ledger = simulate_monthly_load(units, cfg, assumed_disposition="対策中")
    monthly.to_csv("月次件数_過去再生.csv", index=False, encoding="utf-8-sig")
    details.to_csv("月次件数_詳細_過去再生.csv", index=False, encoding="utf-8-sig")

注意
----
- 「発火」で拾われた項目だけをその場で記録して消し込む。「再評価」「保留継続」は
  そのまま件数に数える（実際の運用でも毎月チェック対象になるはずのため）。
- assumed_disposition の選び方で結果が変わる:
    "ノイズ"   : reset=True, watch=False, hold=False。記録したらそれで終わり、
                以後の追加負荷なし（最も楽観的な件数になる）。
    "対策中"   : reset=True, watch=True。reeval_months ヶ月後に「再評価」として
                再度カウントされる（追跡フォローの負荷を含む、より現実的な件数）。
  実際の運用でどちらが多いかは処置区分の使い方次第なので、両方試して幅を見るのが安全。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import state_logic_cusum as s

LEDGER_COLS = ["事業コード", "開発コード", "部番", "判定年月", "記録日", "処置区分",
              "再評価年月", "上書きR", "上書きh", "新ベースライン値",
              "ベースライン窓起点", "ベースライン窓長", "原因メモ", "確認者"]


def simulate_monthly_load(units_df: pd.DataFrame, cfg: dict,
                          start_ym: int | None = None, end_ym: int | None = None,
                          assumed_disposition: str = "ノイズ",
                          reeval_months: int = 3,
                          verbose: bool = True):
    """過去データを月送りで再生し、毎月の要確認件数を実測する。

    Returns
    -------
    monthly : DataFrame [年月, 件数, 発火, 再評価, 保留継続, 対象単位数]
    details : DataFrame  … 全月分のインボックス行を積んだもの（内訳確認用）
    ledger  : DataFrame  … シミュレーション終了時点の（自動記録込みの）台帳
    """
    ledger = pd.DataFrame(columns=LEDGER_COLS)
    yms = sorted(units_df["ym"].unique())
    if start_ym is not None:
        yms = [y for y in yms if y >= start_ym]
    if end_ym is not None:
        yms = [y for y in yms if y <= end_ym]

    monthly_rows, detail_frames = [], []
    for ym in yms:
        table, meta = s.evaluate_units(units_df, ledger, cfg, asof=ym)
        if table.empty:
            monthly_rows.append(dict(年月=ym, 件数=0, 発火=0, 再評価=0, 保留継続=0,
                                     対象単位数=0))
            continue
        inbox = s.build_inbox(table, meta, cfg, asof=ym)
        n_total = len(inbox)
        n_fire = int((inbox["載る理由"] == "発火").sum()) if n_total else 0
        n_reeval = int((inbox["載る理由"] == "再評価").sum()) if n_total else 0
        n_hold = int((inbox["載る理由"] == "保留継続").sum()) if n_total else 0
        monthly_rows.append(dict(年月=ym, 件数=n_total, 発火=n_fire, 再評価=n_reeval,
                                 保留継続=n_hold, 対象単位数=len(meta)))

        if n_total:
            detail_frames.append(inbox.assign(シミュレーション年月=ym))

            # 「発火」した項目だけ、その場で台帳に記録して次月以降の再着火を抑える。
            # 「再評価」「保留継続」は元々ledgerに記録済みなのでここでは触らない。
            fired = inbox[inbox["載る理由"] == "発火"]
            if not fired.empty:
                new_rows = []
                for _, r in fired.iterrows():
                    row = dict.fromkeys(LEDGER_COLS, pd.NA)
                    row.update(事業コード=r["事業コード"], 開発コード=r["開発コード"],
                              部番=r["部番"], 判定年月=ym, 記録日=ym,
                              処置区分=assumed_disposition)
                    if assumed_disposition == "対策中":
                        row["再評価年月"] = s._add_months(ym, reeval_months)
                    new_rows.append(row)
                ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)

    monthly = pd.DataFrame(monthly_rows)
    details = (pd.concat(detail_frames, ignore_index=True) if detail_frames
              else pd.DataFrame())
    if verbose and not monthly.empty:
        nz = monthly[monthly["件数"] > 0]
        print(f"[{assumed_disposition}前提] 対象月数={len(monthly)} "
              f"平均件数/月={monthly['件数'].mean():.2f} "
              f"最大件数/月={monthly['件数'].max()} "
              f"(0件だった月={len(monthly) - len(nz)}件)")
    return monthly, details, ledger


def compare_dispositions(units_df: pd.DataFrame, cfg: dict,
                         start_ym: int | None = None, end_ym: int | None = None,
                         reeval_months: int = 3) -> pd.DataFrame:
    """『消し込み方』による違いを一目で比較する（楽観/現実的な幅を見る用）。"""
    out = {}
    for disp in ("ノイズ", "対策中"):
        monthly, _, _ = simulate_monthly_load(
            units_df, cfg, start_ym, end_ym, assumed_disposition=disp,
            reeval_months=reeval_months, verbose=False)
        out[disp] = monthly.set_index("年月")["件数"]
    cmp = pd.DataFrame(out)
    cmp.columns = [f"件数({c}前提)" for c in cmp.columns]
    print(cmp.describe().loc[["mean", "50%", "max"]])
    return cmp.reset_index()
