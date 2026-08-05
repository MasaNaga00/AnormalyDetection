# -*- coding: utf-8 -*-
"""
run_month.py — 月次運用の実行スクリプト（毎月これを叩く）

使い方
------
    python run_month.py panel_初回.csv 台帳_空.xlsx
    python run_month.py panel_翌月.csv 台帳.xlsx
    python run_month.py panel.csv 台帳.xlsx 出力先フォルダ

出力
----
出力は既定で `出力/YYYYMM/` 配下に書き出す（第3引数で変更可）。
プログラム本体と生成物が混ざらないよう、入力と同じ場所には置かない。

    出力/YYYYMM/統合インボックス_YYYYMM.csv … 全発火候補（統合注目度の降順）
    出力/YYYYMM/レビュー用_YYYYMM.csv        … 上位N件。台帳に貼り付ける用
    出力/YYYYMM/信号B詳細_YYYYMM.csv         … 信号Bの群内比較の生データ
    出力/YYYYMM/信号C詳細_YYYYMM.csv         … 信号Cの販社別の生データ
"""
from __future__ import annotations

import os
import sys
import pandas as pd
import unified_inbox as ui

# --- 実データに合わせて変えるのはここだけ ---------------------------------
COLS = dict(biz="事業コード", dev="開発コード", part="部番", dist="販社",
            ym="年月", elapsed="経過月", monthly_use="月次使用数",
            cum_sales="累積販売台数", sf="SF-CODE")

CFG = dict(ui.CONFIG)
CFG.update(
    base_threshold_pct=5.0,      # 閾値: 基準X(%)
    b_elapsed_cap=36,            # 信号B: 経過月0〜36で比較
    b_min_peers=2,               # 信号B: ピア2機種以上で判定
    b_alpha=0.005,
    c_base_len=12,               # 信号C: 直近12ヶ月をベースライン
    c_alpha=0.005,               # ← 実データでは件数を見て絞る
    c_min_count=3,
    top_n=15,
)

OUT_ROOT = "出力"          # 生成物の置き場。月ごとにサブフォルダを切る
# ---------------------------------------------------------------------------

# 台帳に貼り付ける列（インボックスの列 → 台帳の列 と同じ並び）
REVIEW_COLS = ["事業コード", "開発コード", "部番", "検出器", "対象販社",
               "判定年月", "統合注目度", "状態", "指標",
               "観測率", "当月閾値", "提案Y下限",
               "処置区分", "再評価年月", "上書き閾値", "原因メモ", "確認者"]


def main(panel_path: str, ledger_path: str, out_root: str = OUT_ROOT):
    raw = pd.read_csv(panel_path, encoding="utf-8-sig")

    # --- 投入前チェック（既存の手順A）---
    raw["年月"] = raw["年月"].astype(str).str.replace(r"\D", "", regex=True).astype(int)
    raw = raw.sort_values(["事業コード", "開発コード", "部番", "販社", "年月"])
    raw["累積販売台数"] = raw.groupby(
        ["事業コード", "開発コード", "部番", "販社"])["累積販売台数"].cummax()

    p_all = raw[raw["販社"] == "ALL"].copy()
    p_dist = raw[raw["販社"] != "ALL"].copy()
    print(f"パネル: {len(raw)}行  ALL={len(p_all)}  販社別={len(p_dist)}")

    ledger = ui.load_ledger(ledger_path)
    print(f"台帳: {len(ledger)}行")

    r = ui.build_unified_inbox(p_all, p_dist, ledger, CFG, COLS)
    T = r["asof"]
    inbox = r["inbox"]

    outdir = os.path.join(out_root, str(T))
    os.makedirs(outdir, exist_ok=True)

    def _w(df, name):
        path = os.path.join(outdir, f"{name}_{T}.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    _w(inbox, "統合インボックス")
    review_path = _w(inbox.head(CFG["top_n"]).reindex(columns=REVIEW_COLS), "レビュー用")
    if r["b_raw"] is not None and not r["b_raw"].empty:
        _w(r["b_raw"], "信号B詳細")
    if r["c_raw"] is not None and not r["c_raw"].empty:
        _w(r["c_raw"], "信号C詳細")

    print(f"\n=== 判定年月 {T} ／ 発火 {len(inbox)}件（上位{CFG['top_n']}件をレビュー）===")
    if inbox.empty:
        print("（発火なし）")
        return
    print(inbox["検出器"].value_counts().to_string())
    print()
    show = ["開発コード", "部番", "検出器", "対象販社", "統合注目度", "指標"]
    print(inbox.head(CFG["top_n"])[show].to_string(index=False))
    print(f"\n出力先: {outdir}/")
    print(f"→ {review_path} を開き、処置区分を記入して台帳に追記してください。")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2],
         sys.argv[3] if len(sys.argv) > 3 else OUT_ROOT)
