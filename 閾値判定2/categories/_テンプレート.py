# -*- coding: utf-8 -*-
"""
カテゴリ設定のテンプレート。コピーして categories/<カテゴリ名>.py として使う。
（先頭が _ のファイルは run_all.py の対象外）

書き方
------
- 先頭の `from settings import *` は消さない（共通の既定値を読み込む）
- **settings.py と違う値だけ**を書く。書かなかった項目は共通の値になる
- 辞書は部分的に変えるときも丸ごと書き直す:
      COLS = {**COLS, "sf": "SFコード"}          # 1項目だけ変える
      SUPPRESS_MONTHS = {**SUPPRESS_MONTHS, "信号C": 2}
- パスは実行フォルダ（プログラムのあるフォルダ）からの相対、または絶対パス
"""
from settings import *   # noqa: F401,F403

CATEGORY = ""                     # 空ならファイル名がカテゴリ名になる
# パネル。パターン（*）を使うと一致するファイルのうち名前順で最新を使う。
#   月ごとにフォルダを分ける: "data/*/xxx_panel.csv"   （フォルダ名は YYYYMM の6桁）
#   同じ名前で上書きする    : "data/xxx_panel.csv"
PANEL_PATH = "data/*/xxx_panel.csv"
LEDGER_PATH = "台帳/台帳_xxx.xlsx"
EXPECT_BIZ = {"E1"}               # このカテゴリの事業コード。違うものが混ざると停止

# --- ここから下は必要なものだけ ---
# BASE_THRESHOLD_PCT = 3.0
# THRESHOLD_OVERRIDES = {("E1", None, "P-1234"): 8.0}
# B_MIN_OE = 2.0
# C_ALPHA = 0.001
# HORIZON_MARGIN_OVERRIDES = {"A": 0}
