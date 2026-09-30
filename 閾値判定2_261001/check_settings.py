# -*- coding: utf-8 -*-
"""
check_settings.py — 「設定を変えたのに効かない」を切り分ける

設定が反映されない原因はだいたい次の4つ。どれなのかを特定する。

  1. 別の settings.py を読んでいる（同名ファイルが複数ある／カレントディレクトリ違い）
  2. Jupyter / IPython で既に import 済み（ファイルを直しても再読込されない）
  3. __pycache__ が古い（まれ）
  4. 編集したファイルを保存していない

使い方
------
    python check_settings.py

Jupyter で確認するときは、**必ずカーネルを再起動してから**:
    import importlib, settings
    importlib.reload(settings)     # これでも足りないことがある。再起動が確実
"""
from __future__ import annotations

import os
import sys
import glob
import datetime as dt

import settings as st

KEYS = ["USE_HORIZON", "C_REVISIT_MONTHS", "HORIZON_MARGIN_MONTHS",
        "HORIZON_AUTO_MARGIN", "HORIZON_THIN_RATIO", "HORIZON_FIXED",
        "B_MIN_OE", "B_ALPHA", "C_ALPHA", "C_MIN_COUNT",
        "BASE_THRESHOLD_PCT", "TOP_N"]


def main():
    print("=" * 70)
    print("実際に読み込まれている settings.py")
    print("=" * 70)
    path = os.path.abspath(st.__file__)
    print(f"  パス     : {path}")
    if os.path.exists(path):
        ts = dt.datetime.fromtimestamp(os.path.getmtime(path))
        print(f"  最終更新 : {ts:%Y-%m-%d %H:%M:%S}")
    print(f"  作業ディレクトリ : {os.getcwd()}")

    print("\n=== 主要な値 ===")
    for k in KEYS:
        v = getattr(st, k, "（未定義）")
        print(f"  {k:24s} = {v}")

    # --- 実効値（マスタスイッチの効果を反映したもの）---
    use_hz = bool(getattr(st, "USE_HORIZON", False))
    k_set = int(getattr(st, "C_REVISIT_MONTHS", 0))
    print("\n=== 実効値 ===")
    print(f"  horizon 適用        : {'あり' if use_hz else '★なし'}")
    print(f"  信号Cの再評価窓     : {k_set if use_hz else 0} ヶ月"
          + ("" if use_hz else "  ★USE_HORIZON=False のため0に強制"))
    if not use_hz:
        print("\n  → 信号Cは最新月の1ヶ月しか判定しません。")
        print("     遅れて届いた過去月は出ません。")

    # --- 同名ファイルの重複を探す ---
    print("\n=== 他に settings.py が無いか ===")
    seen, hits = set(), []
    for d in [os.getcwd()] + sys.path:
        if not d or not os.path.isdir(d):
            continue
        rp = os.path.realpath(d)
        if rp in seen:
            continue
        seen.add(rp)
        for f in glob.glob(os.path.join(rp, "settings.py")):
            hits.append(os.path.realpath(f))
    hits = list(dict.fromkeys(hits))
    for f in hits:
        ts = dt.datetime.fromtimestamp(os.path.getmtime(f))
        mark = " ← これが読まれている" if os.path.realpath(path) == f else "  ★別ファイル"
        print(f"  {ts:%Y-%m-%d %H:%M}  {f}{mark}")
    if len(hits) > 1:
        print("\n  ★ settings.py が複数あります。編集したのが上の"
              "「これが読まれている」と同じか確認してください。")

    # --- pycache ---
    pc = os.path.join(os.path.dirname(path), "__pycache__")
    if os.path.isdir(pc):
        old = []
        src_m = os.path.getmtime(path)
        for f in glob.glob(os.path.join(pc, "settings.*.pyc")):
            if os.path.getmtime(f) < src_m - 1:
                old.append(f)
        if old:
            print(f"\n=== __pycache__ が古い ===")
            for f in old:
                print(f"  {f}")
            print("  → このフォルダごと削除してください: rm -rf __pycache__")

    # --- Jupyter 判定 ---
    if "ipykernel" in sys.modules or "IPython" in sys.modules:
        print("\n=== 実行環境 ===")
        print("  ★ Jupyter / IPython で動いています。")
        print("    settings.py を編集しても、既に import 済みなら反映されません。")
        print("    カーネルを再起動してから実行し直してください。")

    print("\n" + "=" * 70)
    if use_hz:
        print("USE_HORIZON=True で読めています。設定は反映されています。")
    else:
        print("★ USE_HORIZON=False として読まれています。")
        print("  上の「これが読まれている」パスのファイルを開いて確認してください。")
    print("=" * 70)


if __name__ == "__main__":
    main()
