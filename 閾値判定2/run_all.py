# -*- coding: utf-8 -*-
"""
run_all.py — 全カテゴリの月次実行を一括で回す

使い方
------
    python run_all.py                     # categories/ の全カテゴリ（_ で始まるファイルは除く）
    python run_all.py カメラ レンズ        # 指定したものだけ
    python run_all.py --no-snapshot       # 試し実行（run_month にそのまま渡す）
    python run_all.py --dir 別フォルダ     # カテゴリ設定の置き場所を変える

動き
----
- カテゴリごとに run_month.py --cat <名前> を**別プロセス**で実行する
  （設定が前のカテゴリから持ち越されることがない）。
- 途中のカテゴリが失敗しても残りは続行し、最後に成功／失敗の一覧を出す。
- 全出力を 出力/_バッチログ/YYYYMMDD_HHMMSS.log に保存する。
- 1件でも失敗があれば終了コード 1（タスクスケジューラ等で検知できる）。
"""
from __future__ import annotations

import os
import re
import sys
import time
import datetime as dt
import subprocess

import config_loader as cl

HERE = cl.HERE


def main(argv: list[str]) -> int:
    flags, names, cat_dir = [], [], None
    it = iter(argv)
    for a in it:
        if a == "--dir":
            cat_dir = os.path.abspath(next(it))
        elif a.startswith("--"):
            flags.append(a)
        else:
            names.append(a)
    cats = names or cl.list_categories(cat_dir)
    if not cats:
        print(f"[エラー] カテゴリ設定がありません: {cat_dir or cl.CATEGORY_DIR}")
        return 1

    base = cl.activate(None)
    logdir = os.path.join(base.OUT_ROOT, "_バッチログ")
    os.makedirs(logdir, exist_ok=True)
    logpath = os.path.join(logdir, f"{dt.datetime.now():%Y%m%d_%H%M%S}.log")

    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    rows = []
    with open(logpath, "w", encoding="utf-8") as log:
        for cat in cats:
            target = cl.category_path(cat, cat_dir) if cat_dir else cat
            head = f"\n{'#' * 70}\n# {cat}\n{'#' * 70}"
            print(head); log.write(head + "\n")
            t0 = time.time()
            p = subprocess.run([sys.executable, "run_month.py", "--cat", target, *flags],
                               cwd=HERE, env=env, capture_output=True)
            out = (p.stdout + p.stderr).decode("utf-8", errors="replace")
            print(out); log.write(out); log.flush()
            m = re.search(r"判定年月 (\d{6}) ／ 発火 (\d+)件", out)
            mp = re.search(r"使用パネル: (.+)", out)
            again = "判定年月" in out and "実行済みです" in out
            note = "" if p.returncode == 0 else _last_error(out)
            if p.returncode == 0 and again:
                note = "★同じ判定年月を再実行（新しいパネル未配置？）"
            rows.append((cat, "成功" if p.returncode == 0 else "★失敗",
                         m.group(1) if m else "-", m.group(2) if m else "-",
                         f"{time.time() - t0:.0f}s", note,
                         mp.group(1).strip() if mp else "-"))

        w = max(len(r[0]) for r in rows) + 2
        summ = ["", "=" * 70, "一括実行の結果", "=" * 70,
                f"{'カテゴリ':<{w}}結果    判定年月  発火  時間  注意・エラー"]
        summ += [f"{c:<{w}}{s:<6}  {t:<8}  {n:>4}  {sec:>4}  {e}\n{'':<{w}}  パネル: {pp}"
                 for c, s, t, n, sec, e, pp in rows]
        summ.append(f"\nログ: {os.path.abspath(logpath)}")
        txt = "\n".join(summ)
        print(txt); log.write(txt + "\n")
    return 0 if all(r[1] == "成功" for r in rows) else 1


def _last_error(out: str) -> str:
    lines = [l.strip() for l in out.strip().splitlines() if l.strip()]
    for l in reversed(lines):
        if l.startswith(("[停止]", "[エラー]")) or "Error" in l:
            return l[:80]
    return lines[-1][:80] if lines else ""


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
