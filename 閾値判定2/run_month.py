# -*- coding: utf-8 -*-
"""
run_month.py — 月次運用の実行スクリプト（毎月これを叩く）

使い方
------
    python run_month.py panel_初回.csv 台帳_空.xlsx
    python run_month.py panel_翌月.csv 台帳.xlsx
    python run_month.py panel.csv 台帳.xlsx 出力先フォルダ
    python run_month.py panel_一部.csv 台帳.xlsx --no-snapshot   # 試し実行
    python run_month.py --cat レンズ                # カテゴリ設定で実行（推奨）

カテゴリ実行（--cat）
--------------------
categories/<名前>.py の設定を使う。パネル・台帳は PANEL_PATH / LEDGER_PATH から
読むので引数は不要（渡した場合はそちらを使い、設定と違えば注意を出す）。
出力は `出力/<カテゴリ>/YYYYMM/`、ファイル名は `..._YYYYMM_<カテゴリ>.csv`、
スナップショットも `出力/<カテゴリ>/スナップショット/` に分かれる。
各CSVの末尾に `カテゴリ` 列が付き、フォルダに `実行記録.txt` が残る。
全カテゴリの一括実行は run_all.py。

試し実行・切り出したパネルで回すときは --no-snapshot を付ける
（その月の正規のスナップショットが痩せたデータで置き換わるのを防ぐ）。

出力
----
出力は既定で `出力/YYYYMM/` 配下に書き出す（第3引数で変更可）。
プログラム本体と生成物が混ざらないよう、入力と同じ場所には置かない。

    出力/YYYYMM/統合インボックス_YYYYMM.csv … 全発火候補（統合注目度の降順）
    出力/YYYYMM/レビュー用_YYYYMM.csv        … 上位N件。台帳に貼り付ける用
    出力/YYYYMM/信号B詳細_YYYYMM.csv         … 信号Bの群内比較の生データ
    出力/YYYYMM/信号C詳細_YYYYMM.csv         … 信号Cの販社別の生データ
    出力/YYYYMM/受領状況_YYYYMM.csv          … 販社別の horizon・遅れ月数（USE_HORIZON時）
    出力/YYYYMM/完全性_YYYYMM.csv            … 分子が欠けている販社（閾値の率が過小な部品）
    出力/スナップショット/snap_YYYYMM.csv    … 熟成カーブ較正用（panel_maturity）
"""
from __future__ import annotations

import os
import sys
import hashlib
import datetime as dt
import pandas as pd
import unified_inbox as ui
import panel_maturity as pm

import config_loader as cl
cl.activate_from_argv()          # --cat を取り除いて設定を切り替える
import settings as st

COLS = st.COLS
CFG = st.build_cfg()
OUT_ROOT = st.OUT_ROOT

# 台帳に貼り付ける列（インボックスの列 → 台帳の列 と同じ並び）
REVIEW_COLS = ["事業コード", "開発コード", "部番", "検出器", "対象販社",
               "判定年月", "対象月内訳", "遅延月", "run年月", "統合注目度", "状態", "指標",
               "観測率", "当月閾値", "提案Y下限",
               "処置区分", "再評価年月", "上書き閾値", "原因メモ", "確認者"]


def _fileinfo(path) -> str:
    try:
        ts = dt.datetime.fromtimestamp(os.path.getmtime(path))
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()[:8]
        return f"{os.path.abspath(path)}  (更新 {ts:%Y-%m-%d %H:%M:%S}, sha256 {h})"
    except OSError:
        return f"{path}  (読めない)"


def main(panel_path: str, ledger_path: str, out_root: str = OUT_ROOT,
         snapshot: bool = True):
    c = COLS
    cat = cl.category()
    if cat:
        print(f"=== カテゴリ: {cat} ===")
        for what, given, conf in (("パネル", panel_path, getattr(st, "PANEL_PATH", None)),
                                  ("台帳", ledger_path, getattr(st, "LEDGER_PATH", None))):
            if conf and cl.is_pattern(conf):
                conf = cl.resolve_path(conf, what, quiet=True)
            if conf and os.path.abspath(given) != os.path.abspath(conf):
                print(f"[注意] {what}がカテゴリ設定と違います: 指定={given} / 設定={conf}")
    print(f"使用パネル: {panel_path}")
    raw = pd.read_csv(panel_path, encoding="utf-8-sig")

    # --- 取り違え検知（カテゴリ設定の EXPECT_BIZ）---
    exp = getattr(st, "EXPECT_BIZ", None)
    if exp:
        got = set(raw[c["biz"]].astype(str).unique())
        bad = sorted(got - {str(x) for x in exp})
        if bad:
            raise SystemExit(f"[停止] 想定外の{c['biz']}がパネルに含まれています: {bad}\n"
                             f"       想定={sorted(exp)}  パネル={panel_path}\n"
                             "       別カテゴリのファイルを渡していないか確認してください。")

    # --- 投入前チェック（既存の手順A）---
    key = [c["biz"], c["dev"], c["part"], c["dist"]]
    raw[c["ym"]] = raw[c["ym"]].astype(str).str.replace(r"\D", "", regex=True).astype(int)
    raw = raw.sort_values(key + [c["ym"]])
    raw[c["cum_sales"]] = raw.groupby(key)[c["cum_sales"]].cummax()

    tok = st.ALL_TOKEN
    p_all = raw[raw[c["dist"]].astype(str) == tok].copy()
    p_dist = raw[raw[c["dist"]].astype(str) != tok].copy()
    print(f"パネル: {len(raw)}行  ALL={len(p_all)}  販社別={len(p_dist)}")

    ledger = ui.load_ledger(ledger_path)
    print(f"台帳: {len(ledger)}行")

    r = ui.build_unified_inbox(p_all, p_dist, ledger, CFG, COLS)
    T = r["asof"]
    inbox = r["inbox"]

    bdir = cl.base_dir(out_root)
    outdir = os.path.join(bdir, str(T))
    rec = os.path.join(outdir, "実行記録.txt")
    if os.path.exists(rec):
        # 同じ判定年月の再実行。新しい月のパネルがまだ置かれていない可能性がある
        ts = dt.datetime.fromtimestamp(os.path.getmtime(rec))
        print(f"[注意] 判定年月 {T} は {ts:%Y-%m-%d %H:%M} に実行済みです（出力を上書き）。"
              "新しい月のパネルが置かれているか確認してください。")
    os.makedirs(outdir, exist_ok=True)
    sfx = cl.suffix(T)

    def _w(df, name):
        path = os.path.join(outdir, f"{name}{sfx}.csv")
        if cat:
            df = df.copy()
            df["カテゴリ"] = cat
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    _w(inbox, "統合インボックス")
    review_path = _w(inbox.head(CFG["top_n"]).reindex(columns=REVIEW_COLS), "レビュー用")
    if r["b_raw"] is not None and not r["b_raw"].empty:
        _w(r["b_raw"], "信号B詳細")
    if r["c_raw"] is not None and not r["c_raw"].empty:
        _w(r["c_raw"], "信号C詳細")

    # --- 販社の報告遅れ ---
    if CFG.get("use_horizon", False):
        rep = r["horizon_report"]
        if rep is not None and not rep.empty:
            _w(rep, "受領状況")
            print(f"\n=== 販社の報告遅れ（global horizon = {r['horizon_global']}）===")
            print(rep.to_string(index=False))
            bad = rep[rep["判定"] == "再評価窓不足"]
            if len(bad):
                print(f"[警告] 再評価窓不足: {'/'.join(bad['販社'].astype(str))} "
                      f"→ C_REVISIT_MONTHS を {int(rep['遅れ月数'].max())} 以上に")
        if r["completeness"] is not None:
            _w(r["completeness"], "完全性")
        if snapshot:
            sp = pm.save_snapshot(p_dist, COLS, T,
                                  outdir=os.path.join(bdir, "スナップショット"),
                                  all_token=st.ALL_TOKEN)
            print(f"スナップショット保存: {sp}")
    else:
        print("\n[注意] USE_HORIZON=False。信号Cは最新月の1ヶ月しか判定しません。")

    _write_record(outdir, T, cat, panel_path, ledger_path, raw, ledger, inbox, snapshot)

    print(f"\n=== 判定年月 {T} ／ 発火 {len(inbox)}件（上位{CFG['top_n']}件をレビュー）===")
    if inbox.empty:
        print("（発火なし）")
        return
    print(inbox["検出器"].value_counts().to_string())
    print()
    show = ["開発コード", "部番", "検出器", "対象販社", "遅延月", "統合注目度", "指標"]
    print(inbox.head(CFG["top_n"])[show].to_string(index=False))
    print(f"\n出力先: {outdir}/")
    print(f"→ {review_path} を開き、処置区分を記入して台帳に追記してください。")


def _write_record(outdir, T, cat, panel_path, ledger_path, raw, ledger, inbox, snapshot):
    """実行記録.txt。どの設定・どの入力で回したかを残す（同月の再実行では上書き）。"""
    c = COLS
    L = [f"実行日時   : {dt.datetime.now():%Y-%m-%d %H:%M:%S}",
         f"カテゴリ   : {cat or '（なし）'}",
         f"判定年月   : {T}",
         f"共通設定   : {_fileinfo(getattr(st, '_BASE_FILE', st.__file__))}"]
    if cat:
        L.append(f"カテゴリ設定: {_fileinfo(st.__file__)}")
    pp = getattr(st, "PANEL_PATH", None)
    if cat and cl.is_pattern(pp):
        L.append(f"PANEL_PATH : {pp}（パターン → 最新を採用）")
    L += [f"パネル     : {_fileinfo(panel_path)}",
          f"             {len(raw)}行  年月 {int(raw[c['ym']].min())}〜{int(raw[c['ym']].max())}"
          f"  {c['biz']}={sorted(raw[c['biz']].astype(str).unique())}",
          f"台帳       : {_fileinfo(ledger_path)}  {len(ledger)}行",
          f"スナップショット: {'保存' if snapshot and CFG.get('use_horizon') else 'なし'}",
          f"発火       : {len(inbox)}件"
          + (f"  {inbox['検出器'].value_counts().to_dict()}" if len(inbox) else ""),
          "", "--- 実効設定（build_cfg）---"]
    L += [f"{k} = {CFG[k]!r}" for k in sorted(CFG)]
    L += ["", "--- 列名（COLS）---"] + [f"{k} = {v!r}" for k, v in COLS.items()]
    with open(os.path.join(outdir, "実行記録.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    no_snap = "--no-snapshot" in sys.argv[1:]
    cat = cl.category()
    if cat and len(args) < 2:
        # --cat のときは PANEL_PATH / LEDGER_PATH を既定にする
        args = [cl.panel_arg(args, 0),
                args[1] if len(args) > 1 else getattr(st, "LEDGER_PATH", None)] + args[2:]
        if not args[1]:
            raise SystemExit("[エラー] 台帳のパスがありません。カテゴリ設定に LEDGER_PATH を書いてください。")
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)
    main(args[0], args[1], args[2] if len(args) > 2 else OUT_ROOT,
         snapshot=not no_snap)
