# -*- coding: utf-8 -*-
"""
run_month.py — 月次運用の実行スクリプト（毎月これを叩く）

使い方
------
    python run_month.py panel_初回.csv 台帳_空.xlsx
    python run_month.py panel_翌月.csv 台帳.xlsx
    python run_month.py panel.csv 台帳.xlsx 出力先フォルダ
    python run_month.py panel.csv 台帳.xlsx --no-snapshot   # 試し実行（熟成カーブを汚さない）

出力
----
出力は既定で `出力/YYYYMM/` 配下に書き出す（第3引数で変更可）。
プログラム本体と生成物が混ざらないよう、入力と同じ場所には置かない。

    出力/YYYYMM/統合インボックス_YYYYMM.csv … 全発火候補（統合注目度の降順）
    出力/YYYYMM/レビュー用_YYYYMM.csv        … 上位N件。台帳に貼り付ける用
    出力/YYYYMM/信号B詳細_YYYYMM.csv         … 信号Bの群内比較の生データ
    出力/YYYYMM/信号C詳細_YYYYMM.csv         … 信号Cの販社別の生データ
    出力/YYYYMM/受領状況_YYYYMM.csv          … 販社ごとの報告遅れ（horizon）
    出力/YYYYMM/完全性_YYYYMM.csv            … 部品ごとの欠測販社（率の読み方の注記）
    出力/スナップショット/snap_YYYYMM.csv     … 熟成カーブ較正用（毎月ためる）
"""
from __future__ import annotations

import os
import sys
import pandas as pd
import unified_inbox as ui
import panel_maturity as pm

import settings as st

COLS = st.COLS
CFG = st.build_cfg()
OUT_ROOT = st.OUT_ROOT

# 台帳に貼り付ける列（インボックスの列 → 台帳の列 と同じ並び）
REVIEW_COLS = ["事業コード", "開発コード", "部番", "検出器", "対象販社",
               "判定年月", "対象月内訳", "遅延月", "run年月",
               "統合注目度", "状態", "指標",
               "観測率", "当月閾値", "提案Y下限",
               "処置区分", "再評価年月", "上書き閾値", "原因メモ", "確認者"]

# 熟成カーブ較正用スナップショットの保存先（出力ルート直下に貯める）
SNAP_DIR = "スナップショット"


def main(panel_path: str, ledger_path: str, out_root: str = OUT_ROOT,
         save_snapshot: bool = True):
    raw = pd.read_csv(panel_path, encoding="utf-8-sig")

    # --- 投入前チェック（既存の手順A）---
    # 列名は settings.COLS を参照する。実データ移行で触るのは settings.py だけ、
    # という原則をここでも守る（旧版はハードコードだった）。
    c_ym, c_dist = COLS["ym"], COLS["dist"]
    c_key = [COLS["biz"], COLS["dev"], COLS["part"], c_dist]
    raw[c_ym] = raw[c_ym].astype(str).str.replace(r"\D", "", regex=True).astype(int)
    bad = raw[c_ym].astype(str).str.len().value_counts()
    if len(bad) > 1:
        print(f"[警告] 年月の桁数がばらついています: {bad.to_dict()}"
              "  ゼロ埋めされていない月（2023-2 など）が混じっていないか確認してください。")
    raw = raw.sort_values(c_key + [c_ym])
    raw[COLS["cum_sales"]] = raw.groupby(c_key)[COLS["cum_sales"]].cummax()

    p_all = raw[raw[c_dist].astype(str) == st.ALL_TOKEN].copy()
    p_dist = raw[raw[c_dist].astype(str) != st.ALL_TOKEN].copy()
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
    if r.get("horizon_report") is not None and not r["horizon_report"].empty:
        _w(r["horizon_report"], "受領状況")
    if r.get("completeness") is not None and not r["completeness"].empty:
        _w(r["completeness"], "完全性")

    # --- 熟成カーブ較正用のスナップショット（毎月ためる。1回数KB）---
    # これだけが「受付月の値が抽出のたびにどれだけ増えるか」を測る手段。
    # 貯め始めが遅れるとその分だけ較正が後ろにずれるので、初回から必ず残す。
    # 同月に複数回実行すると snap_YYYYMM.csv は**上書き**される（意図的。
    # 遅れ軸が月単位なので、同じ月の複数時点を残すと ρ(k) が歪むため）。
    # 上書き前のファイルは 履歴/ に退避され、snapshot_log.csv に追記される。
    # 部分的なパネルや切り出しデータで試すときは --no-snapshot を付けること。
    if not save_snapshot:
        print("スナップショット保存: スキップ（--no-snapshot）")
    else:
        try:
            snapdir = os.path.join(out_root, SNAP_DIR)
            existed = os.path.exists(os.path.join(snapdir, f"snap_{T}.csv"))
            snap = pm.save_snapshot(p_dist, COLS, T, outdir=snapdir,
                                    all_token=st.ALL_TOKEN)
            if existed:
                print(f"スナップショット保存: {snap}"
                      f"  （同月の既存分を上書き。旧版は 履歴/ に退避）")
            else:
                print(f"スナップショット保存: {snap}")
        except Exception as e:
            print(f"[警告] スナップショット保存に失敗: {e}")

    # --- 報告遅れの状況を先に見せる（発火件数の読み方が変わるため）---
    hr = r.get("horizon_report")
    if not getattr(st, "USE_HORIZON", False):
        print("\n[注意] USE_HORIZON=False です。horizon も C_REVISIT_MONTHS も無効で、"
              "\n       信号Cは最新月の1ヶ月しか判定しません"
              "（遅れて届いた過去月は出ません）。")
    if hr is not None and not hr.empty:
        print(f"\n=== 販社の報告遅れ（global horizon = {r.get('horizon_global')}）===")
        print(hr.to_string(index=False))
        ng = hr[hr["判定"] == "再評価窓不足"]
        if len(ng):
            print(f"[警告] 再評価窓が足りない販社: {list(ng['販社'])}")
            print(f"       settings.C_REVISIT_MONTHS を {int(hr['遅れ月数'].max())} 以上に上げてください。"
                  " このままだとその販社の月は一度も検定されません。")

    print(f"\n=== 判定年月 {T} ／ 発火 {len(inbox)}件（上位{CFG['top_n']}件をレビュー）===")
    if inbox.empty:
        print("（発火なし）")
        return
    print(inbox["検出器"].value_counts().to_string())
    print()
    if "遅延月" in inbox.columns:
        late = inbox[inbox["遅延月"] > 0]
        if len(late):
            print(f"\n（うち {len(late)}件 は過去月の発火＝報告遅れで今回はじめて判定できた分）")
    show = [c for c in ["開発コード", "部番", "検出器", "対象販社",
                        "判定年月", "遅延月", "統合注目度", "指標"]
            if c in inbox.columns]
    print(inbox.head(CFG["top_n"])[show].to_string(index=False))
    print(f"\n出力先: {outdir}/")
    print(f"→ {review_path} を開き、処置区分を記入して台帳に追記してください。")


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if len(argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(argv[0], argv[1],
         argv[2] if len(argv) > 2 else OUT_ROOT,
         save_snapshot=("--no-snapshot" not in flags))
