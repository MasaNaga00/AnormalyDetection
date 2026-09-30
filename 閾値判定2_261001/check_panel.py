# -*- coding: utf-8 -*-
"""
check_panel.py — 本番パネルの投入前チェック

「渡す前に直しておくべきもの」と「プログラム側が自動で直すもの」を切り分ける。

自動で直るもの（渡すデータを加工する必要なし）
    ・年月のハイフン区切り → 数字抽出で変換
    ・累積販売台数の非単調 → cummax で補正
    run_month.py / tune_c_alpha.py / tune_b_minoe.py がすべて同じ前処理を通す。

自動で直らないもの（人が判断して直す）
    ・月次使用数の負値（返品・訂正）
    ・ALL行と販社別合計の不一致
    ・キーの重複
    ・SF-CODE の欠損

使い方
------
    python check_panel.py panel.csv
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

import settings as st

COLS = st.COLS
NG, WARN, OK = "✗ 要対処", "△ 確認", "○"


def _hdr(n, title):
    print(f"\n[{n}] {title}")


def main(path: str):
    raw = pd.read_csv(path, encoding="utf-8-sig")
    print("=" * 70)
    print(f"{path}: {len(raw)}行 × {len(raw.columns)}列")
    print("=" * 70)
    issues = []

    # ---------------------------------------------------------------- 1
    _hdr(1, "必須列の存在")
    need = ["biz", "dev", "part", "dist", "ym", "monthly_use", "cum_sales"]
    opt = ["elapsed", "sf"]
    miss = [COLS[k] for k in need if COLS[k] not in raw.columns]
    miss_o = [COLS[k] for k in opt if COLS.get(k) and COLS[k] not in raw.columns]
    if miss:
        print(f"  {NG} 必須列が無い: {miss}")
        print("     settings.COLS を実データの列名に合わせてください。")
        return
    print(f"  {OK} 必須列はそろっています")
    if miss_o:
        print(f"  {WARN} 任意列が無い: {miss_o}"
              "（経過月が無いと信号B、SF-CODEが無いと信号Bが動きません）")
        issues.append("任意列の欠落")

    c_biz, c_dev = COLS["biz"], COLS["dev"]
    c_part, c_dist = COLS["part"], COLS["dist"]
    c_ym, c_use, c_fl = COLS["ym"], COLS["monthly_use"], COLS["cum_sales"]
    keys = [c_biz, c_dev, c_part, c_dist]

    # ---------------------------------------------------------------- 2
    _hdr(2, "年月の書式")
    ymt = raw[c_ym].astype(str).str.replace(r"\D", "", regex=True)
    lens = ymt.str.len().value_counts().to_dict()
    if len(lens) > 1:
        print(f"  {NG} 桁数がばらついています: {lens}")
        print("     ゼロ埋めされていない月（2023-2 など）が混じると5桁になり、"
              "\n     まったく違う年月に化けます。元データ側で直してください。")
        issues.append("年月の桁数ばらつき")
    else:
        print(f"  {OK} 桁数は一定: {lens}")
    raw[c_ym] = ymt.astype(int)
    print(f"     範囲: {int(raw[c_ym].min())} 〜 {int(raw[c_ym].max())}"
          f"  ユニーク月数={raw[c_ym].nunique()}")

    # ---------------------------------------------------------------- 3
    _hdr(3, "キーの重複（同一 機種×部番×販社×年月 が複数行）")
    dup = raw.duplicated(subset=keys + [c_ym], keep=False)
    if dup.any():
        n = int(dup.sum())
        print(f"  {NG} 重複 {n}行")
        print(raw[dup].sort_values(keys + [c_ym]).head(6).to_string(index=False))
        print("     ★プログラムは重複を合算しません。露出も件数も二重になります。"
              "\n     元データ側で1行にまとめてください。")
        issues.append("キーの重複")
    else:
        print(f"  {OK} 重複なし")

    # ---------------------------------------------------------------- 4
    _hdr(4, "月次使用数の負値（★自動では直りません）")
    use = pd.to_numeric(raw[c_use], errors="coerce")
    n_nan = int(use.isna().sum())
    neg = use < 0
    if n_nan:
        print(f"  {WARN} 数値に変換できない値が {n_nan}行（0として扱われます）")
    if neg.any():
        n = int(neg.sum())
        print(f"  {NG} 負値 {n}行 / {len(raw)}行  "
              f"最小={use.min():.0f}  合計={use[neg].sum():.0f}")
        g = raw[neg].groupby(c_dist).size().sort_values(ascending=False)
        print(f"     販社別: {g.to_dict()}")
        print("     影響: 信号Cのベースライン窓の合計が過小になり、"
              "\n           **その後の月が過大に鳴ります**（誤報方向）。"
              "\n           信号B・閾値は逆に過小になり見逃し方向。")
        print("     対処: 返品・訂正が原因なら、可能なら元の月に付け替える。"
              "\n           できない場合は0クリップではなく、"
              "件数と大きさを見て許容範囲か判断する。")
        issues.append("月次使用数の負値")
    else:
        print(f"  {OK} 負値なし")

    # ---------------------------------------------------------------- 5
    _hdr(5, "累積販売台数の非単調（プログラムが cummax で自動補正）")
    fl = pd.to_numeric(raw[c_fl], errors="coerce")
    if fl.isna().any():
        print(f"  {WARN} 数値に変換できない値が {int(fl.isna().sum())}行")
    raw[c_fl] = fl
    s = raw.sort_values(keys + [c_ym]).copy()
    s["_cm"] = s.groupby(keys)[c_fl].cummax()
    d = s[s["_cm"] > s[c_fl]]
    if len(d):
        gap = (s["_cm"] - s[c_fl])
        print(f"  {WARN} 逆転 {len(d)}行 / {len(s)}行  "
              f"最大の落ち込み={gap.max():.0f}台  中央={gap[gap>0].median():.0f}台")
        rel = (gap[gap > 0] / s.loc[gap > 0, "_cm"].replace(0, np.nan))
        print(f"     落ち込みの相対幅: 中央={rel.median():.1%}  最大={rel.max():.1%}")
        g = d.groupby(c_dist).size().sort_values(ascending=False)
        print(f"     販社別: {g.to_dict()}")
        print("     → 小さな逆転（数%以下）なら cummax 補正で問題なし。")
        print("       ★相対幅が大きい／特定の販社や機種に集中している場合は、")
        print("         「本当に台数が減った」可能性がある（市場撤退・廃棄など）。")
        print("         その場合 cummax は分母を高止まりさせ、率を過小にします。")
        if rel.median() > 0.05:
            issues.append("累積販売台数の大きな逆転")
    else:
        print(f"  {OK} 逆転なし")

    # ---------------------------------------------------------------- 6
    _hdr(6, "ALL行と販社別合計の一致")
    a = raw[raw[c_dist].astype(str) == st.ALL_TOKEN]
    b = raw[raw[c_dist].astype(str) != st.ALL_TOKEN]
    print(f"     ALL行={len(a)}  販社別行={len(b)}")
    if a.empty:
        print(f"  {NG} ALL行がありません。閾値・信号Bが動きません。")
        issues.append("ALL行なし")
    elif b.empty:
        print(f"  {NG} 販社別行がありません。信号Cが動きません。")
        issues.append("販社別行なし")
    else:
        k4 = [c_biz, c_dev, c_part, c_ym]
        m = (a.groupby(k4)[c_use].sum().rename("all").to_frame()
             .join(b.groupby(k4)[c_use].sum().rename("sum"), how="inner"))
        ng = int((m["all"] != m["sum"]).sum())
        if ng:
            diff = (m["all"] - m["sum"]).abs()
            print(f"  {WARN} 不一致 {ng} / {len(m)}行  最大差={diff.max():.0f}")
            print("     → ALLが別ソースの可能性。分子の欠け方が販社別行と違うので、")
            print("       報告遅れの扱い（閾値検出器）を別途検討する必要があります。")
            issues.append("ALL行と販社別合計の不一致")
        else:
            print(f"  {OK} 完全一致（{len(m)}行で照合）")

    # ---------------------------------------------------------------- 7
    _hdr(7, "販社別の報告カバレッジ")
    if not b.empty:
        T = int(raw[c_ym].max())
        cov = b.groupby(b[c_dist].astype(str)).agg(
            最終月=(c_ym, "max"), 最古月=(c_ym, "min"),
            月数=(c_ym, "nunique"), 行数=(c_ym, "size"))
        cov["Tとの差"] = [(T // 100 * 12 + T % 100) - (v // 100 * 12 + v % 100)
                       for v in cov["最終月"]]
        print(cov.to_string())
        mx = int(cov["Tとの差"].max())
        print(f"     → C_REVISIT_MONTHS は最低 {mx} 以上必要"
              "（熟成カーブでさらに大きくなることがあります）")

    # ---------------------------------------------------------------- 8
    _hdr(8, "経過月・SF-CODE")
    if COLS.get("elapsed") in raw.columns:
        e = pd.to_numeric(raw[COLS["elapsed"]], errors="coerce")
        print(f"     経過月: 範囲 {e.min():.0f}〜{e.max():.0f}  "
              f"欠損={int(e.isna().sum())}  負値={int((e < 0).sum())}")
        if (e < 0).any() or e.isna().any():
            print(f"  {WARN} 欠損または負値があります（信号Bの窓に影響）")
            issues.append("経過月の異常")
    if COLS.get("sf") in raw.columns:
        n = int(raw[COLS["sf"]].isna().sum())
        print(f"     SF-CODE: ユニーク={raw[COLS['sf']].nunique()}  欠損={n}行")
        if n:
            print(f"  {WARN} 欠損行は信号Bの判定対象から外れます")
            issues.append("SF-CODEの欠損")

    # ---------------------------------------------------------------- まとめ
    print("\n" + "=" * 70)
    if issues:
        print("要対処・要確認:")
        for i in issues:
            print(f"  ・{i}")
    else:
        print("問題は見つかりませんでした。")
    print("\n渡すデータを事前に加工する必要はありません。")
    print("年月の変換と累積販売台数の cummax は、run_month / tune_c_alpha /")
    print("tune_b_minoe がすべて同じ前処理として自動で通します。")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
