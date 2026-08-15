# -*- coding: utf-8 -*-
"""check_rank.py — 過去の検出例が上位N件に入るかを確認する

各ラベルについて、報告月の lookback ヶ月前から報告月までを asof で再現し、
「インボックスに載った最初の月」と「上位N件に入った最初の月」を出す。
遅れ月がマイナス = 販社報告より先に捕捉できた。
"""
import pandas as pd, unified_inbox as ui, settings as st

LOOKBACK, TOPN = st.LOOKBACK_M, st.TOP_N


def _shift(ym, k):
    y, m = divmod(int(ym), 100); i = y * 12 + (m - 1) + k
    return (i // 12) * 100 + (i % 12) + 1


def _diff(a, b):
    ya, ma = divmod(int(a), 100); yb, mb = divmod(int(b), 100)
    return (yb * 12 + mb) - (ya * 12 + ma)


def check(panel_path, labels_path, cfg=None, cols=None,
          lookback=LOOKBACK, top_n=TOPN, ahead=0):
    cfg = dict(cfg or st.build_cfg()); cols = cols or st.COLS
    raw = pd.read_csv(panel_path, encoding="utf-8-sig")
    raw["年月"] = raw["年月"].astype(str).str.replace(r"\D", "", regex=True).astype(int)
    p_all, p_dist = raw[raw.販社 == st.ALL_TOKEN], raw[raw.販社 != st.ALL_TOKEN]

    lab = pd.read_csv(labels_path, encoding="utf-8-sig")
    lab["発生年月"] = (lab["発生年月"].astype(str)
                   .str.replace(r"\D", "", regex=True).str[:6].astype(int))

    # 評価に必要な月をまとめて1回ずつ実行（ラベルごとに回すと重複するため）
    have = set(raw["年月"].unique())
    months = set()
    for y in lab["発生年月"]:
        for k in range(-lookback, ahead + 1):
            m = _shift(y, k)
            if m in have:
                months.add(m)
    months = sorted(months)
    ranks = {}
    for m in months:
        ib = ui.build_unified_inbox(p_all, p_dist, ui.empty_ledger(),
                                    cfg, cols, asof_ym=m)["inbox"]
        for i, r in enumerate(ib.itertuples(), 1):
            ranks[(m, r.事業コード, r.開発コード, r.部番)] = (i, r.検出器, r.統合注目度)

    out = []
    for r in lab.itertuples():
        k = (r.事業コード, r.開発コード, r.部番)
        seen = [(m, *ranks[(m, *k)]) for m in months if (m, *k) in ranks]
        top = [s for s in seen if s[1] <= top_n]
        out.append(dict(
            機種=r.開発コード, 部番=r.部番, 報告年月=r.発生年月,
            載った月=seen[0][0] if seen else None,
            最良順位=min(s[1] for s in seen) if seen else None,
            検出器=seen[0][2] if seen else "",
            上位N入り月=top[0][0] if top else None,
            遅れ月=_diff(r.発生年月, top[0][0]) if top else None,
            判定="○" if top else ("△載るが沈む" if seen else "×未検知")))
    res = pd.DataFrame(out)
    print(f"=== 上位{top_n}件に入ったか（ラベル{len(lab)}件）===")
    print(res.to_string(index=False))
    print("\n" + res["判定"].value_counts().to_string())
    return res


if __name__ == "__main__":
    import sys
    check(sys.argv[1], sys.argv[2])
