# -*- coding: utf-8 -*-
"""
diagnose_signal_c.py — 信号Cで「この部品が出ない」理由を段階ごとに切り分ける

信号Cは通過すべき関門が多く、どこで落ちたかが結果からは分からない。
生パネル → 前処理 → horizon → 評価窓 → ベースライン条件 → p値計算条件
→ alpha判定 → 台帳抑制 の順に、どこまで生き残ったかを表示する。

使い方
------
    python diagnose_signal_c.py panel.csv M02 101-0 D 202410
    python diagnose_signal_c.py panel.csv M02 101-0 D          # 全月を表示
    python diagnose_signal_c.py panel.csv                      # 全体サマリのみ

台帳も見たいとき:
    python diagnose_signal_c.py panel.csv M02 101-0 D 202410 台帳.xlsx
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

import reporting_horizon as rh
import signal_c_dist as sd
import settings as st

COLS = st.COLS


def _revisit() -> int:
    """本番（build_unified_inbox）と同じ規則で再評価窓の月数を返す。

    USE_HORIZON はマスタスイッチで、False のとき horizon も再評価窓も無効になる。
    ここを settings.C_REVISIT_MONTHS のまま使うと、本番より広い窓で診断してしまい
    「診断では発火するのにインボックスに出ない」という食い違いになる。
    """
    return int(getattr(st, "C_REVISIT_MONTHS", 0)) \
        if getattr(st, "USE_HORIZON", False) else 0


def _warn_switch():
    if not getattr(st, "USE_HORIZON", False):
        print("!" * 70)
        print("[警告] settings.USE_HORIZON = False です。")
        print("       horizon も C_REVISIT_MONTHS も無効。信号Cは最新月の1ヶ月しか")
        print("       判定しません（遅れて届いた過去月は出ません）。")
        print("       遅延対応を試すなら USE_HORIZON = True にしてください。")
        print("!" * 70)


def _hz(dist_panel: pd.DataFrame) -> dict:
    if not getattr(st, "USE_HORIZON", False):
        return None
    return rh.estimate_horizon(
        dist_panel, COLS, all_token=st.ALL_TOKEN,
        margin_months=getattr(st, "HORIZON_MARGIN_MONTHS", 0),
        margin_overrides=getattr(st, "HORIZON_MARGIN_OVERRIDES", {}),
        fixed=getattr(st, "HORIZON_FIXED", {}),
        auto_margin=getattr(st, "HORIZON_AUTO_MARGIN", True),
        thin_ratio=getattr(st, "HORIZON_THIN_RATIO", 0.7))


def overview(panel_path: str):
    raw = pd.read_csv(panel_path, encoding="utf-8-sig")
    raw[COLS["ym"]] = (raw[COLS["ym"]].astype(str)
                       .str.replace(r"\D", "", regex=True).astype(int))
    d = raw[raw[COLS["dist"]].astype(str) != st.ALL_TOKEN].copy()
    T = int(raw[COLS["ym"]].max())
    K = _revisit()
    lo = sd._shift_ym(T, -K)
    hz = _hz(d)

    _warn_switch()
    print("=" * 70)
    print(f"パネル最新月 T = {T}")
    print(f"USE_HORIZON={getattr(st,'USE_HORIZON',False)}  "
          f"C_REVISIT_MONTHS(設定)={getattr(st,'C_REVISIT_MONTHS',0)}  "
          f"→ 実効={K}  "
          f"HORIZON_MARGIN_MONTHS={getattr(st,'HORIZON_MARGIN_MONTHS',0)}")
    print(f"評価窓の左端 lo = T - {K} = {lo}")
    print(f"C_ALPHA={st.C_ALPHA}  C_MIN_COUNT={st.C_MIN_COUNT}  "
          f"C_BASE_LEN={st.C_BASE_LEN}  "
          f"C_MIN_BASE_MONTHS={st.C_MIN_BASE_MONTHS}  "
          f"C_MIN_BASE_COUNT={st.C_MIN_BASE_COUNT}")
    print("-" * 70)
    if hz:
        rows = []
        for dist in sorted(hz):
            Td = min(T, hz[dist])
            n = sd._diff_ym(Td, lo) + 1
            rows.append(dict(販社=dist, horizon=hz[dist], 右端Td=Td,
                             評価月数=max(n, 0),
                             判定=("★評価対象ゼロ" if n <= 0 else "OK")))
        t = pd.DataFrame(rows)
        print(t.to_string(index=False))
        if (t["評価月数"] <= 0).any():
            print("\n[警告] 評価対象がゼロの販社があります。"
                  "horizon が評価窓の左端より過去です。")
            print("       HORIZON_MARGIN_MONTHS を小さくするか、"
                  "C_REVISIT_MONTHS を大きくしてください。")
    else:
        print("horizon 無効（USE_HORIZON=False）。全販社 右端 = T")

    res = sd.run_signal_c(d, COLS, base_len=st.C_BASE_LEN, alpha=st.C_ALPHA,
                          min_count=st.C_MIN_COUNT,
                          min_base_months=st.C_MIN_BASE_MONTHS,
                          min_base_count=st.C_MIN_BASE_COUNT,
                          asof_ym=T, all_token=st.ALL_TOKEN,
                          horizon=hz, revisit_months=K)
    print("-" * 70)
    if res.empty:
        print("★ 判定行 0 件。上の評価月数を確認してください。")
        return res
    print(f"判定行={len(res)}  p値を計算した行={int(res['p'].notna().sum())}  "
          f"発火={int(res['alert_dist'].sum())}")
    a = res[res["alert_dist"]]
    if len(a):
        print("\n発火した行:")
        print(a[["dev", "part", "dist", "ym", "遅延月", "use", "expected",
                 "O_E", "p"]].to_string(index=False))
    return res


def trace(panel_path: str, dev: str, part: str, dist: str,
          ym: int | None = None, ledger_path: str | None = None):
    raw = pd.read_csv(panel_path, encoding="utf-8-sig")
    raw[COLS["ym"]] = (raw[COLS["ym"]].astype(str)
                       .str.replace(r"\D", "", regex=True).astype(int))
    d = raw[raw[COLS["dist"]].astype(str) != st.ALL_TOKEN].copy()
    T = int(raw[COLS["ym"]].max())
    K = _revisit()
    lo = sd._shift_ym(T, -K)
    hz = _hz(d)

    _warn_switch()
    print("=" * 70)
    print(f"対象: {dev} / {part} / 販社{dist}" + (f" / {ym}" if ym else ""))
    print("=" * 70)

    # --- ① 生パネル ---
    sel = ((d[COLS["dev"]].astype(str) == dev) &
           (d[COLS["part"]].astype(str) == part) &
           (d[COLS["dist"]].astype(str) == dist))
    g = d[sel]
    print(f"① 生パネルの行数: {len(g)}")
    if g.empty:
        print("   ★ 行が無い。機種/部番/販社の指定を確認（大文字小文字・全角半角）。")
        cand = d[(d[COLS['dev']].astype(str) == dev) &
                 (d[COLS['part']].astype(str) == part)]
        if len(cand):
            print(f"   同じ機種×部番の販社: "
                  f"{sorted(set(cand[COLS['dist']].astype(str)))}")
        return
    print(f"   年月の範囲: {int(g[COLS['ym']].min())} 〜 {int(g[COLS['ym']].max())}")

    # --- ② horizon ---
    Td = T if hz is None else min(T, int(hz.get(dist, T)))
    print(f"② horizon[{dist}] = {hz.get(dist) if hz else '無効'}  → 右端 Td = {Td}")
    print(f"   評価窓 = [{lo}, {Td}]  (左端 lo = T{-K:+d})")
    if sd._diff_ym(Td, lo) < 0:
        print("   ★ 右端が左端より過去。この販社は1行も評価されない。")
        return
    if ym is not None:
        if int(ym) > Td:
            print(f"   ★ {ym} は horizon より後。まだ届いていない扱いで切られている。")
            return
        if int(ym) < lo:
            need = sd._diff_ym(T, int(ym))
            if not getattr(st, "USE_HORIZON", False):
                print(f"   ★ {ym} は評価窓の左端より前。USE_HORIZON=False なので"
                      " 窓が最新月の1ヶ月しかありません。True にしてください。")
            else:
                print(f"   ★ {ym} は評価窓の左端より前。C_REVISIT_MONTHS を"
                      f" {need} 以上にすれば入る。")
            return
        print(f"   {ym} は評価窓の中 ✓")

    # --- ③ 検定結果 ---
    res = sd.run_signal_c(d, COLS, base_len=st.C_BASE_LEN, alpha=st.C_ALPHA,
                          min_count=st.C_MIN_COUNT,
                          min_base_months=st.C_MIN_BASE_MONTHS,
                          min_base_count=st.C_MIN_BASE_COUNT,
                          asof_ym=T, all_token=st.ALL_TOKEN,
                          horizon=hz, revisit_months=K)
    if res.empty:
        print("③ ★ 判定行が全体でゼロ。overview() を見てください。")
        return
    q = res[(res.dev == dev) & (res.part == part) & (res.dist == dist)]
    if ym is not None:
        q = q[q.ym == int(ym)]
    if q.empty:
        print("③ ★ 判定結果に行が無い。ベースライン条件で落ちている可能性:")
        print(f"   ・窓の月数 < C_MIN_BASE_MONTHS({st.C_MIN_BASE_MONTHS})"
              "  ← 系列の左側打ち切り（初回修理までの行が無い）")
        print(f"   ・窓内の使用数合計 < C_MIN_BASE_COUNT({st.C_MIN_BASE_COUNT})")
        print("   ・窓内の台数合計 = 0")
        return
    print("③ 判定結果:")
    print(q[["ym", "use", "fleet", "base_rate", "expected", "O_E", "p",
             "alert_dist", "遅延月"]].to_string(index=False))

    for r in q.itertuples():
        print(f"\n   [{r.ym}] ", end="")
        if np.isnan(r.p):
            if r.use < st.C_MIN_COUNT:
                print(f"p値未計算: 使用数{r.use:.0f} < C_MIN_COUNT({st.C_MIN_COUNT})")
            elif not np.isnan(r.expected) and r.use <= r.expected:
                print(f"p値未計算: 使用数{r.use:.0f} ≤ 期待値{r.expected:.1f}"
                      "（下振れは検定しない）")
            else:
                print("p値未計算: ベースライン条件を満たしていない")
        elif r.p > st.C_ALPHA:
            print(f"p={r.p:.2e} > C_ALPHA({st.C_ALPHA}) で沈黙")
        else:
            print(f"発火 ✓ (p={r.p:.2e}, O/E={r.O_E:.2f})")

    # --- ④ 台帳 ---
    if ledger_path:
        import unified_inbox as ui
        lv = ui.LedgerView(ui.load_ledger(ledger_path), T, st.build_cfg())
        biz = str(g[COLS["biz"]].iloc[0])
        print("\n④ 台帳による抑制:")
        for r in q.itertuples():
            s = lv.status(biz, dev, part, "信号C", event_ym=int(r.ym))
            mark = "★抑制されている" if s["suppressed"] else "抑制なし"
            print(f"   [{r.ym}] {mark}  state={s['state']} until={s['until']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    if len(sys.argv) < 5:
        overview(sys.argv[1])
    else:
        trace(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
              int(sys.argv[5]) if len(sys.argv) > 5 else None,
              sys.argv[6] if len(sys.argv) > 6 else None)
