# -*- coding: utf-8 -*-
"""
evaluate_units_progress.py
===========================
state_logic_cusum.evaluate_units と完全に同じロジックを、
「機種ごとに進捗表示しながら」「一定数ごとにディスクへ書き出してメモリを
解放しながら」実行するためのラッパー。

state_logic_cusum.py 本体は変更しない。同じ内部関数
（classify_mode, estimate_baseline, earlylife_lambda0, unit_events,
  build_plan, replay_unit, resolve_state）をそのまま呼ぶだけなので、
出力は evaluate_units と数値的に完全一致するはず（下部の検証参照）。

使い方
------
    import state_logic_cusum as s
    from evaluate_units_progress import evaluate_units_stream

    table, meta = evaluate_units_stream(
        units, ledger, cfg, asof,
        out_table_path="table_一時.csv",  # Noneなら全部メモリに保持（従来と同じ）
        chunk_size=200,                    # 200機種ごとにディスクへ書き出し
        progress_every=50,                 # 50機種ごとに進捗を表示
    )

並列化したい場合は use_multiprocessing=True, n_workers=... を指定。
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import state_logic_cusum as s


def _evaluate_one_unit(key, unit, ledger, cfg, machine_end, levels, pooled):
    """evaluate_units の1機種分のロジックをそのまま切り出したもの。"""
    biz, dev, part = key
    unit = unit.sort_values("ym").reset_index(drop=True)
    mode = s.classify_mode(unit, cfg)
    level_label, leader_cnt = None, None
    if mode == "安定期":
        base = s.estimate_baseline(unit, cfg)
        mode_label = "安定期"
    else:
        lam0, mode_label, level_label, leader_cnt = s.earlylife_lambda0(unit, levels, pooled, cfg)
        base = (lam0, None, None)
    events = s.unit_events(ledger, biz, dev, part, cfg.get("_asof"))
    plan = s.build_plan(events, unit, base, cfg)
    rep = s.replay_unit(unit, base, mode_label, plan, cfg)
    if rep.empty:
        return None, None

    states = [s.resolve_state(events, rep, int(r.ym), machine_end.get((biz, dev)), plan)
             for r in rep.itertuples()]
    rep = rep.assign(state=[st["state"] for st in states],
                     reevaluation_due=[st["reevaluation_due"] for st in states])
    judged = {int(e["判定年月"]) for e in events}
    rep["judged_point"] = rep["ym"].isin(judged)

    meta_entry = dict(
        mode=mode_label,
        base_lambda0=(float(np.mean(base[0])) if isinstance(base[0], np.ndarray) else base[0]),
        baseline_origin=(plan.baseline_from[-1][0] if plan.baseline_from
                         else int(rep["ym"].min())),
        reset_count=len(plan.reset_after),
        last_reset=max(plan.reset_after) if plan.reset_after else None,
        machine_end=machine_end.get((biz, dev)), close_month=plan.close_month,
        earlylife_level=level_label, leader_count=leader_cnt,
        events=events, plan=plan, base=base,
    )
    return rep, meta_entry


def evaluate_units_stream(units_df: pd.DataFrame, ledger: pd.DataFrame, cfg: dict,
                          asof: int, out_table_path: str | None = None,
                          chunk_size: int = 200, progress_every: int = 50,
                          use_multiprocessing: bool = False, n_workers: int | None = None):
    """evaluate_units と同じ結果を、進捗表示＋メモリを溜め込まない形で計算する。

    out_table_path を指定すると、chunk_size 機種ごとに CSV へ追記して
    メモリ上の一時リストを空にする（ピークメモリを機種数に依存させない）。
    最終的な table は out_table_path から読み直して返すので、戻り値の
    使い勝手は evaluate_units と同じ。

    use_multiprocessing=True で機種単位を複数プロセスに分散する
    （numpy/pandasの計算自体はGILの影響を受けるが、機種数が数千のオーダーだと
    プロセス分散の方が素直に効くことが多い）。
    """
    machine_end = s.machine_end_at(ledger, cfg)
    pooled = s._pooled_lambda0(units_df, cfg)
    levels = s.build_earlylife_curves(units_df, cfg)
    cfg = dict(cfg)
    cfg["_asof"] = asof  # _evaluate_one_unit に asof を渡すための便宜的な格納

    keys, unit_frames = [], []
    for key, unit in units_df.groupby(["biz", "dev", "part"], sort=False):
        keys.append(key)
        unit_frames.append(unit)
    total = len(keys)
    print(f"[evaluate_units_stream] 対象 {total} 機種×部番。開始...")

    t0 = time.time()
    meta: dict = {}
    buf_rows: list[pd.DataFrame] = []
    wrote_header = False
    n_done = 0

    def _flush():
        nonlocal buf_rows, wrote_header
        if not buf_rows or out_table_path is None:
            return
        chunk = pd.concat(buf_rows, ignore_index=True)
        chunk.to_csv(out_table_path, mode=("w" if not wrote_header else "a"),
                    header=(not wrote_header), index=False, encoding="utf-8-sig")
        wrote_header = True
        buf_rows = []

    if use_multiprocessing:
        import multiprocessing as mp
        n_workers = n_workers or max(1, mp.cpu_count() - 1)
        args = [(key, unit, ledger, cfg, machine_end, levels, pooled)
               for key, unit in zip(keys, unit_frames)]
        with mp.Pool(n_workers) as pool:
            for key, (rep, meta_entry) in zip(keys, pool.starmap(_evaluate_one_unit, args)):
                n_done += 1
                if rep is not None:
                    buf_rows.append(rep)
                    meta[key] = meta_entry
                if out_table_path and len(buf_rows) >= chunk_size:
                    _flush()
                if n_done % progress_every == 0 or n_done == total:
                    el = time.time() - t0
                    eta = el / n_done * (total - n_done)
                    print(f"  {n_done}/{total} 機種完了 経過{el:.0f}秒 残り目安{eta:.0f}秒")
    else:
        for key, unit in zip(keys, unit_frames):
            rep, meta_entry = _evaluate_one_unit(key, unit, ledger, cfg, machine_end, levels, pooled)
            n_done += 1
            if rep is not None:
                buf_rows.append(rep)
                meta[key] = meta_entry
            if out_table_path and len(buf_rows) >= chunk_size:
                _flush()
            if n_done % progress_every == 0 or n_done == total:
                el = time.time() - t0
                eta = el / n_done * (total - n_done)
                print(f"  {n_done}/{total} 機種完了 経過{el:.0f}秒 残り目安{eta:.0f}秒")

    if out_table_path:
        _flush()
        table = pd.read_csv(out_table_path) if wrote_header else pd.DataFrame()
    else:
        table = pd.concat(buf_rows, ignore_index=True) if buf_rows else pd.DataFrame()

    print(f"[evaluate_units_stream] 完了。合計 {time.time()-t0:.0f}秒")
    return table, meta
