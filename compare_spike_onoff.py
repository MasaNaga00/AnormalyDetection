# -*- coding: utf-8 -*-
"""
compare_spike_onoff.py — スパイク検定のオン/オフを実データで比較評価する。

「過去のスパイク系インシデントも実はドリフトで拾えていた」という観察を、
ラベル済みデータで定量確認するためのツール。スパイクをオフにして本当に
取りこぼしが増えないかを、インシデント単位で1件ずつ突き合わせる。

スパイクのオフは cfg['alpha_spike'] = None。これで state_logic_cusum は
ドリフトのみで監視する（alpha_spikeを小さくするのとは違い、min_count経由の
残存発火も注目度への副作用も無い、完全なオフ）。

出力
----
1. サマリ: 検知率 / インボックス遅れ中央値 / 月次発火候補中央値（負荷）を
   オン・オフで並べる。
2. インシデント別突き合わせ: 各ラベルが
     - オンでもオフでも検知（=ドリフトで足りる。スパイク不要）
     - オンのみ検知（=スパイクでしか拾えない。オフにすると取りこぼす）★要注意
     - どちらも未検知（=そもそも操作点の外。スパイクの話ではない）
   のどれかを判定し、★の件だけ詳細に出す。★が0件ならスパイクオフは安全。
3. 種別内訳: オンのとき各ラベルが drift / spike / burst のどれで最初に載ったか。
"""
import numpy as np
import pandas as pd
import state_logic_cusum as s
from simulate_capped_triage import simulate_capped_triage, month_diff


def _detect_map(units, cfg, labels, top_n, disp, lookback_m):
    """ラベルごとの (検知有無, インボックス入り月, インボックス遅れ月) を返す。"""
    _, inc, _, _ = simulate_capped_triage(
        units, cfg, top_n=top_n, assumed_disposition=disp,
        labels=labels, lookback_m=lookback_m, verbose=False)
    m = {}
    for _, r in inc.iterrows():
        key = (r["事業コード"], r["開発コード"], r["部番"], int(r["販社報告月"]))
        m[key] = (r["検知"] == "あり", r["インボックス入り月"], r["インボックス遅れ月"])
    return m, inc


def compare_spike_onoff(panel_path: str, labels_path: str, cfg_base: dict,
                        top_n=None, assumed_disposition="対策中", lookback_m=6):
    labels = pd.read_csv(labels_path, encoding="utf-8-sig")

    cfg_on = dict(cfg_base)                       # alpha_spike はそのまま
    if cfg_on.get("alpha_spike") is None:
        raise ValueError("cfg_base の alpha_spike が None です。オンの値を入れてください。")
    cfg_off = dict(cfg_base); cfg_off["alpha_spike"] = None

    panel = s._prepare_panel(pd.read_csv(panel_path), cfg_on)
    units = s.aggregate_units(panel)

    on_map, inc_on = _detect_map(units, cfg_on, labels, top_n, assumed_disposition, lookback_m)
    off_map, inc_off = _detect_map(units, cfg_off, labels, top_n, assumed_disposition, lookback_m)

    # 1. サマリ ------------------------------------------------------------
    def _summ(inc):
        d = (inc["検知"] == "あり")
        return (d.mean(),
                inc.loc[d, "インボックス遅れ月"].median(),
                len(inc))
    on_det, on_lag, n = _summ(inc_on)
    off_det, off_lag, _ = _summ(inc_off)
    print("=" * 64)
    print(f"ラベル総数: {n}")
    print(f"{'':10}{'検知率':>10}{'遅れ中央値':>12}")
    print(f"{'スパイクオン':10}{on_det:>10.3f}{on_lag:>12.1f}")
    print(f"{'スパイクオフ':10}{off_det:>10.3f}{off_lag:>12.1f}")

    # 2. インシデント別突き合わせ -----------------------------------------
    only_on, both, neither, earlier_on = [], [], [], []
    for key, (on_ok, on_ib, on_lg) in on_map.items():
        off_ok, off_ib, off_lg = off_map.get(key, (False, None, None))
        if on_ok and off_ok:
            both.append(key)
            # オンの方が早く載っていた（スパイクが早期化に効いていた）か
            if on_ib is not None and off_ib is not None and on_ib < off_ib:
                earlier_on.append((key, on_ib, off_ib, on_lg, off_lg))
        elif on_ok and not off_ok:
            only_on.append((key, on_ib, on_lg))
        elif not on_ok and not off_ok:
            neither.append(key)

    print("\n" + "=" * 64)
    print("スパイクをオフにする影響（インシデント単位）")
    print("=" * 64)
    print(f"  両方で検知（ドリフトで足りる, スパイク不要）: {len(both)}件")
    print(f"  オンのみ検知（スパイクでしか拾えない）★    : {len(only_on)}件")
    print(f"  どちらも未検知（操作点の外, スパイク無関係）: {len(neither)}件")

    if only_on:
        print("\n  ★ スパイクオフで取りこぼすインシデント:")
        df = pd.DataFrame([dict(開発コード=k[1], 部番=k[2], 販社報告月=k[3],
                                インボックス入り月=ib, 遅れ月=lg)
                           for k, ib, lg in only_on])
        print(df.to_string(index=False))
    else:
        print("\n  → ★は0件。このデータではスパイクをオフにしても取りこぼしは増えない。")

    if earlier_on:
        print("\n  参考: スパイクが検知を早めていたインシデント（オフでも拾えるが遅くなる）:")
        df = pd.DataFrame([dict(開発コード=k[1], 部番=k[2], 販社報告月=k[3],
                                入り月_オン=a, 入り月_オフ=b,
                                遅れ_オン=lo, 遅れ_オフ=lf)
                           for k, a, b, lo, lf in earlier_on])
        print(df.to_string(index=False))

    # 3. オン時の種別内訳（各ラベルが何で載ったか）-------------------------
    #    inc_on の「検知あり」ラベルについて、実テーブルで最初の載り種別を見る。
    print("\n" + "=" * 64)
    print("参考: オンのとき各インシデントが最初に載った種別")
    print("=" * 64)
    kind = _first_alert_kind(units, cfg_on, labels, lookback_m)
    if not kind.empty:
        vc = kind["最初の種別"].value_counts()
        print(vc.to_string())
        spike_only_kind = kind[kind["最初の種別"].isin(["spike", "burst", "spike+burst"])]
        if not spike_only_kind.empty:
            print("\n  スパイク系が先行して載ったインシデント:")
            print(spike_only_kind.to_string(index=False))
    return dict(both=both, only_on=only_on, neither=neither, earlier_on=earlier_on)


def _first_alert_kind(units, cfg_on, labels, lookback_m):
    """オンのとき、各ラベルが最初に発火した月の種別（drift/spike/burst）を返す。"""
    ledger = pd.DataFrame(columns=["事業コード", "開発コード", "部番", "判定年月",
                                   "記録日", "処置区分", "再評価年月", "上書きR",
                                   "上書きh", "新ベースライン値", "ベースライン窓起点",
                                   "ベースライン窓長"])
    asof = int(units["ym"].max())
    tbl, _ = s.evaluate_units(units, ledger, cfg_on, asof)
    lab = labels.copy()
    lab["発生年月"] = lab["発生年月"].map(s.to_yyyymm)
    rows = []
    for _, r in lab.iterrows():
        floor = s._add_months(int(r["発生年月"]), -lookback_m)
        sub = tbl[(tbl["dev"] == r["開発コード"]) & (tbl["part"] == r["部番"]) &
                  (tbl["ym"] >= floor) & (tbl["total_alert"])].sort_values("ym")
        if sub.empty:
            continue
        first = sub.iloc[0]
        kind = s._alert_kind(first["alert_drift"], first["alert_spike"], first["alert_burst"])
        rows.append(dict(開発コード=r["開発コード"], 部番=r["部番"],
                         販社報告月=int(r["発生年月"]), 最初の種別=kind,
                         最初の月=int(first["ym"])))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cfg = dict(s.CONFIG)
    cfg.update(R=1.5, h=5.0, alpha_spike=0.005, min_count=3, burst_window=0)
    compare_spike_onoff("scale_panel.csv", "scale_labels.csv", cfg,
                        top_n=None, assumed_disposition="対策中")
