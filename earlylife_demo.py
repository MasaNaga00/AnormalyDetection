"""
earlylife_demo.py
=================
安定化前(初期不良期)の異常検知を疑似データで動作確認するデモ。

シナリオ:
  - あるSF-コードに先行機種が数台(ランク差あり)。各機種は経過月とともに
    故障率が「初期高→減衰→安定」する初期不良カーブを持つ。
  - これら先行機種から lambda0(t) カーブを経験ベイズで推定。
  - 新機種を2タイプ用意:
      正常な新機種: 先行と同じ初期不良カーブ(集団どおり)。誤報が出ないか見る。
      異常な新機種: 初期不良の水準が集団より高い(初期から悪い)。検知できるか見る。
"""

from __future__ import annotations
import numpy as np
import pandas as pd

import cusum_monitor as cm
from earlylife_baseline import estimate_earlylife_curve, attach_curve_to_unit


def earlylife_rate(months, lam_floor, init_mult, decay_m):
    """初期不良カーブ: lam_floor * (1 + (init_mult-1)*exp(-t/decay_m))。
    t=0 で lam_floor*init_mult、十分大きい t で lam_floor に漸近。"""
    t = np.arange(months)
    return lam_floor * (1 + (init_mult - 1) * np.exp(-t / decay_m))


def make_earlylife_data(
    months: int = 36,
    n_leaders: int = 5,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    rows = []

    def fleet_curve(scale):
        # 発売直後は台数が小さく、立ち上がる。scaleで機種規模差(ランク)を表現。
        return (np.linspace(300, 18000, months) * scale).round().astype(float)

    def emit(dev, lam_floor, init_mult, decay_m, fleet_scale, rank, kind="leader"):
        fleet = fleet_curve(fleet_scale)
        rate = earlylife_rate(months, lam_floor, init_mult, decay_m)
        usage = rng.poisson(rate * fleet)
        for m in range(months):
            rows.append({
                "事業コード": "BIZ1", "SF-コード": "SF100", "ランク": rank,
                "開発コード": dev, "部番": "100-0001-001", "販社": "S_ALL",
                "経過月": m, "月次使用数": int(usage[m]),
                "累積販売台数": float(fleet[m]),
            })
        return dev

    # --- 先行機種(ランク差つき): 同じ初期不良の形、水準と規模はばらつく ---
    leaders = []
    for i in range(n_leaders):
        rank = "H" if i % 2 == 0 else "L"
        lam_floor = (1.2e-4 if rank == "H" else 7e-5) * rng.uniform(0.85, 1.15)
        init_mult = rng.uniform(3.0, 4.5)      # 初期は安定の3〜4.5倍
        decay_m = rng.uniform(6, 10)           # 半年〜10か月で落ち着く
        fleet_scale = (1.3 if rank == "H" else 0.8) * rng.uniform(0.9, 1.1)
        leaders.append(emit(f"LEAD{i}", lam_floor, init_mult, decay_m, fleet_scale, rank))

    # --- 新機種(評価対象) ---
    # 正常: ランクHの集団に従う(初期不良はあるが集団どおり)
    emit("NEW_OK", 1.2e-4, 3.7, 8.0, 1.3, "H")
    # 異常: 初期の水準が集団より明確に高い(init_mult を大きく、floorも上げる)
    emit("NEW_BAD", 1.2e-4 * 1.5, 6.0, 8.0, 1.3, "H")

    return pd.DataFrame(rows)


def main():
    months = 36
    df = make_earlylife_data(months=months, seed=3)
    group_keys = ["事業コード", "SF-コード", "ランク"]  # ランクを分けてプール

    leaders = df[df["開発コード"].str.startswith("LEAD")]
    print(f"先行機種 {leaders['開発コード'].nunique()} 台から lambda0(t) を推定 "
          f"(集団キー={group_keys})")

    # --- カーブ推定(経験ベイズ部分プーリング) ---
    curve = estimate_earlylife_curve(
        leaders, group_keys=group_keys, max_keizoku=months - 1,
    )
    cH = curve[curve["ランク"] == "H"].sort_values("経過月")
    print("\n推定カーブ(ランクH, 抜粋): 経過月ごとの lambda0_hat")
    print(cH[["経過月", "lambda0_hat", "n_models"]].head(12).to_string(index=False))

    # --- 新機種を安定化前モードで監視 ---
    # スパイク/ドリフト両方を初期から見る。spike側は直接ポアソン検定になる。
    params = dict(stable_start_m=0, baseline_len=0, monitor_end_m=months - 1,
                  R=2.0, h=6.0, alpha_spike=0.005, min_count=3, burst_window=2)

    for dev in ["NEW_OK", "NEW_BAD"]:
        unit = df[df["開発コード"] == dev].sort_values("経過月").reset_index(drop=True)
        lam_curve = attach_curve_to_unit(unit, curve, group_keys)
        res = cm.monitor_unit(unit, lambda0_curve=lam_curve, **params)
        n_alarm = int(res["総合アラート"].sum())
        first = res[res["総合アラート"]]["経過月"].min() if n_alarm else None
        print(f"\n=== 新機種 {dev} (安定化前モード) ===")
        print(f"  総アラート月数: {n_alarm}"
              + (f", 初アラート経過月: {int(first)}" if n_alarm else ""))
        show = res[["経過月", "月次使用数", "期待故障数", "CUSUM", "しきい値h",
                    "月次p値", "アラート種別"]].copy()
        show["期待故障数"] = show["期待故障数"].round(2)
        show["CUSUM"] = show["CUSUM"].round(2)
        show["月次p値"] = show["月次p値"].round(4)
        print(show.head(14).to_string(index=False))

    print("\n読み方: NEW_OK は集団どおりの初期不良なので鳴らない(誤報が出ない)のが理想。")
    print("        NEW_BAD は初期から集団水準を超えるので早期に鳴るのが理想。")


if __name__ == "__main__":
    main()
