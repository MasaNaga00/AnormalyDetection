# -*- coding: utf-8 -*-
"""条件付き二項 p >= 素朴ポアソン p を構造的に確認（ベースラインが薄いほど保守側に開く）。"""
import numpy as np
import cusum_monitor as cm

e_t = 1500.0
baseline_len = 12
E = baseline_len * e_t

print("当月カウント x=4 固定。ベースライン窓の総使用数 C（=ベースラインの厚み）を変える。")
print("lam0_hat=C/E を真値扱いした素朴ポアソン p と、推定不確実性を織り込む条件付き二項 p を比較。\n")
print(f"{'C(窓内総数)':>10} {'lam0_hat':>10} {'素朴p':>12} {'条件付二項p':>14} {'条件付/素朴':>12}")
x = 4.0
viol = 0
for C in [1, 2, 4, 8, 16, 32, 64, 128]:
    lam0 = max(C / E, 1e-12)
    p_naive = cm.poisson_sf(x, lam0 * e_t)
    p_s, _, _, _ = cm.spike_test(np.array([x]), np.array([e_t]), lam0,
                                 alpha_spike=0.005, min_count=3,
                                 baseline_count=float(C), baseline_exposure=E)
    p_cond = float(p_s[0])
    if p_cond < p_naive - 1e-12:
        viol += 1
    ratio = p_cond / p_naive if p_naive > 0 else float("inf")
    print(f"{C:>10} {lam0:>10.2e} {p_naive:>12.2e} {p_cond:>14.2e} {ratio:>12.1f}x")

print(f"\n条件付き二項 < 素朴 となった違反: {viol} 件（0なら、条件付き二項は決して鳴りやすくならない）")
print("→ Cが小さい(ベースラインが薄い)ほど比が大きい＝条件付き二項が保守側に開く。")
print("  薄いベースラインでの lambda0 過小推定が誤報を生む状況を、検定自体が吸収する。")
