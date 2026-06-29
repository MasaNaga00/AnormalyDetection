# -*- coding: utf-8 -*-
"""フェーズ1の検証: (a)ドリフト回帰 (b)条件付き二項によるスパイク誤報抑制"""
import numpy as np, pandas as pd
import cusum_monitor as cm
import state_logic_cusum as m

cfg = dict(m.CONFIG); cfg["h"] = 4.0; cfg["burst_window"] = 3

# ---- (a) 回帰: 委譲版のドリフトS が、設計式の独立再計算と一致するか ----
_, _, _, meta, units = m.run_demo(cfg)
a_unit = units[units["part"] == "PART-A"]
base = m.estimate_baseline(a_unit, cfg)
rep = m.replay_unit(a_unit, base, "安定期", m.ReplayPlan(), cfg)  # 台帳なし＝1セグメント

# 設計式 S_t = max(0, S_{t-1}+use-k), k=(R-1)*lam*fleet/lnR を素直に独立実装
lam, R, h = base[0], cfg["R"], cfg["h"]
S_ref, s = [], 0.0
for u, f in zip(rep["use"], rep["fleet"]):
    k = (R-1)*lam*f/np.log(R)
    s = max(0.0, s + u - k); S_ref.append(s)
S_ref = np.array(S_ref)
max_abs_diff = float(np.max(np.abs(rep["S"].to_numpy() - S_ref)))
print(f"(a) ドリフト回帰: 委譲版S と 独立再計算S の最大絶対差 = {max_abs_diff:.2e}")
print(f"    → {'一致（回帰OK）' if max_abs_diff < 1e-9 else '不一致！'}")

# ---- (b) 条件付き二項 vs 素朴ポアソン: 薄いベースラインで lambda0 過小推定の単位 ----
# ベースライン窓: 12か月でたまたま使用数ゼロが続き lambda0 が極小に推定される低頻度部品
np.random.seed(0)
fleet = 1500.0
# ベースライン窓 12か月: 使用数合計 C=2（ほぼゼロ）, 露出 E=12*fleet
C, E = 2.0, 12 * fleet
lam0_hat = C / E   # 過小推定気味の自己ベースライン
# 当月: 5個出た（真の平常が月3〜4個なら何でもない値だが、過小推定lam0だと異常に見える）
x_obs, e_t = 5.0, fleet

# 素朴ポアソン（lam0を真値扱い）
mu0 = lam0_hat * e_t
p_naive = cm.poisson_sf(x_obs, mu0)
# 条件付き二項（ベースラインの推定不確実性を織り込む）
_, _, _, _ = (None,)*4
p_s, a_s, _, _ = cm.spike_test(np.array([x_obs]), np.array([e_t]), lam0_hat,
                               alpha_spike=cfg["alpha_spike"], min_count=cfg["min_count"],
                               baseline_count=C, baseline_exposure=E)
p_cond = float(p_s[0])
alpha = cfg["alpha_spike"]
print(f"\n(b) 薄いベースライン(C={C:.0f}, E={E:.0f} → lam0_hat={lam0_hat:.2e})、当月 x={x_obs:.0f} のスパイクp値")
print(f"    素朴ポアソン   p = {p_naive:.5f}  → {'誤発火(p<=α)' if p_naive<=alpha else '非発火'}")
print(f"    条件付き二項   p = {p_cond:.5f}  → {'発火' if p_cond<=alpha else '抑制(p>α)'}  (α={alpha})")
print(f"    → 素朴は鳴り、条件付き二項は{'鳴らない' if p_cond>alpha else '鳴る'}: 推定誤差を織り込んで誤報を抑制")
