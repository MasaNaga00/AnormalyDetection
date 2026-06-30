# -*- coding: utf-8 -*-
"""フェーズ3: 統合 + asof再現性。
(A) 安定期と安定化前が1回の実行で共存し、台帳のリセット注入・状態解決がモードに依らず効く。
(B1) asof再現性: 同一パネルで asof を変えても、asof以前の月の判定(S/発火/状態)は不変。
(B2) リアルタイム再現: パネルを asof で打ち切っても、安定期は完全一致。安定化前のカーブ依存性を測る。"""
import math, numpy as np, pandas as pd
import state_logic_cusum as s

rng = np.random.default_rng(3)
def brate(t): return 0.0012 + 0.0050*math.exp(-t/3.0)
def gen(dev, part, sf, n, start, mult=1.0, drift_from=None, drift_mult=2.6,
        early=False, spike_at=None, spike_add=0):
    # 安定期も安定化前も同じ初期不良カーブ brate(t)（初期高→減衰→安定）に従う。
    rows=[]
    for t in range(n):
        F = (300+160*t) if early else (800+70*t)
        rate = brate(t)*mult
        if drift_from is not None and t >= drift_from: rate *= drift_mult
        cnt = rng.poisson(rate*F)
        if spike_at is not None and t==spike_at: cnt += spike_add
        rows.append(dict(事業コード="E1", 開発コード=dev, 部番=part, 販社="D0",
                         **{"SF-コード": sf, "ランク":"B"},
                         年月=s._add_months(start,t), 経過月=t,
                         月次使用数=int(cnt), 累積販売台数=int(F)))
    return rows

rows=[]
# 安定期の先行機種3つ（30ヶ月, 2022-01〜）= 安定化前のカーブ源 兼 監視対象
rows+=gen("L1","P1","SF-100",30,202201)
rows+=gen("L2","P1","SF-100",30,202201)
rows+=gen("L3","P1","SF-100",30,202201)
# 安定期・ドリフト機種（経過月20からレート2.5倍）→ 台帳『対策中』
rows+=gen("DR","P1","SF-100",30,202201,drift_from=20,drift_mult=2.8)
# 安定期・ノイズ単月スパイク → 台帳『ノイズ』
rows+=gen("NZ","P1","SF-100",30,202201,spike_at=18,spike_add=18)
# 安定化前・正常/異常（9ヶ月, 2024-01〜）
rows+=gen("EOK","P1","SF-100",9,202401,early=True,mult=1.0)
rows+=gen("ENG","P1","SF-100",9,202401,early=True,mult=2.8)

panel_raw = pd.DataFrame(rows)
cfg=dict(s.CONFIG); cfg["h"]=4.0; cfg["alpha_spike"]=0.01; cfg["min_count"]=3; cfg["burst_window"]=3

# 台帳: DR=対策中(発火月付近), NZ=ノイズ
def build(panel_raw, cfg):
    panel=s._prepare_panel(panel_raw,cfg); return s.aggregate_units(panel)
units=build(panel_raw,cfg)
# DR/NZ の発火月を見て台帳判定年月を決める
t0,_=s.evaluate_units(units, s.load_ledger(cfg, df=pd.DataFrame(columns=["事業コード","開発コード","部番","判定年月","記録日","処置区分"])), cfg, int(units["ym"].max()))
dr_fire=t0[(t0["dev"]=="DR")&(t0["total_alert"])]["ym"].min()
nz_fire=t0[(t0["dev"]=="NZ")&(t0["total_alert"])]["ym"].min()
ledger=s.load_ledger(cfg, df=pd.DataFrame([
    dict(事業コード="E1",開発コード="DR",部番="P1",判定年月=int(dr_fire),記録日=int(dr_fire),処置区分="対策中",再評価年月=None),
    dict(事業コード="E1",開発コード="NZ",部番="P1",判定年月=int(nz_fire),記録日=int(nz_fire),処置区分="ノイズ"),
]))
B=int(units["ym"].max())

# ---- (A) 統合 ----
tableB, metaB = s.evaluate_units(units, ledger, cfg, B)
print(f"=== (A) 統合: 基準月 {B}（安定期5 + 安定化前2 が共存）===")
print(f"{'機種':>5} {'モード':>14} {'最新状態':>14} {'CUSUM':>8} {'リセット':>6}")
for dev in ["L1","DR","NZ","EOK","ENG"]:
    sub=tableB[tableB["dev"]==dev].sort_values("ym"); key=("E1",dev,"P1")
    last=sub.iloc[-1]
    print(f"{dev:>5} {metaB[key]['mode']:>14} {last['state']:>14} {last['S']:>8.2f} {metaB[key]['reset_count']:>6}")
print(f"  DR発火月={dr_fire}(→対策中でリセット), NZ発火月={nz_fire}(→ノイズでリセット)")

# ---- (B1) asof再現性: 同一パネル, asof=A vs B ----
A=202405
tableA,_=s.evaluate_units(units, ledger, cfg, A)
m=tableA.merge(tableB, on=["dev","part","ym"], suffixes=("_A","_B"))
m=m[m["ym"]<=A]
dS=np.abs(m["S_A"]-m["S_B"]).max()
dAlarm=(m["total_alert_A"]!=m["total_alert_B"]).sum()
dState=(m["state_A"]!=m["state_B"]).sum()
print(f"\n=== (B1) asof再現性（同一パネル, asof={A} と {B} で ym<={A} を比較）===")
print(f"  Sの最大差={dS:.2e}  発火不一致={dAlarm}件  状態不一致={dState}件")
print(f"  → {'完全一致（状態を保存せず台帳から再生＝過去判定が再現）' if dS<1e-12 and dAlarm==0 and dState==0 else '不一致あり'}")

# ---- (B2) リアルタイム再現: パネルを<=Aで打ち切り ----
panel_trunc = panel_raw[panel_raw["年月"].map(s.to_yyyymm)<=A]
units_t = build(panel_trunc, cfg)
tableAt,_=s.evaluate_units(units_t, ledger, cfg, A)
m2=tableA.merge(tableAt, on=["dev","part","ym"], suffixes=("_full","_trunc"))
for grp,label in [(["L1","L2","L3","DR","NZ"],"安定期"),(["EOK","ENG"],"安定化前")]:
    mm=m2[m2["dev"].isin(grp)]
    if mm.empty: continue
    d=np.abs(mm["S_full"]-mm["S_trunc"]).max()
    print(f"\n=== (B2) リアルタイム再現 {label}: フル vs <=A打ち切り (ym<={A}) ===")
    print(f"  Sの最大差={d:.2e}  → {'完全一致' if d<1e-9 else f'差あり（カーブ再推定の影響）'}")
