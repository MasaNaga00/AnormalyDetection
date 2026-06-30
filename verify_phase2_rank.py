# -*- coding: utf-8 -*-
"""フェーズ2: ランクが一部の機種にしか無い場合の階層フォールバック検証。
階層 [["biz","sf","rank"],["biz","sf"]]:
  - ランクで先行機種が十分 → ランク単位 (biz+sf+rank)
  - ランクが薄い/欠損        → SF単位 (biz+sf) に自動フォールバック
  - SF単位でも先行機種不足   → 監視保留"""
import math, numpy as np, pandas as pd
import state_logic_cusum as m

rng = np.random.default_rng(11)
def base_rate(t): return 0.0012 + 0.0050*math.exp(-t/3.0)
def fleet_at(t): return 300 + 180*t
def gen(dev, part, sf, rank, n, start, mult=1.0, jit=1.0):
    out=[]
    for t in range(n):
        F=fleet_at(t); u=int(rng.poisson(base_rate(t)*mult*jit*F))
        row=dict(事業コード="E1", 開発コード=dev, 部番=part, 販社="D0",
                 **{"SF-コード": sf}, 年月=m._add_months(start,t), 経過月=t,
                 月次使用数=u, 累積販売台数=F)
        if rank is not None:           # ランク欠損を None で表現（列はあるが値が無い）
            row["ランク"]=rank
        else:
            row["ランク"]=np.nan
        out.append(row)
    return out

rows=[]
# SF-100: ランクA の先行機種3つ（→ランク単位が成立）
rows+=gen("A1","PART-X","SF-100","A",24,202101,jit=0.95)
rows+=gen("A2","PART-X","SF-100","A",24,202101,jit=1.0)
rows+=gen("A3","PART-X","SF-100","A",24,202101,jit=1.1)
# SF-100: ランクB の先行機種は1つだけ（→ランク単位は薄い）＋ ランク欠損の先行機種2つ
rows+=gen("B1","PART-X","SF-100","B",24,202101,jit=1.0)
rows+=gen("M1","PART-X","SF-100",None,24,202101,jit=1.05)
rows+=gen("M2","PART-X","SF-100",None,24,202101,jit=0.9)
# 監視対象（安定化前、9ヶ月）
rows+=gen("mon-A","PART-X","SF-100","A",9,202401,mult=1.0)    # ランクA → ランク単位を使うはず
rows+=gen("mon-B","PART-X","SF-100","B",9,202401,mult=1.0)    # ランクBは薄い → SF単位へ
rows+=gen("mon-none","PART-X","SF-100",None,9,202401,mult=1.0)# ランク欠損 → SF単位へ
rows+=gen("mon-NG","PART-X","SF-100","A",9,202401,mult=2.6)   # ランクA異常 → ランク単位で点灯

cfg=dict(m.CONFIG); cfg["h"]=4.0; cfg["alpha_spike"]=0.01; cfg["min_count"]=3; cfg["burst_window"]=3
panel=m._prepare_panel(pd.DataFrame(rows), cfg)
units=m.aggregate_units(panel)
ledger=m.load_ledger(cfg, df=pd.DataFrame(columns=["事業コード","開発コード","部番","判定年月","記録日","処置区分"]))
asof=int(units["ym"].max())

levels=m.build_earlylife_curves(units, cfg)
print("構築された階層と先行機種数:")
for lv in levels:
    print(f"  {'+'.join(lv['keys'])}: {lv['counts']}")

table, meta = m.evaluate_units(units, ledger, cfg, asof)
print("\n監視対象がどの階層に落ちたか:")
print(f"{'機種':>10} {'ランク':>5} {'採用レベル':>14} {'先行機種数':>8} {'判定':>16}")
for dev, rk in [("mon-A","A"),("mon-B","B"),("mon-none","(欠損)"),("mon-NG","A")]:
    sub=table[table["dev"]==dev].sort_values("ym"); key=("E1",dev,"PART-X")
    fired=sub[sub["total_alert"]]
    first=int(fired["elapsed"].min()) if not fired.empty else None
    verdict = "鳴らない(正常)" if first is None else f"経過月{first}で点灯"
    print(f"{dev:>10} {rk:>5} {str(meta[key].get('earlylife_level')):>14} "
          f"{str(meta[key].get('leader_count')):>8} {verdict:>16}")

print("\n期待: mon-A,mon-NG=biz+sf+rank(3) / mon-B,mon-none=biz+sf(6) にフォールバック / mon-NGのみ点灯")
