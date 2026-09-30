# -*- coding: utf-8 -*-
"""統合インボックスの動作確認: 3検出器が同居し、台帳で抑制されるか。"""
import numpy as np, pandas as pd
import unified_inbox as ui

rng = np.random.default_rng(3)
MONTHS, START = 40, 202101
DISTS = {"A":0.35,"B":0.25,"C":0.20,"D":0.15,"E":0.05}
def ym(k):
    y,m = divmod(START,100); i = y*12+(m-1)+k
    return (i//12)*100+(i%12)+1

# SF-100: 5機種。M3 は高水準(信号B)。M0 は累積率が高い(閾値)。M1は販社Cで最終月にスパイク(信号C)
SPEC = {"M0":(0.0045,1.0,None), "M1":(0.0012,1.0,("C",MONTHS-1,6.0)),
        "M2":(0.0012,1.0,None), "M3":(0.0012,3.0,None), "M4":(0.0012,1.0,None)}
rows=[]
for dev,(lam,fac,spk) in SPEC.items():
    for t in range(MONTHS):
        Y=ym(t); F=3000+120*t; tu=0
        for dist,frac in DISTS.items():
            Fd=int(round(F*frac)); l=lam*fac
            if spk and dist==spk[0] and t==spk[1]: l*=spk[2]
            c=int(rng.poisson(max(l*Fd,0.0)))
            rows.append(dict(事業コード="E1",開発コード=dev,部番=f"{dev}-P0",販社=dist,
                             **{"SF-コード":"SF-100"},年月=Y,経過月=t,月次使用数=c,累積販売台数=Fd))
            tu+=c
        rows.append(dict(事業コード="E1",開発コード=dev,部番=f"{dev}-P0",販社="ALL",
                         **{"SF-コード":"SF-100"},年月=Y,経過月=t,月次使用数=tu,累積販売台数=F))
raw=pd.DataFrame(rows)
p_all=raw[raw.販社=="ALL"].copy(); p_dist=raw[raw.販社!="ALL"].copy()
cols=dict(biz="事業コード",dev="開発コード",part="部番",dist="販社",ym="年月",
          elapsed="経過月",monthly_use="月次使用数",cum_sales="累積販売台数",sf="SF-コード")

cfg=dict(ui.CONFIG); cfg.update(base_threshold_pct=1.5, b_elapsed_cap=36, c_alpha=0.005)

print("=== 1回目: 空台帳 ===")
r1=ui.build_unified_inbox(p_all,p_dist,ui.empty_ledger(),cfg,cols)
print(r1["inbox"][["開発コード","部番","検出器","対象販社","指標","統合注目度"]].to_string(index=False))

print("\n=== 2回目: M3 を『対策中』で記録（再評価=+3ヶ月）→ 抑制されるか ===")
led=ui.empty_ledger()
led.loc[0]={**{c:np.nan for c in ui.LEDGER_COLS},
            "記録日":"2024-04-01","事業コード":"E1","開発コード":"M3","部番":"M3-P0",
            "検出器":"信号B","判定年月":r1["asof"],"処置区分":"対策中",
            "再評価年月":ui.sc._add_months(r1["asof"],3)}
r2=ui.build_unified_inbox(p_all,p_dist,led,cfg,cols)
print(r2["inbox"][["開発コード","部番","検出器","統合注目度"]].to_string(index=False))
