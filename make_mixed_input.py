"""混在データ生成: 履歴の長い機種(安定期)と浅い機種(安定化前)を、
SF-コード・ランク付きで作る。振り分けランナーのデモ入力。"""
import numpy as np, pandas as pd

rng = np.random.default_rng(11)
rows = []

def earlylife_rate(months, floor, mult, decay):
    t = np.arange(months)
    return floor * (1 + (mult - 1) * np.exp(-t / decay))

def emit(dev, sf, rank, months, floor, mult, decay, scale, n_sha=2):
    fleet_tot = (np.linspace(300, 30000, months) * scale).round()
    rate = earlylife_rate(months, floor, mult, decay)
    usage_tot = rng.poisson(rate * fleet_tot)
    for m in range(months):
        u = int(usage_tot[m]); f = float(fleet_tot[m])
        # 2販社へ分割
        u1 = int(rng.binomial(u, 0.6)) if u>0 else 0
        splits = [("S01", u1, round(f*0.6)), ("S02", u-u1, f-round(f*0.6))]
        for sha, uu, ff in splits:
            rows.append({"事業コード":"BIZ1","SF-コード":sf,"ランク":rank,
                         "開発コード":dev,"部番":"100-0001-001","販社":sha,
                         "年月":202000+m,"経過月":m,
                         "月次使用数":uu,"累積販売台数":ff})

# SF100: ランクH/L 各複数の先行機種(履歴96か月=安定期)
for i in range(3):
    emit(f"H_OLD{i}","SF100","H",96, 1.2e-4, 3.8, 8, 1.3*rng.uniform(.9,1.1))
for i in range(3):
    emit(f"L_OLD{i}","SF100","L",96, 7e-5, 3.5, 8, 0.8*rng.uniform(.9,1.1))

# 若い新機種(履歴24か月=安定化前): 正常1・異常1 を各ランク
emit("H_NEW_OK","SF100","H",24, 1.2e-4, 3.8, 8, 1.3)        # 集団どおり
emit("H_NEW_BAD","SF100","H",24, 1.2e-4*1.6, 6.0, 8, 1.3)   # 初期から高い
emit("L_NEW_OK","SF100","L",24, 7e-5, 3.5, 8, 0.8)

# 先行機種が足りないSF(監視保留になるはず): SF200にランクHの若い機種だけ
emit("H_LONE","SF200","H",18, 1.0e-4, 4.0, 8, 1.0)

df = pd.DataFrame(rows)
df.to_csv("mixed_input.csv", index=False, encoding="utf-8-sig")
print("mixed_input.csv:", len(df), "行,", df["開発コード"].nunique(), "機種")
print(df.groupby(["SF-コード","ランク","開発コード"])["経過月"].max().to_string())
