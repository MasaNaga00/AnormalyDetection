"""
修理データ特徴カード Step 1（構造化データのみ・コメント未使用）

対象: 検知された交換部番を含む修理データ（1修理 = 1行、交換部番は1セルに複数）
比較: 同じデータ内で「直近期間」vs「過去期間」
出力: Excel 1ファイル（各シートとも縦持ち＝Tableauでそのまま読める形）

依存: pandas, numpy, openpyxl（scipy 不使用）
"""
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 設定（ここだけ編集）
# ============================================================
REPAIR_FILE = "repair.xlsx"        # 修理データ（部番一致で抽出済み）
SNTYPE_FILE = "sn_type.xlsx"       # 開発コード → SN type (B/C)
PRODMAP_FILE = None                # 任意: 開発コード×プレフィックス → 製造年月（無ければ None）
OUT_FILE = "feature_card.xlsx"

TARGET_PART = "ABC-1234"           # 検知された部番（同時交換部品から除外するため）

# 列名マッピング（左は内部名・右を実データの列名に合わせる）
COLS = {
    "repair_id": "修理ID",
    "model": "開発コード",
    "date": "修理日",
    "country": "国",
    "serial": "シリアル",
    "parts": "交換部番",
}
SNTYPE_COLS = {"model": "開発コード", "sn_type": "SN type"}
PRODMAP_COLS = {"model": "開発コード", "prefix": "プレフィックス", "prod_ym": "製造年月"}

PARTS_SEP = r"[,、;/\s]+"          # 交換部番セル内の区切り
SN_DIGITS = {"B": 2, "C": 3}       # SN type → 頭何桁が製造年月

RECENT_MONTHS = 6                  # 直近期間の長さ（月）
END_MONTH = None                   # 直近期間の最終月 "2026-09"。None=データ最大月
BASE_MONTHS = None                 # 過去期間の長さ（月）。None=直近より前すべて

P_FLAG = 0.05                      # 片側Fisher p値の目安
MIN_COUNT = 3                      # 直近件数がこれ未満は印を付けない
AGE_BINS = [0, 6, 12, 24, 36, 60, 999]  # 使用期間ビン（月）


# ============================================================
# 統計ヘルパー
# ============================================================
def _log_comb(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_upper(a, n_recent, k_total, n_total):
    """直近で a 件以上になる確率（超幾何・片側）。"""
    hi = min(k_total, n_recent)
    if a > hi:
        return 1.0
    denom = _log_comb(n_total, n_recent)
    xs = np.arange(a, hi + 1)
    logp = np.array([_log_comb(k_total, x) + _log_comb(n_total - k_total, n_recent - x)
                     for x in xs]) - denom
    m = logp.max()
    return float(min(1.0, math.exp(m) * np.exp(logp - m).sum()))


def binom_upper(a, n, q):
    """二項分布で a 件以上になる確率。"""
    if a <= 0:
        return 1.0
    if q <= 0:
        return 0.0
    if q >= 1:
        return 1.0
    xs = np.arange(a, n + 1)
    logp = np.array([_log_comb(n, x) for x in xs]) + xs * math.log(q) + (n - xs) * math.log(1 - q)
    m = logp.max()
    return float(min(1.0, math.exp(m) * np.exp(logp - m).sum()))


def add_flag(df, p_col="p値(片側)", n_col="直近件数", lift_col="lift"):
    """表内のカテゴリ数で Bonferroni 補正 → ★。"""
    m = max(int((df[n_col] >= MIN_COUNT).sum()), 1)
    df["p値(補正)"] = np.minimum(1.0, df[p_col] * m)
    df["注目"] = np.where((df["p値(補正)"] < P_FLAG) & (df[n_col] >= MIN_COUNT)
                        & (df[lift_col] > 1), "★", "")
    return df


def compare(cnt_recent, cnt_base, n_recent, n_base, label, flag=True):
    """カテゴリ別件数（直近・過去）→ 割合・lift・p値の表。"""
    df = pd.concat([cnt_recent.rename("直近件数"), cnt_base.rename("過去件数")], axis=1).fillna(0)
    df = df.astype(int)
    df.index.name = label
    df["直近割合"] = df["直近件数"] / max(n_recent, 1)
    df["過去割合"] = df["過去件数"] / max(n_base, 1)
    # 0件対策で0.5を足した lift
    df["lift"] = ((df["直近件数"] + 0.5) / (n_recent + 1)) / ((df["過去件数"] + 0.5) / (n_base + 1))
    n_total = n_recent + n_base
    df["p値(片側)"] = [fisher_upper(a, n_recent, a + b, n_total)
                     for a, b in zip(df["直近件数"], df["過去件数"])]
    if not flag:
        return df.reset_index()
    df = add_flag(df)
    return df.reset_index().sort_values(["注目", "p値(片側)"], ascending=[False, True])


def lot_table(d, recent_start, end, obs_start):
    """
    製造ロット比較（経過時間補正あり）。
    新しいロットは直近に修理が集中するのが当然なので、単純な直近/過去比較は使わない。
    機種全体の「使用期間ごとの故障件数」を形状として、各ロットの製造年月から
    「直近期間に入るはずの割合 q」を計算し、実際の直近件数と二項検定で比べる。
    製造年月が不明なロットは件数のみ表示（判定なし）。
    """
    x = d[d["期間"] != "対象外"].dropna(subset=["prefix"])
    t = x.groupby("prefix").agg(
        直近件数=("期間", lambda s: int((s == "直近").sum())),
        過去件数=("期間", lambda s: int((s == "過去").sum())),
        製造年月=("prod_ym", "first"),
        初出修理月=("repair_ym", "min"),
    )
    t["合計"] = t["直近件数"] + t["過去件数"]

    q = pd.Series(np.nan, index=t.index)
    months = pd.date_range(obs_start, end, freq="MS")
    is_recent = months >= recent_start
    prods = t["製造年月"].dropna()
    if len(prods):
        # 各ロットの「月ごとの使用期間」行列（観測窓内のみ）
        age_mat = {lot: (months.year - p.year) * 12 + (months.month - p.month) for lot, p in prods.items()}
        max_age = max(int(a.max()) for a in age_mat.values()) + 1
        # 使用期間ごとの故障件数 / その使用期間を観測できたロット数（観測打ち切りの補正）
        cnt_lot = {lot: np.zeros(max_age, dtype=int) for lot in age_mat}
        xa = x.dropna(subset=["使用期間_月"])
        for lot, a in xa.groupby("prefix")["使用期間_月"]:
            if lot in cnt_lot:
                cnt_lot[lot] = np.bincount(a.astype(int), minlength=max_age)[:max_age]
        obs_lot = {lot: np.bincount(a[a >= 0], minlength=max_age)[:max_age] for lot, a in age_mat.items()}
        cnt_all = sum(cnt_lot.values())
        obs_all = sum(obs_lot.values())
        for lot in prods.index:
            # 検定対象ロット自身は形状推定から除く（leave-one-out）
            c = cnt_all - cnt_lot[lot]
            o = obs_all - obs_lot[lot]
            g = np.where(o > 0, (c + 0.5) / np.maximum(o, 1), 0.0)
            a = age_mat[lot]
            w = np.where(a >= 0, g[np.clip(a, 0, max_age - 1)], 0.0)
            if w.sum() > 0:
                q[lot] = w[is_recent].sum() / w.sum()
    t["期待直近割合"] = q
    t["実直近割合"] = t["直近件数"] / t["合計"].clip(lower=1)
    t["lift"] = (t["直近件数"] + 0.5) / (t["合計"] * t["期待直近割合"] + 0.5)
    t["p値(片側)"] = [binom_upper(a, n, qq) if pd.notna(qq) else np.nan
                    for a, n, qq in zip(t["直近件数"], t["合計"], t["期待直近割合"])]
    has = t["期待直近割合"].notna()
    t["p値(補正)"], t["注目"] = np.nan, ""
    if has.any():
        f = add_flag(t.loc[has].copy())
        t.loc[has, "p値(補正)"] = f["p値(補正)"]
        t.loc[has, "注目"] = f["注目"]
    t.loc[~has, "注目"] = "(製造年月不明・判定なし)"
    t["製造年月"] = t["製造年月"].dt.strftime("%Y-%m")
    t["初出修理月"] = t["初出修理月"].dt.strftime("%Y-%m")
    t.index.name = "プレフィックス"
    t["_k"] = t["注目"].map({"★": 0, "": 1}).fillna(2)
    return (t.reset_index().sort_values(["_k", "p値(片側)", "直近件数"], ascending=[True, True, False])
            .drop(columns="_k"))


# ============================================================
# 読み込み・前処理
# ============================================================
def read_any(path):
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p, dtype=str)
    return pd.read_csv(p, dtype=str)


def to_ym(s):
    """日付文字列 → 月初 Timestamp（失敗は NaT）。"""
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp()


def load(log):
    raw = read_any(REPAIR_FILE)
    missing = [v for v in COLS.values() if v not in raw.columns]
    if missing:
        raise KeyError(f"修理データに列がありません: {missing}  実在列: {list(raw.columns)}")
    df = raw.rename(columns={v: k for k, v in COLS.items()})
    for c in ("model", "country", "serial", "parts"):
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["country"] = df["country"].replace("", "(空欄)")
    df["repair_ym"] = to_ym(df["date"])
    log.append(("修理データ行数", len(df)))
    log.append(("修理日の解釈失敗", int(df["repair_ym"].isna().sum())))

    # 重複修理ID
    dup = int(df["repair_id"].duplicated().sum())
    log.append(("修理ID重複（除外）", dup))
    df = df.drop_duplicates("repair_id")

    # SN type
    snt = read_any(SNTYPE_FILE).rename(columns={v: k for k, v in SNTYPE_COLS.items()})
    snt["model"] = snt["model"].astype(str).str.strip()
    snt["sn_type"] = snt["sn_type"].astype(str).str.strip().str.upper()
    df = df.merge(snt[["model", "sn_type"]].drop_duplicates("model"), on="model", how="left")
    df["digits"] = df["sn_type"].map(SN_DIGITS)
    log.append(("SN type 不明（ロット解析対象外）", int(df["digits"].isna().sum())))

    # シリアル → プレフィックス（数値化で落ちた ".0" を除去）
    ser = df["serial"].str.replace(r"\.0$", "", regex=True)
    df["prefix"] = [s[: int(d)] if pd.notna(d) and len(s) >= d else np.nan
                    for s, d in zip(ser, df["digits"])]
    log.append(("プレフィックス取得不可", int(df["prefix"].isna().sum())))

    # 任意: プレフィックス → 製造年月
    df["prod_ym"] = pd.NaT
    if PRODMAP_FILE:
        pm = read_any(PRODMAP_FILE).rename(columns={v: k for k, v in PRODMAP_COLS.items()})
        pm["model"] = pm["model"].astype(str).str.strip()
        pm["prefix"] = pm["prefix"].astype(str).str.strip()
        pm["prod_ym"] = to_ym(pm["prod_ym"])
        df = df.drop(columns="prod_ym").merge(
            pm[["model", "prefix", "prod_ym"]].drop_duplicates(["model", "prefix"]),
            on=["model", "prefix"], how="left")
    log.append(("製造年月 変換できた行", int(df["prod_ym"].notna().sum())))

    # 使用期間（月）
    df["使用期間_月"] = ((df["repair_ym"].dt.year - df["prod_ym"].dt.year) * 12
                     + (df["repair_ym"].dt.month - df["prod_ym"].dt.month))
    neg = int((df["使用期間_月"] < 0).sum())
    log.append(("使用期間が負（変換表かシリアルを要確認）", neg))
    df.loc[df["使用期間_月"] < 0, "使用期間_月"] = np.nan

    # 同時交換部品
    tgt = TARGET_PART.strip().upper()
    df["co_parts"] = [sorted({p for p in re.split(PARTS_SEP, s.upper()) if p and p != tgt})
                      for s in df["parts"]]
    return df


def assign_period(df, log):
    end = pd.Timestamp(END_MONTH) if END_MONTH else df["repair_ym"].max()
    start = end - pd.DateOffset(months=RECENT_MONTHS - 1)
    base_start = start - pd.DateOffset(months=BASE_MONTHS) if BASE_MONTHS else pd.Timestamp.min
    df["期間"] = np.select(
        [(df["repair_ym"] >= start) & (df["repair_ym"] <= end),
         (df["repair_ym"] >= base_start) & (df["repair_ym"] < start)],
        ["直近", "過去"], default="対象外")
    obs_start = max(base_start, df["repair_ym"].min())
    log.append(("直近期間", f"{start:%Y-%m} 〜 {end:%Y-%m}"))
    log.append(("過去期間", f"{obs_start:%Y-%m} 〜 {start - pd.DateOffset(months=1):%Y-%m}"))
    return df, (start, end, obs_start)


# ============================================================
# 機種ごとのカード
# ============================================================
def card_for_model(d, period):
    recent_start, end, obs_start = period
    r, b = d[d["期間"] == "直近"], d[d["期間"] == "過去"]
    nr, nb = len(r), len(b)
    out = {}

    out["国"] = compare(r["country"].value_counts(), b["country"].value_counts(), nr, nb, "国")

    lot = lot_table(d, recent_start, end, obs_start)
    out["製造ロット"] = lot

    # 使用期間ビン（参考表示のみ・判定なし）
    #   発売からの年数が経つほど長い使用期間の修理が自然に増えるため、直近/過去の差は判定に使わない
    labels = [f"{AGE_BINS[i]}-{AGE_BINS[i+1]}ヶ月" for i in range(len(AGE_BINS) - 1)]
    age = d.dropna(subset=["使用期間_月"]).copy()
    age["使用期間区分"] = pd.cut(age["使用期間_月"], AGE_BINS, right=False, labels=labels)
    ar, ab = age[age["期間"] == "直近"], age[age["期間"] == "過去"]
    if len(age):
        u = compare(ar["使用期間区分"].value_counts(sort=False), ab["使用期間区分"].value_counts(sort=False),
                    len(ar), len(ab), "使用期間区分", flag=False)
        out["使用期間"] = u[["使用期間区分", "直近件数", "過去件数", "直近割合", "過去割合"]].assign(
            直近中央値_月=ar["使用期間_月"].median(), 過去中央値_月=ab["使用期間_月"].median())

    # 同時交換部品（1修理で同じ部品は1回と数える）
    out["同時交換部品"] = compare(r["co_parts"].explode().dropna().value_counts(),
                            b["co_parts"].explode().dropna().value_counts(), nr, nb, "同時交換部番")

    # 国 × ロット（直近・過去とも、件数のみ）
    out["国x製造ロット"] = (d[d["期間"] != "対象外"].dropna(subset=["prefix"])
                       .groupby(["期間", "country", "prefix"]).size()
                       .rename("件数").reset_index()
                       .rename(columns={"country": "国", "prefix": "プレフィックス"}))

    summary = {"直近件数": nr, "過去件数": nb,
               "注目_国": ", ".join(out["国"].loc[out["国"]["注目"] == "★", "国"].astype(str)),
               "注目_ロット": ", ".join(lot.loc[lot["注目"] == "★", "プレフィックス"].astype(str)),
               "製造年月不明ロット数": int((lot["注目"] == "(製造年月不明・判定なし)").sum()),
               "注目_同時交換": ", ".join(out["同時交換部品"].loc[
                   out["同時交換部品"]["注目"] == "★", "同時交換部番"].astype(str))}
    return out, summary


def main():
    log = []
    df = load(log)
    df, period = assign_period(df, log)

    sheets, summaries = {}, []
    for model, d in df.groupby("model"):
        out, s = card_for_model(d, period)
        summaries.append({"開発コード": model, "対象部番": TARGET_PART, **s})
        for name, t in out.items():
            t = t.copy()
            t.insert(0, "開発コード", model)
            sheets.setdefault(name, []).append(t)

    detail = df.assign(
        修理年月=df["repair_ym"].dt.strftime("%Y-%m"),
        製造年月=df["prod_ym"].dt.strftime("%Y-%m"),
        同時交換部品=df["co_parts"].str.join(", "),
    )[["repair_id", "model", "修理年月", "期間", "country", "serial", "sn_type",
       "prefix", "製造年月", "使用期間_月", "同時交換部品"]]
    detail = detail.rename(columns={"repair_id": COLS["repair_id"], "model": COLS["model"],
                                    "country": COLS["country"], "serial": COLS["serial"]})

    with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as w:
        pd.DataFrame(summaries).to_excel(w, sheet_name="概要", index=False)
        for name, ts in sheets.items():
            pd.concat(ts, ignore_index=True).to_excel(w, sheet_name=name, index=False)
        detail.to_excel(w, sheet_name="明細", index=False)
        pd.DataFrame(log, columns=["項目", "値"]).to_excel(w, sheet_name="読込ログ", index=False)

    for k, v in log:
        print(f"{k}: {v}")
    print(f"出力: {OUT_FILE}")


if __name__ == "__main__":
    main()
