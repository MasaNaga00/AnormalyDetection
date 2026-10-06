"""
修理コメント解析 Step 2-① 探索（分類の型づくり）

流れ:
  extract : サンプル抽出 → Dify(抽出アプリ)で 症状/状況/所見 の自由フレーズ化 → キャッシュ
  propose : フレーズ頻度 → Dify(カテゴリ案アプリ)で軸ごとにカテゴリ案 → カテゴリ辞書_案.xlsx
  (引数なし = 両方)

  python 20261006_step2_explore.py            # 両方
  python 20261006_step2_explore.py extract    # 抽出だけ（結果を見てから propose したい場合）
  python 20261006_step2_explore.py propose

APIキーは環境変数 DIFY_KEY_EXTRACT / DIFY_KEY_CATEGORY に設定（コードに書かない）。
依存: pandas, numpy, openpyxl（HTTP は標準ライブラリ urllib）
"""
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 設定
# ============================================================
REPAIR_FILE = "repair.xlsx"
EXISTING_CAT_FILE = None          # 既存カテゴリ（列: 軸, カテゴリ, 定義）。無ければ None
OUT_DIR = Path("step2_out")

COLS = {"repair_id": "修理ID", "model": "開発コード", "date": "修理日", "country": "国"}
COMMENT_COLS = ["お客様コメント", "修理者コメント"]   # 結合して1テキストにする（欄の区別は信用しない）

N_SAMPLE = 150                    # 探索に使う件数
RECENT_MONTHS = 6                 # 直近期間（サンプルの半分を直近から取る）
MIN_CHARS = 5                     # これ未満のコメントは探索から除外
MAX_CHARS = 1500                  # 1件あたりのテキスト上限
SEED = 0

# エラーコード（言語非依存なので LLM を使わず正規表現で全件から抽出）
#   前後が英数字でないことを条件にする（日本語に隣接しても拾えるよう \b は使わない）
CODE_PATTERNS = [r"(?<![A-Za-z0-9])Err(?:or)?\s*[-:#]?\s*\d{1,3}(?![0-9])",
                 r"(?<![A-Za-z0-9])E[-\s]?\d{2,3}(?![0-9])"]

DIFY_BASE_URL = "https://dify.example.co.jp/v1"   # 社内 Dify の API ベース URL
DIFY_OUTPUT_VAR = "result"        # 終了ノードの出力変数名
CA_BUNDLE = None                  # 社内証明書が必要ならパス
BATCH_SIZE = 8
MAX_RETRY = 3
SLEEP_SEC = 0.5
TIMEOUT = 180
MAX_PHRASES = 400                 # カテゴリ案に渡すフレーズ上限（頻度順）

AXES = {"symptom": "症状", "situation": "状況", "finding": "所見"}
FIXED_LABELS = ["記載なし", "その他"]


# ============================================================
# Dify 呼び出し
# ============================================================
def call_workflow(api_key, inputs):
    """ワークフローを blocking 実行し、出力変数の文字列を返す。"""
    body = json.dumps({"inputs": inputs, "response_mode": "blocking", "user": "step2-explore"}).encode()
    req = urllib.request.Request(f"{DIFY_BASE_URL}/workflows/run", data=body, method="POST",
                                 headers={"Authorization": f"Bearer {api_key}",
                                          "Content-Type": "application/json"})
    ctx = ssl.create_default_context(cafile=CA_BUNDLE) if CA_BUNDLE else None
    with urllib.request.urlopen(req, timeout=TIMEOUT, context=ctx) as r:
        res = json.load(r)
    data = res.get("data", {})
    if data.get("status") != "succeeded":
        raise RuntimeError(f"workflow {data.get('status')}: {data.get('error')}")
    return str(data.get("outputs", {}).get(DIFY_OUTPUT_VAR, ""))


def call_with_retry(api_key, inputs):
    for i in range(MAX_RETRY):
        try:
            return call_workflow(api_key, inputs)
        except Exception as e:  # 通信・タイムアウト・応答不正をまとめて再試行

            msg = e.read().decode(errors="ignore")[:300] if isinstance(e, urllib.error.HTTPError) else str(e)
            print(f"  API失敗({i + 1}/{MAX_RETRY}): {msg}")
            time.sleep(2 ** i)
    return None


def parse_json(text):
    """```json フェンスや前後の文章を除いて JSON を取り出す。失敗は None。"""
    if not text:
        return None
    s = re.sub(r"```(?:json)?", "", text)
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j < i:
        return None
    try:
        return json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return None


# ============================================================
# 抽出（キャッシュ付き・ID照合・再送）
# ============================================================
def as_list(v):
    if isinstance(v, list):
        out = [str(x).strip() for x in v if str(x).strip()]
    elif v is None or str(v).strip() == "":
        out = []
    else:
        out = [str(v).strip()]
    return out or ["記載なし"]


def clean_result(r):
    return {"id": str(r.get("id", "")).strip(),
            **{k: as_list(r.get(k)) for k in AXES},
            "external": str(r.get("external", "不明")).strip() or "不明",
            "summary": str(r.get("summary", "")).strip(),
            "evidence": str(r.get("evidence", "")).strip()}


def run_extract(records, cache_path, api_key):
    """records: [{"id","text"}]。キャッシュ済みはスキップ。返り値: id→結果"""
    done = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["id"]] = r
    # 最初は BATCH_SIZE ずつ、欠落・失敗分は1件ずつ再送
    pending = [r for r in records if r["id"] not in done]
    batches = [pending[i:i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]
    tries = {r["id"]: 0 for r in pending}
    failed = []
    print(f"抽出: 対象 {len(records)} 件 / キャッシュ済 {len(done)} 件 / 送信 {len(pending)} 件")
    with cache_path.open("a", encoding="utf-8") as f:
        while batches:
            b = batches.pop(0)
            out = call_with_retry(api_key, {"records": json.dumps(b, ensure_ascii=False)})
            obj = parse_json(out)
            got = {}
            if obj and isinstance(obj.get("results"), list):
                for r in obj["results"]:
                    c = clean_result(r)
                    if c["id"] in {x["id"] for x in b}:
                        got[c["id"]] = c
            for c in got.values():
                done[c["id"]] = c
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
            f.flush()
            for r in b:
                if r["id"] in got:
                    continue
                tries[r["id"]] += 1
                if tries[r["id"]] < MAX_RETRY:
                    batches.append([r])          # 1件で再送
                else:
                    failed.append(r["id"])
            print(f"  {len(done)}/{len(records)} 完了", end="\r")
            time.sleep(SLEEP_SEC)
    print()
    return done, failed


# ============================================================
# データ準備
# ============================================================
def read_any(path):
    p = Path(path)
    return pd.read_excel(p, dtype=str) if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p, dtype=str)


def load_repairs(log):
    raw = read_any(REPAIR_FILE)
    need = list(COLS.values()) + COMMENT_COLS
    miss = [c for c in need if c not in raw.columns]
    if miss:
        raise KeyError(f"列がありません: {miss}  実在列: {list(raw.columns)}")
    df = raw.rename(columns={v: k for k, v in COLS.items()})
    df["repair_id"] = df["repair_id"].astype(str).str.strip()
    df = df.drop_duplicates("repair_id")
    parts, n_chars = [], 0
    for c in COMMENT_COLS:
        v = raw.loc[df.index, c].fillna("").astype(str).str.strip()
        n_chars = n_chars + v.str.len()
        parts.append(np.where(v != "", "[" + c + "] " + v, ""))
    df["text"] = [" ".join(p for p in ps if p)[:MAX_CHARS] for ps in zip(*parts)]
    df["n_chars"] = n_chars
    ym = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M")
    start = ym.max() - (RECENT_MONTHS - 1)
    df["期間"] = np.where(ym >= start, "直近", "過去")
    log += [("修理データ行数", len(df)),
            ("コメントが短すぎる/空（探索から除外）", int((df["n_chars"] < MIN_CHARS).sum())),
            ("直近期間", f"{start} 〜 {ym.max()}")]
    return df


def sample(df):
    ok = df[df["n_chars"] >= MIN_CHARS]
    n_r = min(N_SAMPLE // 2, (ok["期間"] == "直近").sum())
    n_b = min(N_SAMPLE - n_r, (ok["期間"] == "過去").sum())
    return pd.concat([ok[ok["期間"] == "直近"].sample(n_r, random_state=SEED),
                      ok[ok["期間"] == "過去"].sample(n_b, random_state=SEED)])


def error_codes(df):
    rows = []
    for rid, per, t in zip(df["repair_id"], df["期間"], df["text"]):
        for p in CODE_PATTERNS:
            for m in set(re.findall(p, t, flags=re.I)):
                rows.append({"repair_id": rid, "期間": per,
                             "コード": re.sub(r"[\s\-:#]", "", m).upper()})
    if not rows:
        return pd.DataFrame(columns=["コード", "直近", "過去", "合計"])
    e = pd.DataFrame(rows).drop_duplicates()
    t = e.pivot_table(index="コード", columns="期間", values="repair_id", aggfunc="count", fill_value=0)
    for c in ("直近", "過去"):
        if c not in t:
            t[c] = 0
    t["合計"] = t["直近"] + t["過去"]
    return t[["直近", "過去", "合計"]].sort_values("合計", ascending=False).reset_index()


# ============================================================
# フレーズ集計・カテゴリ案
# ============================================================
def norm(s):
    return re.sub(r"[\s「」『』。、・]", "", str(s))


def phrase_freq(res):
    rows = []
    for r in res.values():
        for k, ax in AXES.items():
            for p in set(r[k]):
                rows.append({"軸": ax, "フレーズ": p})
    f = pd.DataFrame(rows).value_counts().rename("件数").reset_index()
    return f.sort_values(["軸", "件数"], ascending=[True, False])


def propose(freq, res, api_key, existing):
    dict_rows, unassigned, maps = [], [], {}
    for k, ax in AXES.items():
        f = freq[(freq["軸"] == ax) & (~freq["フレーズ"].isin(FIXED_LABELS))].head(MAX_PHRASES)
        if f.empty:
            continue
        ex = existing[existing["軸"] == ax] if existing is not None else pd.DataFrame()
        ex_txt = "\n".join(f"{r['カテゴリ']}: {r.get('定義', '')}" for _, r in ex.iterrows()) or "（なし）"
        print(f"カテゴリ案: {ax}（フレーズ {len(f)} 種）")
        out = call_with_retry(api_key, {
            "axis": ax,
            "phrases": "\n".join(f"{p}\t{n}" for p, n in zip(f["フレーズ"], f["件数"])),
            "existing": ex_txt})
        obj = parse_json(out)
        if not obj or not isinstance(obj.get("categories"), list):
            print(f"  {ax}: カテゴリ案の取得に失敗（出力を確認してください）")
            continue
        m = {}
        for c in obj["categories"]:
            name = str(c.get("name", "")).strip()
            for p in c.get("phrases", []):
                m[norm(p)] = name
            dict_rows.append({"軸": ax, "カテゴリ": name, "定義": c.get("definition", ""),
                              "新設": "新設" if c.get("new") else "既存",
                              "含まれるフレーズ": " / ".join(map(str, c.get("phrases", [])))})
        maps[k] = m
        unassigned += [{"軸": ax, "フレーズ": p} for p in obj.get("unassigned", [])]

    # 記録単位の件数（1記録で同じカテゴリは1回）
    cnt = {}
    for r in res.values():
        for k, ax in AXES.items():
            cats = set()
            for p in r[k]:
                cats.add(p if p in FIXED_LABELS else maps.get(k, {}).get(norm(p), "未分類"))
            for c in cats:
                cnt[(ax, c)] = cnt.get((ax, c), 0) + 1
    d = pd.DataFrame(dict_rows)
    for ax in AXES.values():
        for lab in FIXED_LABELS:
            if d.empty or not ((d["軸"] == ax) & (d["カテゴリ"] == lab)).any():
                d = pd.concat([d, pd.DataFrame([{"軸": ax, "カテゴリ": lab, "新設": "固定",
                                                  "定義": "該当記述なし" if lab == "記載なし" else "どのカテゴリにも当てはまらない"}])])
    d["サンプル内件数"] = [cnt.get((a, c), 0) for a, c in zip(d["軸"], d["カテゴリ"])]
    d["_o"] = d["軸"].map({v: i for i, v in enumerate(AXES.values())})
    d["_f"] = (d["新設"] == "固定").astype(int)
    d = d.sort_values(["_o", "_f", "サンプル内件数"], ascending=[True, True, False]).drop(columns=["_o", "_f"])
    n_un = {ax: cnt.get((ax, "未分類"), 0) for ax in AXES.values()}
    return d.reset_index(drop=True), pd.DataFrame(unassigned), n_un


# ============================================================
def main(mode="all"):
    OUT_DIR.mkdir(exist_ok=True)
    cache = OUT_DIR / "extract_cache.jsonl"
    log = []
    df = load_repairs(log)
    smp = sample(df)
    log.append(("探索サンプル件数", len(smp)))

    if mode in ("all", "extract"):
        key = os.environ.get("DIFY_KEY_EXTRACT")
        if not key:
            sys.exit("環境変数 DIFY_KEY_EXTRACT が未設定です")
        recs = [{"id": i, "text": t} for i, t in zip(smp["repair_id"], smp["text"])]
        res, failed = run_extract(recs, cache, key)
        log.append(("抽出失敗（再送上限）", len(failed)))
        if failed:
            log.append(("失敗ID", ", ".join(failed[:50])))

    # キャッシュから結果を読み直す（サンプル分のみ）
    res = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            res[r["id"]] = r
    res = {i: res[i] for i in smp["repair_id"] if i in res}
    freq = phrase_freq(res) if res else pd.DataFrame(columns=["軸", "フレーズ", "件数"])

    ext = pd.DataFrame(res.values())
    if not ext.empty:
        for k, ax in AXES.items():
            ext[ax] = ext[k].str.join(" / ")
        ext = smp[["repair_id", "model", "country", "期間", "text"]].merge(
            ext.rename(columns={"id": "repair_id", "external": "外因", "summary": "要約", "evidence": "根拠"})
            [["repair_id", *AXES.values(), "外因", "要約", "根拠"]], on="repair_id", how="inner")
        ext = ext.rename(columns={"repair_id": COLS["repair_id"], "model": COLS["model"], "country": COLS["country"]})

    with pd.ExcelWriter(OUT_DIR / "step2_explore.xlsx", engine="openpyxl") as w:
        ext.to_excel(w, sheet_name="抽出結果", index=False)
        freq.to_excel(w, sheet_name="フレーズ頻度", index=False)
        error_codes(df).to_excel(w, sheet_name="エラーコード(全件)", index=False)
        pd.DataFrame(log, columns=["項目", "値"]).to_excel(w, sheet_name="読込ログ", index=False)

    if mode in ("all", "propose") and res:
        key = os.environ.get("DIFY_KEY_CATEGORY")
        if not key:
            sys.exit("環境変数 DIFY_KEY_CATEGORY が未設定です")
        existing = read_any(EXISTING_CAT_FILE) if EXISTING_CAT_FILE else None
        d, un, n_un = propose(freq, res, key, existing)
        with pd.ExcelWriter(OUT_DIR / "カテゴリ辞書_案.xlsx", engine="openpyxl") as w:
            d.to_excel(w, sheet_name="辞書", index=False)
            un.to_excel(w, sheet_name="未分類フレーズ", index=False)
        log += [(f"未分類の記録数_{ax}", n) for ax, n in n_un.items()]

    for k, v in log:
        print(f"{k}: {v}")
    print(f"出力: {OUT_DIR}/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
