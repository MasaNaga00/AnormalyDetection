"""
1. 最終確認（投入前の1回限りの作業）
CONFIGを固定してファイルに残す（口頭やノートブックの中だけで持たない）。
"""
import json
import state_logic_cusum as s

cfg = dict(s.CONFIG)
cfg.update(R=..., h=..., alpha_spike=..., min_count=..., burst_window=...)  # 決定値

# JSON化できない値（Noneやパス）は除いて、パラメータ部分だけ保存
frozen = {k: v for k, v in cfg.items()
         if k in ("R","h","alpha_spike","min_count","burst_window",
                  "reset_after_alarm","stable_start_m","baseline_len",
                  "monitor_end_m","lambda0_floor","min_leaders",
                  "earlylife_group_levels","earlylife_smooth_window",
                  "earlylife_rate_floor_frac")}
with open("CONFIG_確定版_2026-07.json", "w", encoding="utf-8") as f:
    json.dump(frozen, f, ensure_ascii=False, indent=2)

# 動作確認スクリプトの最終実行（結線が壊れていないか）：
#python verify_phase3.py   # asof再現性の差0確認

"""
2. 初回本番投入
実データ全体（過去分すべて）に対して、**まず空の台帳で1回run()**します。
これはいきなり月次運用に入るのではなく、「今溜まっている問題を一度全部洗い出す」ための特別な回です。
"""
import state_logic_cusum as s
import json

cfg = dict(s.CONFIG)
with open("CONFIG_確定版_2026-07.json") as f:
    cfg.update(json.load(f))

cfg["panel_path"] = "実データpanel_補正済み.csv"
cfg["ledger_path"] = "台帳テンプレート_CUSUM版.xlsx"   # まだ0行の状態
cfg["ledger_sheet"] = "台帳"
cfg["out_inbox"] = "要確認インボックス_初回.csv"
cfg["out_tableau"] = "tableau_監視テーブル.csv"

inbox, tableau, meta = s.run(cfg)
print(f"初回の要確認件数: {len(inbox)}")

"""
3. 月次運用サイクルの型化
ここからが「本番」です。毎月、同じ手順を繰り返します。
[毎月X営業日]
1. 最新パネルデータを取得・整形（列名確認、cummax補正など）
2. panel_path を最新ファイルに差し替えて s.run(cfg) を実行
3. 出てきた 要確認インボックス.csv をレビュー
4. 各行に対して処置区分を判断し、台帳（Excel）の「台帳」シートに追記
   （インボックスの行をそのままコピーして、処置区分・原因メモ・確認者・記録日を埋める）
5. tableau_監視テーブル.csv をTableauで開き（または既存ワークブックのデータソースを更新）、
   チームへ共有
"""

"""
④ 継続監視の仕組み化（定期的に、月次より低頻度で）

半年〜1年に1回：run_backtestを最新パネルで再実行し、パラメータがまだ適切か確認（誤報率や検出力が当初の想定からズレていないか）
simulate_monthly_loadで定期的に振り返り：実際の月次件数の推移が想定レンジに収まっているか
台帳のバックアップ：Excel単一ファイルが唯一のマスタなので、月次でバージョン管理（日付付きコピー保存、またはOneDrive/SharePointの履歴機能）をルール化しておくと、誤操作からの復旧が効きます
"""
