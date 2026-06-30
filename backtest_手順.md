# 操作点決め 手順（backtest_cusum.py）

データはローカルから出せないので、本ハーネスを**手元で実データに当てて**、出てきたトレードオフ表で
操作点（R, h, alpha_spike, min_count, burst_window）を選ぶ。検知エンジンは本番（state_logic_cusum）と
同じ `cusum_monitor` / `earlylife_baseline` に委譲するので、ここで選んだ操作点はそのまま本番に効く。

## 0. 置き場所
`cusum_monitor.py` / `earlylife_baseline.py` / `state_logic_cusum.py` / `backtest_cusum.py` を同じフォルダに。

## 1. 入力
- **パネル**: state_logic_cusum と同じ縦長形式（事業コード・開発コード・部番・販社・年月・経過月・
  月次使用数・累積販売台数、任意で SF-コード・ランク）。台帳は不要（素の検出特性を測るため使わない）。
- **既知異常ラベル（任意）**: 列 `事業コード, 開発コード, 部番, 発生年月`。確定した過去のロット/設計問題。

## 2. 実行
```python
import state_logic_cusum as s
from backtest_cusum import run_backtest, recommend_drift, recommend_spike

cfg = dict(s.CONFIG)                       # 列名マッピング等は本番と同じものを使う
drift_tbl, spike_tbl = run_backtest("panel.xlsx", cfg=cfg, labels_path=None)
#                       labels_path="labels.csv" があれば検出遅れも測る
```
`backtest_drift.csv` / `backtest_spike.csv`（ラベルありなら `backtest_labeled.csv`）が出る。

## 3. 表の読み方
ドリフトとスパイクは独立した検出器なので別々に決める。

### ドリフト（CUSUM）
| 列 | 意味 |
|---|---|
| ARL0_単位月 | 正常データで誤報が出るまでの平均“単位×月”。大きいほど誤報が稀 |
| 月間誤報件数 | 艦隊全体での月あたり誤報件数（= 誤報率 × 監視単位数）。**チームが捌ける量で線引き** |
| 検出率 | 真倍率（1.5/2/3倍のレート上昇）を注入したときに捕まえた割合 |
| 遅れ月_中央値 | 発生から検出までの月数の中央値 |

選び方: まず捕まえたい最小のドリフト（例: 1.5倍）を決め、その倍率で検出率が十分高く、かつ
月間誤報件数が予算内に収まる最大の h を選ぶ。R は「最速で捕まえたい倍率」に合わせる
（R≈その倍率）。h が誤報・遅れの主つまみ、R が感度の形を決める。

### スパイク（Shewhart）
| 列 | 意味 |
|---|---|
| 誤報率_単位月 / 月間誤報件数 | 正常データでスパイク発火する率・件数 |
| 検出率 | 注入スパイク（期待故障数の3倍/5倍）を捕まえた割合 |
| burst_window | 0=単月のみ。3=直近3月の合算も見る（2〜3月の急増を拾うが件数は増える） |

選び方: alpha_spike を下げると誤報も検出力も下がる。min_count は低カウント単位の偽陽性ガード
（実データの低頻度部品で効く）。burst を入れるかは「2〜3月かけて出るロット不良」を拾いたいか次第。

## 4. 推薦
```python
recommend_drift(drift_tbl, max_alarms_per_month=20, factor=2.0)   # 予算20件/月で2倍ドリフト前提
recommend_spike(spike_tbl, max_alarms_per_month=20, mult=5.0)
```
予算（月間誤報件数）以下の中から、検出率→遅れ→誤報の順で並べた候補を返す。

## 5. CONFIG へ反映
決めた値を本番 `state_logic_cusum.CONFIG`（または運用ランナーの cfg）に入れる:
`R, h, alpha_spike, min_count, burst_window`。安定化前カーブのつまみ
（`earlylife_smooth_window, earlylife_rate_floor_frac, earlylife_group_levels`）も
必要なら同じ枠組みで振れる（既定で十分なら触らない）。

## 注意・限界
- **誤報率は「実データ＝ほぼ正常」前提**。参照期間に本物の異常が混じると誤報率は上振れ（保守側）に出る。
  既知異常の期間をパネルから除く、または別系列にすると、より正確な誤報率が得られる。
- **検出力は合成注入による近似**。真倍率・注入倍率・onset 位置は `BT` で調整できる。
  既知異常ラベルがあれば `labels_path` で実測の検出遅れも併せて見る（こちらが本筋）。
- ARL0/遅れの測定は各アラートを離散イベントとして数えるため `reset_after_alarm=True` で測る
  （ARL 測定の流儀）。本番のリセットは台帳由来だが、(R,h) の誤報特性づけはこの流儀で正しい。
- 多重検定: 監視単位が数百〜数千なら、per-test の alpha より**艦隊全体の月間件数**で見るのが実務的。
  「月間誤報件数」列がその数。状態機械＋ランク付きワークリストの triage 能力に合わせて予算を決める。

## グリッドの調整
`backtest_cusum.BT` の各 `*_grid` を編集すれば探索範囲を変えられる。実データの単位数が多いと
時間がかかるので、まず粗いグリッドで当たりをつけ、近傍を細かく刻むとよい。
