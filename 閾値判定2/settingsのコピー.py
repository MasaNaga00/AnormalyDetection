# -*- coding: utf-8 -*-
"""
settings.py — 設定はすべてここ。他のファイルは触らない。

run_month.py / tune_c_alpha.py / check_rank.py はここを読む。
実データに移すときも、変えるのはこのファイルだけ。
"""

# ============================================================================
# 1. 列名 — 実データのカラム名に合わせる（最初に必ず確認）
# ============================================================================
COLS = dict(
    biz="事業コード",          # 事業部コード
    dev="開発コード",          # 機種
    part="部番",
    dist="販社",
    ym="年月",                 # YYYYMM。ハイフン区切りでも自動変換される
    elapsed="経過月",          # 発売からの経過月
    monthly_use="月次使用数",
    cum_sales="累積販売台数",  # 露出(E)の素になる。販社別行は販社ごとの台数
    sf="SF-CODE",              # 部品の種類コード。信号Bのピアキー
)

ALL_TOKEN = "ALL"              # 販社列の「全販社合計」を表す値


# ============================================================================
# 2. 閾値検知（メイン）
# ============================================================================
# 累積使用率がこの値(%)を超えたら発火。現行運用の基準に合わせる。
BASE_THRESHOLD_PCT = 5.0

# 部品ごとに基準を変えたいとき。{(事業コード, 開発コード, 部番): 閾値%}
# 開発コードに None を入れると機種を問わず適用。
#   例: {("E1", None, "P-1234"): 8.0}
THRESHOLD_OVERRIDES = {}

# 「新常態受容」で上書き閾値を提案するときの上乗せ幅(%)
MARGIN_PCT = 0.5

# 累積販売台数がこれ未満の単位は判定しない（母数が薄いと率が暴れるため）
MIN_DENOMINATOR = 0


# ============================================================================
# 3. 信号B（SF群内ピア比較）
# ============================================================================
# 経過月0〜この値の区間で比較する。発売年次の差を消すために必須。
# 短くすると反応は速いが、窓を完走した機種だけが対象になるので母数が減る。
B_ELAPSED_CAP = 36

B_MIN_PEERS = 2        # ピア機種がこの数以上そろって初めて判定（未満は沈黙）

# ★主レバー: ピア比がこの倍率以上のときだけ発火。
#   件数Cが数百になるとp値は桁で飛ぶ（O/E=1.27でp=6e-4、O/E=2.48でp=1e-58）。
#   機種は正当な理由でも差が出る（過分散）ので、統計的有意性でなく効果量で切る。
#   tune_b_minoe.py で実測してから決める。
B_MIN_OE = 1.5

B_ALPHA = 0.005        # 補助レバー。件数の少ない群での偶然を落とすガード
B_MIN_COUNT = 3        # この使用数未満は発火させない
B_NB_STRAT = True      # nb(部番数)が一致する機種同士だけ比較。原則 True のまま
B_TWO_PASS = True      # 発火機種をピアプールから外して再計算。原則 True のまま


# ============================================================================
# 4. 信号C（販社別の月次急増）
# ============================================================================
C_BASE_LEN = 12        # 直近何ヶ月をベースラインにするか

# ★実データ投入前に tune_c_alpha.py で必ず実測してから決める。
#   15,000系列を毎月検定するので 0.005 のままだと月数十件出る。
C_ALPHA = 0.005

C_MIN_COUNT = 3        # この使用数未満は発火させない
C_MIN_BASE_MONTHS = 6  # ベースラインに必要な最低月数（販社行の左側打ち切り対策）
C_MIN_BASE_COUNT = 3.0 # ベースライン窓の最低使用数


# ============================================================================
# 4.5 販社の報告遅れ（reporting_horizon.py）
# ============================================================================
# 販社ごとに修理データの送付頻度が違う（毎日／週次／月次）。3月上旬に集計すると
# 月次送付の販社は12月分までしか入っていない、ということが起こる。
#
# ★ これを入れないと、遅れている販社の月は信号Cで**一度も検定されない**。
#   run_signal_c は既定で基準月Tの1ヶ月だけを判定し、翌月もまたTしか見ないため。

USE_HORIZON = True

# 各販社の末尾から落とす月数の既定値。0 のままでよい（下の自動判定が効く）。
HORIZON_MARGIN_MONTHS = 0

# True: 最終月の使用数計が直前6ヶ月の中央値の HORIZON_THIN_RATIO 未満なら
#       「まだ月の途中」と見なして margin を +1 する。
#       月次一括送付の販社は最終月が完結しているので margin 0、
#       毎日/週次送付の販社は月の途中で切れて薄いので margin 1、が自動で付く。
HORIZON_AUTO_MARGIN = True
HORIZON_THIN_RATIO = 0.7

# 送付実態が分かっている販社は手で固定できる（自動判定より優先）。
#   例: HORIZON_MARGIN_OVERRIDES = {"日本": 0, "米国": 1}
HORIZON_MARGIN_OVERRIDES = {}

# 受領管理表がある場合はこれが正解。{販社: YYYYMM}
HORIZON_FIXED = {}

# ★信号Cが毎回さかのぼって再判定する月数。
#   **販社の最大遅れ月数以上**にすること。足りないと月が永久に未検定になる。
#   reporting_horizon.report(...) の「遅れ月数」の最大値を見て決める。
#   迷ったら大きめ（6）にしてよい。既に台帳に記録済みの月は抑制で消える。
C_REVISIT_MONTHS = 6

# 信号Bを「全販社が揃っている月」で打ち切る。累積O/Eの機種間比較なので、
# 直近の欠測量が機種の販社構成によって違うと比較が不公平になる。
# 12ヶ月抑制の遅い検出器なので数ヶ月遅れても実害はない。
B_TRUNCATE_TO_HORIZON = True


# ============================================================================
# 5. 統合インボックス
# ============================================================================
TOP_N = 15             # 毎月レビューする件数（人間の運用ルール）
MULTI_BONUS = 0.5      # 検出器が1つ増えるごとの加点
SCORE_CAP = 3.0        # 各検出器の生スコアの上限

# 台帳の再評価年月が空欄のときの既定抑制期間（月）。検出器ごとに独立。
# 信号Bが長いのは、累積O/Eが月次でほぼ動かず毎月鳴り続けるため。
SUPPRESS_MONTHS = {"閾値": 6, "信号B": 12, "信号C": 3, "CUSUM": 1}

MACHINE_ALL_PART_TOKEN = "機種全体"   # 機種終了を表す部番の値


# ============================================================================
# 6. 出力先
# ============================================================================
OUT_ROOT = "出力"      # 生成物は 出力/YYYYMM/ 配下に置かれる


# ============================================================================
# 7. 検証ツール用
# ============================================================================
LOOKBACK_M = 6         # 過去例の評価: 報告月の何ヶ月前まで遡って先行検知を認めるか
MONTHS_BACK = 24       # c_alpha実測: 何ヶ月さかのぼるか
SCAN_MIN_COUNT = 2     # c_alpha実測: p値を計算する最低使用数（1にすると重い）

# ラベルCSVの列名。発生年月は必ず YYYYMM（年月日だと月がズレる）
LABEL_COLS = dict(biz="事業コード", dev="開発コード", part="部番",
                  ym="発生年月", dist="販社")


# ============================================================================
# 以下は組み立て。触らない。
# ============================================================================
def build_cfg() -> dict:
    """unified_inbox に渡す設定を組み立てる。"""
    import unified_inbox as ui
    cfg = dict(ui.CONFIG)
    cfg.update(
        base_threshold_pct=BASE_THRESHOLD_PCT,
        threshold_overrides=THRESHOLD_OVERRIDES,
        margin_pct=MARGIN_PCT,
        min_denominator=MIN_DENOMINATOR,
        b_elapsed_cap=B_ELAPSED_CAP, b_min_peers=B_MIN_PEERS,
        b_alpha=B_ALPHA, b_min_count=B_MIN_COUNT, b_min_oe=B_MIN_OE,
        c_base_len=C_BASE_LEN, c_alpha=C_ALPHA, c_min_count=C_MIN_COUNT,
        c_min_base_months=C_MIN_BASE_MONTHS, c_min_base_count=C_MIN_BASE_COUNT,
        multi_bonus=MULTI_BONUS, score_cap=SCORE_CAP, top_n=TOP_N,
        suppress_months=SUPPRESS_MONTHS,
        all_token=ALL_TOKEN,
        machine_all_part_token=MACHINE_ALL_PART_TOKEN,
        use_horizon=USE_HORIZON,
        horizon_margin_months=HORIZON_MARGIN_MONTHS,
        horizon_margin_overrides=HORIZON_MARGIN_OVERRIDES,
        horizon_fixed=HORIZON_FIXED,
        horizon_auto_margin=HORIZON_AUTO_MARGIN,
        horizon_thin_ratio=HORIZON_THIN_RATIO,
        c_revisit_months=C_REVISIT_MONTHS,
        b_truncate_to_horizon=B_TRUNCATE_TO_HORIZON,
    )
    return cfg


if __name__ == "__main__":
    cfg = build_cfg()
    print("=== 現在の設定 ===")
    for k in ("base_threshold_pct", "b_elapsed_cap", "b_min_peers", "b_alpha",
              "c_base_len", "c_alpha", "c_min_count", "top_n",
              "use_horizon", "c_revisit_months", "b_truncate_to_horizon"):
        print(f"  {k:22s} = {cfg[k]}")
    print(f"  suppress_months        = {cfg['suppress_months']}")
    print(f"  列名 sf                = {COLS['sf']}")
    print(f"  出力先                 = {OUT_ROOT}/")
