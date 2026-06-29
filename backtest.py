"""
backtest.py
===========
過去のアラート相当事例(ラベル)に対して検知性能を測り、感度パラメータを
チューニングするための検証ハーネス。対象パラメータ:
  ドリフト側 : R = 悪化倍率, h = しきい値 (CUSUM)
  スパイク側 : alpha_spike, min_count, burst_window (単月/バースト検定)

データは一切外部に出さない。御社環境でラベル付きデータを入力すれば、
こちらで中身を見ることなく性能指標を出せる。

ラベルの与え方
--------------
labels: DataFrame。1行 = 1異常事例。最低限の列:
  - 監視単位を特定するキー (事業コード, 開発コード, 部番[, 販社])
  - 基準月 "ラベル経過月": 「ここで確認すべきだった」と分かっている経過月
    (社内で把握している発覚時点・確認時点の経過月)。
  - 任意で "タイプ" 列 (例: drift / spike)。あればタイプ別の検知率が
    結果に power_drift / power_spike として併記され、detail にも引き継がれる。

検知遅れ(delay)の符号
----------------------
  検知判定: ラベル前 early_margin か月 〜 ラベル後 late_margin か月 の窓内に
  アラートがあれば検知成功。delay は窓内の最初のアラートを基準に
  delay = アラート経過月 - ラベル経過月 で計算する。
    delay < 0 : 人手の発覚より「早く」検知できた(早期検知として望ましい)
    delay = 0 : 同時
    delay > 0 : 人手より遅れて検知

誤報(false alarm)
-----------------
  ラベルの無い監視単位(平常系列)でアラートが出たら誤報。
  単位時間あたりに正規化して「監視単位・年あたり誤報数」で評価する(低頻度向け)。
  注意: ラベルを一部タイプだけに絞って渡すと、他タイプの異常単位が
  「平常」扱いになり誤報が過大計上される。誤報を正しく測るには
  全タイプのラベルを渡し、関心タイプの検知率は power_<タイプ> 列で読む。

検知タイプ(alert_types)
-----------------------
  ドリフト(CUSUM)とスパイク(単月/バースト検定)は別エンジンなので、
  評価時にどのアラートを「検知」とみなすか alert_types で選べる:
    None                       : 総合アラート(全タイプのOR、既定)
    ("drift",)                 : CUSUMのみ(従来互換)
    ("spike", "burst")         : スパイク系のみ(スパイク側の単独チューニング用)
  ドリフト側の (R, h) とスパイク側の (alpha_spike, min_count, burst_window) は
  狙う信号が違うため独立にチューニングしてよい。grid_search が前者、
  spike_grid_search が後者を担当する。
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from cusum_monitor import monitor_all, UNIT_KEYS_WITH_SHA, UNIT_KEYS_AGG_SHA

# アラート種別 → 結果テーブルの真偽列 の対応
_ALERT_COLS = {
    "drift": "アラート",
    "spike": "スパイクアラート",
    "burst": "バーストアラート",
}


def _alarm_mask(unit_result: pd.DataFrame, alert_types=None) -> pd.Series:
    """指定タイプのアラートのORマスクを返す。None なら総合アラート。"""
    if alert_types is None:
        if "総合アラート" in unit_result.columns:
            return unit_result["総合アラート"]
        return unit_result["アラート"]
    mask = pd.Series(False, index=unit_result.index)
    for t in alert_types:
        col = _ALERT_COLS[t]
        if col in unit_result.columns:
            mask = mask | unit_result[col].fillna(False)
    return mask


def _alarm_months(unit_result: pd.DataFrame, col_keizoku="経過月",
                  alert_types=None) -> np.ndarray:
    a = unit_result[_alarm_mask(unit_result, alert_types)]
    return np.sort(a[col_keizoku].to_numpy())


def evaluate(
    df: pd.DataFrame,
    labels: pd.DataFrame,
    unit_keys: list[str],
    stable_start_m: int,
    baseline_len: int,
    monitor_end_m: int,
    R: float,
    h: float,
    alpha_spike: float | None = None,
    min_count: int = 2,
    burst_window: int = 0,
    alpha_burst: float | None = None,
    alert_types=None,
    early_margin: int = 24,
    late_margin: int = 6,
    col_keizoku: str = "経過月",
    label_keizoku_col: str = "ラベル経過月",
) -> dict:
    """単一のパラメータ組で全単位を監視し、検知性能をまとめて返す。

    alpha_spike を指定するとスパイク検定も併走する。alert_types で
    「どのアラートを検知とみなすか」を選ぶ(既定: 全タイプのOR)。

    Returns(dict)
      power            : ラベル事例のうち検知できた割合
      power_<タイプ>    : labels に「タイプ」列がある場合のタイプ別検知率
      n_labeled        : ラベル事例数
      n_detected       : 検知できた事例数
      median_delay     : 検知事例の delay の中央値(負ほど早期、月)
      mean_delay
      fa_per_unit_year : 平常単位での誤報数 / 監視単位・年
      n_clean_units    : 平常(ラベル無し)単位数
      detail           : 事例ごとの明細 DataFrame
    """
    results = monitor_all(
        df, unit_keys, stable_start_m, baseline_len, monitor_end_m, R, h,
        alpha_spike=alpha_spike, min_count=min_count,
        burst_window=burst_window, alpha_burst=alpha_burst,
    )

    # 監視単位ごとに結果を引けるよう辞書化
    res_by_unit = {key: g for key, g in results.groupby(unit_keys, dropna=False)}

    # ラベルキー(存在する列だけ使う)
    label_join_keys = [k for k in unit_keys if k in labels.columns]

    detail_rows = []
    labeled_unit_set = set()
    for _, lab in labels.iterrows():
        key = tuple(lab[k] for k in label_join_keys)
        if len(label_join_keys) < len(unit_keys):
            # 販社合算監視でラベルが販社別の場合などは合算キーに丸める
            key = tuple(lab[k] for k in unit_keys if k in labels.columns)
        # 単一キーの groupby は tuple でなくスカラになるため整合させる
        lookup = key if len(key) > 1 else key[0]
        labeled_unit_set.add(lookup)

        extra = {"タイプ": lab["タイプ"]} if "タイプ" in labels.columns else {}

        g = res_by_unit.get(lookup)
        lab_m = int(lab[label_keizoku_col])
        if g is None:
            detail_rows.append({"key": lookup, "ラベル経過月": lab_m,
                                "初アラート経過月": None, "delay": None,
                                "検知": False, **extra})
            continue
        # 検知判定: 窓 [ラベル-early_margin, ラベル+late_margin] 内にアラートが
        # あれば検知成功。delay は窓内の最初のアラートを基準に計算する。
        # (系列全体の最初のアラートで判定すると、窓より前の無関係なアラートが
        #  あるだけで「未検知」になってしまうため)
        am = _alarm_months(g, col_keizoku, alert_types)
        in_win = am[(am >= lab_m - early_margin) & (am <= lab_m + late_margin)]
        first_overall = int(am[0]) if len(am) else None
        if len(in_win) == 0:
            detail_rows.append({"key": lookup, "ラベル経過月": lab_m,
                                "初アラート経過月": first_overall, "delay": None,
                                "検知": False, **extra})
            continue
        delay = int(in_win[0]) - lab_m
        detail_rows.append({"key": lookup, "ラベル経過月": lab_m,
                            "初アラート経過月": first_overall, "delay": delay,
                            "検知": True, **extra})

    detail = pd.DataFrame(detail_rows)
    n_labeled = len(detail)
    detected_mask = detail["検知"] if n_labeled else pd.Series(dtype=bool)
    n_detected = int(detected_mask.sum()) if n_labeled else 0
    power = n_detected / n_labeled if n_labeled else float("nan")
    delays = detail.loc[detail["検知"], "delay"].dropna() if n_labeled else pd.Series(dtype=float)
    median_delay = float(delays.median()) if len(delays) else float("nan")
    mean_delay = float(delays.mean()) if len(delays) else float("nan")

    # 誤報: ラベルの無い単位でアラートが立った回数を、監視月数で正規化
    n_clean_units = 0
    fa_count = 0
    monitored_months = 0
    for key, g in res_by_unit.items():
        if key in labeled_unit_set:
            continue
        n_clean_units += 1
        monitored_months += len(g)
        fa_count += int(_alarm_mask(g, alert_types).sum())
    fa_per_unit_year = (
        fa_count / (monitored_months / 12.0) if monitored_months > 0 else float("nan")
    )

    out = {
        "R": R, "h": h,
        "alpha_spike": alpha_spike, "min_count": min_count,
        "burst_window": burst_window,
        "power": power, "n_labeled": n_labeled, "n_detected": n_detected,
        "median_delay": median_delay, "mean_delay": mean_delay,
        "fa_per_unit_year": fa_per_unit_year, "n_clean_units": n_clean_units,
        "detail": detail,
    }
    # ラベルに「タイプ」列があれば、タイプ別の検知率も併記する
    # (例: power_drift, power_spike)。全ラベルを渡したうえで
    # 関心のあるタイプの検知率だけを読む、という使い方ができる。
    if n_labeled and "タイプ" in detail.columns:
        for t, sub in detail.groupby("タイプ"):
            out[f"power_{t}"] = float(sub["検知"].mean())
    return out


def grid_search(
    df: pd.DataFrame,
    labels: pd.DataFrame,
    unit_keys: list[str],
    stable_start_m: int,
    baseline_len: int,
    monitor_end_m: int,
    R_grid,
    h_grid,
    **eval_kwargs,
) -> pd.DataFrame:
    """ドリフト側 (R, h) の格子で性能を計算し、サマリ表を返す。

    返り値の各行が1つの (R, h)。power が高く、median_delay が小さく(=早期)、
    fa_per_unit_year が許容内、の操作点を選ぶ。
    ドリフト側を単独でチューニングするなら alert_types=("drift",) を渡し、
    labels は全タイプを渡して power_drift 列を読むこと(誤報の過大計上を防ぐ)。
    """
    rows = []
    for R in R_grid:
        for h in h_grid:
            r = evaluate(
                df, labels, unit_keys, stable_start_m, baseline_len,
                monitor_end_m, R, h, **eval_kwargs,
            )
            r.pop("detail")
            rows.append(r)
    out = pd.DataFrame(rows)
    return out.sort_values(["fa_per_unit_year", "median_delay"]).reset_index(drop=True)


def spike_grid_search(
    df: pd.DataFrame,
    labels: pd.DataFrame,
    unit_keys: list[str],
    stable_start_m: int,
    baseline_len: int,
    monitor_end_m: int,
    alpha_grid,
    min_count_grid=(2, 3),
    burst_window: int = 0,
    **eval_kwargs,
) -> pd.DataFrame:
    """スパイク側パラメータ (alpha_spike, min_count) のグリッドサーチ。

    ドリフト側とは独立にチューニングするため、評価は alert_types=("spike","burst")
    に限定する(CUSUMのアラートは検知にも誤報にも数えない)。

    labels には【全タイプのラベル】を渡すこと。スパイク事例だけに絞ると、
    ドリフト事例の単位が「平常」扱いになり、そこで鳴った本物のアラートが
    誤報として計上されてしまう。全ラベルを渡せば異常単位は誤報計算から
    除外され、スパイク事例の検知率は power_spike 列で読める。

    スパイクは「発生月の近くで鳴る」のが期待動作なので、検知窓の既定を
    early_margin=3, late_margin=3 と狭めてある(eval_kwargsで上書き可)。
    """
    eval_kwargs.setdefault("early_margin", 3)
    eval_kwargs.setdefault("late_margin", 3)
    rows = []
    for alpha in alpha_grid:
        for mc in min_count_grid:
            r = evaluate(
                df, labels, unit_keys, stable_start_m, baseline_len,
                monitor_end_m, R=2.0, h=float("inf"),  # ドリフト側は実質無効
                alpha_spike=alpha, min_count=mc,
                burst_window=burst_window,
                alert_types=("spike", "burst"),
                **eval_kwargs,
            )
            r.pop("detail")
            rows.append(r)
    out = pd.DataFrame(rows)
    return out.sort_values(["fa_per_unit_year", "median_delay"]).reset_index(drop=True)


def suggest_operating_point(
    grid_result: pd.DataFrame,
    max_fa_per_unit_year: float,
    min_power: float = 0.8,
    power_col: str = "power",
) -> pd.DataFrame:
    """誤報予算と最低検知率の制約を満たす操作点を、早期性(delay)順に並べて返す。

    power_col で判定に使う検知率列を選べる(例: 全ラベルを渡したグリッドで
    ドリフト側操作点を選ぶなら power_col="power_drift")。
    """
    cand = grid_result[
        (grid_result["fa_per_unit_year"] <= max_fa_per_unit_year)
        & (grid_result[power_col] >= min_power)
    ].copy()
    return cand.sort_values(["median_delay", "fa_per_unit_year"]).reset_index(drop=True)
