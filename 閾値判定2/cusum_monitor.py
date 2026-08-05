"""
cusum_monitor.py
================
部品出庫の異常検知コアモジュール。
2種類の検知エンジンを同じベースライン(lambda0)の上で併走させる:

  (1) ドリフト検知: 時変平均ポアソン上側CUSUM
      「小さいが持続する上昇」に最速。安定期に入った機種の緩やかな劣化が主ターゲット。
  (2) スパイク検知: 単月/バーストの正確検定 (Shewhart相当)
      「大きく突発的な上昇」に最速。CUSUMが反応する前に単月〜数か月で出る急増を拾う。
      件数が少ない部番でも、正規近似に頼らず正確な裾確率(p値)で判定できる。

両者は得意領域が逆の補完関係(combined Shewhart-CUSUM)。共有するもの:
  - 平常レート lambda0(安定期初期で推定し固定)
  - 期待故障数 = lambda0 * 累積販売台数(保有台数増を自動吸収)
  - 妥当性ウィンドウ(退役で分母が過大になる末期は監視対象外)
  - 監視単位の二段構え(販社合算が主、販社別が副)

設計の要点（詳細は設計ドキュメント・検証手順書を参照）:
  - 監視指標は「1台あたり月次故障ペース」。ただし生レートを正規分布扱いせず、
    月次使用数(離散カウント)を時変平均のポアソン過程として監視する。
  - 平常レート lambda0 は「安定期初期」で推定し固定する(ドリフト吸収を防ぐ)。
  - 期待故障数 = lambda0 * 累積販売台数 は保有台数の増加を自動で吸収するので、
    アラートはレートの上昇分にのみ反応する。
  - 累積販売台数が実稼働台数の近似として成立する経過月レンジに監視を限定する
    (退役が進む末期は分母が過大になり、本物の上昇をマスクするため)。
  - スパイク検定は lambda0 を真値扱いせず、ベースライン窓のカウントとの
    条件付き二項検定(2つのポアソンレート比較の正確検定)で行う。
    短いベースラインによる推定誤差が検定に正確に織り込まれる。
  - lambda0 <= 0 (ベースライン窓で使用実績ゼロ) の単位は両エンジンとも監視不能と
    して扱う。スパイク側だけ救う場合は別途ベースラインの取り方を検討する(未決)。

このモジュールは外部にデータを送らない。すべてローカルで完結する。
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd


# 監視単位を一意に決めるキー。販社合算で監視するときは販社をキーから外す。
UNIT_KEYS_WITH_SHA = ["事業コード", "開発コード", "部番", "販社"]
UNIT_KEYS_AGG_SHA = ["事業コード", "開発コード", "部番"]


def estimate_lambda0(usage: np.ndarray, fleet: np.ndarray) -> float:
    """安定期初期の窓から平常レート lambda0(1台・1か月あたり故障率)を推定する。

    lambda0 = (窓内の月次使用数合計) / (窓内の累積販売台数合計)
    台数で重みづけした平均レートに相当する。
    """
    total_fleet = float(np.sum(fleet))
    if total_fleet <= 0:
        return 0.0
    return float(np.sum(usage)) / total_fleet


def poisson_cusum(
    usage: np.ndarray,
    fleet: np.ndarray,
    lambda0,
    R: float,
    h: float,
    reset_after_alarm: bool = True,
):
    """時変平均ポアソン上側CUSUM。

    Parameters
    ----------
    usage  : 各月の月次使用数(カウント)
    fleet  : 各月の累積販売台数(保有台数近似)。期待値の分母。
    lambda0: 平常レート。スカラ(安定期: 1つの固定レート)でも、
             配列 lambda0_t(安定化前: 経過月ごとに変化する期待カーブ)でも可。
             配列の場合は len(usage) と長さを揃える。
    R      : 検知したい悪化倍率 lambda1/lambda0 ( > 1 )。例: 2.0 は「レート倍化を狙う」
    h      : 判定しきい値(感度)。小さいほど早く鳴るが誤報増。
    reset_after_alarm: アラート後に統計量を0へ戻すか(再検知を許す)。

    Returns
    -------
    S      : CUSUM統計量の時系列
    alarm  : 各月のアラート真偽(S >= h)
    k      : 各月の参照値(時変)

    時変λ0について
    --------------
    安定期は lambda0 がスカラで、分母(累積販売台数)の増加だけが時変だった。
    安定化前は故障率そのものが「初期高→減衰」と下がるので lambda0_t を配列で渡す。
    悪化倍率Rは各月の基準レートに対する倍率として展開する:
      lambda1_t = R * lambda0_t
      k_t = (lambda1_t - lambda0_t) * fleet_t / ln(R)
            = (R - 1) * lambda0_t * fleet_t / ln(R)
    Rが時間によらず一定なら ln(R) は共通。基準カーブが下がれば k_t も下がり、
    「その経過月で期待される水準からの上振れ」を一貫して検出する。
    """
    usage = np.asarray(usage, dtype=float)
    fleet = np.asarray(fleet, dtype=float)
    n = len(usage)
    S = np.zeros(n)
    alarm = np.zeros(n, dtype=bool)
    k = np.zeros(n)

    lam0 = np.asarray(lambda0, dtype=float)
    if lam0.ndim == 0:
        lam0 = np.full(n, float(lam0))

    if R <= 1.0 or np.all(lam0 <= 0):
        # 悪化倍率が不正、または基準レートが全月0なら監視不能(全て非アラート)。
        return S, alarm, k

    log_ratio = np.log(R)
    # 時変参照値 k_t = (R - 1) * lambda0_t * fleet_t / ln(R)
    # lambda0_t <= 0 の月は k_t=0 とし、その月は実質ノーカウント基準にする。
    lam0_safe = np.where(lam0 > 0, lam0, 0.0)
    k = (R - 1.0) * lam0_safe * fleet / log_ratio

    s_prev = 0.0
    for t in range(n):
        s = max(0.0, s_prev + usage[t] - k[t])
        S[t] = s
        if s >= h:
            alarm[t] = True
            s_prev = 0.0 if reset_after_alarm else s
        else:
            s_prev = s
    return S, alarm, k


def poisson_sf(x: float, mu: float) -> float:
    """正確なポアソン上側裾確率 P(X >= x), X ~ Poisson(mu)。scipy非依存。

    低頻度カウント(muが小さい)で正規近似を使わずに「この月のx個は偶然か」を
    評価するためのもの。muが非常に大きい場合のみ連続性補正つき正規近似に
    フォールバックする。
    """
    if x <= 0:
        return 1.0
    if mu <= 0:
        return 0.0
    x = int(x)
    if mu < 700.0:  # exp(-mu) がアンダーフローしない範囲は正確に計算
        term = math.exp(-mu)
        cdf = term
        for i in range(1, x):
            term *= mu / i
            cdf += term
        return min(1.0, max(0.0, 1.0 - cdf))
    # 大mu(本用途ではほぼ出ない): 連続性補正つき正規近似
    z = (x - 0.5 - mu) / math.sqrt(mu)
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def binom_sf(x: int, n: int, p: float) -> float:
    """正確な二項上側裾確率 P(X >= x), X ~ Binomial(n, p)。scipy非依存。"""
    if x <= 0:
        return 1.0
    if x > n:
        return 0.0
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    log_p = math.log(p)
    log_q = math.log1p(-p)
    total = 0.0
    for i in range(int(x), int(n) + 1):
        log_pmf = (
            math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + i * log_p + (n - i) * log_q
        )
        total += math.exp(log_pmf)
    return min(1.0, max(0.0, total))


def spike_test(
    usage: np.ndarray,
    fleet: np.ndarray,
    lambda0: float,
    alpha_spike: float,
    min_count: int = 2,
    burst_window: int = 0,
    alpha_burst: float | None = None,
    baseline_count: float | None = None,
    baseline_exposure: float | None = None,
):
    """単月スパイク検定 + (任意)バーストウィンドウ検定。

    検定の中身 (条件付き二項検定):
      lambda0 はベースライン窓からの「推定値」であり、低頻度部番では推定誤差が
      大きい。これを真値扱いした単純なポアソン検定は、lambda0 を過小推定した
      単位で誤報を量産する。そこで baseline_count (ベースライン窓の総使用数 C) と
      baseline_exposure (同窓の累積販売台数合計 E) が与えられた場合は、
      「2つのポアソンレートの比較」の古典的な正確検定を使う:
        H0: 当月のレート = ベースラインのレート のもとで、
        合計 n = x_t + C を固定すると x_t ~ Binomial(n, e_t/(e_t+E))
        (e_t = 当月の累積販売台数)。p_t = P(X >= x_t)。
      ベースラインの推定不確実性が正確に織り込まれるため、カウントが少ない
      単位ほど自動的に保守的になる。C・E が無い場合のみ、lambda0 を既知と
      みなす単純ポアソン検定にフォールバックする。

    単月検定 (Shewhart相当):
      アラート条件: p_t <= alpha_spike かつ usage_t >= min_count。
      min_count は「期待値が極小のとき、たまたま数個出ただけで鳴る」のを防ぐ
      ガード。低頻度部番ほど効く。

    バーストウィンドウ検定 (burst_window >= 2 のとき):
      直近W=burst_window か月の合算カウントを同じ方法で検定する。
      月境界で割れたスパイクや、単月では足りないがCUSUMには短すぎる
      2〜3か月の急増を拾う中間レンジ担当。
      アラート条件: p <= alpha_burst かつ 合算カウント >= min_count + 1
      (単月より窓が広いぶん、最低個数を1段引き上げて単発ノイズを弾く)。
      系列先頭のW-1か月は窓が揃わないため判定しない(p値=NaN)。

    Returns
    -------
    p_spike     : 各月の単月p値
    alarm_spike : 単月スパイクアラート真偽
    p_burst     : 各月のバーストp値 (burst_window<2 なら全NaN)
    alarm_burst : バーストアラート真偽
    """
    usage = np.asarray(usage, dtype=float)
    fleet = np.asarray(fleet, dtype=float)
    n = len(usage)
    p_spike = np.full(n, np.nan)
    alarm_spike = np.zeros(n, dtype=bool)
    p_burst = np.full(n, np.nan)
    alarm_burst = np.zeros(n, dtype=bool)

    lam0 = np.asarray(lambda0, dtype=float)
    lam0_is_array = lam0.ndim > 0
    if lam0_is_array and len(lam0) != n:
        raise ValueError("lambda0 配列の長さが usage と一致しません")

    if (not lam0_is_array and lam0 <= 0) or (lam0_is_array and np.all(lam0 <= 0)):
        # 基準レートが無い単位は監視不能(ドリフト側と同じ扱い)。
        return p_spike, alarm_spike, p_burst, alarm_burst

    # 時変λ0(安定化前カーブ)が渡されたら、ベースライン窓ではなく
    # 各月の期待値 mu0_t = lambda0_t * fleet_t に対する直接ポアソン検定を使う。
    # (安定化前は「自分の過去」が無いので、条件付き二項検定の物差しが作れない)
    if lam0_is_array:
        mu0 = lam0 * fleet
        for t in range(n):
            p_spike[t] = poisson_sf(usage[t], mu0[t])
            if p_spike[t] <= alpha_spike and usage[t] >= min_count:
                alarm_spike[t] = True
        if burst_window >= 2:
            ab = alpha_burst if alpha_burst is not None else alpha_spike
            w = int(burst_window)
            for t in range(w - 1, n):
                u_sum = float(np.sum(usage[t - w + 1 : t + 1]))
                m_sum = float(np.sum(mu0[t - w + 1 : t + 1]))
                p_burst[t] = poisson_sf(u_sum, m_sum)
                if p_burst[t] <= ab and u_sum >= min_count + 1:
                    alarm_burst[t] = True
        return p_spike, alarm_spike, p_burst, alarm_burst

    use_conditional = (
        baseline_count is not None and baseline_exposure is not None
        and baseline_exposure > 0
    )

    def tail_prob(x_obs: float, exposure: float) -> float:
        if use_conditional:
            n_total = int(round(x_obs + baseline_count))
            pi = exposure / (exposure + baseline_exposure)
            return binom_sf(int(round(x_obs)), n_total, pi)
        return poisson_sf(x_obs, lam0 * exposure)

    for t in range(n):
        p_spike[t] = tail_prob(usage[t], fleet[t])
        if p_spike[t] <= alpha_spike and usage[t] >= min_count:
            alarm_spike[t] = True

    if burst_window >= 2:
        ab = alpha_burst if alpha_burst is not None else alpha_spike
        w = int(burst_window)
        for t in range(w - 1, n):
            u_sum = float(np.sum(usage[t - w + 1 : t + 1]))
            f_sum = float(np.sum(fleet[t - w + 1 : t + 1]))
            p_burst[t] = tail_prob(u_sum, f_sum)
            if p_burst[t] <= ab and u_sum >= min_count + 1:
                alarm_burst[t] = True

    return p_spike, alarm_spike, p_burst, alarm_burst


def monitor_unit(
    df_unit: pd.DataFrame,
    stable_start_m: int,
    baseline_len: int,
    monitor_end_m: int,
    R: float,
    h: float,
    alpha_spike: float | None = None,
    min_count: int = 2,
    burst_window: int = 0,
    alpha_burst: float | None = None,
    lambda0_curve: np.ndarray | None = None,
    col_keizoku: str = "経過月",
    col_usage: str = "月次使用数",
    col_fleet: str = "累積販売台数",
) -> pd.DataFrame:
    """1監視単位の時系列に対して、基準レート設定 → CUSUM + スパイク監視を行う。

    2つのモード:
      安定期モード(既定, lambda0_curve=None):
        ベースライン窓 [stable_start_m, stable_start_m+baseline_len) から
        スカラ lambda0 を推定して固定。監視レンジは [stable_start_m, monitor_end_m]。
      安定化前モード(lambda0_curve を渡す):
        先行機種から推定した時変カーブ lambda0(t) を基準にする
        (earlylife_baseline.attach_curve_to_unit の出力)。配列長は df_unit の
        全行と一致させること。監視レンジは [0, monitor_end_m](初期から見る)。
        スパイク検定は条件付き二項でなく各月の直接ポアソン検定になる。

    alpha_spike が None ならスパイク検定はオフ(従来どおりCUSUMのみ)。
    指定すると単月スパイク検定が走り、burst_window >= 2 でバースト検定も加わる。

    返り値は監視レンジの各月に対する結果テーブル。主な追加列:
      月次p値 / スパイクアラート / バーストp値 / バーストアラート /
      アラート種別("drift"・"spike"・"burst" を + 連結) / 総合アラート
    従来の「アラート」列はドリフト(CUSUM)アラートのまま据え置き(後方互換)。
    """
    d = df_unit.sort_values(col_keizoku).reset_index(drop=True)

    early_mode = lambda0_curve is not None

    if early_mode:
        # 安定化前: カーブを基準に、初期から監視する。
        lam_curve = np.asarray(lambda0_curve, dtype=float)
        if len(lam_curve) != len(d):
            raise ValueError("lambda0_curve の長さが df_unit の行数と一致しません")
        mon_mask = d[col_keizoku] <= monitor_end_m
        mon = d[mon_mask].copy()
        lam_mon = lam_curve[mon_mask.to_numpy()]
        base = d.iloc[0:0]  # 安定化前はベースライン窓を使わない
    else:
        base_mask = (d[col_keizoku] >= stable_start_m) & (
            d[col_keizoku] < stable_start_m + baseline_len
        )
        mon_mask = (d[col_keizoku] >= stable_start_m) & (d[col_keizoku] <= monitor_end_m)
        base = d[base_mask]
        lambda0 = estimate_lambda0(
            base[col_usage].to_numpy(), base[col_fleet].to_numpy()
        )
        mon = d[mon_mask].copy()

    if len(mon) == 0:
        return mon

    usage = mon[col_usage].to_numpy()
    fleet = mon[col_fleet].to_numpy()

    lam_arg = lam_mon if early_mode else lambda0

    S, alarm, k = poisson_cusum(usage, fleet, lam_arg, R, h)
    mon["lambda0"] = lam_mon if early_mode else lambda0
    mon["期待故障数"] = (lam_mon if early_mode else lambda0) * fleet
    mon["参照値k"] = k
    mon["CUSUM"] = S
    mon["しきい値h"] = h
    mon["アラート"] = alarm  # ドリフト(CUSUM)アラート
    mon["基準モード"] = "earlylife" if early_mode else "stable"

    if alpha_spike is not None:
        if early_mode:
            p_s, a_s, p_b, a_b = spike_test(
                usage, fleet, lam_mon, alpha_spike,
                min_count=min_count, burst_window=burst_window,
                alpha_burst=alpha_burst,
            )
        else:
            p_s, a_s, p_b, a_b = spike_test(
                usage, fleet, lambda0, alpha_spike,
                min_count=min_count, burst_window=burst_window,
                alpha_burst=alpha_burst,
                baseline_count=float(base[col_usage].sum()),
                baseline_exposure=float(base[col_fleet].sum()),
            )
        mon["月次p値"] = p_s
        mon["スパイクアラート"] = a_s
        mon["バーストp値"] = p_b
        mon["バーストアラート"] = a_b

        kinds = []
        for dr, sp, bu in zip(alarm, a_s, a_b):
            parts = []
            if dr:
                parts.append("drift")
            if sp:
                parts.append("spike")
            if bu:
                parts.append("burst")
            kinds.append("+".join(parts))
        mon["アラート種別"] = kinds
        mon["総合アラート"] = alarm | a_s | a_b
    else:
        mon["アラート種別"] = np.where(alarm, "drift", "")
        mon["総合アラート"] = alarm

    return mon


def monitor_all(
    df: pd.DataFrame,
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
    **cols,
) -> pd.DataFrame:
    """全監視単位に monitor_unit を適用して結果を縦に結合する。

    unit_keys は UNIT_KEYS_WITH_SHA(販社別) か UNIT_KEYS_AGG_SHA(販社合算)。
    販社合算の場合は、呼び出し前に df を unit_keys + [年月, 経過月] で集約しておくこと
    (使用数・販売台数を合計する。aggregate_over_sha を参照)。
    """
    out = []
    for _, g in df.groupby(unit_keys, dropna=False):
        res = monitor_unit(
            g, stable_start_m, baseline_len, monitor_end_m, R, h,
            alpha_spike=alpha_spike, min_count=min_count,
            burst_window=burst_window, alpha_burst=alpha_burst,
            **cols,
        )
        if len(res):
            out.append(res)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def aggregate_over_sha(
    df: pd.DataFrame,
    col_usage: str = "月次使用数",
    col_fleet: str = "累積販売台数",
) -> pd.DataFrame:
    """販社を合算した監視単位(機種×部番)を作る。

    使用数と累積販売台数を販社横断で合計する。経過月・年月は機種側で共通の想定。
    """
    keys = UNIT_KEYS_AGG_SHA + ["年月", "経過月"]
    agg = (
        df.groupby(keys, dropna=False)[[col_usage, col_fleet]]
        .sum()
        .reset_index()
    )
    return agg
