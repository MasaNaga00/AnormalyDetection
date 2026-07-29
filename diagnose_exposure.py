"""
diagnose_exposure.py
====================
露出（分母）定義の比較診断。監視レンジを monitor_end_m 以上に延ばせるかを数字で判定する。

■ 問題設定
現行の露出は `fleet` = 累積販売台数（= 生存率 S(a)≡1 = 永久に退役しない仮定）。
レンズのように製品寿命が10〜15年あると、後半は退役した台数を数え続けるので

    E_t が過大  →  mu0 = lambda0 * E_t が過大
                →  k = (R-1)*lambda0*fleet/lnR が過大
                →  CUSUM 増分 (use - k) が負に寄る
                →  S が0に張り付き **本物の上昇をマスクする**（= 見逃し方向）

cusum_monitor.py の設計メモ「退役が進む末期は分母が過大になり、本物の上昇をマスクする」
がまさにこれ。monitor_end_m=60 の打ち切りは「鳴りすぎるから止める」ではなく
「検知能力が保証できないから止める」という性格の歯止めである。

■ このスクリプトが出す答え
露出定義ごとに 観測/期待比 O/E を経過月別に出し、
  (1) O/E が 1 の近傍に留まる経過月の上限 = その定義で信用できる monitor_end_m
  (2) 末期 O/E の向き（<1 なら見逃し方向 / >1 なら誤報方向）
を測る。O/E は CUSUM が見ている量そのもの（O/E>1 で S が伸び、<1 で沈む）なので、
そのまま検知の挙動に翻訳できる。

■ 比較する定義（w(a) = 販売から a ヶ月後の1台が今月生む期待出庫の重み）
    cum        w(a)=1                       現行（累積販売台数）
    roll_L     w(a)=1 (a<L), 0 (a>=L)       矩形。L ヶ月でピタリ全数退役
    lin_L      w(a)=max(0, 1-a/L)           線形減衰（崖を鈍らせた版）
    exp_H      w(a)=0.5**(a/H)              指数減衰（半減期 H）。0 にならない
    wbl(e,b)   w(a)=exp(-(a/e)**b)          ワイブル生存（当てはめ用・2パラメータ）
    E_t = sum_a s_{t-a} * w(a)   （月次販売台数 s との畳み込み）

w(a) は退役 S(a) と経年劣化 h(a) を分離せず **積 S(a)*h(a) を1本のネットカーネル**として
持つ。退役データが無いので個別同定はできないが、検知に必要なのは積だけなので支障はない。

■ スケール不変性（この診断が成立する根拠）
lambda0 は自己参照（ベースライン窓の C/E）なので、E を定数倍 c しても lambda0 が 1/c に
なって mu0 は不変。つまり **露出の絶対水準は検知に一切効かず、時間方向の「形」だけが効く**。
よって w の正規化は自由で、比較すべきは形状のみ。本スクリプトは全定義について
lambda0 を production と同じ手順（ベースライン窓の C/E）で取り直してから O/E を測る。

■ 使い方
    python diagnose_exposure.py                                  # 合成データでデモ（動作確認）
    python diagnose_exposure.py panel.csv                        # 実データ（全体をまとめて）
    python diagnose_exposure.py panel.csv --group-keys biz sf    # 集団別に分けて
    python diagnose_exposure.py panel.csv --max-elapsed 200 --exclude-trunc

Python は numpy / pandas のみ（scipy 非依存）。結果は CSV にも保存し Tableau で重ねられる。
"""

from __future__ import annotations

import sys
import types
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# state_logic_cusum の読み込み
# この診断は _prepare_panel / aggregate_units / CONFIG だけを使い、安定化前カーブ
# （earlylife_baseline）には触らない。同フォルダに無い環境でも動くようスタブを差す。
# ----------------------------------------------------------------------------
try:
    import state_logic_cusum as s
except ModuleNotFoundError as e:
    if "earlylife_baseline" not in str(e):
        raise
    _stub = types.ModuleType("earlylife_baseline")
    _stub.estimate_earlylife_curve = lambda *a, **k: None
    _stub.attach_curve_to_unit = lambda *a, **k: None
    sys.modules["earlylife_baseline"] = _stub
    import state_logic_cusum as s
    print("[注記] earlylife_baseline.py が見つからないためスタブで代替しました。"
          "この診断は安定化前カーブを使わないので結果に影響しません。")


# ============================================================================
# 露出カーネル
# ============================================================================
def kernel_cum(max_a: int) -> np.ndarray:
    """現行：w(a)=1。畳み込むと累積販売台数と厳密に一致する（恒等性検証に使う）。"""
    return np.ones(max_a + 1, dtype=float)


def kernel_rect(max_a: int, L: int) -> np.ndarray:
    """矩形生存：L ヶ月でピタリ全数退役。崖があるので販売終了後に E=0 になり得る。"""
    return (np.arange(max_a + 1) < L).astype(float)


def kernel_linear(max_a: int, L: int) -> np.ndarray:
    """線形減衰：a=L で 0 に到達。矩形の崖を鈍らせた版。"""
    a = np.arange(max_a + 1, dtype=float)
    return np.maximum(0.0, 1.0 - a / float(L))


def kernel_exp(max_a: int, half_life: float) -> np.ndarray:
    """指数減衰：半減期 half_life ヶ月。厳密に 0 にならないので E=0 病理が起きない。"""
    a = np.arange(max_a + 1, dtype=float)
    return np.power(0.5, a / float(half_life))


def kernel_weibull(max_a: int, eta: float, beta: float) -> np.ndarray:
    """ワイブル生存 exp(-(a/eta)^beta)。beta=1 で指数、beta を上げると矩形に近づく。"""
    a = np.arange(max_a + 1, dtype=float)
    return np.exp(-np.power(a / float(eta), float(beta)))


def build_default_kernels(max_a: int, life_candidates=(48, 72, 96, 120, 144, 180)) -> dict:
    """既定の比較セット。life_candidates は製品寿命の想定候補（ヶ月）。"""
    ks = {"cum": kernel_cum(max_a)}
    for L in life_candidates:
        ks[f"roll_{L}"] = kernel_rect(max_a, L)
    for L in life_candidates:
        ks[f"lin_{L}"] = kernel_linear(max_a, L)
    for H in life_candidates:
        ks[f"exp_{H}"] = kernel_exp(max_a, H)
    return ks


# ============================================================================
# 月次販売台数の復元
# ============================================================================
def build_unit_sales(units: pd.DataFrame, truncation_tol: int = 2,
                     verbose: bool = True) -> pd.DataFrame:
    """単位（機種×部番, 販社合算）ごとに月次販売台数 s を累積販売台数の階差で復元する。

    入力 units は s.aggregate_units() の出力（fleet = 累積販売台数の販社合算）。
    production の露出はこの fleet なので、ここを起点にすれば定義比較が production と
    厳密に対応する。

    - 各単位の先頭月の fleet はストック。その時点の経過月が truncation_tol 以下なら
      「それ以前の販売は無い」と見て先頭月の販売とみなす。超えていれば **左打ち切り**
      （初期の販売年齢構成が復元できない）として flag を立てる。
    - 階差が負になる月は 0 にクリップし件数を報告する（累積の非単調 = cummax 補正漏れ検出）。

    Returns: units に s_new(月次販売台数), left_trunc(bool), uid(文字列キー) を付けたもの
    """
    out, n_trunc, n_neg = [], 0, 0
    for key, g in units.groupby(["biz", "dev", "part"], sort=False):
        g = g.sort_values("ym").reset_index(drop=True)
        fleet = g["fleet"].to_numpy(dtype=float)
        e0 = g["elapsed"].iloc[0]
        trunc = (not pd.notna(e0)) or (float(e0) > truncation_tol)
        n_trunc += int(trunc)

        s_new = np.diff(fleet, prepend=0.0)      # 先頭月は fleet[0] そのもの
        n_neg += int((s_new < 0).sum())
        s_new = np.maximum(s_new, 0.0)

        g = g.copy()
        g["s_new"] = s_new
        g["left_trunc"] = trunc
        g["uid"] = "|".join(str(x) for x in (key if isinstance(key, tuple) else (key,)))
        out.append(g)

    res = pd.concat(out, ignore_index=True)
    if verbose:
        n_unit = res["uid"].nunique()
        print(f"[月次販売台数の復元] 単位数={n_unit}  左打ち切り={n_trunc}単位  "
              f"階差負(0クリップ)={n_neg}月")
        if n_trunc:
            print("  ※ 左打ち切り単位は初期の販売年齢構成が不明。--exclude-trunc で除外可")
        if n_neg:
            print("  ※ 階差負が多い場合は投入前チェックの cummax 補正が漏れている可能性")
    return res


# ============================================================================
# 畳み込みの一括化（性能）
# ============================================================================
def _lag_matrix(s_new: np.ndarray) -> np.ndarray:
    """M[t, a] = s_{t-a}（t<a なら0）を stride トリックでコピーなしに作る。

    E_t^(j) = sum_a s_{t-a} w_j(a) なので、複数カーネルの露出は M @ W の1回の
    行列積でまとめて出せる（カーネルごとに np.convolve を回すより2桁速い）。
    """
    n = len(s_new)
    pad = np.concatenate([np.zeros(n - 1, dtype=float), np.asarray(s_new, dtype=float)])
    it = pad.strides[0]
    M = np.lib.stride_tricks.as_strided(pad[n - 1:], shape=(n, n),
                                        strides=(it, -it), writeable=False)
    return M


def _exposures(s_new: np.ndarray, W: np.ndarray) -> np.ndarray:
    """W: (max_a+1, K) のカーネル行列 → 露出 (n, K) を一括計算。"""
    n = len(s_new)
    return _lag_matrix(s_new) @ W[:n, :]


# ============================================================================
# 露出定義ごとの O/E（コア）
# ============================================================================
def _prepare_units(units_sales: pd.DataFrame, cfg: dict,
                   group_keys: list[str] | None, max_elapsed: int | None,
                   exclude_trunc: bool, verbose: bool):
    """単位ごとの配列を一度だけ取り出す（ループ不変量のホイスト）。

    compute_oe と fit_kernel_grid で共有する。カーネル候補ごとに作り直すと
    実データ規模（2000単位）で groupby が支配的になるため。
    """
    df = units_sales
    if exclude_trunc:
        before = df["uid"].nunique()
        df = df[~df["left_trunc"]]
        if verbose:
            print(f"[左打ち切り除外] {before} → {df['uid'].nunique()} 単位")
    if df.empty:
        raise ValueError("対象単位が0件です")
    if group_keys:
        miss = [k for k in group_keys if k not in df.columns]
        if miss:
            raise ValueError(f"group_keys に無い列: {miss}")

    lo = int(cfg["stable_start_m"])
    hi_win = lo + int(cfg["baseline_len"])
    max_e = int(max_elapsed) if max_elapsed else int(df["elapsed"].max())

    prepared = []
    for uid, g in df.groupby("uid", sort=False):
        g = g.sort_values("ym")
        elapsed = g["elapsed"].to_numpy(dtype=float)
        use = g["use"].to_numpy(dtype=float)
        win = (elapsed >= lo) & (elapsed < hi_win)
        e_int = elapsed.astype(int)
        sel = (elapsed >= lo) & (elapsed <= max_e) & (e_int >= 0) & (e_int <= max_e)
        if not win.any() or not sel.any():
            continue
        gk = tuple(str(g[k].iloc[0]) for k in group_keys) if group_keys else ("ALL",)
        prepared.append(dict(
            uid=uid, gk=gk,
            s_new=g["s_new"].to_numpy(dtype=float),
            use=use, fleet=g["fleet"].to_numpy(dtype=float),
            win=win, sel=sel, eidx=e_int[sel], C=float(use[win].sum()),
        ))
    if not prepared:
        raise ValueError("O/E を計算できる単位がありません（ベースライン窓が空？）")
    meta = dict(lo=lo, hi_win=hi_win, max_e=max_e)
    return prepared, meta


def _trim_set(prepared: list, trim_frac: float, meta: dict, verbose: bool) -> set:
    """全期間 O/E が上位 trim_frac の単位を特定する（現行定義 cum で判定）。

    真の異常単位が末期の O/E を押し上げるのを防ぐ。判定は cum 一本で行い、
    同じ除外集合を全定義に適用するので定義間の比較は公平に保たれる。
    """
    if not trim_frac or trim_frac <= 0:
        return set()
    oes = []
    for u in prepared:
        Ew = float(u["fleet"][u["win"]].sum())
        if Ew <= 0:
            continue
        mu0 = (u["C"] / Ew) * u["fleet"]
        tot_E = float(mu0[u["sel"]].sum())
        if tot_E > 0:
            oes.append((u["uid"], float(u["use"][u["sel"]].sum()) / tot_E))
    if len(oes) < 10:
        return set()
    arr = pd.DataFrame(oes, columns=["uid", "oe"])
    thr = float(arr["oe"].quantile(1.0 - trim_frac))
    drop = set(arr.loc[arr["oe"] > thr, "uid"])
    if verbose:
        print(f"[トリム] 全期間 O/E > {thr:.2f} の {len(drop)}単位をプールから除外"
              f"（真の異常が末期 O/E を押し上げるのを防ぐ）")
    return drop

def compute_oe(units_sales: pd.DataFrame, cfg: dict, kernels: dict,
               group_keys: list[str] | None = None,
               max_elapsed: int | None = None,
               exclude_trunc: bool = False,
               trim_frac: float = 0.10,
               verbose: bool = True):
    """露出定義ごとに経過月別の観測/期待比 O/E をプールして返す。

    production と同じ手順を、露出定義だけ差し替えて再現する:
        1. E_t^(def)      = sum_a s_{t-a} w(a)
        2. lambda0^(def)  = (窓内 use 合計) / (窓内 E^(def) 合計)     窓=[stable_start_m, +baseline_len)
        3. mu0_t^(def)    = lambda0^(def) * E_t^(def)
        4. O/E(t)         = sum_i use_{i,t} / sum_i mu0_{i,t}         期待値重みでプール

    ベースライン窓では構成上 O/E≈1 になる。見るべきは窓より後の乖離。

    trim_frac: 全期間 O/E が上位 trim_frac の単位をプールから除外（真の異常単位が
        末期の O/E を押し上げるのを防ぐ）。判定は現行定義 cum で行い全定義に同じ集合を適用
        するので、定義間の比較は公平に保たれる。0 で無効。

    Returns (oe, patho, meta)
        oe    : group, kernel, elapsed, O, E, n_unit, oe
        patho : kernel ごとの E<=0 病理カウント
        meta  : 窓・レンジ・恒等性検証の結果
    """
    prepared, meta0 = _prepare_units(units_sales, cfg, group_keys, max_elapsed,
                                     exclude_trunc, verbose)
    lo, hi_win, max_e = meta0["lo"], meta0["hi_win"], meta0["max_e"]
    kn_list = list(kernels.keys())
    drop = _trim_set(prepared, trim_frac, meta0, verbose)

    max_a = max(len(k) for k in kernels.values())
    W = np.column_stack([np.asarray(kernels[kn], dtype=float)[:max_a]
                         if len(kernels[kn]) >= max_a
                         else np.pad(kernels[kn], (0, max_a - len(kernels[kn])))
                         for kn in kn_list])          # (max_a, K)
    ci = kn_list.index("cum") if "cum" in kn_list else None

    groups = sorted({u["gk"] for u in prepared if u["uid"] not in drop})
    gidx = {g: i for i, g in enumerate(groups)}
    nG, nE, K = len(groups), max_e + 1, len(kn_list)
    O_sum = np.zeros((K, nG, nE))
    E_sum = np.zeros((K, nG, nE))
    U_cnt = np.zeros((K, nG, nE))
    p_zero = np.zeros(K, dtype=np.int64)
    p_zero_use = np.zeros(K, dtype=np.int64)
    identity_max_diff = 0.0

    for u in prepared:
        if u["uid"] in drop:
            continue
        gi = gidx[u["gk"]]
        s_new, use, win, sel, eidx = u["s_new"], u["use"], u["win"], u["sel"], u["eidx"]
        n = len(s_new)

        E_all = _exposures(s_new, W)                      # (n, K) 全定義まとめて
        if ci is not None:                                # 恒等性検証: w=1 → 累積販売台数
            identity_max_diff = max(identity_max_diff,
                                    float(np.max(np.abs(E_all[:, ci] - u["fleet"]))))

        zero = E_all <= 0
        p_zero += zero.sum(axis=0)
        p_zero_use += (zero & (use > 0)[:, None]).sum(axis=0)

        Ew = E_all[win].sum(axis=0)                       # (K,) 窓内の露出合計
        good = Ew > 0
        if not good.any():
            continue
        mu0 = np.zeros_like(E_all)
        mu0[:, good] = E_all[:, good] * (u["C"] / Ew[good])   # production と同じ自己参照

        wO = np.bincount(eidx, weights=use[sel], minlength=nE)
        for j in np.flatnonzero(good):
            O_sum[j, gi] += wO
            E_sum[j, gi] += np.bincount(eidx, weights=mu0[sel, j], minlength=nE)
            U_cnt[j, gi] += np.bincount(eidx, minlength=nE)

    rows = []
    for j, kn in enumerate(kn_list):
        for g, gi in gidx.items():
            el = np.flatnonzero((O_sum[j, gi] > 0) | (E_sum[j, gi] > 0))
            if el.size == 0:
                continue
            rows.append(pd.DataFrame(dict(
                group="+".join(g), kernel=kn, elapsed=el,
                O=O_sum[j, gi][el], E=E_sum[j, gi][el],
                n_unit=U_cnt[j, gi][el].astype(int))))
    oe = pd.concat(rows, ignore_index=True)
    oe["oe"] = np.where(oe["E"] > 0, oe["O"] / oe["E"], np.nan)

    patho = pd.DataFrame([dict(kernel=kn, months_E0=int(p_zero[j]),
                               months_E0_with_use=int(p_zero_use[j]))
                          for j, kn in enumerate(kn_list)])
    meta = dict(identity_max_diff=identity_max_diff, lo=lo, hi_win=hi_win,
                max_e=max_e, n_unit=len(prepared) - len(drop))

    if verbose and ci is not None:
        ok = identity_max_diff < 1e-6
        print(f"[恒等性検証] w(a)=1 の畳み込み露出 vs 累積販売台数  最大差={identity_max_diff:.6g}"
              f"  {'→ 差0（実装OK）' if ok else '→ ★不一致。階差復元/左打ち切りを確認'}")
    return oe, patho, meta


# ============================================================================
# 平坦性の評価
# ============================================================================
def summarize_flatness(oe: pd.DataFrame, meta: dict, tol: float = 0.25,
                       eval_from: int | None = None, min_expected: float = 20.0):
    """定義ごとに O/E の平坦性を評価し「信用できる経過月の上限」を出す。

    tol        : |log(O/E)| の許容幅。0.25 ≒ O/E が 0.78〜1.28。
    eval_from  : 評価開始経過月（既定はベースライン窓の直後）。
    min_expected: 期待件数がこれ未満の経過月はノイズが大きいので判定に使わない。

    usable_max : eval_from から**連続して** |log(O/E)|<=tol を保てた最大の経過月。
                 = その露出定義で monitor_end_m をどこまで延ばせるかの目安。
    mad_log    : 平均 |log(O/E)|。小さいほど形が合っている。
    oe_tail    : 末期（レンジ上位30%）のプール O/E。<1 で見逃し方向、>1 で誤報方向。
    """
    hi_win, max_e = meta["hi_win"], meta["max_e"]
    ef = int(eval_from) if eval_from is not None else hi_win
    tail_from = max(ef, int(max_e * 0.7))

    rows = []
    for (grp, kn), g in oe.groupby(["group", "kernel"], sort=False):
        g = g.sort_values("elapsed")
        gv = g[(g["elapsed"] >= ef) & (g["E"] >= min_expected)]
        if gv.empty:
            rows.append(dict(group=grp, kernel=kn, usable_max=np.nan, mad_log=np.nan,
                             oe_tail=np.nan, n_pts=0))
            continue
        logoe = np.log(np.clip(gv["oe"].to_numpy(dtype=float), 1e-12, None))
        el = gv["elapsed"].to_numpy()
        usable = np.nan
        for i, flag in enumerate(np.abs(logoe) <= tol):
            if not flag:
                break
            usable = el[i]
        gt = gv[gv["elapsed"] >= tail_from]
        oe_tail = (float(gt["O"].sum() / gt["E"].sum())
                   if len(gt) and gt["E"].sum() > 0 else np.nan)
        rows.append(dict(group=grp, kernel=kn, usable_max=usable,
                         mad_log=float(np.mean(np.abs(logoe))),
                         oe_tail=oe_tail, n_pts=int(len(gv))))
    return pd.DataFrame(rows).sort_values(["group", "mad_log"]).reset_index(drop=True)


def _sweep_deviance(prepared: list, drop: set, meta: dict,
                    cand: list[tuple[float, float]], ef: int) -> pd.DataFrame:
    """候補 (eta,beta) 全点の deviance を **単位ループ1回** で評価する。

    候補ごとに compute_oe を呼び直すと groupby と前処理が候補数だ�け繰り返され、
    実データ規模（2000単位×177候補）で数分かかる。カーネルを行列 W にまとめ、
    露出を M @ W の1回の行列積で全候補ぶん同時に出すことで1パスに畳む。
    """
    max_e = meta["max_e"]
    nE = max_e + 1
    max_a = max(len(u["s_new"]) for u in prepared)
    W = np.column_stack([kernel_weibull(max_a - 1, eta, beta) for eta, beta in cand])
    K = len(cand)
    O_sum = np.zeros(nE)
    E_sum = np.zeros((K, nE))

    for u in prepared:
        if u["uid"] in drop:
            continue
        s_new, use, win, sel, eidx = u["s_new"], u["use"], u["win"], u["sel"], u["eidx"]
        E_all = _exposures(s_new, W)                       # (n, K)
        Ew = E_all[win].sum(axis=0)
        good = Ew > 0
        if not good.any():
            continue
        O_sum += np.bincount(eidx, weights=use[sel], minlength=nE)
        mu0 = E_all[:, good] * (u["C"] / Ew[good])
        for col, j in enumerate(np.flatnonzero(good)):
            E_sum[j] += np.bincount(eidx, weights=mu0[sel, col], minlength=nE)

    el = np.arange(nE)
    rows = []
    for j, (eta, beta) in enumerate(cand):
        m = (el >= ef) & (E_sum[j] > 0)
        if not m.any():
            continue
        O = O_sum[m]
        E = E_sum[j][m]
        E = E * (O.sum() / E.sum())                        # スケールをプロファイル・アウト
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(O > 0, O * np.log(np.clip(O / E, 1e-300, None)), 0.0)
        rows.append(dict(eta=float(eta), beta=float(beta),
                         deviance=2.0 * float(np.sum(term - (O - E))),
                         n_pts=int(m.sum())))
    return pd.DataFrame(rows)


def fit_kernel_grid(units_sales: pd.DataFrame, cfg: dict,
                    group_keys: list[str] | None = None,
                    max_elapsed: int | None = None,
                    eta_grid=(36, 48, 60, 72, 84, 96, 108, 120, 144, 168, 192, 240),
                    beta_grid=(0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.6, 3.0),
                    refine: bool = True,
                    exclude_trunc: bool = False, trim_frac: float = 0.10,
                    eval_from: int | None = None, verbose: bool = True):
    """ワイブル生存カーネル exp(-(a/eta)^beta) を格子探索で当てる（scipy 非依存）。

    目的関数はポアソン deviance。測りたいのは「形」なのでスケールはプロファイル・アウト
    （全期間の O/E で正規化してから測る）:
        dev = 2 * sum[ O*ln(O/E) - (O-E) ]
    ※ production の lambda0 はベースライン窓から取るが、当てはめの段では形状だけを
       評価したいのでこの扱いにする。O/E テーブル側は production 準拠（窓で正規化）。

    refine=True で2段階探索（粗格子 → 最良点の周りを細格子）。粗格子だけだと真値を
    外すことがあるため既定で有効（合成データで eta=108,beta=2.0 の厳密復元を確認済み）。

    Returns (best, grid_tbl)
    """
    ef = eval_from if eval_from is not None else int(cfg["stable_start_m"]) + int(cfg["baseline_len"])
    prepared, meta = _prepare_units(units_sales, cfg, group_keys, max_elapsed,
                                    exclude_trunc, verbose=False)
    drop = _trim_set(prepared, trim_frac, meta, verbose=False)

    cand = [(e, b) for e in eta_grid for b in beta_grid]
    grid = _sweep_deviance(prepared, drop, meta, cand, ef)
    if grid.empty:
        return None, grid
    grid["stage"] = "coarse"

    if refine:
        b = grid.sort_values("deviance").iloc[0]
        de, db = max(3.0, b["eta"] * 0.12), max(0.1, b["beta"] * 0.2)
        etas = sorted({int(x) for x in np.round(np.linspace(b["eta"] - de, b["eta"] + de, 9))
                       if x >= 6})
        betas = sorted({float(x) for x in
                        np.round(np.linspace(max(0.4, b["beta"] - db), b["beta"] + db, 9), 2)})
        fine = _sweep_deviance(prepared, drop, meta, [(e, bb) for e in etas for bb in betas], ef)
        if not fine.empty:
            fine["stage"] = "fine"
            grid = pd.concat([grid, fine], ignore_index=True)

    grid = grid.sort_values("deviance").reset_index(drop=True)
    best = grid.iloc[0].to_dict()
    if verbose:
        half = best["eta"] * (np.log(2.0) ** (1.0 / best["beta"]))
        print(f"[カーネル当てはめ] 最良 ワイブル生存 eta={best['eta']:.0f}ヶ月 "
              f"beta={best['beta']:.2f}  (deviance={best['deviance']:.1f})")
        print(f"  → 半減齢 ≈ {half:.0f}ヶ月（{half/12:.1f}年）で稼働台数が半分になる形")
        print(f"  → 実装するなら kernel_weibull(max_a, {best['eta']:.0f}, {best['beta']:.2f})")
        print("  ※ これは退役 S(a) と経年劣化 h(a) の**積**の形。個別の同定はできない")
    return best, grid


# ============================================================================
# 読み方の自動判定
# ============================================================================
def interpret(flat: pd.DataFrame, patho: pd.DataFrame, meta: dict, cfg: dict,
              max_a: int | None = None):
    print("\n" + "=" * 78)
    print("判定")
    print("=" * 78)
    cur_end = int(cfg["monitor_end_m"])

    for grp, g in flat.groupby("group", sort=False):
        cum = g[g["kernel"] == "cum"]
        if cum.empty:
            continue
        cum = cum.iloc[0]
        print(f"\n■ 集団 {grp}")
        print(f"  現行(cum): 使用可能上限={cum['usable_max']}ヶ月  "
              f"末期O/E={cum['oe_tail']:.2f}  平均|log O/E|={cum['mad_log']:.3f}")

        if pd.notna(cum["oe_tail"]):
            if cum["oe_tail"] < 0.8:
                print("  → 末期 O/E < 1：累積露出が実稼働台数を**過大評価**（退役の影響が支配的）。")
                print("     mu0 が大きすぎて CUSUM が沈む＝**見逃し方向**。露出定義の修正が有効。")
            elif cum["oe_tail"] > 1.25:
                print("  → 末期 O/E > 1：期待を上回って出庫（経年劣化 h(a) の上昇が支配的）。")
                print("     露出を絞ると O/E がさらに上がり誤報方向。**露出定義では直らない**。")
                print("     経年上昇を正常扱いするなら lambda0(t) 側（カーブ）で吸収する設計が必要。")
            else:
                print("  → 末期 O/E ≈ 1：累積露出のままでも形が保てている。"
                      "レンジ延長の障害は露出定義ではない可能性。")

        alt = g[g["kernel"] != "cum"].sort_values("mad_log")
        if not alt.empty:
            b = alt.iloc[0]
            gain = cum["mad_log"] - b["mad_log"]
            print(f"  最良の代替: {b['kernel']}  使用可能上限={b['usable_max']}ヶ月  "
                  f"平均|log O/E|={b['mad_log']:.3f}（改善 {gain:+.3f}）")
            if pd.notna(b["usable_max"]):
                if b["usable_max"] > cur_end:
                    print(f"     → monitor_end_m を {cur_end} → {b['usable_max']:.0f} ヶ月へ"
                          f"延ばせる見込み")
                else:
                    print(f"     → この定義でも現行 monitor_end_m={cur_end} を超えられない")
            if gain < 0.02:
                print("     → 改善が小さい。露出定義の変更は費用対効果が低い")

    bad = patho[patho["months_E0_with_use"] > 0]
    print("\n■ E=0 病理（露出0なのに出庫がある月）")
    if not bad.empty:
        print("  この月は mu0=0 → p≈0 で**必ず誤発火**する（設計メモ4.7節の分母側での再来）。")
        for _, r in bad.sort_values("months_E0_with_use", ascending=False).head(8).iterrows():
            print(f"    {r['kernel']:<10} {int(r['months_E0_with_use']):>6}月"
                  f"  (E=0 の月 計{int(r['months_E0'])})")
        print("  → 崖のある定義(roll_*/lin_*)は販売終了後にこれを起こす。exp_*/ワイブルは起きない。")
    else:
        print("  該当なし")

    if max_a is not None:
        safe = [r["kernel"] for _, r in patho.iterrows()
                if r["months_E0"] == 0 and str(r["kernel"]).startswith(("roll_", "lin_"))]
        if safe:
            print(f"  ※ 注意：{', '.join(safe[:6])} は E=0 が0件だが、崖の位置が観測レンジ"
                  f"(最大経過月{max_a})の外にあるだけの可能性がある。")
            print("     レンジを延ばすと崖に到達して誤発火し始めるので、本番採用は避けるのが安全。")

    ok = flat[(flat["kernel"].str.startswith(("exp_", "wbl_")))].sort_values("mad_log")
    if not ok.empty:
        b = ok.iloc[0]
        print(f"\n■ 本番採用の推奨（崖なし定義に限定）: {b['kernel']}"
              f"  使用可能上限={b['usable_max']}ヶ月  平均|log O/E|={b['mad_log']:.3f}")

    print("\n■ 次の一手")
    print("  1. 末期 O/E<1 が確認できたら、崖の無い定義(exp_* / ワイブル)で露出を差し替える")
    print("  2. カテゴリ別に monitor_end_m を分ける（カメラ約3年 / レンズ約10年）")
    print("  3. 切替は並走で。monitor_end_m は現行のまま新露出を入れ、backtest_labeled で")
    print("     既知インシデントの検知が落ちないことを確認してから延長する")


# ============================================================================
# メイン
# ============================================================================
def run_diagnosis(panel_path: str | None = None, cfg: dict | None = None,
                  group_keys: list[str] | None = None,
                  max_elapsed: int | None = None,
                  life_candidates=(48, 72, 96, 120, 144, 180),
                  tol: float = 0.25, exclude_trunc: bool = False,
                  trim_frac: float = 0.10, do_fit: bool = True,
                  out_prefix: str = "exposure_diag"):
    """診断一式を実行して CSV に保存する。"""
    cfg = dict(s.CONFIG) if cfg is None else dict(cfg)

    if panel_path is None:
        print("[入力] 合成デモデータ（動作確認モード：真の生存 = ワイブル eta=108, beta=2.0）")
        panel_raw = _demo_panel()
    else:
        print(f"[入力] {panel_path}")
        if str(panel_path).lower().endswith((".xlsx", ".xls")):
            panel_raw = pd.read_excel(panel_path, sheet_name=cfg.get("panel_sheet", 0))
        else:
            panel_raw = pd.read_csv(panel_path, encoding="utf-8-sig")

    panel = s._prepare_panel(panel_raw, cfg)
    units = s.aggregate_units(panel)
    print(f"  パネル {len(panel)}行 → {units.groupby(['biz','dev','part']).ngroups}単位 "
          f"/ {len(units)}単位月  経過月 max={int(units['elapsed'].max())}")

    us = build_unit_sales(units)
    max_a = int(max_elapsed) if max_elapsed else int(us["elapsed"].max())
    kernels = build_default_kernels(max_a + 2, life_candidates=life_candidates)

    # 当てはめは先に済ませ、得られたカーネルを比較表にも並べる（既存定義との勝ち負けを見る）
    best, grid = (None, None)
    if do_fit:
        print()
        best, grid = fit_kernel_grid(us, cfg, group_keys=group_keys, max_elapsed=max_elapsed,
                                    exclude_trunc=exclude_trunc, trim_frac=trim_frac)
        if best is not None:
            kernels[f"wbl_fit"] = kernel_weibull(max_a + 2, best["eta"], best["beta"])
        if grid is not None and not grid.empty:
            grid.to_csv(f"{out_prefix}_kernel_grid.csv", index=False, encoding="utf-8-sig")

    print(f"\n[比較定義] {len(kernels)}種（cum + roll/lin/exp × {list(life_candidates)}"
          f"{' + wbl_fit' if best is not None else ''}）  経過月 0〜{max_a}")
    oe, patho, meta = compute_oe(us, cfg, kernels, group_keys=group_keys,
                                max_elapsed=max_elapsed, exclude_trunc=exclude_trunc,
                                trim_frac=trim_frac)
    flat = summarize_flatness(oe, meta, tol=tol)

    print("\n" + "=" * 78)
    print(f"観測/期待比 O/E（ベースライン窓=経過月{meta['lo']}〜{meta['hi_win']-1} で O/E≈1 に正規化）")
    print(f"  対象 {meta['n_unit']}単位  1.00 から離れるほど期待とズレている")
    print("=" * 78)
    show = ["cum"] + flat[flat["kernel"] != "cum"].sort_values("mad_log")["kernel"] \
                        .drop_duplicates().head(3).tolist()
    bins = list(range(0, max_a + 13, 12))
    tmp = oe[oe["kernel"].isin(show)].copy()
    tmp["経過月"] = pd.cut(tmp["elapsed"], bins=bins, right=False,
                           labels=[f"{b:>3}-{b+11}" for b in bins[:-1]])
    piv = (tmp.groupby(["group", "経過月", "kernel"], observed=True)
              .agg(O=("O", "sum"), E=("E", "sum")).reset_index())
    piv["oe"] = piv["O"] / piv["E"]
    tbl = piv.pivot_table(index=["group", "経過月"], columns="kernel", values="oe", observed=True)
    tbl = tbl[[k for k in show if k in tbl.columns]]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(tbl.round(3).to_string())

    print("\n" + "=" * 78)
    print(f"平坦性サマリ（|log O/E|<={tol} を連続で保てる上限 = monitor_end_m の目安）")
    print("=" * 78)
    with pd.option_context("display.width", 200):
        print(flat.head(12).to_string(index=False))

    interpret(flat, patho, meta, cfg, max_a)

    oe.to_csv(f"{out_prefix}_oe.csv", index=False, encoding="utf-8-sig")
    flat.to_csv(f"{out_prefix}_flatness.csv", index=False, encoding="utf-8-sig")
    patho.to_csv(f"{out_prefix}_pathology.csv", index=False, encoding="utf-8-sig")
    print(f"\n[保存] {out_prefix}_oe.csv（Tableau で経過月×定義の O/E を重ねる）"
          f" / _flatness.csv / _pathology.csv"
          + (" / _kernel_grid.csv" if best is not None else ""))
    return dict(oe=oe, flatness=flat, pathology=patho, best_kernel=best, grid=grid, meta=meta)


# ============================================================================
# 合成デモ（動作確認用）
# ============================================================================
def _demo_panel(n_dev: int = 20, n_part: int = 5, n_month: int = 180,
                life_eta: float = 108.0, life_beta: float = 2.0,
                seed: int = 7) -> pd.DataFrame:
    """真の生存カーネルが分かっている合成パネル（レンズ想定）。

    販売は最初の42ヶ月のみ、その後は保有だけが残り退役していく。
    真の生存 = ワイブル(eta=108, beta=2.0)。故障率 h(a) は一定＝真のドリフトは無い。
    → 診断が「cum は末期 O/E<1（見逃し方向）」「真値近傍のカーネルが最良」を
      復元できれば実装が正しい、という自己検証になる。
    """
    rng = np.random.default_rng(seed)
    a = np.arange(n_month + 1, dtype=float)
    surv = np.exp(-np.power(a / life_eta, life_beta))
    h = 3.0e-4                                        # 1台1ヶ月あたり故障率（一定）

    rows = []
    for d in range(n_dev):
        s_new = np.zeros(n_month)
        peak = rng.uniform(3000, 12000)
        for t in range(42):                           # 販売カーブ（立ち上がり→減衰）
            s_new[t] = peak * np.exp(-((t - 8) ** 2) / (2 * 12.0 ** 2))
        cum = np.cumsum(s_new)
        live = np.convolve(s_new, surv)[:n_month]     # 真の稼働台数
        for p in range(n_part):
            scale = rng.uniform(0.4, 2.0)
            for t in range(n_month):
                rows.append(dict(
                    事業コード="B1", 開発コード=f"DEV{d:02d}", 部番=f"P{p:02d}",
                    販社="SUM", 年月=_ym_seq(202001, t), 経過月=t,
                    月次使用数=int(rng.poisson(max(h * scale * live[t], 0.0))),
                    累積販売台数=float(cum[t]),
                    **{"SF-コード": f"SF{d % 4}", "ランク": f"R{d % 2}"},
                ))
    return pd.DataFrame(rows)


def _ym_seq(start_ym: int, k: int) -> int:
    y, m = divmod(start_ym, 100)
    tot = (y * 12 + (m - 1)) + k
    return (tot // 12) * 100 + (tot % 12) + 1


# ============================================================================
if __name__ == "__main__":
    args = sys.argv[1:]
    panel, group_keys, max_elapsed, exclude_trunc = None, None, None, False
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--group-keys":
            group_keys = []
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                group_keys.append(args[i]); i += 1
            continue
        if a == "--max-elapsed":
            max_elapsed = int(args[i + 1]); i += 2; continue
        if a == "--exclude-trunc":
            exclude_trunc = True; i += 1; continue
        if a == "--no-fit":
            i += 1; continue
        panel = a; i += 1

    run_diagnosis(panel, group_keys=group_keys, max_elapsed=max_elapsed,
                  exclude_trunc=exclude_trunc, do_fit=("--no-fit" not in args))
