# -*- coding: utf-8 -*-
"""
config_loader.py — カテゴリ別設定の切り替え

設定は「共通（settings.py）＋ カテゴリ差分（categories/<名前>.py）」の2段。
カテゴリファイルは先頭で `from settings import *` し、違う値だけ上書きする。

使い方
------
コマンドライン（全ツール共通）:
    python run_month.py --cat レンズ
    python tune_c_alpha.py --cat レンズ              # パネルは PANEL_PATH を使う
    python tune_c_alpha.py panel.csv --cat レンズ    # 明示も可

Jupyter:
    import config_loader as cl
    st = cl.use("レンズ")          # 以降の import settings はレンズ設定を指す
    import tune_c_alpha as t       # （import 済みのツールは自動で読み直す）
    scan = pd.read_csv(cl.work_path("scan_c.csv"), encoding="utf-8-sig")

`--cat` を付けなければ settings.py だけが使われ、従来と完全に同じ挙動になる。

仕組み
------
カテゴリファイルを読み込んだモジュールを sys.modules["settings"] に差し込む。
各ツールは `import settings as st` のままでカテゴリ設定を受け取れる。
build_cfg() はカテゴリ側の値で組み立て直す（そのままだと共通側の値を読むため）。
"""
from __future__ import annotations

import os
import sys
import glob
import importlib
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
CATEGORY_DIR = os.path.join(HERE, "categories")

# settings を import 時に束縛するツール。use() で切り替えたとき読み直す。
_TOOLS = ("run_month", "tune_c_alpha", "tune_b_minoe", "check_rank",
          "inject_power", "check_panel", "diagnose_signal_c", "check_settings")

_base = None          # 共通の settings モジュール
_active = None        # いま sys.modules["settings"] にあるもの


def _load_base(reload: bool = False):
    global _base
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    if _base is None:
        cur = sys.modules.get("settings")
        if cur is not None and getattr(cur, "_IS_CATEGORY", False):
            cur = None
        _base = cur or importlib.import_module("settings")
    if reload:
        sys.modules["settings"] = _base
        importlib.reload(_base)
    return _base


def category_path(cat: str, cat_dir: str | None = None) -> str:
    """カテゴリ名（またはファイルパス）→ ファイルの絶対パス。"""
    if cat.endswith(".py") or os.sep in cat or "/" in cat:
        p = os.path.abspath(cat)
    else:
        p = os.path.join(cat_dir or CATEGORY_DIR, f"{cat}.py")
    if not os.path.exists(p):
        have = ", ".join(list_categories(cat_dir)) or "（なし）"
        raise SystemExit(f"[エラー] カテゴリ設定が見つかりません: {p}\n"
                         f"        あるカテゴリ: {have}")
    return p


def list_categories(cat_dir: str | None = None) -> list[str]:
    """categories/ にあるカテゴリ名。先頭が _ のファイル（テンプレート等）は除く。"""
    d = cat_dir or CATEGORY_DIR
    fs = sorted(glob.glob(os.path.join(d, "*.py")))
    return [os.path.splitext(os.path.basename(f))[0] for f in fs
            if not os.path.basename(f).startswith("_")]


def activate(cat: str | None = None, reload: bool = False):
    """カテゴリ設定を有効にして、そのモジュールを返す。None なら共通のみ。"""
    global _active
    base = _load_base(reload=reload)
    if not cat:
        sys.modules["settings"] = base
        _active = base
        return base

    path = category_path(cat)
    name = os.path.splitext(os.path.basename(path))[0]
    # カテゴリファイル内の `from settings import *` が共通側を読むように戻しておく
    sys.modules["settings"] = base
    spec = importlib.util.spec_from_file_location(f"_category_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod._IS_CATEGORY = True
    mod._BASE_FILE = os.path.abspath(base.__file__)
    if not getattr(mod, "CATEGORY", ""):
        mod.CATEGORY = name
    mod.build_cfg = lambda: base.build_cfg(mod)
    sys.modules["settings"] = mod
    _active = mod
    return mod


def activate_from_argv(argv: list | None = None):
    """sys.argv から `--cat X` / `--cat=X` を取り除いて有効化する。

    位置引数の並びは変わらないので、各ツールの argv 処理はそのまま使える。
    すでに有効化済みで --cat が無ければ何もしない（Jupyter の use() を尊重）。
    """
    argv = sys.argv if argv is None else argv
    cat = None
    for i, a in enumerate(list(argv)):
        if a == "--cat" and i + 1 < len(argv):
            cat = argv[i + 1]
            del argv[i:i + 2]
            break
        if a.startswith("--cat="):
            cat = a.split("=", 1)[1]
            del argv[i]
            break
    if cat is None and _active is not None:
        return _active
    return activate(cat)


def use(cat: str | None):
    """Jupyter 用。設定ファイルを読み直して切り替え、import 済みツールも読み直す。

    settings.py / カテゴリファイルを編集した直後でもカーネル再起動は不要。
    ただし `from tune_c_alpha import step1_scan` のように関数を直接取り出した
    ものは古いまま残るので、`import tune_c_alpha as t` の形で使うこと。
    """
    mod = activate(cat, reload=True)
    for t in _TOOLS:
        if t in sys.modules:
            importlib.reload(sys.modules[t])
    print(f"[config_loader] カテゴリ = {getattr(mod, 'CATEGORY', '') or '（共通のみ）'}")
    return mod


def current():
    return _active or _load_base()


def category() -> str:
    return str(getattr(current(), "CATEGORY", "") or "")


def base_dir(out_root: str | None = None) -> str:
    """出力のルート。カテゴリありなら OUT_ROOT/<カテゴリ>、なしなら OUT_ROOT。"""
    st = current()
    root = out_root or st.OUT_ROOT
    c = category()
    return os.path.join(root, c) if c else root


def suffix(T) -> str:
    """ファイル名の末尾。カテゴリありなら _YYYYMM_<カテゴリ>。"""
    c = category()
    return f"_{T}_{c}" if c else f"_{T}"


def work_path(name: str) -> str:
    """チューニングの中間ファイル（scan_c.csv など）の置き場所。

    カテゴリありなら OUT_ROOT/<カテゴリ>/チューニング/<name>（カテゴリ間で混ざらない）。
    なしなら従来どおりカレントフォルダ。
    """
    c = category()
    if not c:
        return name
    d = os.path.join(base_dir(), "チューニング")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def is_pattern(p) -> bool:
    return bool(p) and any(ch in str(p) for ch in "*?[")


def resolve_path(p: str, what: str = "パネル", quiet: bool = False) -> str:
    """パターン（* ? [ ] を含む）なら、一致するファイルのうち**パス名順で最後**を返す。

    月ごとにフォルダを分ける運用:   PANEL_PATH = "data/*/レンズ.csv"
        → data/202412/レンズ.csv, data/202501/レンズ.csv … の最後（=最新月）
    月ごとにファイル名を変える運用: PANEL_PATH = "data/レンズ_*.csv"
    同じ名前で上書きする運用:       PANEL_PATH = "data/レンズ.csv"（そのまま使う）

    フォルダ名・ファイル名の年月は **YYYYMM の6桁** で揃えること
    （2025-1 のような桁の揺れがあると名前順が月順にならない）。
    """
    if not is_pattern(p):
        return p
    hits = sorted(f for f in glob.glob(p) if os.path.isfile(f))
    if not hits:
        raise SystemExit(f"[エラー] {what}のパターンに一致するファイルがありません: {p}")
    if not quiet:
        prev = f"  （前回候補: {hits[-2]}）" if len(hits) > 1 else ""
        print(f"[{what}] パターン {p} → {len(hits)}件中の最新: {hits[-1]}{prev}")
    return hits[-1]


def panel_arg(args: list, i: int = 0, what: str = "パネル") -> str:
    """位置引数 i を返す。無い／"-" ならカテゴリ設定の PANEL_PATH（パターン可）。"""
    if len(args) > i and args[i] != "-":
        return resolve_path(args[i], what)
    p = getattr(current(), "PANEL_PATH", None)
    if not p:
        raise SystemExit(f"[エラー] {what}のパスがありません。引数で渡すか、"
                         "カテゴリ設定に PANEL_PATH を書いてください。")
    return resolve_path(p, what)
