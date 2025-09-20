import os
import random
from typing import List
import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語フォントを自動設定

# ==========================
# ユーティリティ
# ==========================
DICE_FACES = [1, 2, 3, 4, 5, 6]

def parse_weights(env_val: str | None) -> List[float]:
    """環境変数 DICE_WEIGHTS="1,1,1,1,1,1" を [1,1,1,1,1,1] に変換。
    不正な場合は等確率を返す。
    """
    if not env_val:
        return [1, 1, 1, 1, 1, 1]
    try:
        parts = [p.strip() for p in env_val.split(",")]
        vals = [float(p) for p in parts]
        if len(vals) != 6 or any(v <= 0 for v in vals):
            return [1, 1, 1, 1, 1, 1]
        return vals
    except Exception:
        return [1, 1, 1, 1, 1, 1]


def roll_once(weights: List[float]) -> int:
    return random.choices(DICE_FACES, weights=weights, k=1)[0]


def roll_many(n: int, weights: List[float]) -> List[int]:
    return random.choices(DICE_FACES, weights=weights, k=n)


def count_results(seq: List[int]) -> dict[int, int]:
    c = {f: 0 for f in DICE_FACES}
    for x in seq:
        c[x] += 1
    return c


def counts_df(seq: List[int]) -> pd.DataFrame:
    counts = count_results(seq)
    df = pd.DataFrame({"目": list(counts.keys()), "回数": list(counts.values())})
    df = df.sort_values("目").reset_index(drop=True)
    return df


def plot_bar(df: pd.DataFrame, title: str):
    """ラベルが寝ない縦棒グラフ＋上に数値表示。"""
    # x 軸は文字列にして固定（1,2,3,4,5,6）
    x_labels = df["目"].astype(str).tolist()
    y_vals = df["回数"].tolist()
    x_pos = list(range(len(x_labels)))

    fig, ax = plt.subplots()
    bars = ax.bar(x_pos, y_vals, edgecolor="black")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_xlabel("目")
    ax.set_ylabel("回数")
    ax.set_title(title)

    # 各棒の上に値を描画
    for rect, v in zip(bars, y_vals):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.05, str(v), ha="center", va="bottom")

    plt.tight_layout()
    st.pyplot(fig)

# ==========================
# 初期設定（隠しパラメータ）
# ==========================
# 1) 既定の重みは環境変数から（授業前に先生が設定）
DEFAULT_WEIGHTS = parse_weights(os.environ.get("DICE_WEIGHTS"))
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")  # 任意

# 2) セッション状態
if "weights" not in st.session_state:
    st.session_state.weights = DEFAULT_WEIGHTS
if "history" not in st.session_state:
    st.session_state.history = []  # これまでの出目履歴

# ==========================
# UI（2カラム：左=操作 / 右=結果）
# ==========================
st.set_page_config(page_title="サイコロシミュレーション", page_icon="🎲", layout="centered")
st.title("🎲 サイコロシミュレーション")

# 結果を保持する（再描画でも右カラムに表示を残す）
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_df" not in st.session_state:
    st.session_state.last_df = None

# 左右レイアウト
col_left, col_right = st.columns([1, 1])  # 必要なら [1, 2] や [2, 3] に調整

with col_left:
    st.subheader("操作")
    with st.form("many_form", clear_on_submit=False):
        n = st.number_input("回数", min_value=1, max_value=100000, value=100, step=10)
        submitted = st.form_submit_button("振る", use_container_width=True)

    if submitted:
        results = roll_many(int(n), st.session_state.weights)
        st.session_state.history.extend(results)
        st.session_state.last_results = results
        st.session_state.last_df = counts_df(results)

    # ===== プレビューを左に表示 =====
    if st.session_state.last_results is not None:
        results = st.session_state.last_results
        preview_len = 200
        if len(results) <= preview_len:
            st.success(f"結果: {results}")
        else:
            st.success(
                f"結果（先頭 {preview_len} 件）: {results[:preview_len]} …… ほか {len(results)-preview_len} 件"
            )

with col_right:
    st.subheader("集計")
    if st.session_state.last_df is None:
        st.info("左で回数を入力して「振る」を押すと、ここに結果が表示されます。")
    else:
        # ===== 表（中央寄せ, インデックス非表示・幅を抑えめに）=====
        df_now = st.session_state.last_df
        html = df_now.to_html(index=False)
        html = html.replace(
            "<table",
            '<table style="width:100%; max-width:420px; border-collapse:collapse; text-align:center;"'
        ).replace(
            "<th>", '<th style="text-align:center; padding:4px;">'
        ).replace(
            "<td>", '<td style="text-align:center; padding:4px;">'
        )
        st.markdown(html, unsafe_allow_html=True)

        # ===== 棒グラフ（小さめにしてスクロール削減）=====
        def plot_bar_compact(df, title):
            x_labels = df["目"].astype(str).tolist()
            y_vals = df["回数"].tolist()
            x_pos = list(range(len(x_labels)))
            fig, ax = plt.subplots(figsize=(4.5, 3.2))  # ←小さめ
            bars = ax.bar(x_pos, y_vals, edgecolor="black")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=0)
            ax.set_xlabel("目")
            ax.set_ylabel("回数")
            ax.set_title(title)
            for rect, v in zip(bars, y_vals):
                ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.05, str(v),
                        ha="center", va="bottom")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        plot_bar_compact(df_now, "集計（今回の試行）")

st.divider()


# ==========================
# 教員用セクション（パスワードで保護）
# ==========================
with st.expander("管理用", expanded=False):
    if ADMIN_PASSWORD:
        pwd = st.text_input("管理用パスワード", type="password", help="環境変数 ADMIN_PASSWORD で設定")
        auth_ok = (pwd == ADMIN_PASSWORD)
    else:
        # パスワード未設定の場合は、ローカル利用前提で自分だけが触れる想定
        st.caption("※ ADMIN_PASSWORD が未設定のため、ローカル授業用の簡易モードです。")
        auth_ok = True

    if auth_ok:
        st.markdown("#### 重み")
        mode = st.radio("設定方法", ["スライダーで設定", "直接入力（カンマ区切り）"], horizontal=True)

        if mode == "スライダーで設定":
            w1 = st.slider("1 の重み", 0.01, 10.0, float(st.session_state.weights[0]))
            w2 = st.slider("2 の重み", 0.01, 10.0, float(st.session_state.weights[1]))
            w3 = st.slider("3 の重み", 0.01, 10.0, float(st.session_state.weights[2]))
            w4 = st.slider("4 の重み", 0.01, 10.0, float(st.session_state.weights[3]))
            w5 = st.slider("5 の重み", 0.01, 10.0, float(st.session_state.weights[4]))
            w6 = st.slider("6 の重み", 0.01, 10.0, float(st.session_state.weights[5]))
            new_weights = [w1, w2, w3, w4, w5, w6]
        else:
            raw = st.text_input("重みを 6 つカンマ区切りで", value=",".join(str(x) for x in st.session_state.weights))
            new_weights = parse_weights(raw)

        if st.button("重みを反映"):
            st.session_state.weights = new_weights
            st.success(f"重みを更新しました: {st.session_state.weights}")

        st.markdown("#### 既定値の管理（任意）")
        st.write("起動前に環境変数 **DICE_WEIGHTS** を設定しておくと、既定の重みを隠し持てます。例: `export DICE_WEIGHTS=1,1,1,1,1,5`")
    else:
        st.warning("管理用設定を表示するには正しいパスワードを入力してください。")

st.caption("© サイコロシミュレーション")
