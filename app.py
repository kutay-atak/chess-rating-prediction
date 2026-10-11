"""Demo: paste a Lichess blitz game and watch both players' rating estimates sharpen move by move.

Usage: streamlit run app.py
"""

import chess
import chess.svg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402

from demo import bands, fetch_pgn, game_id, load_model, read_pgn  # noqa: E402

SIDES = [("White", "#2a78d6"), ("Black", "#eb6834")]  # validated categorical slots 1-2


@st.cache_resource
def model():
    return load_model()


def chart(b, ply, true=None):
    fig, ax = plt.subplots(figsize=(6, 3.4), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    x = range(1, len(b) + 1)
    for side, (name, color) in enumerate(SIDES):
        ax.fill_between(x, b[:, side, 0], b[:, side, 2], color=color, alpha=0.15, lw=0)
        ax.plot(x, b[:, side, 1], color=color, lw=2, label=f"{name}: {b[ply - 1, side, 1]:.0f} "
                f"({b[ply - 1, side, 0]:.0f}–{b[ply - 1, side, 2]:.0f})")
        if true:
            ax.axhline(true[side], color=color, lw=1, ls="--")
    ax.axvline(ply, color="#52514e", lw=1, ls=":")
    ax.set(xlabel="plies played", ylabel="rating", xlim=(1, len(b)))
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


st.set_page_config(page_title="Rating from the moves", layout="wide")
st.title("How strong are they? Rating estimates, move by move")
st.caption("A GRU trained on September 2025 Lichess blitz games guesses both players' ratings from their moves and "
           "clock use only, with a 90% band. Ratings stay hidden until the last move.")

link = st.text_input("Lichess game link (blitz)", placeholder="https://lichess.org/abcdefgh")
with st.expander("…or paste a PGN with clocks"):
    pasted = st.text_area("PGN", height=120)
use_example = st.button("Try an example game")

text = None
try:
    if pasted.strip():
        text = pasted
    elif link.strip():
        gid = game_id(link)
        if gid is None:
            raise ValueError("That isn't a lichess.org game link.")
        text = fetch_pgn(gid)
    elif use_example or st.session_state.get("example"):
        st.session_state["example"] = True
        text = open("example.pgn").read()
    if text:
        headers, sans, clocks, base, inc = read_pgn(text)
except ValueError as e:
    st.error(str(e))
    text = None
except OSError as e:
    st.error(f"Couldn't fetch the game from Lichess ({e}).")
    text = None

if text:
    net, saved = model()
    b = bands(net, saved, sans, clocks, base, inc)
    ply = st.slider("move", 1, len(sans), len(sans) // 2 if len(sans) > 2 else 1, format="ply %d")
    board = chess.Board()
    for san in sans[:ply]:
        board.push_san(san)
    last = board.peek() if board.move_stack else None
    left, right = st.columns([2, 3])
    with left:
        st.image(chess.svg.board(board, size=360, lastmove=last), width=360)
    reveal = ply == len(sans) or st.toggle("Reveal the true ratings")
    true = None
    if reveal and headers.get("WhiteElo", "?").isdigit() and headers.get("BlackElo", "?").isdigit():
        true = int(headers["WhiteElo"]), int(headers["BlackElo"])
    with right:
        st.pyplot(chart(b, ply, true))
        if true:
            st.write(f"True ratings: White **{true[0]}**, Black **{true[1]}** (dashed lines).")
    st.caption(f"{base // 60}+{inc} · {len(sans)} plies · intervals calibrated on held-out players to cover 90%; "
               "they cover only about 60–70% of players under 1000 or over 2200, whose estimates are pulled to the middle.")
