import streamlit as st
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.train_generator import TrainState, simulate_step, TRAIN_IDS, BLOCKS

st.set_page_config(
    page_title="Rail Signal Optimizer",
    layout="wide",
    page_icon="🚂"
)

st.title("🚂 Real-Time Rail Corridor — Signal State")
st.caption("Harrisburg Subdivision | Live Block Occupancy")

SIGNAL_COLORS = {
    "clear":    "#16a34a",  # green
    "approach": "#ca8a04",  # yellow
    "stop":     "#dc2626",  # red
}

SIGNAL_LABELS = {
    "clear":    "CLR",
    "approach": "APP",
    "stop":     "STP",
}

# Initialize sim state once
if "states" not in st.session_state:
    st.session_state.states = [
        TrainState(tid, i * 3)
        for i, tid in enumerate(TRAIN_IDS)
    ]

placeholder = st.empty()

while True:
    events = simulate_step(st.session_state.states)
    occupied_blocks = {e.block_id: e for e in events}

    with placeholder.container():

        # ── Track Map ──────────────────────────────────────────
        st.subheader("Track Map")
        cols = st.columns(len(BLOCKS))

        for col, block_id in zip(cols, BLOCKS):
            if block_id in occupied_blocks:
                e = occupied_blocks[block_id]
                color = SIGNAL_COLORS[e.signal_aspect_ahead]
                label = SIGNAL_LABELS[e.signal_aspect_ahead]
                col.markdown(
                    f"""
                    <div style="
                        background:{color};
                        padding:6px 2px;
                        border-radius:5px;
                        text-align:center;
                        font-size:9px;
                        color:white;
                        line-height:1.6;
                    ">
                    <b>{block_id}</b><br>
                    {e.train_id}<br>
                    {e.current_speed_mph} mph<br>
                    <b>{label}</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                col.markdown(
                    f"""
                    <div style="
                        background:#166534;
                        padding:6px 2px;
                        border-radius:5px;
                        text-align:center;
                        font-size:9px;
                        color:#86efac;
                        line-height:1.6;
                    ">
                    <b>{block_id}</b><br>
                    —<br>
                    CLEAR<br>
                    &nbsp;
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.divider()

        # ── Advisory Feed ──────────────────────────────────────
        st.subheader("Train Advisory Feed")

        for e in events:
            adh = e.schedule_adherence_sec
            adh_str = f"+{adh:.0f}s" if adh >= 0 else f"{adh:.0f}s"
            adh_color = "green" if adh >= 0 else "red"
            signal_emoji = {"clear": "🟢", "approach": "🟡", "stop": "🔴"}

            col1, col2, col3, col4, col5 = st.columns([1, 1.5, 1.5, 1.5, 2])
            col1.write(f"**{e.train_id}**")
            col2.write(f"Block: `{e.block_id}`")
            col3.write(f"Speed: **{e.current_speed_mph} mph**")
            col4.write(f"{signal_emoji[e.signal_aspect_ahead]} {e.signal_aspect_ahead.upper()}")
            col5.markdown(
                f"Schedule: :{adh_color}[**{adh_str}**]"
            )

        st.divider()
        st.caption(f"Last updated: {time.strftime('%H:%M:%S')} | Refresh: 2s")

    time.sleep(2)
    st.rerun()