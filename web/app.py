import streamlit as st
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from confluent_kafka import Consumer
from simulator.train_event import TrainEvent
from simulator.train_generator import BLOCKS

KAFKA_BROKER = "localhost:9092"
TOPIC = "train_events"

SIGNAL_COLORS = {"clear": "#16a34a", "approach": "#ca8a04", "stop": "#dc2626"}
SIGNAL_LABELS = {"clear": "CLR", "approach": "APP", "stop": "STP"}

st.set_page_config(page_title="Rail Signal Optimizer", layout="wide", page_icon="🚂")
st.title("🚂 Real-Time Rail Corridor — Signal State")
st.caption("Harrisburg Subdivision | Live Block Occupancy  |  Source: Kafka → train_events")

@st.cache_resource
def get_consumer() -> Consumer:
    c = Consumer({
        "bootstrap.servers": KAFKA_BROKER,
        "group.id": "streamlit-ui",
        "auto.offset.reset": "latest",
        "enable.auto.commit": True,
    })
    c.subscribe([TOPIC])
    return c

if "latest_events" not in st.session_state:
    st.session_state.latest_events = {}

# Poll Kafka
consumer = get_consumer()
msgs = consumer.consume(num_messages=20, timeout=0.5)
for msg in msgs:
    value = msg.value()
    if msg.error() is None and value is not None:
        event = TrainEvent.model_validate_json(value)
        st.session_state.latest_events[event.train_id] = event

events = list(st.session_state.latest_events.values())
occupied_blocks = {e.block_id: e for e in events}

if not events:
    st.info("Waiting for data… run `python -m simulator.kafka_producer`")
else:
    # ── Track Map ──────────────────────────────────────────────────
    st.subheader("Track Map")
    cols = st.columns(len(BLOCKS))
    for col, block_id in zip(cols, BLOCKS):
        if block_id in occupied_blocks:
            e = occupied_blocks[block_id]
            color = SIGNAL_COLORS[e.signal_aspect_ahead]
            label = SIGNAL_LABELS[e.signal_aspect_ahead]
            col.markdown(
                f"""<div style="background:{color};padding:6px 2px;border-radius:5px;
                text-align:center;font-size:9px;color:white;line-height:1.6;">
                <b>{block_id}</b><br>{e.train_id}<br>{e.current_speed_mph} mph<br><b>{label}</b>
                </div>""",
                unsafe_allow_html=True
            )
        else:
            col.markdown(
                f"""<div style="background:#166534;padding:6px 2px;border-radius:5px;
                text-align:center;font-size:9px;color:#86efac;line-height:1.6;">
                <b>{block_id}</b><br>—<br>CLEAR<br>&nbsp;
                </div>""",
                unsafe_allow_html=True
            )

    st.divider()

    # ── Advisory Feed ───────────────────────────────────────────────
    st.subheader("Train Advisory Feed")
    signal_emoji = {"clear": "🟢", "approach": "🟡", "stop": "🔴"}
    for e in sorted(events, key=lambda x: x.train_id):
        adh = e.schedule_adherence_sec
        adh_str = f"+{adh:.0f}s" if adh >= 0 else f"{adh:.0f}s"
        adh_color = "green" if adh >= 0 else "red"
        col1, col2, col3, col4, col5 = st.columns([1, 1.5, 1.5, 1.5, 2])
        col1.write(f"**{e.train_id}**")
        col2.write(f"Block: `{e.block_id}`")
        col3.write(f"Speed: **{e.current_speed_mph} mph**")
        col4.write(f"{signal_emoji[e.signal_aspect_ahead]} {e.signal_aspect_ahead.upper()}")
        col5.markdown(f"Schedule: :{adh_color}[**{adh_str}**]")

st.divider()
st.caption(f"Last updated: {time.strftime('%H:%M:%S')} | Refresh: ~2s")

time.sleep(1.5)
st.rerun()
