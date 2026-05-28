"""
Day 22 — Streamlit dashboard reading from Gold Delta table.

Reads the latest block-occupancy window per train from the Gold parquet files,
shows the track map with signal aspects, advisory speeds, and delay severity.
No Kafka/Docker required — runs entirely from local Delta table files.

Day 23 adds the Marey time-distance diagram.
Day 24 adds the purple RL advisory overlay.
"""
import sys
import os
import glob
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

from simulator.train_generator import BLOCKS
from web.components.time_distance_plot import render_marey

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_PATH = os.path.join(PROJECT_ROOT, "data", "gold", "train_advisories")

SIGNAL_COLORS = {"clear": "#16a34a", "approach": "#ca8a04", "stop": "#dc2626"}
SIGNAL_LABELS = {"clear": "CLR", "approach": "APP", "stop": "STP"}
SEVERITY_COLORS = {"nominal": "#16a34a", "moderate": "#ca8a04", "critical": "#dc2626"}

st.set_page_config(
    page_title="Rail Signal Optimizer",
    layout="wide",
    page_icon="🚂",
)
st.title("Real-Time Rail Corridor — Signal State & Speed Advisory")
st.caption("Harrisburg Subdivision  |  Source: Gold Delta Table  |  Refreshes every 5s")


# ── Data loading ───────────────────────────────────────────────────────

@st.cache_data(ttl=4)
def load_gold() -> pd.DataFrame:
    """Read latest Gold parquet files, return one row per train (most recent window)."""
    parquet_files = glob.glob(
        os.path.join(GOLD_PATH, "**", "*.parquet"), recursive=True
    )
    if not parquet_files:
        return pd.DataFrame()

    parquet_files.sort(key=os.path.getmtime, reverse=True)
    dfs = [pd.read_parquet(f) for f in parquet_files[:8]]
    df = pd.concat(dfs, ignore_index=True)

    if "window_start" in df.columns:
        df = (
            df.sort_values("window_start")
            .groupby("train_id", as_index=False)
            .last()
        )
    return df


df = load_gold()

if df.empty:
    st.warning(
        "No Gold data found. Run the pipeline first:\n\n"
        "```\npython -m pipelines.silver_transform\npython -m pipelines.gold_aggregation\n```"
    )
    time.sleep(5)
    st.rerun()

occupied_blocks: dict[str, pd.Series] = {}
for _, row in df.iterrows():
    occupied_blocks[row["block_id"]] = row


# ── Track Map ─────────────────────────────────────────────────────────

st.subheader("Track Map")
cols = st.columns(len(BLOCKS))
for col, block_id in zip(cols, BLOCKS):
    if block_id in occupied_blocks:
        row = occupied_blocks[block_id]
        sig = row.get("signal_aspect", "clear")
        color = SIGNAL_COLORS.get(sig, "#166534")
        label = SIGNAL_LABELS.get(sig, "CLR")
        speed = row.get("speed_mph", 0.0)
        train = row.get("train_id", "?")
        col.markdown(
            f"""<div style="background:{color};padding:6px 2px;border-radius:5px;
            text-align:center;font-size:9px;color:white;line-height:1.6;">
            <b>{block_id}</b><br>{train}<br>{speed:.0f} mph<br><b>{label}</b>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        col.markdown(
            f"""<div style="background:#166534;padding:6px 2px;border-radius:5px;
            text-align:center;font-size:9px;color:#86efac;line-height:1.6;">
            <b>{block_id}</b><br>&mdash;<br>CLEAR<br>&nbsp;
            </div>""",
            unsafe_allow_html=True,
        )

st.divider()


# ── Advisory Table ─────────────────────────────────────────────────────

st.subheader("Train Advisory Feed")

for _, row in df.sort_values("train_id").iterrows():
    adh = float(row.get("schedule_adherence_sec", 0.0))
    adh_str = f"+{adh:.0f}s" if adh >= 0 else f"{adh:.0f}s"
    adh_color = "green" if adh >= -60 else "red"
    sig = row.get("signal_aspect", "clear")
    sig_icon = {"clear": "🟢", "approach": "🟡", "stop": "🔴"}.get(sig, "⚪")
    sev = str(row.get("delay_severity", "nominal")).lower()
    sev_color = SEVERITY_COLORS.get(sev, "#6b7280")
    advisory = float(row.get("advisory_speed_mph", 79.0))
    shockwave = float(row.get("shockwave_delay_sec", 0.0))

    c1, c2, c3, c4, c5, c6 = st.columns([1, 1.5, 1.5, 1.5, 2, 2])
    c1.write(f"**{row['train_id']}**")
    c2.write(f"Block: `{row.get('block_id','?')}`")
    c3.write(f"Speed: **{float(row.get('speed_mph',0)):.0f} mph**")
    c4.write(f"{sig_icon} {sig.upper()}")
    c5.write(f"Advisory: **{advisory:.0f} mph**  |  Shockwave: {shockwave:.0f}s")
    c6.markdown(
        f"Adherence: :{adh_color}[**{adh_str}**]  "
        f"<span style='color:{sev_color};font-weight:bold'>{sev.upper()}</span>",
        unsafe_allow_html=True,
    )

st.divider()


# ── Summary Metrics ────────────────────────────────────────────────────

st.subheader("Corridor Summary")
m1, m2, m3, m4 = st.columns(4)

avg_adh = df["schedule_adherence_sec"].mean() if "schedule_adherence_sec" in df.columns else 0.0
avg_speed = df["speed_mph"].mean() if "speed_mph" in df.columns else 0.0
critical_count = (df["delay_severity"] == "critical").sum() if "delay_severity" in df.columns else 0
avg_advisory = df["advisory_speed_mph"].mean() if "advisory_speed_mph" in df.columns else 0.0

m1.metric("Avg Schedule Adherence", f"{avg_adh:+.0f}s", delta=None)
m2.metric("Avg Train Speed", f"{avg_speed:.1f} mph")
m3.metric("Critical Trains", int(critical_count))
m4.metric("Avg Advisory Speed", f"{avg_advisory:.0f} mph")

# ── Marey Time-Distance Diagram (Day 23) ──────────────────────────────

st.subheader("Marey Time-Distance Diagram")
render_marey(GOLD_PATH)

st.divider()

# ── RL Advisory Overlay (Day 24) ───────────────────────────────────────

st.subheader("RL Speed Advisory Overlay")
st.caption("Purple = RL advisory differs from rule-based signal policy")

rl_cols = st.columns(len(BLOCKS))
for col, block_id in zip(rl_cols, BLOCKS):
    if block_id in occupied_blocks:
        row = occupied_blocks[block_id]
        advisory = float(row.get("advisory_speed_mph", 79.0))
        sig = row.get("signal_aspect", "clear")
        # Rule-based would give: stop=0, approach=20, clear=79
        rule_speed = {"stop": 0.0, "approach": 20.0, "clear": 79.0}.get(sig, 79.0)
        rl_differs = abs(advisory - rule_speed) > 5.0
        color = "#7c3aed" if rl_differs else SIGNAL_COLORS.get(sig, "#166534")
        col.markdown(
            f"""<div style="background:{color};padding:6px 2px;border-radius:5px;
            text-align:center;font-size:9px;color:white;line-height:1.6;">
            <b>{block_id}</b><br>{row.get('train_id','?')}<br>
            <b>{advisory:.0f} mph</b><br>{'RL' if rl_differs else 'STD'}
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        col.markdown(
            f"""<div style="background:#166534;padding:6px 2px;border-radius:5px;
            text-align:center;font-size:9px;color:#86efac;line-height:1.6;">
            <b>{block_id}</b><br>&mdash;<br>&nbsp;<br>&nbsp;
            </div>""",
            unsafe_allow_html=True,
        )

st.divider()
st.caption(f"Last updated: {time.strftime('%H:%M:%S')}  |  Trains: {len(df)}")

time.sleep(5)
st.rerun()
