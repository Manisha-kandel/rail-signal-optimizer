"""
Day 22 — Streamlit dashboard reading from Gold Delta table.
Day 23 — Marey time-distance diagram via Plotly.
Day 24 — Purple RL advisory overlay.

Reads Gold parquet files directly via pandas (no JVM).
Falls back to data/sample/ when live data is absent (e.g. Streamlit Cloud).
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
GOLD_PATH    = os.path.join(PROJECT_ROOT, "data", "gold", "train_advisories")
SAMPLE_PATH  = os.path.join(PROJECT_ROOT, "data", "sample")

SIGNAL_COLORS   = {"clear": "#16a34a", "approach": "#ca8a04", "stop": "#dc2626"}
SIGNAL_LABELS   = {"clear": "CLR",     "approach": "APP",     "stop": "STP"}
SEVERITY_COLORS = {"nominal": "#16a34a", "moderate": "#ca8a04", "critical": "#dc2626"}


# ── Data loading (defined before use) ─────────────────────────────────

@st.cache_data(ttl=4)
def load_gold() -> tuple[pd.DataFrame, bool]:
    """
    Return (df, using_sample).
    df = one row per train (most recent window).
    Falls back to data/sample/ when live Gold table is absent.
    """
    parquet_files = glob.glob(
        os.path.join(GOLD_PATH, "**", "*.parquet"), recursive=True
    )
    sample = False

    if not parquet_files:
        parquet_files = glob.glob(os.path.join(SAMPLE_PATH, "*.parquet"))
        sample = bool(parquet_files)
        if not parquet_files:
            return pd.DataFrame(), sample

    parquet_files.sort(key=os.path.getmtime, reverse=True)
    dfs = [pd.read_parquet(f) for f in parquet_files[:8]]
    df = pd.concat(dfs, ignore_index=True)

    if "window_start" in df.columns:
        df = (
            df.sort_values("window_start")
            .groupby("train_id", as_index=False)
            .last()
        )
    return df, sample


# ── Page setup ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Rail Signal Optimizer",
    layout="wide",
    page_icon="🚂",
)
st.title("Real-Time Rail Corridor — Signal State & Speed Advisory")

df, using_sample = load_gold()
source_label = "Sample data (demo)" if using_sample else "Gold Delta Table (live)"
st.caption(f"Harrisburg Subdivision  |  Source: {source_label}  |  Refreshes every 5s")

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
        sig   = str(row.get("signal_aspect", "clear"))
        color = SIGNAL_COLORS.get(sig, "#166534")
        label = SIGNAL_LABELS.get(sig, "CLR")
        speed = float(row.get("speed_mph", 0.0))
        train = str(row.get("train_id", "?"))
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


# ── Advisory Feed ──────────────────────────────────────────────────────

st.subheader("Train Advisory Feed")

for _, row in df.sort_values("train_id").iterrows():
    adh      = float(row.get("schedule_adherence_sec", 0.0))
    adh_str  = f"+{adh:.0f}s" if adh >= 0 else f"{adh:.0f}s"
    adh_color = "green" if adh >= -60 else "red"
    sig      = str(row.get("signal_aspect", "clear"))
    sig_icon = {"clear": "🟢", "approach": "🟡", "stop": "🔴"}.get(sig, "⚪")
    sev      = str(row.get("delay_severity", "nominal")).lower()
    sev_color = SEVERITY_COLORS.get(sev, "#6b7280")
    advisory  = float(row.get("advisory_speed_mph", 79.0))
    shockwave = float(row.get("shockwave_delay_sec", 0.0))

    c1, c2, c3, c4, c5, c6 = st.columns([1, 1.5, 1.5, 1.5, 2, 2])
    c1.write(f"**{row['train_id']}**")
    c2.write(f"Block: `{row.get('block_id', '?')}`")
    c3.write(f"Speed: **{float(row.get('speed_mph', 0)):.0f} mph**")
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

avg_adh      = df["schedule_adherence_sec"].mean() if "schedule_adherence_sec" in df.columns else 0.0
avg_speed    = df["speed_mph"].mean()               if "speed_mph"    in df.columns else 0.0
crit_count   = (df["delay_severity"] == "critical").sum() if "delay_severity" in df.columns else 0
avg_advisory = df["advisory_speed_mph"].mean()      if "advisory_speed_mph" in df.columns else 0.0

m1.metric("Avg Schedule Adherence", f"{avg_adh:+.0f}s")
m2.metric("Avg Train Speed",        f"{avg_speed:.1f} mph")
m3.metric("Critical Trains",        int(crit_count))
m4.metric("Avg Advisory Speed",     f"{avg_advisory:.0f} mph")

st.divider()


# ── Marey Time-Distance Diagram (Day 23) ──────────────────────────────

st.subheader("Marey Time-Distance Diagram")
marey_path = SAMPLE_PATH if using_sample else GOLD_PATH
render_marey(marey_path)

st.divider()


# ── RL Advisory Overlay (Day 24) ───────────────────────────────────────

st.subheader("RL Speed Advisory Overlay")
st.caption("Purple block = RL advisory differs from rule-based signal policy by >5 mph")

rl_cols = st.columns(len(BLOCKS))
for col, block_id in zip(rl_cols, BLOCKS):
    if block_id in occupied_blocks:
        row      = occupied_blocks[block_id]
        advisory = float(row.get("advisory_speed_mph", 79.0))
        sig      = str(row.get("signal_aspect", "clear"))
        rule_speed = {"stop": 0.0, "approach": 20.0, "clear": 79.0}.get(sig, 79.0)
        rl_differs = abs(advisory - rule_speed) > 5.0
        color = "#7c3aed" if rl_differs else SIGNAL_COLORS.get(sig, "#166534")
        col.markdown(
            f"""<div style="background:{color};padding:6px 2px;border-radius:5px;
            text-align:center;font-size:9px;color:white;line-height:1.6;">
            <b>{block_id}</b><br>{row.get('train_id', '?')}<br>
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
