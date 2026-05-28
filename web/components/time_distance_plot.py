"""
Day 23 — Marey time-distance (space-time) diagram using Plotly.

A Marey diagram plots block position (Y-axis) against time (X-axis).
Each train is a diagonal line; slope = speed. Bunched lines = congestion.
Horizontal segments = stopped train (shockwave upstream).
"""
import glob
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

TRAIN_COLORS = {
    "T_001": "#3b82f6",   # blue
    "T_002": "#10b981",   # green
    "T_003": "#f59e0b",   # amber
    "T_004": "#8b5cf6",   # purple
    "T_005": "#ef4444",   # red
}


def load_gold_history(gold_path: str, max_files: int = 20) -> pd.DataFrame:
    """Read recent Gold parquet files and return full time-series (all windows)."""
    parquet_files = glob.glob(
        os.path.join(gold_path, "**", "*.parquet"), recursive=True
    )
    if not parquet_files:
        return pd.DataFrame()

    parquet_files.sort(key=os.path.getmtime, reverse=True)
    dfs = [pd.read_parquet(f) for f in parquet_files[:max_files]]
    df = pd.concat(dfs, ignore_index=True)

    if "window_start" in df.columns:
        df["window_start"] = pd.to_datetime(df["window_start"])
        df = df.sort_values(["train_id", "window_start"])

    return df


def render_marey(gold_path: str) -> None:
    """Render the Marey time-distance diagram in the active Streamlit context."""
    df = load_gold_history(gold_path)

    if df.empty or "block_idx" not in df.columns or "window_start" not in df.columns:
        st.info("Marey diagram: no historical data yet. Run the pipeline first.")
        return

    fig = go.Figure()

    severity_marker = {"nominal": "circle", "moderate": "diamond", "critical": "x"}

    for train_id, group in df.groupby("train_id"):
        color = TRAIN_COLORS.get(str(train_id), "#6b7280")
        sev = group["delay_severity"].map(
            lambda s: severity_marker.get(str(s).lower(), "circle")
        ) if "delay_severity" in group.columns else ["circle"] * len(group)

        # Main trace (line)
        fig.add_trace(go.Scatter(
            x=group["window_start"],
            y=group["block_idx"],
            mode="lines+markers",
            name=str(train_id),
            line={"color": color, "width": 2},
            marker={"size": 5, "color": color},
            hovertemplate=(
                f"<b>{train_id}</b><br>"
                "Time: %{x}<br>"
                "Block: %{y}<br>"
                "Speed: %{customdata[0]:.0f} mph<br>"
                "Adherence: %{customdata[1]:+.0f}s<br>"
                "Severity: %{customdata[2]}<extra></extra>"
            ),
            customdata=group[["speed_mph", "schedule_adherence_sec", "delay_severity"]].values
            if all(c in group.columns for c in ["speed_mph", "schedule_adherence_sec", "delay_severity"])
            else None,
        ))

    fig.update_layout(
        title="Marey Time-Distance Diagram — Harrisburg Subdivision",
        xaxis_title="Time (UTC)",
        yaxis_title="Block Index (0 = origin, 19 = terminal)",
        yaxis={"range": [-0.5, 19.5], "dtick": 1, "gridcolor": "#e5e7eb"},
        xaxis={"gridcolor": "#e5e7eb"},
        legend_title="Train",
        height=500,
        plot_bgcolor="#f9fafb",
        paper_bgcolor="#ffffff",
        hovermode="closest",
    )

    st.plotly_chart(fig, use_container_width=True)
