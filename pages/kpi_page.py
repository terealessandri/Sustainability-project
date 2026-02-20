"""
KPI Dashboard Page — Extracted Metrics Visualization
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def render():
    """Render the KPI dashboard page."""
    st.markdown("## 📊 KPI Dashboard — Extracted Metrics")

    if not st.session_state.processing_complete:
        st.warning("⚠️ Please upload and process reports first!")
        st.info("👈 Go to **Upload Reports** to get started")
        return

    st.write("Explore quantitative metrics extracted from ESG reports.")

    enriched_chunks = st.session_state.enriched_chunks

    # Calculate metrics summary
    total_chunks = len(enriched_chunks)
    chunks_with_metrics = sum(
        1 for c in enriched_chunks
        if c.get("metrics", {}).get("percentages") or
           c.get("metrics", {}).get("emissions") or
           c.get("metrics", {}).get("currency")
    )

    commitment_counts = defaultdict(int)
    for chunk in enriched_chunks:
        commitment_type = chunk.get("metrics", {}).get("commitment_type", "vague")
        commitment_counts[commitment_type] += 1

    # Summary metrics
    st.markdown("### 📈 Metrics Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Chunks", total_chunks)

    with col2:
        st.metric("With Metrics", chunks_with_metrics)

    with col3:
        coverage = chunks_with_metrics / total_chunks * 100 if total_chunks > 0 else 0
        st.metric("Coverage", f"{coverage:.1f}%")

    with col4:
        st.metric("Targets", commitment_counts.get("target", 0))

    # Commitment type distribution
    st.markdown("### 🎯 Commitment Type Distribution")

    if commitment_counts:
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = {
            "target": "#2E7D32",    # Green
            "actual": "#1976D2",    # Blue
            "vague": "#D32F2F"      # Red
        }

        types = list(commitment_counts.keys())
        counts = list(commitment_counts.values())
        bar_colors = [colors.get(t, "#666") for t in types]

        bars = ax.bar(types, counts, color=bar_colors, alpha=0.7, edgecolor='black')

        ax.set_ylabel("Count", fontsize=12)
        ax.set_xlabel("Commitment Type", fontsize=12)
        ax.set_title("Distribution of Commitment Types", fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add count labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        st.pyplot(fig)

    # Metrics table
    st.markdown("### 📋 Extracted Metrics Table")

    # Build table data
    table_data = []

    for chunk in enriched_chunks[:100]:  # Limit to first 100 for performance
        metrics = chunk.get("metrics", {})

        if metrics.get("percentages") or metrics.get("emissions") or metrics.get("currency"):
            table_data.append({
                "Source": chunk.get("source", ""),
                "Page": chunk.get("page", ""),
                "Type": metrics.get("commitment_type", ""),
                "Percentages": ", ".join(metrics.get("percentages", [])),
                "Emissions": ", ".join(metrics.get("emissions", [])),
                "Currency": ", ".join(metrics.get("currency", [])),
                "Years": ", ".join(map(str, metrics.get("years", []))),
                "Text Preview": chunk.get("text", "")[:100] + "..."
            })

    if table_data:
        df = pd.DataFrame(table_data)

        # Add filters
        col1, col2 = st.columns(2)

        with col1:
            source_filter = st.selectbox(
                "Filter by source:",
                options=["All"] + list(df["Source"].unique())
            )

        with col2:
            type_filter = st.selectbox(
                "Filter by type:",
                options=["All"] + list(df["Type"].unique())
            )

        # Apply filters
        filtered_df = df.copy()
        if source_filter != "All":
            filtered_df = filtered_df[filtered_df["Source"] == source_filter]
        if type_filter != "All":
            filtered_df = filtered_df[filtered_df["Type"] == type_filter]

        st.dataframe(filtered_df, use_container_width=True, height=400)

        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="esg_metrics.csv",
            mime="text/csv"
        )

    else:
        st.info("No metrics extracted from the reports.")

    # Timeline analysis
    st.markdown("### 📅 Timeline Analysis")

    all_years = []
    for chunk in enriched_chunks:
        years = chunk.get("metrics", {}).get("years", [])
        all_years.extend(years)

    if all_years:
        year_counts = pd.Series(all_years).value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(year_counts.index, year_counts.values, marker='o',
               color='#2E7D32', linewidth=2, markersize=8)
        ax.fill_between(year_counts.index, year_counts.values, alpha=0.3, color='#2E7D32')

        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Mentions", fontsize=12)
        ax.set_title("Target Year Distribution", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

        st.write(f"**Year Range**: {min(all_years)} - {max(all_years)}")
        st.write(f"**Most Common**: {year_counts.idxmax()} ({year_counts.max()} mentions)")

    else:
        st.info("No year mentions found in the reports.")
