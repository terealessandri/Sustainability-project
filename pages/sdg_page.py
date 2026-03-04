"""
SDG Coverage Page — Visualization of SDG Distribution
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sdg_classifier import SDG_DEFINITIONS

# Full label: "SDG 13 – Climate Action"
SDG_LABEL = {sdg_id: f"{sdg_id} – {name}" for sdg_id, name, _ in SDG_DEFINITIONS}
# Short name only: "Climate Action"
SDG_SHORT_NAME = {sdg_id: name for sdg_id, name, _ in SDG_DEFINITIONS}


def render():
    """Render the SDG coverage page."""
    st.markdown("## 🎯 SDG Coverage Analysis")

    if not st.session_state.processing_complete:
        st.warning("⚠️ Please upload and process reports first!")
        st.info("👈 Go to **Upload Reports** to get started")
        return

    st.write("Explore how ESG reports map to the UN's 17 Sustainable Development Goals.")

    # Get classified chunks
    classified_chunks = st.session_state.classified_chunks

    # Calculate SDG statistics
    sdg_counts = defaultdict(int)
    sdg_by_source = defaultdict(lambda: defaultdict(int))

    for chunk in classified_chunks:
        source = chunk.get("source")
        for match in chunk.get("sdg_matches", []):
            sdg_id = match["sdg_id"]
            sdg_counts[sdg_id] += 1
            sdg_by_source[source][sdg_id] += 1

    # Overall statistics
    st.markdown("### 📊 Overall SDG Coverage")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Mentions", sum(sdg_counts.values()))

    with col2:
        st.metric("Unique SDGs", len(sdg_counts))

    with col3:
        most_common = max(sdg_counts, key=sdg_counts.get) if sdg_counts else "N/A"
        st.metric("Most Common", SDG_SHORT_NAME.get(most_common, most_common))

    with col4:
        coverage_rate = len([c for c in classified_chunks if c.get("sdg_matches")]) / len(classified_chunks) * 100
        st.metric("Coverage Rate", f"{coverage_rate:.1f}%")

    # SDG distribution chart
    st.markdown("### 📈 SDG Distribution")

    if sdg_counts:
        # Create DataFrame with full labels
        df = pd.DataFrame(list(sdg_counts.items()), columns=["SDG", "Count"])
        df["Label"] = df["SDG"].map(lambda x: SDG_LABEL.get(x, x))
        df = df.sort_values("Count", ascending=False)

        # Bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(df["Label"], df["Count"], color='#2E7D32')

        ax.set_xlabel("Number of Mentions", fontsize=12)
        ax.set_ylabel("SDG", fontsize=12)
        ax.set_title("SDG Mention Frequency", fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add count labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f' {int(width)}',
                   ha='left', va='center', fontsize=10)

        st.pyplot(fig)

    else:
        st.info("No SDG matches found in the reports.")

    # Per-source breakdown
    st.markdown("### 📚 Coverage by Document")

    sources = list(sdg_by_source.keys())

    if len(sources) > 1:
        # Heatmap
        sdg_ids = sorted(set(sdg_id for counts in sdg_by_source.values() for sdg_id in counts.keys()))
        sdg_col_labels = [SDG_LABEL.get(sid, sid) for sid in sdg_ids]

        # Create matrix
        matrix = []
        for source in sources:
            row = [sdg_by_source[source].get(sdg_id, 0) for sdg_id in sdg_ids]
            matrix.append(row)

        df_heatmap = pd.DataFrame(matrix, index=sources, columns=sdg_col_labels)

        fig, ax = plt.subplots(figsize=(14, len(sources) * 0.5 + 2))
        sns.heatmap(df_heatmap, annot=True, fmt='d', cmap='Greens',
                   cbar_kws={'label': 'Mentions'}, ax=ax)
        ax.set_title("SDG Coverage Heatmap by Document", fontsize=14, fontweight='bold')
        ax.set_xlabel("SDG", fontsize=12)
        ax.set_ylabel("Document", fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

        st.pyplot(fig)

    else:
        # Single source table
        if sources:
            source = sources[0]
            df_single = pd.DataFrame(
                list(sdg_by_source[source].items()),
                columns=["SDG", "Mentions"]
            ).sort_values("Mentions", ascending=False)

            st.dataframe(df_single, use_container_width=True)

    # Top chunks per SDG
    st.markdown("### 🔍 Sample Chunks by SDG")

    selected_sdg = st.selectbox(
        "Select an SDG to view sample mentions:",
        options=sorted(sdg_counts.keys()) if sdg_counts else [],
        format_func=lambda x: SDG_LABEL.get(x, x)
    )

    if selected_sdg:
        # Find chunks mentioning this SDG
        matching_chunks = [
            c for c in classified_chunks
            if any(m["sdg_id"] == selected_sdg for m in c.get("sdg_matches", []))
        ]

        st.write(f"**{SDG_LABEL.get(selected_sdg, selected_sdg)}**: {len(matching_chunks)} mentions")

        # Show top 3 samples
        for i, chunk in enumerate(matching_chunks[:3], 1):
            with st.expander(f"Sample {i} — {chunk['source']} (page {chunk['page']})"):
                st.write(chunk['text'])

                # Show SDG confidence
                for match in chunk.get("sdg_matches", []):
                    if match["sdg_id"] == selected_sdg:
                        st.caption(f"Confidence: {match['score']:.2f}")
