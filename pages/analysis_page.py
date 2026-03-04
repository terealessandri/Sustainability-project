"""
Greenwash Analysis Page — Transparency Scoring
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from greenwash_scorer import GreenwashScorer


def render():
    """Render the greenwash analysis page."""
    st.markdown("## 🔍 Greenwashing Transparency Analysis")

    if not st.session_state.processing_complete:
        st.warning("⚠️ Please upload and process reports first!")
        st.info("👈 Go to **Upload Reports** to get started")
        return

    st.write("Comprehensive transparency scoring to detect potential greenwashing.")

    # Initialize scorer
    similarity_analyzer = st.session_state.get("similarity_analyzer")
    scorer = GreenwashScorer(similarity_analyzer)

    enriched_chunks = st.session_state.enriched_chunks

    # Get unique sources
    sources = list(set(c.get("source") for c in enriched_chunks))

    # Calculate scores — cache in session_state to avoid recomputation on page reloads
    if "greenwash_comparison" not in st.session_state:
        with st.spinner("Calculating transparency scores..."):
            st.session_state.greenwash_comparison = scorer.compare_documents(enriched_chunks)
    comparison = st.session_state.greenwash_comparison

    # Overall statistics
    st.markdown("### 📊 Overall Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Average Score", f"{comparison['statistics']['average_score']:.1f}/100")

    with col2:
        st.metric("Best Score", f"{comparison['statistics']['best_score']:.1f}/100")

    with col3:
        st.metric("Worst Score", f"{comparison['statistics']['worst_score']:.1f}/100")

    with col4:
        st.metric("Documents", comparison['statistics']['total_documents'])

    # Document rankings
    st.markdown("### 🏆 Transparency Rankings")

    for i, doc in enumerate(comparison['documents'], 1):
        with st.expander(
            f"**#{i} — {doc['source']}** | Score: {doc['overall_score']:.1f}/100",
            expanded=(i == 1)
        ):
            # Risk level badge
            risk_class = {
                "low": "risk-low",
                "medium": "risk-medium",
                "high": "risk-high",
                "very_high": "risk-very-high"
            }.get(doc['risk_level'], "")

            st.markdown(
                f"<p class='{risk_class}'>Risk Level: {doc['risk_level'].replace('_', ' ').upper()}</p>",
                unsafe_allow_html=True
            )

            # Component scores
            st.markdown("#### Component Scores")

            components = doc['component_scores']

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Metric Specificity", f"{components.get('metric_specificity', 0):.1f}")
                st.metric("Temporal Clarity", f"{components.get('temporal_clarity', 0):.1f}")

            with col2:
                st.metric("SDG Coverage", f"{components.get('sdg_coverage', 0):.1f}")
                st.metric("Uniqueness", f"{components.get('uniqueness', 0):.1f}")

            with col3:
                st.metric("Actual Achievements", f"{components.get('actual_achievements', 0):.1f}")
                st.metric("Chunks with Metrics", f"{doc['chunks_with_metrics']}/{doc['total_chunks']}")

            # SDG-specific scores
            if doc['sdg_scores']:
                st.markdown("#### SDG-Specific Scores")

                sdg_df = pd.DataFrame([
                    {
                        "SDG": sdg_id,
                        "Score": data['score'],
                        "Risk": data['risk_level'],
                        "Mentions": data['chunk_count']
                    }
                    for sdg_id, data in doc['sdg_scores'].items()
                ]).sort_values("Score", ascending=False)

                st.dataframe(sdg_df, use_container_width=True)

            # Red flags
            if doc['red_flags']:
                st.markdown("#### ⚠️ Red Flags (Potential Greenwashing)")

                red_flag_df = pd.DataFrame([
                    {"Issue": flag, "Count": count}
                    for flag, count in sorted(doc['red_flags'].items(), key=lambda x: x[1], reverse=True)
                ])

                st.dataframe(red_flag_df, use_container_width=True)

            # Green flags
            if doc['green_flags']:
                st.markdown("#### ✅ Green Flags (Positive Indicators)")

                green_flag_df = pd.DataFrame([
                    {"Indicator": flag, "Count": count}
                    for flag, count in sorted(doc['green_flags'].items(), key=lambda x: x[1], reverse=True)
                ])

                st.dataframe(green_flag_df, use_container_width=True)

    # Comparison chart
    st.markdown("### 📊 Score Comparison")

    if len(comparison['documents']) > 1:
        fig, ax = plt.subplots(figsize=(12, max(6, len(comparison['documents']) * 0.5)))

        sources_list = [doc['source'] for doc in comparison['documents']]
        scores_list = [doc['overall_score'] for doc in comparison['documents']]

        # Color bars by risk level
        colors = []
        for doc in comparison['documents']:
            risk = doc['risk_level']
            if risk == "low":
                colors.append('#2E7D32')
            elif risk == "medium":
                colors.append('#F57C00')
            elif risk == "high":
                colors.append('#D32F2F')
            else:
                colors.append('#B71C1C')

        bars = ax.barh(sources_list, scores_list, color=colors, alpha=0.7, edgecolor='black')

        ax.set_xlabel("Transparency Score", fontsize=12)
        ax.set_ylabel("Document", fontsize=12)
        ax.set_title("Transparency Score Comparison", fontsize=14, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.grid(axis='x', alpha=0.3)

        # Add score labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 2, bar.get_y() + bar.get_height()/2,
                   f'{width:.1f}',
                   ha='left', va='center', fontsize=10, fontweight='bold')

        # Add threshold lines
        ax.axvline(80, color='green', linestyle='--', alpha=0.5, label='Low Risk')
        ax.axvline(60, color='orange', linestyle='--', alpha=0.5, label='Medium Risk')
        ax.axvline(40, color='red', linestyle='--', alpha=0.5, label='High Risk')

        ax.legend()

        st.pyplot(fig)

    # Export reports
    st.markdown("### 📥 Export Report")

    # Generate full report text — pass precomputed comparison to avoid double computation
    report_text = scorer.format_score_report(enriched_chunks, source=None,
                                             precomputed_comparison=comparison)

    st.download_button(
        label="📄 Download Full Report (TXT)",
        data=report_text,
        file_name="greenwashing_analysis.txt",
        mime="text/plain"
    )

    # Key insights
    st.markdown("### 💡 Key Insights")

    high_risk_docs = [d for d in comparison['documents'] if d['risk_level'] in ['high', 'very_high']]
    low_risk_docs = [d for d in comparison['documents'] if d['risk_level'] == 'low']

    if high_risk_docs:
        st.warning(f"⚠️ **{len(high_risk_docs)} document(s) show high greenwashing risk**")
        for doc in high_risk_docs:
            st.write(f"- {doc['source']}: {doc['overall_score']:.1f}/100")

    if low_risk_docs:
        st.success(f"✅ **{len(low_risk_docs)} document(s) show high transparency**")
        for doc in low_risk_docs:
            st.write(f"- {doc['source']}: {doc['overall_score']:.1f}/100")

    if not high_risk_docs and not low_risk_docs:
        st.info("Most documents fall in the medium transparency range.")
