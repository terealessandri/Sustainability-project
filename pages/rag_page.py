"""
RAG Q&A Page — Semantic Search Over Reports
"""

import streamlit as st
from rag_query import RAGQueryEngine, format_results


def render():
    """Render the RAG Q&A page."""
    st.markdown("## 💬 Ask Questions About the Reports")

    if not st.session_state.processing_complete:
        st.warning("⚠️ Please upload and process reports first!")
        st.info("👈 Go to **Upload Reports** to get started")
        return

    st.write("Ask any question about the ESG reports and get a direct answer with source citations.")

    # Initialize query engine
    manager = st.session_state.embedding_manager
    engine = RAGQueryEngine(manager, default_top_k=5)

    # Query interface
    col1, col2 = st.columns([4, 1])

    with col1:
        query = st.text_input(
            "Your question:",
            placeholder="What are the carbon reduction targets and by what year?",
            key="rag_query"
        )

    with col2:
        top_k = st.number_input("Passages", min_value=1, max_value=20, value=5,
                                help="Number of source passages to retrieve")

    # Source filter — show original filenames
    sources = list(set(c.get("source") for c in st.session_state.chunks))
    selected_sources = st.multiselect(
        "Filter by document (optional):",
        options=sources,
        help="Leave empty to search across all uploaded reports"
    )

    # Search button
    if st.button("🔍 Ask", type="primary") and query:
        with st.spinner("Searching and generating answer..."):
            try:
                results = engine.query(
                    query,
                    top_k=top_k,
                    sources=selected_sources if selected_sources else None,
                    include_scores=True
                )

                if results:
                    # --- Synthesized Answer ---
                    answer_data = engine.synthesize_answer(query, results)

                    st.markdown("### 💡 Answer")
                    st.info(answer_data['answer'])

                    if answer_data['citations']:
                        cites = " · ".join(
                            f"**{c['source']}**, p. {c['page']}"
                            for c in answer_data['citations']
                        )
                        st.caption(f"📎 Sources: {cites}")

                    # --- Source Passages ---
                    st.markdown("---")
                    with st.expander(f"📄 View {len(results)} source passages used"):
                        for i, result in enumerate(results, 1):
                            st.markdown(f"**[{i}] {result['source']} — page {result['page']}**"
                                        f"  *(relevance score: {result['score']:.3f})*")
                            st.write(result['text'])
                            st.markdown("---")

                else:
                    st.info("No relevant passages found. Try rephrasing your question.")

            except Exception as e:
                st.error(f"Error during search: {str(e)}")

    # Example queries
    with st.expander("💡 Example Questions"):
        st.write("""
        - What are the carbon emissions reduction targets and by what year?
        - How much has been invested in renewable energy?
        - What governance practices are mentioned?
        - What are the goals for 2030?
        - How does the company address climate change?
        - What social responsibility initiatives are described?
        - What is the Scope 1, 2, and 3 emissions breakdown?
        """)

    # Compare sources (only shown when 2+ docs uploaded)
    if len(sources) >= 2:
        st.markdown("---")
        st.markdown("### 🔄 Compare Documents")
        st.write("Ask the same question across two different reports side by side.")

        col1, col2 = st.columns(2)

        with col1:
            source_a = st.selectbox("Document A:", sources, key="compare_a")

        with col2:
            source_b = st.selectbox("Document B:", sources, index=min(1, len(sources)-1), key="compare_b")

        compare_query = st.text_input(
            "Question to compare:",
            placeholder="What are the carbon reduction targets?",
            key="compare_query"
        )

        if st.button("Compare") and compare_query:
            with st.spinner("Comparing..."):
                comparison = engine.compare_sources(
                    compare_query,
                    source_a,
                    source_b,
                    top_k_per_source=3
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**{source_a}**")
                    if comparison["source_a"]["results"]:
                        answer_a = engine.synthesize_answer(compare_query, comparison["source_a"]["results"])
                        st.info(answer_a['answer'])
                        with st.expander("View passages"):
                            for r in comparison["source_a"]["results"]:
                                st.write(f"*p. {r['page']}:* {r['text'][:200]}...")
                    else:
                        st.write("No relevant content found.")

                with col2:
                    st.markdown(f"**{source_b}**")
                    if comparison["source_b"]["results"]:
                        answer_b = engine.synthesize_answer(compare_query, comparison["source_b"]["results"])
                        st.info(answer_b['answer'])
                        with st.expander("View passages"):
                            for r in comparison["source_b"]["results"]:
                                st.write(f"*p. {r['page']}:* {r['text'][:200]}...")
                    else:
                        st.write("No relevant content found.")
