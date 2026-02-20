"""
RAG Q&A Page — Semantic Search Over Reports
"""

import streamlit as st
from rag_query import RAGQueryEngine, format_results


def render():
    """Render the RAG Q&A page."""
    st.markdown("## 💬 RAG Q&A — Ask Questions About Reports")

    if not st.session_state.processing_complete:
        st.warning("⚠️ Please upload and process reports first!")
        st.info("👈 Go to **Upload Reports** to get started")
        return

    st.write("Ask natural language questions about the ESG reports. The system will search semantically and return relevant passages.")

    # Initialize query engine
    manager = st.session_state.embedding_manager
    engine = RAGQueryEngine(manager, default_top_k=5)

    # Query interface
    col1, col2 = st.columns([4, 1])

    with col1:
        query = st.text_input(
            "Your question:",
            placeholder="What are the carbon reduction targets?",
            key="rag_query"
        )

    with col2:
        top_k = st.number_input("Results", min_value=1, max_value=20, value=5)

    # Source filter
    sources = list(set(c.get("source") for c in st.session_state.chunks))
    selected_sources = st.multiselect(
        "Filter by source (optional):",
        options=sources,
        help="Leave empty to search all documents"
    )

    # Search button
    if st.button("🔍 Search", type="primary") and query:
        with st.spinner("Searching..."):
            try:
                # Perform search
                results = engine.query(
                    query,
                    top_k=top_k,
                    sources=selected_sources if selected_sources else None,
                    include_scores=True
                )

                if results:
                    st.success(f"Found {len(results)} relevant passages")

                    # Display results
                    for i, result in enumerate(results, 1):
                        with st.expander(f"**[{i}]** {result['source']} (page {result['page']}) — Score: {result['score']:.3f}"):
                            st.write(result['text'])

                            # Metadata
                            st.caption(f"Chunk ID: {result['chunk_id']} | Words: {result.get('word_count', 'N/A')}")

                else:
                    st.info("No results found. Try a different query.")

            except Exception as e:
                st.error(f"Error during search: {str(e)}")

    # Example queries
    with st.expander("💡 Example Questions"):
        st.write("""
        - What are the carbon emissions reduction targets?
        - How much has been invested in renewable energy?
        - What governance practices are mentioned?
        - What are the goals for 2030?
        - How does the company address climate change?
        - What social responsibility initiatives are described?
        """)

    # Compare sources
    if len(sources) >= 2:
        st.markdown("---")
        st.markdown("### 🔄 Compare Sources")

        col1, col2 = st.columns(2)

        with col1:
            source_a = st.selectbox("Source A:", sources, key="compare_a")

        with col2:
            source_b = st.selectbox("Source B:", sources, index=min(1, len(sources)-1), key="compare_b")

        compare_query = st.text_input(
            "Question to compare:",
            placeholder="carbon emissions",
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
                    for result in comparison["source_a"]["results"]:
                        st.write(f"- {result['text'][:150]}...")

                with col2:
                    st.markdown(f"**{source_b}**")
                    for result in comparison["source_b"]["results"]:
                        st.write(f"- {result['text'][:150]}...")
