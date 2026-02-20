"""
Upload Page — PDF Upload and Processing
"""

import streamlit as st
import tempfile
import os
from pdf_parser import parse_multiple_pdfs
from sdg_classifier import SDGClassifier
from metric_extractor import MetricExtractor
from embeddings import EmbeddingManager
from similarity import SimilarityAnalyzer


def render():
    """Render the upload page."""
    st.markdown("## 📤 Upload ESG Reports")
    st.write("Upload one or more ESG reports (PDF format) for analysis.")

    # File uploader
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload ESG sustainability reports in PDF format"
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")

        # Show uploaded files
        with st.expander("📁 Uploaded Files"):
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size / 1024:.1f} KB)")

        # Process button
        if st.button("🚀 Process Reports", type="primary"):
            process_reports(uploaded_files)

    else:
        st.info("👆 Upload PDF files to get started")

        # Sample data option
        st.markdown("---")
        st.markdown("### 📝 Don't have ESG reports?")
        st.write("You can test with sample texts or find ESG reports from:")
        st.write("- Company annual reports (Investor Relations pages)")
        st.write("- ESG/Sustainability reports (usually on company websites)")
        st.write("- CDP (Carbon Disclosure Project) disclosures")

    # Show current state
    if st.session_state.processing_complete:
        st.markdown("---")
        st.success("✅ **Processing Complete!**")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Chunks",
                len(st.session_state.chunks) if st.session_state.chunks else 0
            )

        with col2:
            st.metric(
                "Documents",
                len(set(c.get("source") for c in st.session_state.chunks)) if st.session_state.chunks else 0
            )

        with col3:
            st.metric(
                "Status",
                "Ready"
            )

        st.info("👈 Navigate to other pages to explore the analysis!")


def process_reports(uploaded_files):
    """Process uploaded PDF reports through the full pipeline."""

    with st.spinner("Processing reports... This may take a few minutes."):

        # Save uploaded files temporarily
        temp_paths = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_paths.append(tmp_file.name)

        try:
            # Step 1: Parse PDFs
            with st.spinner("📄 Step 1/5: Parsing PDFs..."):
                chunks = parse_multiple_pdfs(temp_paths)
                st.session_state.chunks = chunks
                st.success(f"✓ Parsed {len(chunks)} chunks from {len(uploaded_files)} document(s)")

            # Step 2: Classify SDGs
            with st.spinner("🎯 Step 2/5: Classifying into SDGs..."):
                classifier = SDGClassifier(confidence_threshold=0.4)
                classified = classifier.classify_chunks(chunks, show_progress=False)
                st.session_state.classified_chunks = classified
                st.success(f"✓ Classified chunks into UN SDGs")

            # Step 3: Extract Metrics
            with st.spinner("📊 Step 3/5: Extracting metrics..."):
                extractor = MetricExtractor()
                enriched = extractor.extract_from_chunks(classified, show_progress=False)
                st.session_state.enriched_chunks = enriched
                st.success(f"✓ Extracted quantitative metrics")

            # Step 4: Build Embeddings
            with st.spinner("🧠 Step 4/5: Building semantic index..."):
                manager = EmbeddingManager()
                manager.build_index(enriched)
                st.session_state.embedding_manager = manager
                st.success(f"✓ Built searchable vector index")

            # Step 5: Analyze Similarity
            with st.spinner("🔍 Step 5/5: Analyzing similarity..."):
                analyzer = SimilarityAnalyzer(manager)
                st.session_state.similarity_analyzer = analyzer
                st.success(f"✓ Similarity analysis ready")

            # Mark as complete
            st.session_state.processing_complete = True
            st.session_state.uploaded_files = [f.name for f in uploaded_files]

            st.balloons()
            st.success("🎉 **All processing complete!** Navigate to other pages to explore results.")

        except Exception as e:
            st.error(f"❌ Error processing reports: {str(e)}")
            st.exception(e)

        finally:
            # Clean up temp files
            for path in temp_paths:
                try:
                    os.unlink(path)
                except:
                    pass
