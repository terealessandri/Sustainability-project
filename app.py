"""
ESG TruthBot Analyzer — Streamlit Application
Main entry point for the greenwashing detection dashboard
"""

import streamlit as st
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure page
st.set_page_config(
    page_title="ESG TruthBot Analyzer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E7D32;
    }
    .risk-low {
        color: #2E7D32;
        font-weight: bold;
    }
    .risk-medium {
        color: #F57C00;
        font-weight: bold;
    }
    .risk-high {
        color: #D32F2F;
        font-weight: bold;
    }
    .risk-very-high {
        color: #B71C1C;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🌱 ESG TruthBot Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">RAG-Powered ESG Intelligence & Greenwashing Detection</p>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")

pages = {
    "🏠 Home": "home",
    "📤 Upload Reports": "upload",
    "💬 RAG Q&A": "rag",
    "🎯 SDG Coverage": "sdg",
    "📊 KPI Dashboard": "kpi",
    "🔍 Greenwash Analysis": "analysis"
}

# Page selection
page = st.sidebar.radio("Go to", list(pages.keys()))

# Initialize session state
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "classified_chunks" not in st.session_state:
    st.session_state.classified_chunks = None
if "enriched_chunks" not in st.session_state:
    st.session_state.enriched_chunks = None
if "embedding_manager" not in st.session_state:
    st.session_state.embedding_manager = None
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

# Page routing
if pages[page] == "home":
    # HOME PAGE
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🎯 What We Do")
        st.write("""
        Analyze ESG reports to detect potential greenwashing by:
        - Classifying claims into UN SDGs
        - Extracting quantitative metrics
        - Detecting vague commitments
        - Comparing company language
        - Calculating transparency scores
        """)

    with col2:
        st.markdown("### 🔬 How It Works")
        st.write("""
        1. **Upload** ESG reports (PDF)
        2. **Chunk** text into semantic passages
        3. **Embed** chunks into vector index (FAISS)
        4. **RAG Q&A**: retrieve + synthesize answers
        5. **Classify** into 17 UN SDGs (keyword search)
        6. **Score** transparency (0-100)
        """)

    with col3:
        st.markdown("### 📊 Key Features")
        st.write("""
        - **RAG Q&A**: direct answers from reports
        - Semantic search (vector embeddings)
        - Keyword-based SDG classification
        - Metric extraction & validation
        - Copy-paste detection
        - Transparency risk scoring (0-100)
        """)

    st.markdown("---")

    # Quick stats
    st.markdown("### 📈 System Capabilities")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("UN SDGs", "17", help="Sustainable Development Goals")
    with col2:
        st.metric("Transparency Score", "0-100", help="Risk-based scoring")
    with col3:
        st.metric("Local Models", "✓", help="No API costs")
    with col4:
        st.metric("Zero-Shot", "✓", help="No training data needed")

    st.markdown("---")

    # Getting started
    st.markdown("### 🚀 Getting Started")
    st.info("👈 Use the sidebar to navigate to **Upload Reports** to begin analysis")

    # Tech stack
    with st.expander("🛠️ Technology Stack"):
        st.write("""
        - **RAG Pipeline**: FAISS retrieval → extractive answer synthesis
        - **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, 384-dim)
        - **Vector Store**: FAISS (cosine similarity search)
        - **SDG Classification**: Keyword-based NLP (TF-IDF matching, 17 SDGs)
        - **Metric Extraction**: spaCy (pattern matching)
        - **PDF Processing**: PyMuPDF
        - **No external API needed** — all models run locally
        """)

    # Academic note
    st.markdown("---")
    st.caption("🎓 Academic POC — Built for educational purposes")

elif pages[page] == "upload":
    # Import pages dynamically
    from pages import upload_page
    upload_page.render()

elif pages[page] == "rag":
    from pages import rag_page
    rag_page.render()

elif pages[page] == "sdg":
    from pages import sdg_page
    sdg_page.render()

elif pages[page] == "kpi":
    from pages import kpi_page
    kpi_page.render()

elif pages[page] == "analysis":
    from pages import analysis_page
    analysis_page.render()
