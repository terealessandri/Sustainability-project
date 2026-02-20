# ESG TruthBot Analyzer

**An NLP-Powered Greenwashing Detection System for ESG Reports**
*Academic Proof of Concept — University Project*

![Status](https://img.shields.io/badge/status-complete-success)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Tests](https://img.shields.io/badge/tests-128%20passing-success)
![License](https://img.shields.io/badge/license-academic-orange)

---

## 📋 Overview

ESG TruthBot Analyzer is a comprehensive NLP system that analyzes ESG (Environmental, Social, Governance) reports to detect potential greenwashing. The system combines multiple detection signals to calculate transparency scores and identify red flags in corporate sustainability claims.

### Key Capabilities

- **📄 PDF Processing**: Automatic text extraction and semantic chunking
- **🎯 SDG Classification**: Zero-shot classification into 17 UN Sustainable Development Goals
- **📊 Metric Extraction**: Automated KPI detection (emissions, percentages, currency, targets)
- **🔍 Greenwashing Detection**: Multi-signal transparency scoring (0-100 scale)
- **💬 Semantic Search**: RAG-based Q&A over multiple reports
- **🔗 Similarity Analysis**: Cross-company comparison and copy-paste detection
- **🌐 Interactive UI**: Streamlit dashboard with 5 analysis pages

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- 8GB RAM minimum (16GB recommended)
- ~5GB disk space for models

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd TrushBot

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Running the Application

```bash
# Launch Streamlit dashboard
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

---

## 📖 Usage Guide

### 1. Upload Reports

- Navigate to **Upload Reports** page
- Select one or more ESG PDF reports
- Click **Process Reports**
- Wait for 5-step pipeline to complete:
  1. PDF parsing & chunking
  2. SDG classification
  3. Metric extraction
  4. Embedding generation
  5. Similarity analysis

### 2. Explore Analysis

**RAG Q&A** 💬
- Ask natural language questions about reports
- Filter by specific documents
- Compare responses across sources

**SDG Coverage** 🎯
- View SDG distribution charts
- Explore heatmaps for multi-company comparison
- Examine sample chunks per SDG

**KPI Dashboard** 📊
- Browse extracted metrics table
- Analyze commitment types (target/actual/vague)
- View timeline of target years
- Export data as CSV

**Greenwash Analysis** 🔍
- Review transparency rankings
- Examine component scores
- Identify red flags (potential greenwashing)
- Spot green flags (positive indicators)
- Download full report

---

## 🏗️ System Architecture

```
┌─────────────┐
│  PDF Upload │
└──────┬──────┘
       │
       ▼
┌─────────────────┐     ┌──────────────────┐
│  PDF Parser     │────▶│  Text Chunks     │
│  (PyMuPDF)      │     │  (300 words)     │
└─────────────────┘     └────────┬─────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
         ┌──────────────────┐    ┌──────────────────┐
         │  SDG Classifier  │    │  Metric          │
         │  (BART-mnli)     │    │  Extractor       │
         │  Zero-shot       │    │  (spaCy)         │
         └────────┬─────────┘    └────────┬─────────┘
                  │                       │
                  └───────────┬───────────┘
                              ▼
                   ┌─────────────────────┐
                   │  Embeddings +       │
                   │  FAISS Index        │
                   │  (sentence-trans.)  │
                   └──────────┬──────────┘
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
         ┌──────────────────┐  ┌──────────────────┐
         │  Similarity      │  │  Greenwashing    │
         │  Analyzer        │  │  Scorer          │
         │  (Cosine Sim.)   │  │  (0-100)         │
         └──────────────────┘  └──────────────────┘
                    │                    │
                    └─────────┬──────────┘
                              ▼
                   ┌─────────────────────┐
                   │  Streamlit UI       │
                   │  (6 pages)          │
                   └─────────────────────┘
```

---

## 🛠️ Technology Stack

### Core NLP
- **sentence-transformers** (all-MiniLM-L6-v2) — 384-dim embeddings
- **FAISS** — Vector similarity search
- **BART-large-mnli** — Zero-shot SDG classification
- **spaCy** (en_core_web_sm) — Pattern-based metric extraction

### Processing
- **PyMuPDF** — PDF text extraction
- **scikit-learn** — Cosine similarity
- **NumPy, pandas** — Data processing

### Interface
- **Streamlit** — Web dashboard
- **matplotlib, seaborn** — Visualizations

**All models run locally — No API costs!**

---

## 📊 Project Structure

```
esg-truthbot/
├── app.py                      # Streamlit main entry
├── src/
│   ├── pdf_parser.py           # PDF → chunks (Step 2)
│   ├── embeddings.py           # Embeddings + FAISS (Step 3)
│   ├── rag_query.py            # Semantic search (Step 4)
│   ├── sdg_classifier.py       # SDG classification (Step 5)
│   ├── metric_extractor.py     # KPI extraction (Step 6)
│   ├── similarity.py           # Cross-company analysis (Step 7)
│   └── greenwash_scorer.py     # Transparency scoring (Step 8)
├── pages/
│   ├── upload_page.py          # PDF upload & processing
│   ├── rag_page.py             # Q&A interface
│   ├── sdg_page.py             # SDG visualization
│   ├── kpi_page.py             # Metrics dashboard
│   └── analysis_page.py        # Greenwashing analysis
├── tests/                      # 128 unit tests (100% passing)
├── docs/
│   ├── architecture.md         # System design
│   └── nlp_pipeline.md         # Technical details
├── DEVLOG.md                   # Development log
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## 🎯 Greenwashing Detection Methodology

### Multi-Signal Scoring

Transparency scores combine 5 weighted signals:

| Signal | Weight | Detection |
|--------|--------|-----------|
| **Metric Specificity** | 30% | Quantified targets? (%, emissions, $) |
| **SDG Coverage** | 25% | Which SDGs addressed? Confidence? |
| **Temporal Clarity** | 20% | Deadlines specified? (by 2030) |
| **Uniqueness** | 15% | Authentic or copy-paste language? |
| **Actual Achievements** | 10% | Evidence of past action? |

### Risk Levels

- **🟢 Low (80-100)**: High transparency, minimal risk
- **🟡 Medium (60-79)**: Some concerns, needs monitoring
- **🟠 High (40-59)**: Significant red flags
- **🔴 Very High (0-39)**: Critical transparency issues

### Red Flags

- ⚠️ SDG claims without quantitative backing
- ⚠️ Vague commitments ("aims to improve")
- ⚠️ No deadlines for goals
- ⚠️ Copy-paste language from competitors
- ⚠️ Low uniqueness score (generic boilerplate)

---

## 🧪 Testing

```bash
# Run all unit tests
python -m pytest tests/ -v

# Run specific module tests
python -m pytest tests/test_greenwash_scorer.py -v

# Test coverage
python -m pytest tests/ --cov=src
```

**Test Statistics**:
- 128 unit tests across 8 modules
- 100% passing rate
- Comprehensive edge case coverage

---

## 📚 Documentation

- **[DEVLOG.md](DEVLOG.md)**: Complete development log with decisions and rationale
- **[docs/architecture.md](docs/architecture.md)**: System architecture and design
- **[docs/nlp_pipeline.md](docs/nlp_pipeline.md)**: Technical NLP pipeline details

---

## 💡 Example Usage

### Command Line

```python
from src.pdf_parser import parse_multiple_pdfs
from src.sdg_classifier import SDGClassifier
from src.metric_extractor import MetricExtractor
from src.embeddings import EmbeddingManager
from src.similarity import SimilarityAnalyzer
from src.greenwash_scorer import GreenwashScorer

# Parse PDFs
chunks = parse_multiple_pdfs(["company_a.pdf", "company_b.pdf"])

# Classify into SDGs
classifier = SDGClassifier()
classified = classifier.classify_chunks(chunks)

# Extract metrics
extractor = MetricExtractor()
enriched = extractor.extract_from_chunks(classified)

# Build search index
manager = EmbeddingManager()
manager.build_index(enriched)

# Analyze similarity
analyzer = SimilarityAnalyzer(manager)

# Score transparency
scorer = GreenwashScorer(analyzer)
comparison = scorer.compare_documents(enriched)

# Print results
for doc in comparison['documents']:
    print(f"{doc['source']}: {doc['overall_score']:.1f}/100 ({doc['risk_level']})")
```

---

## 🎓 Academic Context

This project demonstrates:

- **Zero-shot learning** for domain-specific classification without training data
- **Multi-signal NLP analysis** combining embeddings, classification, and rule-based extraction
- **Explainable AI** with transparent component scoring
- **Production-quality code** with comprehensive testing and documentation

**Developed for**: Academic evaluation
**Purpose**: Demonstrate practical NLP applications in ESG/sustainability domain
**Approach**: Proof-of-concept system with real-world applicability

---

## 🔧 Troubleshooting

### Model Download Issues

```bash
# If BART model fails to download
python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='facebook/bart-large-mnli')"

# If spaCy model missing
python -m spacy download en_core_web_sm
```

### Memory Issues

- Reduce `chunk_size` in `pdf_parser.py` (default: 300 words)
- Process fewer PDFs at once
- Increase system RAM if possible

### Slow Processing

- First run downloads models (~2GB total)
- Subsequent runs use cached models
- BART classification is slowest step (~10 sec per chunk)

---

## 📝 License

Academic project — for educational purposes only.

---

## 👥 Contributors

Built with Claude Sonnet 4.5 for academic demonstration.

---

## 🙏 Acknowledgments

- **UN Sustainable Development Goals** framework
- **HuggingFace** for transformer models
- **spaCy** for NLP toolkit
- **Streamlit** for rapid UI development

---

## 📞 Support

For issues or questions:
1. Check [DEVLOG.md](DEVLOG.md) for implementation details
2. Review [docs/architecture.md](docs/architecture.md) for system design
3. Examine unit tests for usage examples

---

**Built for transparency. Designed for impact. Powered by NLP.** 🌱
