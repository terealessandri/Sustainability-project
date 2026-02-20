# NLP Pipeline — Technical Details

This document explains each NLP component, model choice, and implementation approach.

---

## 1. Text Extraction & Chunking

**Library**: PyMuPDF (fitz)
**Why**: Faster than pdfplumber, better layout preservation than PyPDF2

**Chunking Strategy**:
- **Size**: ~300 words per chunk
- **Overlap**: 50 words (prevents context loss at boundaries)
- **Why 300 words**: Balance between semantic coherence and embedding model context (384 tokens ≈ 500 words max)

**Output Schema**:
```python
{
    "chunk_id": "report_A_p3_c1",
    "source": "company_A_esg_2023.pdf",
    "page": 3,
    "text": "We achieved a 25% reduction in Scope 1 emissions..."
}
```

---

## 2. Sentence Embeddings

**Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Speed**: ~3000 sentences/sec on CPU
- **Quality**: Good balance for semantic search tasks

**Alternative Considered**:
- `all-mpnet-base-v2`: Higher quality (768-dim) but 3x slower
- **Decision**: MiniLM sufficient for POC, faster iteration

**Embedding Process**:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks, show_progress_bar=True)
```

---

## 3. Vector Store (FAISS)

**Index Type**: `IndexFlatL2` (exact search)
- **Why not HNSW/IVF**: Dataset too small (<10K chunks expected), exact search is fast enough

**Search Method**: Cosine similarity via L2 normalization
```python
faiss.normalize_L2(embeddings)  # Convert to cosine space
index.search(query_embedding, k=5)
```

---

## 4. Zero-Shot SDG Classification

**Model**: `facebook/bart-large-mnli`
**Task**: Multi-label classification into 17 UN SDGs

**SDG Labels** (subset shown):
```python
SDG_LABELS = {
    "SDG 7": "Affordable and clean energy, renewable energy access",
    "SDG 12": "Responsible consumption and production, circular economy",
    "SDG 13": "Climate action, emissions reduction, carbon neutrality",
    # ... all 17 SDGs
}
```

**Pipeline**:
```python
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = classifier(text, candidate_labels=SDG_LABELS.values())
```

**Threshold**: Confidence > 0.4 to filter weak predictions
**Aggregation**: Count SDG mentions per report → coverage heatmap

---

## 5. Metric Extraction (spaCy)

**Model**: `en_core_web_sm` (12MB, efficient)
**Patterns**:
```python
# Emissions
{"LOWER": {"IN": ["emissions", "co2", "ghg"]}, "OP": "?"}, {"IS_DIGIT": True}

# Percentages
{"LIKE_NUM": True}, {"ORTH": "%"}

# Currency
{"ORTH": "$"}, {"LIKE_NUM": True}

# Targets
{"LOWER": {"IN": ["target", "goal", "by"]}}, {"IS_DIGIT": True}  # "by 2030"
```

**Classification Logic**:
- **Target**: Contains "by [year]", "target", "goal"
- **Actual**: Contains "achieved", "reduced", "reached"
- **Vague**: "plans to", "commits to", "aims to" (no date)

---

## 6. Similarity Analysis

**Method**: Cosine similarity between company embeddings

**Use Case**:
```
Query: "How do companies A and B compare on SDG 13?"
→ Extract SDG 13 chunks from both
→ Average embeddings per company
→ Compute similarity score
```

**Interpretation**:
- **>0.8**: Nearly identical language (potential copy-paste)
- **0.5-0.8**: Similar strategies
- **<0.5**: Divergent approaches

---

## 7. Greenwashing Transparency Score

**Formula** (per SDG):
```python
score = (
    0.4 * metric_specificity +  # Has quantified targets?
    0.3 * temporal_clarity +     # Clear deadlines?
    0.2 * language_confidence +  # Avoids vague terms?
    0.1 * sdg_coverage           # Mentions SDG explicitly?
)
```

**Flags**:
- ⚠️ **High Risk**: SDG claim with no metrics (score <40)
- ⚡ **Medium**: Vague targets ("net zero ambitions")
- ✅ **Transparent**: "Reduce Scope 1 emissions 50% by 2030 vs. 2020 baseline"

---

## Validation Strategy

**Test Cases**:
1. **Positive Control**: Sample ESG report with known metrics
2. **Negative Control**: Marketing fluff text (should score low)
3. **Edge Cases**: Unusual formatting, tables, non-English snippets

**Evaluation Metrics** (if ground truth available):
- SDG classification: Precision/Recall per SDG
- Metric extraction: F1 score for entity recognition
- Greenwashing: Manual validation against 10-report sample

---

## Limitations

- **No LLM reasoning**: Cannot detect subtle rhetorical greenwashing
- **English-only**: Would need multilingual models for global reports
- **PDF quality**: Scanned images require OCR (not implemented)
- **Domain drift**: Models not fine-tuned on ESG corpus
