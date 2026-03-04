"""
SDG Classifier Module
Zero-shot classification of text into UN's 17 Sustainable Development Goals

Uses facebook/bart-large-mnli for zero-shot classification without training data.
Maps ESG report content to SDGs for coverage analysis and greenwashing detection.
"""

from transformers import pipeline
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import warnings
import streamlit as st

# Suppress transformers warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


@st.cache_resource
def _load_zero_shot_pipeline(model_name: str):
    """Load and cache the zero-shot classification pipeline across Streamlit sessions."""
    return pipeline(
        "zero-shot-classification",
        model=model_name,
        device=-1
    )


# UN Sustainable Development Goals with descriptions
# Format: (SDG_ID, Name, Description for zero-shot classification)
SDG_DEFINITIONS = [
    ("SDG 1", "No Poverty", "End poverty in all its forms, economic development, social protection"),
    ("SDG 2", "Zero Hunger", "Food security, nutrition, sustainable agriculture, hunger elimination"),
    ("SDG 3", "Good Health", "Health and well-being, healthcare access, disease prevention, mental health"),
    ("SDG 4", "Quality Education", "Education access, learning opportunities, skills development, literacy"),
    ("SDG 5", "Gender Equality", "Women's empowerment, gender equality, discrimination elimination"),
    ("SDG 6", "Clean Water", "Water and sanitation access, water quality, wastewater treatment"),
    ("SDG 7", "Clean Energy", "Affordable clean energy, renewable energy, energy efficiency, solar, wind"),
    ("SDG 8", "Economic Growth", "Economic growth, employment, decent work, labor rights"),
    ("SDG 9", "Innovation", "Infrastructure, innovation, industrialization, technology, research"),
    ("SDG 10", "Reduced Inequalities", "Inequality reduction, inclusion, fair opportunities"),
    ("SDG 11", "Sustainable Cities", "Sustainable cities, urban planning, housing, transportation"),
    ("SDG 12", "Responsible Consumption", "Responsible consumption and production, waste reduction, circular economy, recycling"),
    ("SDG 13", "Climate Action", "Climate action, emissions reduction, carbon neutrality, global warming mitigation"),
    ("SDG 14", "Life Below Water", "Ocean conservation, marine resources, aquatic ecosystems"),
    ("SDG 15", "Life on Land", "Terrestrial ecosystems, forests, biodiversity, land degradation"),
    ("SDG 16", "Peace and Justice", "Peace, justice, strong institutions, governance, transparency, accountability"),
    ("SDG 17", "Partnerships", "Global partnerships, cooperation, sustainable development collaboration")
]


class SDGClassifier:
    """
    Zero-shot classifier for mapping text to UN Sustainable Development Goals.

    Uses a zero-shot NLI model to classify text chunks into one or more of the 17 SDGs
    without requiring labeled training data.

    Attributes:
        model_name: HuggingFace model ID (default: cross-encoder/nli-MiniLM2-L6-H768)
        classifier: Transformers zero-shot classification pipeline
        sdg_labels: List of SDG IDs
        sdg_descriptions: Dict mapping SDG IDs to full descriptions
        confidence_threshold: Minimum confidence score to accept classification
    """

    def __init__(self, model_name: str = "cross-encoder/nli-MiniLM2-L6-H768",
                 confidence_threshold: float = 0.4):
        """
        Initialize SDG classifier.

        Args:
            model_name: HuggingFace model for zero-shot classification
            confidence_threshold: Minimum score to accept SDG match (0-1)
                                 Lower = more SDGs matched, higher = stricter
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.classifier = None

        # Prepare SDG labels and descriptions
        self.sdg_labels = []
        self.sdg_descriptions = {}
        self.sdg_names = {}

        for sdg_id, name, description in SDG_DEFINITIONS:
            self.sdg_labels.append(sdg_id)
            self.sdg_descriptions[sdg_id] = description
            self.sdg_names[sdg_id] = name

    def load_model(self):
        """
        Load zero-shot classification model.
        Downloads ~66MB on first call.
        """
        if self.classifier is None:
            print(f"Loading zero-shot classifier: {self.model_name}...")
            print("(This may take a moment on first run - downloading ~300MB)")
            self.classifier = _load_zero_shot_pipeline(self.model_name)
            print(f"✓ Model loaded: {self.model_name}")

    def classify_text(self, text: str,
                     multi_label: bool = True,
                     top_k: Optional[int] = None) -> List[Dict]:
        """
        Classify a single text into SDGs.

        Args:
            text: Text to classify
            multi_label: Allow multiple SDG assignments (default: True)
            top_k: Return only top K SDGs (None = all above threshold)

        Returns:
            List of SDG matches with scores:
            [
                {
                    "sdg_id": "SDG 13",
                    "sdg_name": "Climate Action",
                    "score": 0.87,
                    "description": "Climate action, emissions reduction..."
                },
                ...
            ]

        Example:
            classifier = SDGClassifier()
            results = classifier.classify_text(
                "We reduced carbon emissions by 30% through renewable energy"
            )
        """
        # Handle empty or whitespace-only text
        if not text or not text.strip():
            return []

        self.load_model()

        # Get candidate labels (descriptions for better matching)
        candidate_labels = [self.sdg_descriptions[sdg] for sdg in self.sdg_labels]

        # Classify
        result = self.classifier(
            text,
            candidate_labels,
            multi_label=multi_label
        )

        # Format results
        matches = []
        for label, score in zip(result['labels'], result['scores']):
            # Map description back to SDG ID
            sdg_id = None
            for sid, desc in self.sdg_descriptions.items():
                if desc == label:
                    sdg_id = sid
                    break

            if sdg_id and score >= self.confidence_threshold:
                matches.append({
                    "sdg_id": sdg_id,
                    "sdg_name": self.sdg_names[sdg_id],
                    "score": float(score),
                    "description": self.sdg_descriptions[sdg_id]
                })

        # Apply top_k if specified
        if top_k is not None:
            matches = matches[:top_k]

        return matches

    def classify_chunks(self, chunks: List[Dict],
                       show_progress: bool = True) -> List[Dict]:
        """
        Classify multiple chunks into SDGs.

        Args:
            chunks: List of chunk dicts with 'text' field (from PDFParser)
            show_progress: Print progress during classification

        Returns:
            List of chunks with added 'sdg_matches' field:
            [
                {
                    "chunk_id": "report_p5_c2",
                    "text": "...",
                    "sdg_matches": [
                        {"sdg_id": "SDG 13", "score": 0.87, ...},
                        {"sdg_id": "SDG 7", "score": 0.65, ...}
                    ]
                },
                ...
            ]
        """
        self.load_model()

        classified_chunks = []

        for i, chunk in enumerate(chunks):
            if show_progress and (i + 1) % 10 == 0:
                print(f"  Classified {i + 1}/{len(chunks)} chunks...")

            # Classify chunk
            sdg_matches = self.classify_text(chunk["text"], multi_label=True)

            # Add to chunk data
            classified_chunk = dict(chunk)
            classified_chunk["sdg_matches"] = sdg_matches

            classified_chunks.append(classified_chunk)

        if show_progress:
            print(f"✓ Classified {len(chunks)} chunks")

        return classified_chunks

    def aggregate_by_document(self, classified_chunks: List[Dict]) -> Dict[str, Dict]:
        """
        Aggregate SDG coverage by source document.

        Args:
            classified_chunks: Chunks with 'sdg_matches' field

        Returns:
            Dictionary mapping source -> SDG coverage:
            {
                "company_a.pdf": {
                    "SDG 13": {
                        "count": 15,
                        "avg_score": 0.78,
                        "chunks": [...]
                    },
                    "SDG 7": {
                        "count": 8,
                        "avg_score": 0.65,
                        "chunks": [...]
                    },
                    ...
                }
            }
        """
        aggregation = defaultdict(lambda: defaultdict(lambda: {
            "count": 0,
            "scores": [],
            "chunks": []
        }))

        for chunk in classified_chunks:
            source = chunk.get("source", "unknown")

            for match in chunk.get("sdg_matches", []):
                sdg_id = match["sdg_id"]
                score = match["score"]

                aggregation[source][sdg_id]["count"] += 1
                aggregation[source][sdg_id]["scores"].append(score)
                aggregation[source][sdg_id]["chunks"].append(chunk["chunk_id"])

        # Calculate averages
        result = {}
        for source, sdgs in aggregation.items():
            result[source] = {}
            for sdg_id, data in sdgs.items():
                result[source][sdg_id] = {
                    "count": data["count"],
                    "avg_score": sum(data["scores"]) / len(data["scores"]),
                    "sdg_name": self.sdg_names[sdg_id],
                    "chunks": data["chunks"]
                }

        return dict(result)

    def get_coverage_summary(self, classified_chunks: List[Dict]) -> Dict:
        """
        Get overall SDG coverage statistics across all documents.

        Args:
            classified_chunks: Chunks with 'sdg_matches' field

        Returns:
            Summary statistics:
            {
                "total_chunks": 150,
                "chunks_with_sdgs": 142,
                "coverage_rate": 0.947,
                "sdg_distribution": {
                    "SDG 13": 45,
                    "SDG 7": 32,
                    ...
                },
                "top_sdgs": [
                    ("SDG 13", 45),
                    ("SDG 7", 32),
                    ...
                ]
            }
        """
        total_chunks = len(classified_chunks)
        chunks_with_sdgs = sum(1 for c in classified_chunks if c.get("sdg_matches"))

        sdg_counts = defaultdict(int)
        for chunk in classified_chunks:
            for match in chunk.get("sdg_matches", []):
                sdg_counts[match["sdg_id"]] += 1

        # Sort by count
        top_sdgs = sorted(sdg_counts.items(), key=lambda x: x[1], reverse=True)

        return {
            "total_chunks": total_chunks,
            "chunks_with_sdgs": chunks_with_sdgs,
            "coverage_rate": chunks_with_sdgs / total_chunks if total_chunks > 0 else 0,
            "sdg_distribution": dict(sdg_counts),
            "top_sdgs": top_sdgs,
            "num_unique_sdgs": len(sdg_counts)
        }

    def format_coverage_report(self, classified_chunks: List[Dict],
                              by_document: bool = True) -> str:
        """
        Generate human-readable coverage report.

        Args:
            classified_chunks: Chunks with SDG classifications
            by_document: Include per-document breakdown

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("SDG COVERAGE REPORT")
        lines.append("=" * 60)

        # Overall summary
        summary = self.get_coverage_summary(classified_chunks)
        lines.append(f"\nTotal chunks analyzed: {summary['total_chunks']}")
        lines.append(f"Chunks with SDG matches: {summary['chunks_with_sdgs']} ({summary['coverage_rate']:.1%})")
        lines.append(f"Unique SDGs identified: {summary['num_unique_sdgs']}/17")

        lines.append("\n--- Top SDGs ---")
        for sdg_id, count in summary['top_sdgs'][:10]:
            sdg_name = self.sdg_names[sdg_id]
            lines.append(f"  {sdg_id} ({sdg_name}): {count} mentions")

        # Per-document breakdown
        if by_document:
            doc_aggregation = self.aggregate_by_document(classified_chunks)

            lines.append("\n" + "=" * 60)
            lines.append("PER-DOCUMENT BREAKDOWN")
            lines.append("=" * 60)

            for source, sdgs in sorted(doc_aggregation.items()):
                lines.append(f"\n{source}:")

                # Sort SDGs by count
                sorted_sdgs = sorted(sdgs.items(), key=lambda x: x[1]["count"], reverse=True)

                for sdg_id, data in sorted_sdgs[:5]:  # Top 5 per document
                    lines.append(
                        f"  {sdg_id} ({data['sdg_name']}): "
                        f"{data['count']} mentions (avg score: {data['avg_score']:.2f})"
                    )

        return "\n".join(lines)


# Demo/Testing functionality
if __name__ == "__main__":
    """
    Test SDG classifier with sample texts or PDFs
    """
    import sys

    if len(sys.argv) > 1:
        # Classify PDFs
        from src.pdf_parser import parse_multiple_pdfs

        pdf_paths = sys.argv[1:]
        print("=" * 60)
        print("SDG CLASSIFIER — Document Analysis")
        print("=" * 60)

        try:
            # Parse PDFs
            print("\nStep 1: Parsing PDFs...")
            chunks = parse_multiple_pdfs(pdf_paths)

            # Classify chunks
            print(f"\nStep 2: Classifying {len(chunks)} chunks into SDGs...")
            classifier = SDGClassifier(confidence_threshold=0.4)
            classified_chunks = classifier.classify_chunks(chunks, show_progress=True)

            # Generate report
            print("\n" + classifier.format_coverage_report(classified_chunks))

            # Interactive query
            print("\n" + "=" * 60)
            print("INTERACTIVE MODE (type 'quit' to exit)")
            print("Enter text to classify into SDGs")
            print("=" * 60)

            while True:
                text = input("\n> ").strip()

                if text.lower() in ["quit", "exit", "q"]:
                    break

                if text:
                    results = classifier.classify_text(text)

                    if results:
                        print("\nSDG Matches:")
                        for match in results:
                            print(f"  {match['sdg_id']} ({match['sdg_name']}): {match['score']:.2f}")
                    else:
                        print("No SDG matches found (all scores below threshold)")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    else:
        # Demo with sample texts
        print("SDG Classifier Demo\n")

        classifier = SDGClassifier(confidence_threshold=0.4)

        sample_texts = [
            "We reduced carbon emissions by 50% through renewable energy investments.",
            "Our workforce diversity program ensures equal opportunities for all genders.",
            "We provide clean water access to 10,000 households in rural communities.",
            "Board governance practices ensure transparency and accountability to stakeholders."
        ]

        for i, text in enumerate(sample_texts, 1):
            print(f"\n[{i}] Text: {text}")
            results = classifier.classify_text(text, top_k=2)

            if results:
                print("  SDG Matches:")
                for match in results:
                    print(f"    {match['sdg_id']} ({match['sdg_name']}): {match['score']:.2f}")
            else:
                print("  No matches")
