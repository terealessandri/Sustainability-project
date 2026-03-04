"""
SDG Classifier Module
Keyword-based classification of text into UN's 17 Sustainable Development Goals

Uses TF-IDF-style keyword matching to map ESG report content to SDGs.
Fast, lightweight, no model download required — suitable for CPU-constrained
deployment on Streamlit Community Cloud free tier.
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import re


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

# Keyword sets per SDG for fast matching
SDG_KEYWORDS = {
    "SDG 1":  ["poverty", "poor", "social protection", "inequality", "vulnerable", "livelihood",
                "economic inclusion", "basic needs", "low income", "underprivileged"],
    "SDG 2":  ["hunger", "food security", "nutrition", "agriculture", "farming", "food waste",
                "sustainable agriculture", "crop", "malnutrition", "food system"],
    "SDG 3":  ["health", "well-being", "wellbeing", "disease", "healthcare", "medical",
                "mental health", "safety", "mortality", "pandemic", "epidemic", "wellness"],
    "SDG 4":  ["education", "training", "skills", "learning", "school", "literacy",
                "workforce development", "vocational", "university", "scholarship"],
    "SDG 5":  ["gender", "women", "female", "equality", "diversity", "inclusion",
                "discrimination", "empowerment", "pay gap", "parental leave", "harassment"],
    "SDG 6":  ["water", "sanitation", "wastewater", "clean water", "water quality",
                "freshwater", "water efficiency", "water management", "sewage"],
    "SDG 7":  ["energy", "renewable", "solar", "wind", "clean energy", "efficiency",
                "electricity", "fossil fuel", "power", "carbon free", "net zero energy"],
    "SDG 8":  ["employment", "jobs", "labor", "labour", "wage", "economic growth",
                "decent work", "workforce", "supply chain", "fair trade", "modern slavery"],
    "SDG 9":  ["innovation", "infrastructure", "technology", "research", "development",
                "r&d", "digital", "automation", "patent", "industrial", "engineering"],
    "SDG 10": ["inequality", "inclusion", "diverse", "minority", "refugee", "migration",
                "social mobility", "equal opportunity", "pay equity", "marginalized"],
    "SDG 11": ["urban", "city", "housing", "transport", "mobility", "sustainable city",
                "community", "public transport", "affordable housing", "smart city"],
    "SDG 12": ["consumption", "waste", "recycling", "circular economy", "packaging",
                "sustainable production", "responsible sourcing", "single use", "landfill"],
    "SDG 13": ["climate", "carbon", "emissions", "ghg", "greenhouse gas", "co2",
                "net zero", "paris agreement", "decarbonization", "carbon neutral",
                "scope 1", "scope 2", "scope 3", "global warming", "climate change"],
    "SDG 14": ["ocean", "marine", "sea", "aquatic", "fish", "coral", "plastic pollution",
                "water pollution", "coastal", "blue economy", "overfishing"],
    "SDG 15": ["forest", "biodiversity", "ecosystem", "land", "deforestation", "wildlife",
                "habitat", "species", "nature", "conservation", "reforestation"],
    "SDG 16": ["governance", "transparency", "accountability", "ethics", "compliance",
                "anti-corruption", "rule of law", "human rights", "reporting", "audit"],
    "SDG 17": ["partnership", "collaboration", "stakeholder", "global", "initiative",
                "coalition", "cross-sector", "multilateral", "united nations", "sdg"],
}


class SDGClassifier:
    """
    Keyword-based classifier for mapping text to UN Sustainable Development Goals.

    Uses TF-IDF-style keyword matching to classify text chunks into one or more
    of the 17 SDGs. Runs in milliseconds with no model download required.

    Attributes:
        model_name: Kept for API compatibility (unused)
        confidence_threshold: Minimum score to accept SDG match (0-1)
    """

    def __init__(self, model_name: str = "keyword-matching",
                 confidence_threshold: float = 0.15):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.classifier = True  # Always "loaded"

        self.sdg_labels = []
        self.sdg_descriptions = {}
        self.sdg_names = {}

        for sdg_id, name, description in SDG_DEFINITIONS:
            self.sdg_labels.append(sdg_id)
            self.sdg_descriptions[sdg_id] = description
            self.sdg_names[sdg_id] = name

    def load_model(self):
        """No-op: keyword matching requires no model."""
        pass

    def classify_text(self, text: str,
                     multi_label: bool = True,
                     top_k: Optional[int] = None) -> List[Dict]:
        """
        Classify a single text into SDGs using keyword matching.

        Returns list of SDG matches above confidence_threshold, sorted by score.
        """
        if not text or not text.strip():
            return []

        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        total_words = max(len(words), 1)

        matches = []
        for sdg_id, keywords in SDG_KEYWORDS.items():
            hit_count = sum(1 for kw in keywords if kw in text_lower)
            if hit_count == 0:
                continue

            # Normalize: hits / total keywords, scaled to 0-1
            score = min(hit_count / len(keywords) * 3.0, 1.0)

            if score >= self.confidence_threshold:
                matches.append({
                    "sdg_id": sdg_id,
                    "sdg_name": self.sdg_names[sdg_id],
                    "score": round(score, 3),
                    "description": self.sdg_descriptions[sdg_id]
                })

        matches.sort(key=lambda x: x["score"], reverse=True)

        if top_k is not None:
            matches = matches[:top_k]

        return matches

    def classify_chunks(self, chunks: List[Dict],
                       show_progress: bool = True) -> List[Dict]:
        """Classify multiple chunks into SDGs."""
        classified_chunks = []

        for chunk in chunks:
            sdg_matches = self.classify_text(chunk["text"], multi_label=True)
            classified_chunk = dict(chunk)
            classified_chunk["sdg_matches"] = sdg_matches
            classified_chunks.append(classified_chunk)

        if show_progress:
            print(f"✓ Classified {len(chunks)} chunks")

        return classified_chunks

    def aggregate_by_document(self, classified_chunks: List[Dict]) -> Dict[str, Dict]:
        """Aggregate SDG coverage by source document."""
        aggregation = defaultdict(lambda: defaultdict(lambda: {
            "count": 0,
            "scores": [],
            "chunks": []
        }))

        for chunk in classified_chunks:
            source = chunk.get("source", "unknown")
            for match in chunk.get("sdg_matches", []):
                sdg_id = match["sdg_id"]
                aggregation[source][sdg_id]["count"] += 1
                aggregation[source][sdg_id]["scores"].append(match["score"])
                aggregation[source][sdg_id]["chunks"].append(chunk["chunk_id"])

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
        """Get overall SDG coverage statistics across all documents."""
        total_chunks = len(classified_chunks)
        chunks_with_sdgs = sum(1 for c in classified_chunks if c.get("sdg_matches"))

        sdg_counts = defaultdict(int)
        for chunk in classified_chunks:
            for match in chunk.get("sdg_matches", []):
                sdg_counts[match["sdg_id"]] += 1

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
        """Generate human-readable coverage report."""
        lines = ["=" * 60, "SDG COVERAGE REPORT", "=" * 60]

        summary = self.get_coverage_summary(classified_chunks)
        lines.append(f"\nTotal chunks analyzed: {summary['total_chunks']}")
        lines.append(f"Chunks with SDG matches: {summary['chunks_with_sdgs']} ({summary['coverage_rate']:.1%})")
        lines.append(f"Unique SDGs identified: {summary['num_unique_sdgs']}/17")
        lines.append("\n--- Top SDGs ---")

        for sdg_id, count in summary['top_sdgs'][:10]:
            lines.append(f"  {sdg_id} ({self.sdg_names[sdg_id]}): {count} mentions")

        if by_document:
            doc_aggregation = self.aggregate_by_document(classified_chunks)
            lines += ["\n" + "=" * 60, "PER-DOCUMENT BREAKDOWN", "=" * 60]

            for source, sdgs in sorted(doc_aggregation.items()):
                lines.append(f"\n{source}:")
                for sdg_id, data in sorted(sdgs.items(), key=lambda x: x[1]["count"], reverse=True)[:5]:
                    lines.append(
                        f"  {sdg_id} ({data['sdg_name']}): "
                        f"{data['count']} mentions (avg score: {data['avg_score']:.2f})"
                    )

        return "\n".join(lines)
