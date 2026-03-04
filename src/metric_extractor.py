"""
Metric Extractor Module
Extracts KPIs from ESG reports using spaCy: emissions, percentages, currency, targets

Uses pattern matching and entity recognition to identify quantitative commitments
and classify them as targets (future), actuals (achieved), or vague (unspecific).
"""

import spacy
from spacy.matcher import Matcher
from typing import List, Dict, Optional, Tuple
import re
from collections import defaultdict
import streamlit as st


@st.cache_resource
def _load_spacy_model(model_name: str):
    """Load and cache the spaCy model across Streamlit sessions."""
    return spacy.load(model_name)


class MetricExtractor:
    """
    Extracts quantitative metrics and commitments from ESG text.

    Uses spaCy for NER and pattern matching to identify:
    - Emissions (CO2, CO2e, GHG, tonnes)
    - Percentages (%, reduction targets)
    - Currency (USD, EUR, investment amounts)
    - Dates/years (target years, deadlines)

    Classifies commitments as:
    - Target: Future goals with specific deadlines
    - Actual: Achieved results with evidence
    - Vague: Aspirational without specifics
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize metric extractor with spaCy model.

        Args:
            model_name: spaCy model to load (default: en_core_web_sm)
        """
        self.model_name = model_name
        self.nlp = None
        self.matcher = None

    def load_model(self):
        """Load spaCy model and initialize pattern matchers."""
        if self.nlp is None:
            print(f"Loading spaCy model: {self.model_name}...")
            self.nlp = _load_spacy_model(self.model_name)
            self.matcher = Matcher(self.nlp.vocab)

            # Add patterns
            self._add_emission_patterns()
            self._add_percentage_patterns()
            self._add_currency_patterns()
            self._add_target_patterns()

            print(f"✓ Model loaded with {len(self.matcher)} patterns")

    def _add_emission_patterns(self):
        """Add patterns for emissions mentions (CO2, GHG, tonnes)."""
        emission_patterns = [
            # CO2 / CO2e / GHG with numbers
            [{"LIKE_NUM": True}, {"LOWER": {"IN": ["tonnes", "tons", "mt", "metric tons"]}},
             {"LOWER": {"IN": ["co2", "co2e", "ghg", "carbon"]}}],
            [{"LIKE_NUM": True}, {"LOWER": {"IN": ["co2", "co2e", "ghg"]}},
             {"LOWER": {"IN": ["emissions", "equivalent"]}, "OP": "?"}],

            # Carbon emissions
            [{"LOWER": "carbon"}, {"LOWER": "emissions"}],
            [{"LOWER": {"IN": ["scope", "scopes"]}}, {"LIKE_NUM": True}],

            # Greenhouse gas
            [{"LOWER": "greenhouse"}, {"LOWER": "gas"}, {"LOWER": "emissions", "OP": "?"}],
        ]

        self.matcher.add("EMISSIONS", emission_patterns)

    def _add_percentage_patterns(self):
        """Add patterns for percentage mentions."""
        percentage_patterns = [
            # Number + %
            [{"LIKE_NUM": True}, {"ORTH": "%"}],
            [{"LIKE_NUM": True}, {"LOWER": {"IN": ["percent", "percentage"]}}],

            # Reduction percentages
            [{"LIKE_NUM": True}, {"ORTH": "%"}, {"LOWER": {"IN": ["reduction", "decrease", "increase"]}}],
        ]

        self.matcher.add("PERCENTAGE", percentage_patterns)

    def _add_currency_patterns(self):
        """Add patterns for currency/investment mentions."""
        currency_patterns = [
            # $X million/billion
            [{"ORTH": "$"}, {"LIKE_NUM": True},
             {"LOWER": {"IN": ["million", "billion", "m", "bn", "k"]}, "OP": "?"}],

            # X USD/EUR/GBP
            [{"LIKE_NUM": True}, {"LOWER": {"IN": ["usd", "eur", "gbp", "dollars", "euros"]}}],

            # Investment
            [{"LOWER": "investment"}, {"ORTH": "of", "OP": "?"}, {"ORTH": "$", "OP": "?"}, {"LIKE_NUM": True}],
        ]

        self.matcher.add("CURRENCY", currency_patterns)

    def _add_target_patterns(self):
        """Add patterns for target/goal language."""
        target_patterns = [
            # By YEAR
            [{"LOWER": "by"}, {"LIKE_NUM": True}],

            # Target/goal year
            [{"LOWER": {"IN": ["target", "goal"]}}, {"LIKE_NUM": True, "OP": "?"}],

            # Achieve/reach by
            [{"LOWER": {"IN": ["achieve", "reach", "attain"]}}, {"IS_ALPHA": True, "OP": "*"},
             {"LOWER": "by"}, {"LIKE_NUM": True}],
        ]

        self.matcher.add("TARGET", target_patterns)

    def extract_metrics(self, text: str) -> Dict:
        """
        Extract all metrics from text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with extracted metrics:
            {
                "emissions": [...],
                "percentages": [...],
                "currency": [...],
                "targets": [...],
                "years": [...],
                "commitment_type": "target" | "actual" | "vague"
            }
        """
        if not text or not text.strip():
            return self._empty_metrics()

        self.load_model()

        doc = self.nlp(text)

        metrics = {
            "emissions": self._extract_emissions(doc),
            "percentages": self._extract_percentages(doc),
            "currency": self._extract_currency(doc),
            "targets": self._extract_targets(doc),
            "years": self._extract_years(doc),
            "numbers": self._extract_numbers(doc)
        }

        # Classify commitment type
        metrics["commitment_type"] = self._classify_commitment(text, metrics)

        return metrics

    def _extract_emissions(self, doc) -> List[str]:
        """Extract emission-related mentions."""
        matches = self.matcher(doc, as_spans=True)
        emissions = [span.text for span in matches if span.label_ == "EMISSIONS"]
        return list(set(emissions))  # Unique

    def _extract_percentages(self, doc) -> List[str]:
        """Extract percentage values."""
        matches = self.matcher(doc, as_spans=True)
        percentages = [span.text for span in matches if span.label_ == "PERCENTAGE"]
        return list(set(percentages))

    def _extract_currency(self, doc) -> List[str]:
        """Extract currency/investment amounts."""
        matches = self.matcher(doc, as_spans=True)
        currency = [span.text for span in matches if span.label_ == "CURRENCY"]

        # Also use spaCy's built-in MONEY entity
        money_entities = [ent.text for ent in doc.ents if ent.label_ == "MONEY"]

        all_currency = list(set(currency + money_entities))
        return all_currency

    def _extract_targets(self, doc) -> List[str]:
        """Extract target/goal phrases."""
        matches = self.matcher(doc, as_spans=True)
        targets = [span.text for span in matches if span.label_ == "TARGET"]
        return list(set(targets))

    def _extract_years(self, doc) -> List[int]:
        """Extract year mentions (2020-2100)."""
        years = []

        # From DATE entities
        for ent in doc.ents:
            if ent.label_ == "DATE":
                # Extract 4-digit years
                year_matches = re.findall(r'\b(20\d{2}|2100)\b', ent.text)
                years.extend([int(y) for y in year_matches])

        # Direct pattern matching
        for token in doc:
            if token.like_num and len(token.text) == 4:
                try:
                    year = int(token.text)
                    if 2020 <= year <= 2100:
                        years.append(year)
                except ValueError:
                    pass

        return sorted(list(set(years)))

    def _extract_numbers(self, doc) -> List[float]:
        """Extract all numeric values."""
        numbers = []

        for token in doc:
            if token.like_num:
                try:
                    # Clean and parse number
                    num_str = token.text.replace(',', '').replace('%', '')
                    num = float(num_str)
                    numbers.append(num)
                except ValueError:
                    pass

        return numbers

    def _classify_commitment(self, text: str, metrics: Dict) -> str:
        """
        Classify commitment as target, actual, or vague.

        Classification logic:
        - Target: Has future year (>= current year) + specific goal
        - Actual: Has past achievement language ("achieved", "reduced")
        - Vague: Aspirational language without specifics ("aims to", "plans to")
        """
        text_lower = text.lower()

        # Check for actual achievements
        actual_keywords = ["achieved", "reduced", "reached", "delivered", "completed",
                          "decreased", "increased", "met", "exceeded"]
        if any(keyword in text_lower for keyword in actual_keywords):
            # If has numbers/targets, likely actual achievement
            if metrics["percentages"] or metrics["numbers"]:
                return "actual"

        # Check for future targets
        target_keywords = ["target", "goal", "by 2", "aim to", "plan to", "will",
                          "committed to", "commits to", "pledge"]

        has_target_language = any(keyword in text_lower for keyword in target_keywords)
        has_future_year = any(year >= 2025 for year in metrics["years"])
        has_metrics = bool(metrics["percentages"] or metrics["numbers"])

        if has_target_language and (has_future_year or has_metrics):
            return "target"

        # Check for vague commitments
        vague_keywords = ["aims to", "seeks to", "strives to", "intends to",
                         "endeavors to", "aspires to", "plans to", "hopes to"]

        if any(keyword in text_lower for keyword in vague_keywords):
            # Vague if no specific metrics or deadlines
            if not has_metrics or not has_future_year:
                return "vague"

        # Default to vague if uncertain
        return "vague" if not (metrics["percentages"] or metrics["numbers"]) else "target"

    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure."""
        return {
            "emissions": [],
            "percentages": [],
            "currency": [],
            "targets": [],
            "years": [],
            "numbers": [],
            "commitment_type": "vague"
        }

    def extract_from_chunks(self, chunks: List[Dict],
                           show_progress: bool = True) -> List[Dict]:
        """
        Extract metrics from multiple chunks.

        Args:
            chunks: List of chunk dicts with 'text' field
            show_progress: Print progress during extraction

        Returns:
            Chunks with added 'metrics' field
        """
        self.load_model()

        enriched_chunks = []

        for i, chunk in enumerate(chunks):
            if show_progress and (i + 1) % 50 == 0:
                print(f"  Extracted metrics from {i + 1}/{len(chunks)} chunks...")

            metrics = self.extract_metrics(chunk["text"])

            enriched_chunk = dict(chunk)
            enriched_chunk["metrics"] = metrics

            enriched_chunks.append(enriched_chunk)

        if show_progress:
            print(f"✓ Extracted metrics from {len(chunks)} chunks")

        return enriched_chunks

    def aggregate_metrics(self, enriched_chunks: List[Dict]) -> Dict:
        """
        Aggregate metrics across all chunks.

        Args:
            enriched_chunks: Chunks with 'metrics' field

        Returns:
            Aggregated statistics:
            {
                "total_chunks": 150,
                "chunks_with_metrics": 87,
                "commitment_types": {
                    "target": 45,
                    "actual": 32,
                    "vague": 73
                },
                "total_emissions_mentions": 23,
                "total_percentages": 45,
                "total_currency": 12,
                "year_range": [2025, 2050]
            }
        """
        total_chunks = len(enriched_chunks)
        chunks_with_metrics = 0

        commitment_counts = defaultdict(int)
        all_years = []

        emission_count = 0
        percentage_count = 0
        currency_count = 0

        for chunk in enriched_chunks:
            metrics = chunk.get("metrics", {})

            # Count chunks with any metrics
            has_metrics = (metrics.get("emissions") or
                          metrics.get("percentages") or
                          metrics.get("currency") or
                          metrics.get("targets"))

            if has_metrics:
                chunks_with_metrics += 1

            # Count by commitment type
            commitment_type = metrics.get("commitment_type", "vague")
            commitment_counts[commitment_type] += 1

            # Aggregate counts
            emission_count += len(metrics.get("emissions", []))
            percentage_count += len(metrics.get("percentages", []))
            currency_count += len(metrics.get("currency", []))

            all_years.extend(metrics.get("years", []))

        return {
            "total_chunks": total_chunks,
            "chunks_with_metrics": chunks_with_metrics,
            "metric_coverage_rate": chunks_with_metrics / total_chunks if total_chunks > 0 else 0,
            "commitment_types": dict(commitment_counts),
            "total_emissions_mentions": emission_count,
            "total_percentages": percentage_count,
            "total_currency": currency_count,
            "year_range": [min(all_years), max(all_years)] if all_years else [],
            "unique_years": sorted(list(set(all_years)))
        }

    def format_metrics_report(self, enriched_chunks: List[Dict],
                             by_document: bool = True) -> str:
        """
        Generate human-readable metrics report.

        Args:
            enriched_chunks: Chunks with metrics extracted
            by_document: Include per-document breakdown

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("METRICS EXTRACTION REPORT")
        lines.append("=" * 60)

        agg = self.aggregate_metrics(enriched_chunks)

        lines.append(f"\nTotal chunks analyzed: {agg['total_chunks']}")
        lines.append(f"Chunks with metrics: {agg['chunks_with_metrics']} ({agg['metric_coverage_rate']:.1%})")

        lines.append("\n--- Commitment Types ---")
        for comm_type, count in sorted(agg['commitment_types'].items(), key=lambda x: x[1], reverse=True):
            percentage = count / agg['total_chunks'] * 100 if agg['total_chunks'] > 0 else 0
            lines.append(f"  {comm_type.capitalize()}: {count} ({percentage:.1f}%)")

        lines.append("\n--- Metric Counts ---")
        lines.append(f"  Emissions mentions: {agg['total_emissions_mentions']}")
        lines.append(f"  Percentages: {agg['total_percentages']}")
        lines.append(f"  Currency/investments: {agg['total_currency']}")

        if agg['year_range']:
            lines.append(f"\n--- Timeline ---")
            lines.append(f"  Year range: {agg['year_range'][0]} - {agg['year_range'][1]}")
            lines.append(f"  Years mentioned: {', '.join(map(str, agg['unique_years'][:10]))}")

        # Per-document breakdown
        if by_document:
            lines.append("\n" + "=" * 60)
            lines.append("PER-DOCUMENT BREAKDOWN")
            lines.append("=" * 60)

            # Group by source
            by_source = defaultdict(list)
            for chunk in enriched_chunks:
                by_source[chunk.get("source", "unknown")].append(chunk)

            for source, source_chunks in sorted(by_source.items()):
                source_agg = self.aggregate_metrics(source_chunks)

                lines.append(f"\n{source}:")
                lines.append(f"  Chunks: {source_agg['total_chunks']}")
                lines.append(f"  With metrics: {source_agg['chunks_with_metrics']} ({source_agg['metric_coverage_rate']:.1%})")

                # Commitment breakdown
                for comm_type in ["target", "actual", "vague"]:
                    count = source_agg['commitment_types'].get(comm_type, 0)
                    lines.append(f"    {comm_type.capitalize()}: {count}")

        return "\n".join(lines)


# Demo/Testing functionality
if __name__ == "__main__":
    """
    Test metric extractor with sample texts or PDFs
    """
    import sys

    if len(sys.argv) > 1:
        # Extract from PDFs
        from src.pdf_parser import parse_multiple_pdfs

        pdf_paths = sys.argv[1:]
        print("=" * 60)
        print("METRIC EXTRACTOR — KPI Analysis")
        print("=" * 60)

        try:
            # Parse PDFs
            print("\nStep 1: Parsing PDFs...")
            chunks = parse_multiple_pdfs(pdf_paths)

            # Extract metrics
            print(f"\nStep 2: Extracting metrics from {len(chunks)} chunks...")
            extractor = MetricExtractor()
            enriched = extractor.extract_from_chunks(chunks, show_progress=True)

            # Generate report
            print("\n" + extractor.format_metrics_report(enriched))

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    else:
        # Demo with sample texts
        print("Metric Extractor Demo\n")

        extractor = MetricExtractor()

        sample_texts = [
            "We reduced carbon emissions by 50% by 2030 through renewable energy investments.",
            "Our target is to achieve net zero emissions by 2050.",
            "We aim to improve sustainability across our operations.",
            "Invested $10 million in clean energy infrastructure in 2023.",
            "Achieved a 25% reduction in Scope 1 and 2 emissions last year."
        ]

        for i, text in enumerate(sample_texts, 1):
            print(f"\n[{i}] Text: {text}")
            metrics = extractor.extract_metrics(text)

            print(f"  Type: {metrics['commitment_type']}")
            if metrics['percentages']:
                print(f"  Percentages: {metrics['percentages']}")
            if metrics['years']:
                print(f"  Years: {metrics['years']}")
            if metrics['emissions']:
                print(f"  Emissions: {metrics['emissions']}")
            if metrics['currency']:
                print(f"  Currency: {metrics['currency']}")
