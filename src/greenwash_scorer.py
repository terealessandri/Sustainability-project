"""
Greenwashing Scorer Module
Combines signals to compute transparency score (0-100) per SDG

Analyzes multiple dimensions to detect potential greenwashing:
- SDG claims backed by quantitative metrics
- Language specificity (target vs. vague)
- Temporal clarity (deadlines specified)
- Uniqueness (authentic vs. copied language)
- Evidence of actual achievements
"""

from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import numpy as np


class GreenwashScorer:
    """
    Calculates transparency scores to detect potential greenwashing.

    Combines multiple signals:
    1. Metric Specificity: Are claims backed by numbers?
    2. SDG Coverage: Which goals are addressed?
    3. Temporal Clarity: Are deadlines specified?
    4. Uniqueness: Is language authentic or generic?
    5. Actual Achievements: Is there evidence of past action?

    Outputs transparency scores (0-100):
    - 80-100: High transparency, low greenwashing risk
    - 60-79: Medium transparency, some concerns
    - 40-59: Low transparency, significant concerns
    - 0-39: Very low transparency, high greenwashing risk
    """

    def __init__(self, similarity_analyzer=None):
        """
        Initialize greenwashing scorer.

        Args:
            similarity_analyzer: Optional SimilarityAnalyzer for uniqueness scoring
        """
        self.similarity_analyzer = similarity_analyzer

        # Scoring weights
        self.weights = {
            "metric_specificity": 0.30,    # Has quantified targets?
            "sdg_coverage": 0.25,          # SDG breadth and depth
            "temporal_clarity": 0.20,      # Clear deadlines?
            "uniqueness": 0.15,            # Authentic language?
            "actual_achievements": 0.10     # Evidence of past action?
        }

    def score_chunk(self, chunk: Dict) -> Dict:
        """
        Calculate transparency score for a single chunk.

        Args:
            chunk: Chunk with SDG matches and metrics

        Returns:
            Score breakdown:
            {
                "chunk_id": "...",
                "overall_score": 73.5,
                "risk_level": "medium",
                "component_scores": {
                    "metric_specificity": 85.0,
                    "temporal_clarity": 70.0,
                    "sdg_coverage": 60.0,
                    "commitment_type": "target"
                },
                "red_flags": ["No quantified target for SDG 13"],
                "green_flags": ["Specific deadline mentioned"]
            }
        """
        sdg_matches = chunk.get("sdg_matches", [])
        metrics = chunk.get("metrics", {})
        commitment_type = metrics.get("commitment_type", "vague")

        # Component scores
        metric_score = self._score_metric_specificity(metrics)
        temporal_score = self._score_temporal_clarity(metrics)
        sdg_score = self._score_sdg_coverage(sdg_matches)

        # Calculate weighted overall score
        overall_score = (
            self.weights["metric_specificity"] * metric_score +
            self.weights["temporal_clarity"] * temporal_score +
            self.weights["sdg_coverage"] * sdg_score
        )

        # Note: uniqueness and actual achievements require multi-chunk context
        # They are calculated at document level

        # Identify red/green flags
        red_flags = self._identify_red_flags(sdg_matches, metrics, commitment_type)
        green_flags = self._identify_green_flags(sdg_matches, metrics, commitment_type)

        # Risk level
        risk_level = self._determine_risk_level(overall_score)

        return {
            "chunk_id": chunk.get("chunk_id"),
            "overall_score": overall_score,
            "risk_level": risk_level,
            "component_scores": {
                "metric_specificity": metric_score,
                "temporal_clarity": temporal_score,
                "sdg_coverage": sdg_score,
                "commitment_type": commitment_type
            },
            "red_flags": red_flags,
            "green_flags": green_flags
        }

    def _score_metric_specificity(self, metrics: Dict) -> float:
        """
        Score based on presence and specificity of metrics.

        100: Multiple quantified metrics (%, emissions, currency)
        70: Some metrics present
        40: Only vague numbers
        0: No metrics
        """
        score = 0.0

        # Check for different metric types
        has_percentage = len(metrics.get("percentages", [])) > 0
        has_emissions = len(metrics.get("emissions", [])) > 0
        has_currency = len(metrics.get("currency", [])) > 0
        has_numbers = len(metrics.get("numbers", [])) > 0

        # Weight different types
        if has_percentage:
            score += 40  # Percentages are strong indicators
        if has_emissions:
            score += 30  # Emissions mentions are specific
        if has_currency:
            score += 20  # Investment amounts show commitment
        elif has_numbers:
            score += 10  # Generic numbers are weak

        return min(score, 100.0)

    def _score_temporal_clarity(self, metrics: Dict) -> float:
        """
        Score based on temporal specificity.

        100: Multiple specific years (e.g., "by 2030 and 2050")
        70: Single year mentioned
        30: Vague timeframe ("soon", "near future")
        0: No timeframe
        """
        years = metrics.get("years", [])

        if not years:
            return 0.0

        # More years = more clarity
        if len(years) >= 2:
            return 100.0
        elif len(years) == 1:
            return 70.0
        else:
            return 30.0

    def _score_sdg_coverage(self, sdg_matches: List[Dict]) -> float:
        """
        Score based on SDG coverage quality.

        100: Multiple SDGs with high confidence
        70: Single SDG with high confidence
        40: Multiple SDGs with low confidence
        20: Single SDG with low confidence
        0: No SDGs
        """
        if not sdg_matches:
            return 0.0

        # Average confidence
        avg_confidence = np.mean([m.get("score", 0) for m in sdg_matches])

        # Score based on count and confidence
        count = len(sdg_matches)

        if count >= 2:
            base_score = 100 if avg_confidence > 0.7 else 60
        else:
            base_score = 70 if avg_confidence > 0.7 else 30

        return base_score

    def _identify_red_flags(self, sdg_matches: List[Dict],
                           metrics: Dict,
                           commitment_type: str) -> List[str]:
        """Identify potential greenwashing indicators."""
        flags = []

        # SDG claims without metrics
        if sdg_matches and not any([
            metrics.get("percentages"),
            metrics.get("emissions"),
            metrics.get("currency")
        ]):
            flags.append("SDG claim without quantitative backing")

        # Vague commitment
        if commitment_type == "vague":
            flags.append("Vague commitment without specifics")

        # No deadlines
        if sdg_matches and not metrics.get("years"):
            flags.append("No deadline specified for SDG goals")

        # Multiple SDGs but generic
        if len(sdg_matches) > 3 and commitment_type == "vague":
            flags.append("Many SDGs claimed with minimal substance")

        return flags

    def _identify_green_flags(self, sdg_matches: List[Dict],
                             metrics: Dict,
                             commitment_type: str) -> List[str]:
        """Identify positive transparency indicators."""
        flags = []

        # Specific metrics
        if metrics.get("percentages") and metrics.get("years"):
            flags.append("Quantified target with deadline")

        # Actual achievements
        if commitment_type == "actual":
            flags.append("Evidence of past achievements")

        # Multiple metric types
        metric_types = sum([
            bool(metrics.get("percentages")),
            bool(metrics.get("emissions")),
            bool(metrics.get("currency"))
        ])
        if metric_types >= 2:
            flags.append("Multiple metric types (comprehensive)")

        # Emissions specificity
        if metrics.get("emissions"):
            flags.append("Specific emissions metrics mentioned")

        return flags

    def _determine_risk_level(self, score: float) -> str:
        """Determine greenwashing risk level from score."""
        if score >= 80:
            return "low"
        elif score >= 60:
            return "medium"
        elif score >= 40:
            return "high"
        else:
            return "very_high"

    def score_document(self, enriched_chunks: List[Dict],
                      source: str) -> Dict:
        """
        Calculate comprehensive transparency score for a document.

        Args:
            enriched_chunks: Chunks with SDG+metrics+embeddings
            source: Document filename to score

        Returns:
            Document-level score:
            {
                "source": "company_a.pdf",
                "overall_score": 68.5,
                "risk_level": "medium",
                "component_scores": {...},
                "sdg_scores": {
                    "SDG 13": {"score": 75, "risk": "medium"},
                    ...
                },
                "red_flags": [...],
                "green_flags": [...],
                "uniqueness_score": 0.67
            }
        """
        # Filter chunks for this source
        source_chunks = [c for c in enriched_chunks if c.get("source") == source]

        if not source_chunks:
            return self._empty_score(source)

        # Score each chunk
        chunk_scores = [self.score_chunk(chunk) for chunk in source_chunks]

        # Calculate overall score
        overall_score = np.mean([s["overall_score"] for s in chunk_scores])

        # Component averages
        component_scores = {
            "metric_specificity": np.mean([
                s["component_scores"]["metric_specificity"] for s in chunk_scores
            ]),
            "temporal_clarity": np.mean([
                s["component_scores"]["temporal_clarity"] for s in chunk_scores
            ]),
            "sdg_coverage": np.mean([
                s["component_scores"]["sdg_coverage"] for s in chunk_scores
            ])
        }

        # Add uniqueness if similarity analyzer available
        if self.similarity_analyzer:
            uniqueness = self.similarity_analyzer.calculate_uniqueness_score(
                enriched_chunks, source
            )
            component_scores["uniqueness"] = uniqueness * 100
            overall_score += self.weights["uniqueness"] * uniqueness * 100

        # Add actual achievement score
        actual_ratio = sum(
            1 for s in chunk_scores
            if s["component_scores"]["commitment_type"] == "actual"
        ) / len(chunk_scores)
        component_scores["actual_achievements"] = actual_ratio * 100
        overall_score += self.weights["actual_achievements"] * actual_ratio * 100

        # Per-SDG scores
        sdg_scores = self._calculate_sdg_scores(source_chunks)

        # Aggregate flags
        all_red_flags = []
        all_green_flags = []
        for s in chunk_scores:
            all_red_flags.extend(s["red_flags"])
            all_green_flags.extend(s["green_flags"])

        # Deduplicate and count
        red_flag_counts = {}
        for flag in all_red_flags:
            red_flag_counts[flag] = red_flag_counts.get(flag, 0) + 1

        green_flag_counts = {}
        for flag in all_green_flags:
            green_flag_counts[flag] = green_flag_counts.get(flag, 0) + 1

        return {
            "source": source,
            "overall_score": overall_score,
            "risk_level": self._determine_risk_level(overall_score),
            "component_scores": component_scores,
            "sdg_scores": sdg_scores,
            "red_flags": red_flag_counts,
            "green_flags": green_flag_counts,
            "total_chunks": len(source_chunks),
            "chunks_with_metrics": sum(
                1 for c in source_chunks
                if c.get("metrics", {}).get("percentages") or
                   c.get("metrics", {}).get("emissions") or
                   c.get("metrics", {}).get("currency")
            )
        }

    def _calculate_sdg_scores(self, source_chunks: List[Dict]) -> Dict[str, Dict]:
        """Calculate per-SDG transparency scores."""
        sdg_data = defaultdict(list)

        # Group chunks by SDG
        for chunk in source_chunks:
            for sdg in chunk.get("sdg_matches", []):
                sdg_id = sdg["sdg_id"]
                chunk_score = self.score_chunk(chunk)
                sdg_data[sdg_id].append(chunk_score)

        # Calculate average score per SDG
        sdg_scores = {}
        for sdg_id, scores in sdg_data.items():
            avg_score = np.mean([s["overall_score"] for s in scores])
            sdg_scores[sdg_id] = {
                "score": avg_score,
                "risk_level": self._determine_risk_level(avg_score),
                "chunk_count": len(scores)
            }

        return sdg_scores

    def _empty_score(self, source: str) -> Dict:
        """Return empty score for source with no data."""
        return {
            "source": source,
            "overall_score": 0.0,
            "risk_level": "very_high",
            "component_scores": {},
            "sdg_scores": {},
            "red_flags": {"No data available": 1},
            "green_flags": {},
            "total_chunks": 0,
            "chunks_with_metrics": 0
        }

    def compare_documents(self, enriched_chunks: List[Dict]) -> Dict:
        """
        Compare transparency scores across all documents.

        Returns ranked comparison with insights.
        """
        sources = list(set(c.get("source") for c in enriched_chunks))

        # Score each document
        doc_scores = []
        for source in sources:
            score = self.score_document(enriched_chunks, source)
            doc_scores.append(score)

        # Sort by overall score (descending)
        doc_scores.sort(key=lambda x: x["overall_score"], reverse=True)

        # Calculate statistics
        scores_list = [d["overall_score"] for d in doc_scores]

        return {
            "documents": doc_scores,
            "statistics": {
                "average_score": np.mean(scores_list),
                "median_score": np.median(scores_list),
                "best_score": max(scores_list),
                "worst_score": min(scores_list),
                "total_documents": len(doc_scores)
            }
        }

    def format_score_report(self, enriched_chunks: List[Dict],
                           source: Optional[str] = None,
                           precomputed_comparison: Optional[Dict] = None) -> str:
        """
        Generate human-readable transparency report.

        Args:
            enriched_chunks: Chunks with all enrichments
            source: Optional specific source (if None, compare all)
            precomputed_comparison: Pre-computed result from compare_documents() to avoid
                                    recomputation (used when source=None)

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("GREENWASHING TRANSPARENCY REPORT")
        lines.append("=" * 60)

        if source:
            # Single document report
            score = self.score_document(enriched_chunks, source)

            lines.append(f"\nDocument: {score['source']}")
            lines.append(f"Overall Score: {score['overall_score']:.1f}/100")
            lines.append(f"Risk Level: {score['risk_level'].upper()}")

            lines.append("\n--- Component Scores ---")
            for component, value in score['component_scores'].items():
                lines.append(f"  {component.replace('_', ' ').title()}: {value:.1f}")

            if score['sdg_scores']:
                lines.append("\n--- SDG-Specific Scores ---")
                for sdg_id, sdg_data in sorted(score['sdg_scores'].items()):
                    lines.append(
                        f"  {sdg_id}: {sdg_data['score']:.1f} "
                        f"({sdg_data['risk_level']}, {sdg_data['chunk_count']} mentions)"
                    )

            if score['red_flags']:
                lines.append("\n--- Red Flags (Potential Greenwashing) ---")
                for flag, count in sorted(score['red_flags'].items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"  ⚠️  {flag} ({count}x)")

            if score['green_flags']:
                lines.append("\n--- Green Flags (Positive Indicators) ---")
                for flag, count in sorted(score['green_flags'].items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"  ✅ {flag} ({count}x)")

        else:
            # Comparative report — use precomputed result if available
            comparison = precomputed_comparison if precomputed_comparison is not None \
                else self.compare_documents(enriched_chunks)

            lines.append(f"\nDocuments Analyzed: {comparison['statistics']['total_documents']}")
            lines.append(f"Average Score: {comparison['statistics']['average_score']:.1f}")
            lines.append(f"Best: {comparison['statistics']['best_score']:.1f}")
            lines.append(f"Worst: {comparison['statistics']['worst_score']:.1f}")

            lines.append("\n--- Document Rankings ---")
            for i, doc in enumerate(comparison['documents'], 1):
                lines.append(
                    f"\n[{i}] {doc['source']}: {doc['overall_score']:.1f}/100 "
                    f"({doc['risk_level'].upper()})"
                )
                lines.append(f"    Chunks with metrics: {doc['chunks_with_metrics']}/{doc['total_chunks']}")

                # Top red flag
                if doc['red_flags']:
                    top_flag = max(doc['red_flags'].items(), key=lambda x: x[1])
                    lines.append(f"    Top concern: {top_flag[0]} ({top_flag[1]}x)")

        return "\n".join(lines)


# Demo/Testing functionality
if __name__ == "__main__":
    """
    Test greenwashing scorer with sample data
    """
    import sys

    print("Greenwashing Scorer Demo")
    print("\nUsage: python src/greenwash_scorer.py <pdf1> <pdf2> ...")
    print("\nThis will:")
    print("  1. Parse PDFs and extract text")
    print("  2. Classify into SDGs")
    print("  3. Extract metrics")
    print("  4. Analyze similarity")
    print("  5. Calculate transparency scores")
    print("  6. Generate greenwashing risk report")
