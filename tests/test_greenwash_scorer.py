"""
Unit tests for Greenwashing Scorer module
"""

import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.greenwash_scorer import GreenwashScorer


class TestGreenwashScorer(unittest.TestCase):
    """Test cases for GreenwashScorer class"""

    def setUp(self):
        """Initialize scorer for each test"""
        self.scorer = GreenwashScorer()

    def test_initialization(self):
        """Test scorer initializes with correct weights"""
        self.assertIn("metric_specificity", self.scorer.weights)
        self.assertIn("sdg_coverage", self.scorer.weights)
        self.assertIn("temporal_clarity", self.scorer.weights)

        # Weights should sum to 1.0
        self.assertAlmostEqual(sum(self.scorer.weights.values()), 1.0)

    def test_score_metric_specificity_high(self):
        """Test scoring with multiple metrics"""
        metrics = {
            "percentages": ["50%", "30%"],
            "emissions": ["CO2", "GHG"],
            "currency": ["$10M"],
            "years": [2030]
        }

        score = self.scorer._score_metric_specificity(metrics)

        # Should be high (has percentage, emissions, currency)
        self.assertGreaterEqual(score, 80)

    def test_score_metric_specificity_low(self):
        """Test scoring with no metrics"""
        metrics = {
            "percentages": [],
            "emissions": [],
            "currency": [],
            "numbers": []
        }

        score = self.scorer._score_metric_specificity(metrics)

        # Should be 0 (no metrics)
        self.assertEqual(score, 0.0)

    def test_score_temporal_clarity_multiple_years(self):
        """Test scoring with multiple years"""
        metrics = {"years": [2030, 2040, 2050]}

        score = self.scorer._score_temporal_clarity(metrics)

        # Should be 100 (multiple years)
        self.assertEqual(score, 100.0)

    def test_score_temporal_clarity_single_year(self):
        """Test scoring with single year"""
        metrics = {"years": [2030]}

        score = self.scorer._score_temporal_clarity(metrics)

        # Should be 70 (single year)
        self.assertEqual(score, 70.0)

    def test_score_temporal_clarity_no_years(self):
        """Test scoring with no years"""
        metrics = {"years": []}

        score = self.scorer._score_temporal_clarity(metrics)

        # Should be 0 (no timeframe)
        self.assertEqual(score, 0.0)

    def test_score_sdg_coverage_multiple_high_confidence(self):
        """Test scoring with multiple high-confidence SDGs"""
        sdg_matches = [
            {"sdg_id": "SDG 13", "score": 0.9},
            {"sdg_id": "SDG 7", "score": 0.85},
            {"sdg_id": "SDG 12", "score": 0.8}
        ]

        score = self.scorer._score_sdg_coverage(sdg_matches)

        # Should be high (multiple SDGs, high confidence)
        self.assertGreaterEqual(score, 80)

    def test_score_sdg_coverage_none(self):
        """Test scoring with no SDGs"""
        sdg_matches = []

        score = self.scorer._score_sdg_coverage(sdg_matches)

        # Should be 0 (no SDGs)
        self.assertEqual(score, 0.0)

    def test_identify_red_flags_sdg_without_metrics(self):
        """Test red flag for SDG claim without metrics"""
        sdg_matches = [{"sdg_id": "SDG 13", "score": 0.9}]
        metrics = {
            "percentages": [],
            "emissions": [],
            "currency": []
        }

        flags = self.scorer._identify_red_flags(sdg_matches, metrics, "vague")

        # Should flag SDG without metrics
        self.assertGreater(len(flags), 0)
        self.assertTrue(
            any("quantitative" in flag.lower() for flag in flags)
        )

    def test_identify_red_flags_vague_commitment(self):
        """Test red flag for vague commitment"""
        sdg_matches = []
        metrics = {}

        flags = self.scorer._identify_red_flags(sdg_matches, metrics, "vague")

        # Should flag vague commitment
        self.assertTrue(any("vague" in flag.lower() for flag in flags))

    def test_identify_green_flags_quantified_target(self):
        """Test green flag for quantified target with deadline"""
        sdg_matches = [{"sdg_id": "SDG 13", "score": 0.9}]
        metrics = {
            "percentages": ["50%"],
            "years": [2030]
        }

        flags = self.scorer._identify_green_flags(sdg_matches, metrics, "target")

        # Should have green flag for quantified target
        self.assertGreater(len(flags), 0)

    def test_identify_green_flags_actual_achievement(self):
        """Test green flag for actual achievements"""
        sdg_matches = []
        metrics = {"percentages": ["25%"]}

        flags = self.scorer._identify_green_flags(sdg_matches, metrics, "actual")

        # Should flag evidence of achievement
        self.assertTrue(any("achievement" in flag.lower() for flag in flags))

    def test_determine_risk_level(self):
        """Test risk level determination"""
        self.assertEqual(self.scorer._determine_risk_level(85), "low")
        self.assertEqual(self.scorer._determine_risk_level(70), "medium")
        self.assertEqual(self.scorer._determine_risk_level(50), "high")
        self.assertEqual(self.scorer._determine_risk_level(30), "very_high")

    def test_score_chunk_with_good_data(self):
        """Test scoring chunk with transparent data"""
        chunk = {
            "chunk_id": "test_p1_c0",
            "sdg_matches": [
                {"sdg_id": "SDG 13", "score": 0.9}
            ],
            "metrics": {
                "percentages": ["50%"],
                "emissions": ["CO2"],
                "years": [2030],
                "commitment_type": "target"
            }
        }

        score = self.scorer.score_chunk(chunk)

        # Check structure
        self.assertIn("chunk_id", score)
        self.assertIn("overall_score", score)
        self.assertIn("risk_level", score)
        self.assertIn("component_scores", score)
        self.assertIn("red_flags", score)
        self.assertIn("green_flags", score)

        # Should have decent score (has metrics, SDG, year)
        self.assertGreater(score["overall_score"], 50)

    def test_score_chunk_with_poor_data(self):
        """Test scoring chunk with potential greenwashing"""
        chunk = {
            "chunk_id": "test_p1_c0",
            "sdg_matches": [
                {"sdg_id": "SDG 13", "score": 0.9}
            ],
            "metrics": {
                "percentages": [],
                "emissions": [],
                "years": [],
                "commitment_type": "vague"
            }
        }

        score = self.scorer.score_chunk(chunk)

        # Should have low score (SDG without backing)
        self.assertLess(score["overall_score"], 40)

        # Should have red flags
        self.assertGreater(len(score["red_flags"]), 0)

    def test_score_document(self):
        """Test document-level scoring"""
        enriched_chunks = [
            {
                "chunk_id": "test_p1_c0",
                "source": "test.pdf",
                "sdg_matches": [{"sdg_id": "SDG 13", "score": 0.9}],
                "metrics": {
                    "percentages": ["50%"],
                    "years": [2030],
                    "commitment_type": "target"
                }
            },
            {
                "chunk_id": "test_p2_c0",
                "source": "test.pdf",
                "sdg_matches": [{"sdg_id": "SDG 7", "score": 0.85}],
                "metrics": {
                    "emissions": ["CO2"],
                    "years": [2040],
                    "commitment_type": "actual"
                }
            }
        ]

        score = self.scorer.score_document(enriched_chunks, "test.pdf")

        # Check structure
        self.assertEqual(score["source"], "test.pdf")
        self.assertIn("overall_score", score)
        self.assertIn("risk_level", score)
        self.assertIn("component_scores", score)
        self.assertIn("sdg_scores", score)

        # Should have scores for both SDGs
        self.assertIn("SDG 13", score["sdg_scores"])
        self.assertIn("SDG 7", score["sdg_scores"])

    def test_score_document_empty(self):
        """Test scoring document with no data"""
        enriched_chunks = []

        score = self.scorer.score_document(enriched_chunks, "missing.pdf")

        # Should return empty score
        self.assertEqual(score["overall_score"], 0.0)
        self.assertEqual(score["risk_level"], "very_high")

    def test_compare_documents(self):
        """Test comparing multiple documents"""
        enriched_chunks = [
            {
                "chunk_id": "a_p1_c0",
                "source": "company_a.pdf",
                "sdg_matches": [{"sdg_id": "SDG 13", "score": 0.9}],
                "metrics": {
                    "percentages": ["50%"],
                    "years": [2030],
                    "commitment_type": "target"
                }
            },
            {
                "chunk_id": "b_p1_c0",
                "source": "company_b.pdf",
                "sdg_matches": [{"sdg_id": "SDG 13", "score": 0.9}],
                "metrics": {
                    "percentages": [],
                    "years": [],
                    "commitment_type": "vague"
                }
            }
        ]

        comparison = self.scorer.compare_documents(enriched_chunks)

        # Check structure
        self.assertIn("documents", comparison)
        self.assertIn("statistics", comparison)

        # Should have both documents
        self.assertEqual(len(comparison["documents"]), 2)

        # Documents should be sorted by score
        scores = [d["overall_score"] for d in comparison["documents"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_format_score_report_single_document(self):
        """Test report formatting for single document"""
        enriched_chunks = [
            {
                "chunk_id": "test_p1_c0",
                "source": "test.pdf",
                "sdg_matches": [{"sdg_id": "SDG 13", "score": 0.9}],
                "metrics": {
                    "percentages": ["50%"],
                    "years": [2030],
                    "commitment_type": "target"
                }
            }
        ]

        report = self.scorer.format_score_report(enriched_chunks, source="test.pdf")

        # Should be a string
        self.assertIsInstance(report, str)

        # Should contain key sections
        self.assertIn("GREENWASHING TRANSPARENCY REPORT", report)
        self.assertIn("Overall Score", report)
        self.assertIn("Risk Level", report)

    def test_format_score_report_comparison(self):
        """Test report formatting for comparison"""
        enriched_chunks = [
            {
                "chunk_id": "a_p1_c0",
                "source": "company_a.pdf",
                "sdg_matches": [{"sdg_id": "SDG 13", "score": 0.9}],
                "metrics": {"percentages": ["50%"], "years": [2030], "commitment_type": "target"}
            },
            {
                "chunk_id": "b_p1_c0",
                "source": "company_b.pdf",
                "sdg_matches": [{"sdg_id": "SDG 13", "score": 0.9}],
                "metrics": {"percentages": [], "years": [], "commitment_type": "vague"}
            }
        ]

        report = self.scorer.format_score_report(enriched_chunks, source=None)

        # Should contain comparison sections
        self.assertIn("Document Rankings", report)
        self.assertIn("company_a.pdf", report)
        self.assertIn("company_b.pdf", report)


class TestScoringLogic(unittest.TestCase):
    """Test scoring calculation logic"""

    def setUp(self):
        """Initialize scorer"""
        self.scorer = GreenwashScorer()

    def test_high_transparency_score(self):
        """Test that comprehensive data yields high score"""
        chunk = {
            "chunk_id": "test",
            "sdg_matches": [
                {"sdg_id": "SDG 13", "score": 0.95},
                {"sdg_id": "SDG 7", "score": 0.90}
            ],
            "metrics": {
                "percentages": ["50%", "75%"],
                "emissions": ["CO2", "Scope 1"],
                "currency": ["$10M"],
                "years": [2030, 2040, 2050],
                "commitment_type": "target"
            }
        }

        score = self.scorer.score_chunk(chunk)

        # Should be high transparency
        self.assertGreater(score["overall_score"], 70)
        self.assertIn(score["risk_level"], ["low", "medium"])

    def test_low_transparency_score(self):
        """Test that vague data yields low score"""
        chunk = {
            "chunk_id": "test",
            "sdg_matches": [{"sdg_id": "SDG 13", "score": 0.5}],
            "metrics": {
                "percentages": [],
                "emissions": [],
                "years": [],
                "commitment_type": "vague"
            }
        }

        score = self.scorer.score_chunk(chunk)

        # Should be low transparency
        self.assertLess(score["overall_score"], 50)
        self.assertIn(score["risk_level"], ["high", "very_high"])


if __name__ == "__main__":
    unittest.main()
