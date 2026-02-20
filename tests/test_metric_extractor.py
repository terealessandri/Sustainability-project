"""
Unit tests for Metric Extractor module
"""

import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.metric_extractor import MetricExtractor


class TestMetricExtractor(unittest.TestCase):
    """Test cases for MetricExtractor class"""

    def setUp(self):
        """Initialize extractor for each test"""
        self.extractor = MetricExtractor()

    def test_initialization(self):
        """Test extractor initializes correctly"""
        self.assertEqual(self.extractor.model_name, "en_core_web_sm")
        self.assertIsNone(self.extractor.nlp)
        self.assertIsNone(self.extractor.matcher)

    def test_extract_emissions(self):
        """Test extraction of emission-related metrics"""
        text = "We reduced carbon emissions by 1000 tonnes CO2e last year."

        metrics = self.extractor.extract_metrics(text)

        # Should detect emissions mention
        self.assertGreater(len(metrics["emissions"]), 0)

    def test_extract_percentages(self):
        """Test extraction of percentage values"""
        text = "Achieved a 50% reduction in emissions and 30% increase in efficiency."

        metrics = self.extractor.extract_metrics(text)

        # Should extract percentages
        self.assertGreater(len(metrics["percentages"]), 0)

        # Check if values are captured
        percentages_str = " ".join(metrics["percentages"])
        self.assertTrue("50" in percentages_str or "30" in percentages_str)

    def test_extract_currency(self):
        """Test extraction of currency/investment amounts"""
        text = "Invested $10 million in renewable energy projects."

        metrics = self.extractor.extract_metrics(text)

        # Should detect currency
        self.assertGreater(len(metrics["currency"]), 0)

    def test_extract_years(self):
        """Test extraction of year mentions"""
        text = "We target carbon neutrality by 2050 and 50% reduction by 2030."

        metrics = self.extractor.extract_metrics(text)

        # Should extract years
        self.assertGreater(len(metrics["years"]), 0)

        # Should include both years
        self.assertIn(2030, metrics["years"])
        self.assertIn(2050, metrics["years"])

    def test_commitment_type_target(self):
        """Test classification of target commitments"""
        text = "Our target is to reduce emissions by 50% by 2030."

        metrics = self.extractor.extract_metrics(text)

        self.assertEqual(metrics["commitment_type"], "target")

    def test_commitment_type_actual(self):
        """Test classification of actual achievements"""
        text = "We achieved a 30% reduction in carbon emissions last year."

        metrics = self.extractor.extract_metrics(text)

        self.assertEqual(metrics["commitment_type"], "actual")

    def test_commitment_type_vague(self):
        """Test classification of vague commitments"""
        text = "We aim to improve our sustainability practices."

        metrics = self.extractor.extract_metrics(text)

        self.assertEqual(metrics["commitment_type"], "vague")

    def test_target_with_specific_year(self):
        """Test that specific year + goal = target"""
        text = "Achieve net zero by 2040."

        metrics = self.extractor.extract_metrics(text)

        self.assertEqual(metrics["commitment_type"], "target")
        self.assertIn(2040, metrics["years"])

    def test_vague_without_metrics(self):
        """Test vague language without numbers"""
        text = "We plan to enhance our environmental efforts."

        metrics = self.extractor.extract_metrics(text)

        self.assertEqual(metrics["commitment_type"], "vague")
        self.assertEqual(len(metrics["percentages"]), 0)
        self.assertEqual(len(metrics["years"]), 0)

    def test_actual_with_past_achievement(self):
        """Test actual achievements with past tense"""
        text = "Reduced Scope 1 emissions by 25% in 2023."

        metrics = self.extractor.extract_metrics(text)

        self.assertEqual(metrics["commitment_type"], "actual")

    def test_empty_text(self):
        """Test handling of empty text"""
        metrics = self.extractor.extract_metrics("")

        self.assertEqual(metrics["commitment_type"], "vague")
        self.assertEqual(len(metrics["emissions"]), 0)
        self.assertEqual(len(metrics["percentages"]), 0)

    def test_complex_text(self):
        """Test extraction from complex text with multiple metrics"""
        text = ("We achieved a 30% reduction in Scope 1 and 2 emissions in 2023, "
                "invested $50 million in renewable energy, and target net zero by 2050 "
                "with an interim goal of 75% reduction by 2030.")

        metrics = self.extractor.extract_metrics(text)

        # Should detect multiple metrics
        self.assertGreater(len(metrics["percentages"]), 0)
        self.assertGreater(len(metrics["currency"]), 0)
        self.assertGreater(len(metrics["years"]), 0)

        # Should have multiple years
        self.assertGreaterEqual(len(metrics["years"]), 2)


class TestChunkProcessing(unittest.TestCase):
    """Test batch processing of chunks"""

    def setUp(self):
        """Set up extractor and sample chunks"""
        self.extractor = MetricExtractor()

        self.sample_chunks = [
            {
                "chunk_id": "test_p1_c0",
                "source": "test.pdf",
                "page": 1,
                "text": "Target 50% emissions reduction by 2030.",
                "word_count": 6
            },
            {
                "chunk_id": "test_p2_c0",
                "source": "test.pdf",
                "page": 2,
                "text": "Achieved 25% reduction last year.",
                "word_count": 5
            },
            {
                "chunk_id": "test_p3_c0",
                "source": "test.pdf",
                "page": 3,
                "text": "We plan to improve sustainability.",
                "word_count": 5
            }
        ]

    def test_extract_from_chunks(self):
        """Test extracting metrics from multiple chunks"""
        enriched = self.extractor.extract_from_chunks(
            self.sample_chunks,
            show_progress=False
        )

        # Should return same number of chunks
        self.assertEqual(len(enriched), len(self.sample_chunks))

        # Each chunk should have metrics field
        for chunk in enriched:
            self.assertIn("metrics", chunk)
            self.assertIsInstance(chunk["metrics"], dict)

            # Original fields preserved
            self.assertIn("chunk_id", chunk)
            self.assertIn("text", chunk)

    def test_aggregate_metrics(self):
        """Test aggregating metrics across chunks"""
        enriched = self.extractor.extract_from_chunks(
            self.sample_chunks,
            show_progress=False
        )

        agg = self.extractor.aggregate_metrics(enriched)

        # Check structure
        self.assertIn("total_chunks", agg)
        self.assertIn("chunks_with_metrics", agg)
        self.assertIn("commitment_types", agg)
        self.assertIn("metric_coverage_rate", agg)

        # Check values
        self.assertEqual(agg["total_chunks"], 3)

        # Should have commitment type counts
        self.assertIn("target", agg["commitment_types"])
        self.assertIn("actual", agg["commitment_types"])
        self.assertIn("vague", agg["commitment_types"])

    def test_format_metrics_report(self):
        """Test report formatting"""
        enriched = self.extractor.extract_from_chunks(
            self.sample_chunks,
            show_progress=False
        )

        report = self.extractor.format_metrics_report(enriched)

        # Should be a string
        self.assertIsInstance(report, str)

        # Should contain key sections
        self.assertIn("METRICS EXTRACTION REPORT", report)
        self.assertIn("Commitment Types", report)
        self.assertIn("Metric Counts", report)


class TestMetricPatterns(unittest.TestCase):
    """Test specific pattern matching"""

    def setUp(self):
        """Initialize extractor"""
        self.extractor = MetricExtractor()

    def test_scope_emissions(self):
        """Test detection of Scope 1/2/3 emissions"""
        text = "Reduced Scope 1 and Scope 2 emissions significantly."

        metrics = self.extractor.extract_metrics(text)

        # Should detect emissions or scope mentions
        self.assertTrue(
            len(metrics["emissions"]) > 0 or
            "scope" in " ".join(metrics.get("targets", [])).lower()
        )

    def test_multiple_percentages(self):
        """Test extraction of multiple percentage values"""
        text = "Improved efficiency by 30%, reduced waste by 40%, and cut costs by 15%."

        metrics = self.extractor.extract_metrics(text)

        # Should detect multiple percentages
        self.assertGreaterEqual(len(metrics["percentages"]), 2)

    def test_investment_amounts(self):
        """Test extraction of investment/financial commitments"""
        texts = [
            "Invested $10 million in clean energy.",
            "Allocated 5 million EUR for sustainability projects.",
            "$500k for renewable infrastructure."
        ]

        for text in texts:
            metrics = self.extractor.extract_metrics(text)
            # Should detect some form of currency/number
            has_financial = (len(metrics["currency"]) > 0 or
                           len(metrics["numbers"]) > 0)
            self.assertTrue(has_financial, f"Failed for: {text}")

    def test_target_year_patterns(self):
        """Test various target year expressions"""
        texts = [
            "By 2030",
            "Target 2050",
            "Achieve net zero by 2040",
            "Goal of carbon neutrality by 2035"
        ]

        for text in texts:
            metrics = self.extractor.extract_metrics(text)
            self.assertGreater(
                len(metrics["years"]),
                0,
                f"Should extract year from: {text}"
            )

    def test_no_false_positives(self):
        """Test that unrelated text doesn't trigger false positives"""
        text = "The quick brown fox jumps over the lazy dog."

        metrics = self.extractor.extract_metrics(text)

        # Should have minimal or no metrics
        total_metrics = (len(metrics["emissions"]) +
                        len(metrics["percentages"]) +
                        len(metrics["currency"]))

        self.assertLessEqual(total_metrics, 0)


if __name__ == "__main__":
    unittest.main()
