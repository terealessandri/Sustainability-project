"""
Unit tests for SDG Classifier module
"""

import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.sdg_classifier import SDGClassifier, SDG_DEFINITIONS


class TestSDGDefinitions(unittest.TestCase):
    """Test SDG definitions and setup"""

    def test_all_17_sdgs_defined(self):
        """Test that all 17 SDGs are defined"""
        self.assertEqual(len(SDG_DEFINITIONS), 17)

        # Check IDs are unique and sequential
        sdg_ids = [sdg[0] for sdg in SDG_DEFINITIONS]
        expected_ids = [f"SDG {i}" for i in range(1, 18)]
        self.assertEqual(sdg_ids, expected_ids)

    def test_sdg_structure(self):
        """Test each SDG has required fields"""
        for sdg_id, name, description in SDG_DEFINITIONS:
            # Check format
            self.assertTrue(sdg_id.startswith("SDG "))
            self.assertGreater(len(name), 0)
            self.assertGreater(len(description), 10)

            # Check no empty strings
            self.assertIsInstance(name, str)
            self.assertIsInstance(description, str)


class TestSDGClassifier(unittest.TestCase):
    """Test cases for SDGClassifier class"""

    def setUp(self):
        """Initialize classifier for each test"""
        self.classifier = SDGClassifier(confidence_threshold=0.3)

    def test_initialization(self):
        """Test classifier initializes correctly"""
        self.assertEqual(self.classifier.model_name, "facebook/bart-large-mnli")
        self.assertEqual(self.classifier.confidence_threshold, 0.3)
        self.assertIsNone(self.classifier.classifier)

        # Check SDG data loaded
        self.assertEqual(len(self.classifier.sdg_labels), 17)
        self.assertEqual(len(self.classifier.sdg_descriptions), 17)
        self.assertEqual(len(self.classifier.sdg_names), 17)

    def test_sdg_label_mapping(self):
        """Test SDG labels map correctly"""
        # Check specific SDGs
        self.assertIn("SDG 1", self.classifier.sdg_labels)
        self.assertIn("SDG 13", self.classifier.sdg_labels)
        self.assertIn("SDG 17", self.classifier.sdg_labels)

        # Check names
        self.assertEqual(self.classifier.sdg_names["SDG 1"], "No Poverty")
        self.assertEqual(self.classifier.sdg_names["SDG 13"], "Climate Action")

        # Check descriptions exist
        self.assertIn("climate", self.classifier.sdg_descriptions["SDG 13"].lower())

    def test_classify_text_climate(self):
        """Test classification of climate-related text"""
        text = "We reduced carbon emissions by 50% through renewable energy."

        results = self.classifier.classify_text(text)

        # Should return results
        self.assertGreater(len(results), 0)

        # Check result structure
        for result in results:
            self.assertIn("sdg_id", result)
            self.assertIn("sdg_name", result)
            self.assertIn("score", result)
            self.assertIn("description", result)

            # Score should be in valid range
            self.assertGreaterEqual(result["score"], 0)
            self.assertLessEqual(result["score"], 1)

        # Should identify climate-related SDGs (13 or 7)
        sdg_ids = [r["sdg_id"] for r in results]
        self.assertTrue(
            "SDG 13" in sdg_ids or "SDG 7" in sdg_ids,
            f"Expected SDG 13 or 7 for climate text, got: {sdg_ids}"
        )

    def test_classify_text_governance(self):
        """Test classification of governance-related text"""
        text = "Our board ensures transparency and accountability to stakeholders."

        results = self.classifier.classify_text(text)

        # Should have results
        self.assertGreater(len(results), 0)

        # Should identify governance SDG (16)
        sdg_ids = [r["sdg_id"] for r in results]
        self.assertIn("SDG 16", sdg_ids, f"Expected SDG 16 for governance text, got: {sdg_ids}")

    def test_classify_text_with_top_k(self):
        """Test limiting results with top_k parameter"""
        text = "We invest in renewable energy, education, and healthcare."

        # Get top 2 only
        results = self.classifier.classify_text(text, top_k=2)

        self.assertEqual(len(results), 2)

        # Should be sorted by score (highest first)
        if len(results) == 2:
            self.assertGreaterEqual(results[0]["score"], results[1]["score"])

    def test_confidence_threshold_filtering(self):
        """Test confidence threshold filters low scores"""
        # Create classifier with high threshold
        strict_classifier = SDGClassifier(confidence_threshold=0.8)

        text = "Some vague corporate statement about doing better."

        results = strict_classifier.classify_text(text)

        # With strict threshold, should have fewer or no results
        # (compared to default threshold)
        default_results = self.classifier.classify_text(text)

        self.assertLessEqual(len(results), len(default_results))

    def test_classify_chunks(self):
        """Test classifying multiple chunks"""
        chunks = [
            {
                "chunk_id": "test_p1_c0",
                "source": "test.pdf",
                "page": 1,
                "text": "We reduced carbon emissions through solar energy.",
                "word_count": 8
            },
            {
                "chunk_id": "test_p2_c0",
                "source": "test.pdf",
                "page": 2,
                "text": "Gender equality is central to our workforce policies.",
                "word_count": 9
            }
        ]

        classified = self.classifier.classify_chunks(chunks, show_progress=False)

        # Should return same number of chunks
        self.assertEqual(len(classified), len(chunks))

        # Each chunk should have sdg_matches field
        for chunk in classified:
            self.assertIn("sdg_matches", chunk)
            self.assertIsInstance(chunk["sdg_matches"], list)

            # Original fields should be preserved
            self.assertIn("chunk_id", chunk)
            self.assertIn("text", chunk)

        # First chunk (climate) should match SDG 13 or 7
        first_matches = [m["sdg_id"] for m in classified[0]["sdg_matches"]]
        self.assertTrue(
            "SDG 13" in first_matches or "SDG 7" in first_matches
        )

        # Second chunk (gender) should match SDG 5
        second_matches = [m["sdg_id"] for m in classified[1]["sdg_matches"]]
        self.assertIn("SDG 5", second_matches)


class TestAggregation(unittest.TestCase):
    """Test SDG aggregation and reporting"""

    def setUp(self):
        """Set up classified chunks for testing"""
        self.classifier = SDGClassifier()

        # Mock classified chunks
        self.classified_chunks = [
            {
                "chunk_id": "report_a_p1_c0",
                "source": "company_a.pdf",
                "text": "Climate text",
                "sdg_matches": [
                    {"sdg_id": "SDG 13", "score": 0.85},
                    {"sdg_id": "SDG 7", "score": 0.65}
                ]
            },
            {
                "chunk_id": "report_a_p2_c0",
                "source": "company_a.pdf",
                "text": "More climate",
                "sdg_matches": [
                    {"sdg_id": "SDG 13", "score": 0.75}
                ]
            },
            {
                "chunk_id": "report_b_p1_c0",
                "source": "company_b.pdf",
                "text": "Governance text",
                "sdg_matches": [
                    {"sdg_id": "SDG 16", "score": 0.90}
                ]
            }
        ]

    def test_aggregate_by_document(self):
        """Test aggregating SDG coverage by document"""
        aggregation = self.classifier.aggregate_by_document(self.classified_chunks)

        # Should have both sources
        self.assertIn("company_a.pdf", aggregation)
        self.assertIn("company_b.pdf", aggregation)

        # Check company_a has SDG 13
        self.assertIn("SDG 13", aggregation["company_a.pdf"])

        # Check SDG 13 data for company_a
        sdg13_data = aggregation["company_a.pdf"]["SDG 13"]
        self.assertEqual(sdg13_data["count"], 2)  # 2 chunks mention it
        self.assertAlmostEqual(sdg13_data["avg_score"], (0.85 + 0.75) / 2)
        self.assertEqual(len(sdg13_data["chunks"]), 2)

        # Check company_b has SDG 16
        self.assertIn("SDG 16", aggregation["company_b.pdf"])
        self.assertEqual(aggregation["company_b.pdf"]["SDG 16"]["count"], 1)

    def test_get_coverage_summary(self):
        """Test overall coverage summary"""
        summary = self.classifier.get_coverage_summary(self.classified_chunks)

        # Check structure
        self.assertIn("total_chunks", summary)
        self.assertIn("chunks_with_sdgs", summary)
        self.assertIn("coverage_rate", summary)
        self.assertIn("sdg_distribution", summary)
        self.assertIn("top_sdgs", summary)

        # Check values
        self.assertEqual(summary["total_chunks"], 3)
        self.assertEqual(summary["chunks_with_sdgs"], 3)
        self.assertEqual(summary["coverage_rate"], 1.0)

        # SDG 13 should appear most (2 times)
        self.assertEqual(summary["sdg_distribution"]["SDG 13"], 2)

        # Top SDGs should be sorted
        self.assertEqual(summary["top_sdgs"][0][0], "SDG 13")
        self.assertEqual(summary["top_sdgs"][0][1], 2)

    def test_format_coverage_report(self):
        """Test report formatting"""
        report = self.classifier.format_coverage_report(self.classified_chunks)

        # Should be a string
        self.assertIsInstance(report, str)

        # Should contain key information
        self.assertIn("SDG COVERAGE REPORT", report)
        self.assertIn("Total chunks analyzed: 3", report)
        self.assertIn("SDG 13", report)

        # Should include per-document breakdown
        self.assertIn("company_a.pdf", report)
        self.assertIn("company_b.pdf", report)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def setUp(self):
        """Set up classifier"""
        self.classifier = SDGClassifier()

    def test_empty_text(self):
        """Test classifying empty text"""
        results = self.classifier.classify_text("")

        # Should return list (might be empty or have low-confidence matches)
        self.assertIsInstance(results, list)

    def test_unrelated_text(self):
        """Test text unrelated to SDGs"""
        text = "The quick brown fox jumps over the lazy dog."

        results = self.classifier.classify_text(text)

        # Might have no matches above threshold
        # Just verify it doesn't crash
        self.assertIsInstance(results, list)

    def test_chunks_without_sdg_matches(self):
        """Test aggregation with chunks that have no SDG matches"""
        chunks = [
            {
                "chunk_id": "test_p1_c0",
                "source": "test.pdf",
                "text": "Random text",
                "sdg_matches": []  # No matches
            }
        ]

        summary = self.classifier.get_coverage_summary(chunks)

        self.assertEqual(summary["total_chunks"], 1)
        self.assertEqual(summary["chunks_with_sdgs"], 0)
        self.assertEqual(summary["coverage_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
