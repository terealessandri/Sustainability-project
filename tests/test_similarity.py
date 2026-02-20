"""
Unit tests for Similarity module
"""

import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.similarity import SimilarityAnalyzer
from src.embeddings import EmbeddingManager


class TestSimilarityAnalyzer(unittest.TestCase):
    """Test cases for SimilarityAnalyzer class"""

    def setUp(self):
        """Initialize analyzer with embedding manager"""
        self.manager = EmbeddingManager()

        # Build small index for testing
        self.test_chunks = [
            {"text": "We aim to reduce carbon emissions by 50% by 2030."},
            {"text": "Our goal is to cut carbon emissions 50% by 2030."},  # Very similar
            {"text": "Gender equality is central to our workforce policies."}  # Different
        ]

        self.manager.build_index(self.test_chunks)
        self.analyzer = SimilarityAnalyzer(self.manager)

    def test_initialization(self):
        """Test analyzer initializes correctly"""
        self.assertIsNotNone(self.analyzer.embedding_manager)
        self.assertIn("identical", self.analyzer.similarity_thresholds)
        self.assertIn("very_similar", self.analyzer.similarity_thresholds)

    def test_compare_texts_similar(self):
        """Test comparing very similar texts"""
        text1 = "We aim to reduce carbon emissions by 50%."
        text2 = "Our goal is to cut carbon emissions by 50%."

        similarity = self.analyzer.compare_texts(text1, text2)

        # Should be high similarity (synonyms: aim/goal, reduce/cut)
        self.assertGreater(similarity, 0.70)
        self.assertLessEqual(similarity, 1.0)

    def test_compare_texts_different(self):
        """Test comparing different texts"""
        text1 = "We aim to reduce carbon emissions."
        text2 = "Gender equality is important to us."

        similarity = self.analyzer.compare_texts(text1, text2)

        # Should be low similarity (different topics)
        self.assertLess(similarity, 0.50)

    def test_compare_texts_identical(self):
        """Test comparing identical texts"""
        text = "We aim to reduce carbon emissions by 50% by 2030."

        similarity = self.analyzer.compare_texts(text, text)

        # Should be 1.0 or very close (floating point)
        self.assertGreater(similarity, 0.99)

    def test_interpret_similarity(self):
        """Test similarity score interpretation"""
        # Identical
        self.assertEqual(
            self.analyzer.interpret_similarity(0.97),
            "identical"
        )

        # Very similar
        self.assertEqual(
            self.analyzer.interpret_similarity(0.87),
            "very_similar"
        )

        # Similar
        self.assertEqual(
            self.analyzer.interpret_similarity(0.75),
            "similar"
        )

        # Somewhat similar
        self.assertEqual(
            self.analyzer.interpret_similarity(0.55),
            "somewhat_similar"
        )

        # Different
        self.assertEqual(
            self.analyzer.interpret_similarity(0.30),
            "different"
        )

    def test_compare_sources_on_sdg(self):
        """Test comparing two sources on a specific SDG"""
        # Create test data with SDG matches
        classified_chunks = [
            {
                "chunk_id": "a_p1_c0",
                "source": "company_a.pdf",
                "text": "We reduced emissions by 30%.",
                "sdg_matches": [{"sdg_id": "SDG 13", "score": 0.9}]
            },
            {
                "chunk_id": "b_p1_c0",
                "source": "company_b.pdf",
                "text": "We cut emissions by 30%.",
                "sdg_matches": [{"sdg_id": "SDG 13", "score": 0.9}]
            }
        ]

        comparison = self.analyzer.compare_sources_on_sdg(
            classified_chunks,
            "company_a.pdf",
            "company_b.pdf",
            "SDG 13"
        )

        # Check structure
        self.assertEqual(comparison["sdg_id"], "SDG 13")
        self.assertEqual(comparison["source_a"], "company_a.pdf")
        self.assertEqual(comparison["source_b"], "company_b.pdf")
        self.assertIsNotNone(comparison["average_similarity"])
        self.assertIn(comparison["interpretation"], [
            "identical", "very_similar", "similar",
            "somewhat_similar", "different"
        ])

    def test_compare_sources_insufficient_data(self):
        """Test comparison with no matching chunks"""
        classified_chunks = [
            {
                "chunk_id": "a_p1_c0",
                "source": "company_a.pdf",
                "text": "Test",
                "sdg_matches": [{"sdg_id": "SDG 13", "score": 0.9}]
            }
        ]

        comparison = self.analyzer.compare_sources_on_sdg(
            classified_chunks,
            "company_a.pdf",
            "company_b.pdf",
            "SDG 13"
        )

        self.assertEqual(comparison["interpretation"], "insufficient_data")
        self.assertIsNone(comparison["average_similarity"])

    def test_detect_copy_paste(self):
        """Test copy-paste detection"""
        classified_chunks = [
            {
                "chunk_id": "a_p1_c0",
                "source": "company_a.pdf",
                "text": "We target carbon neutrality by 2050 through renewable energy.",
                "sdg_matches": [{"sdg_id": "SDG 13", "score": 0.9}]
            },
            {
                "chunk_id": "b_p1_c0",
                "source": "company_b.pdf",
                "text": "We target carbon neutrality by 2050 through renewable energy.",  # Identical
                "sdg_matches": [{"sdg_id": "SDG 13", "score": 0.9}]
            },
            {
                "chunk_id": "c_p1_c0",
                "source": "company_c.pdf",
                "text": "Gender equality is our priority.",
                "sdg_matches": [{"sdg_id": "SDG 5", "score": 0.9}]
            }
        ]

        copy_paste = self.analyzer.detect_copy_paste(
            classified_chunks,
            threshold=0.90
        )

        # Should detect the identical texts
        self.assertGreater(len(copy_paste), 0)

        # Check structure
        if copy_paste:
            instance = copy_paste[0]
            self.assertIn("source_a", instance)
            self.assertIn("source_b", instance)
            self.assertIn("similarity", instance)
            self.assertGreaterEqual(instance["similarity"], 0.90)

    def test_detect_copy_paste_no_matches(self):
        """Test copy-paste detection with no similar texts"""
        classified_chunks = [
            {
                "chunk_id": "a_p1_c0",
                "source": "company_a.pdf",
                "text": "We focus on carbon emissions reduction.",
                "sdg_matches": [{"sdg_id": "SDG 13", "score": 0.9}]
            },
            {
                "chunk_id": "b_p1_c0",
                "source": "company_b.pdf",
                "text": "Gender equality is our priority.",
                "sdg_matches": [{"sdg_id": "SDG 5", "score": 0.9}]
            }
        ]

        copy_paste = self.analyzer.detect_copy_paste(
            classified_chunks,
            threshold=0.95
        )

        # Should not detect copy-paste (different topics)
        self.assertEqual(len(copy_paste), 0)

    def test_calculate_uniqueness_score(self):
        """Test uniqueness score calculation"""
        classified_chunks = [
            {
                "chunk_id": "a_p1_c0",
                "source": "company_a.pdf",
                "text": "We have a completely unique approach to sustainability with innovative practices.",
                "sdg_matches": []
            },
            {
                "chunk_id": "b_p1_c0",
                "source": "company_b.pdf",
                "text": "Standard sustainability practices are followed.",
                "sdg_matches": []
            },
            {
                "chunk_id": "c_p1_c0",
                "source": "company_c.pdf",
                "text": "Standard sustainability practices are implemented.",  # Similar to B
                "sdg_matches": []
            }
        ]

        # Company A should have high uniqueness (different from others)
        uniqueness_a = self.analyzer.calculate_uniqueness_score(
            classified_chunks,
            "company_a.pdf"
        )

        # Company B should have lower uniqueness (similar to C)
        uniqueness_b = self.analyzer.calculate_uniqueness_score(
            classified_chunks,
            "company_b.pdf"
        )

        # A should be more unique than B
        self.assertGreater(uniqueness_a, uniqueness_b)

        # Scores should be in valid range
        self.assertGreaterEqual(uniqueness_a, 0.0)
        self.assertLessEqual(uniqueness_a, 1.0)
        self.assertGreaterEqual(uniqueness_b, 0.0)
        self.assertLessEqual(uniqueness_b, 1.0)

    def test_calculate_uniqueness_single_source(self):
        """Test uniqueness with only one source"""
        classified_chunks = [
            {
                "chunk_id": "a_p1_c0",
                "source": "company_a.pdf",
                "text": "Test",
                "sdg_matches": []
            }
        ]

        uniqueness = self.analyzer.calculate_uniqueness_score(
            classified_chunks,
            "company_a.pdf"
        )

        # Should be 1.0 (only source, by definition unique)
        self.assertEqual(uniqueness, 1.0)

    def test_format_similarity_report(self):
        """Test report formatting"""
        classified_chunks = [
            {
                "chunk_id": "a_p1_c0",
                "source": "company_a.pdf",
                "text": "Climate action is important.",
                "sdg_matches": [{"sdg_id": "SDG 13", "score": 0.9}]
            },
            {
                "chunk_id": "b_p1_c0",
                "source": "company_b.pdf",
                "text": "Climate action is important.",
                "sdg_matches": [{"sdg_id": "SDG 13", "score": 0.9}]
            }
        ]

        report = self.analyzer.format_similarity_report(classified_chunks)

        # Should be a string
        self.assertIsInstance(report, str)

        # Should contain key sections
        self.assertIn("SIMILARITY ANALYSIS REPORT", report)
        self.assertIn("COPY-PASTE DETECTION", report)
        self.assertIn("UNIQUENESS SCORES", report)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def test_analyzer_without_embedding_manager(self):
        """Test that analyzer requires embedding manager"""
        analyzer = SimilarityAnalyzer(embedding_manager=None)

        with self.assertRaises(ValueError):
            analyzer.compare_texts("text1", "text2")

    def test_empty_classified_chunks(self):
        """Test handling of empty chunk list"""
        manager = EmbeddingManager()
        manager.build_index([{"text": "test"}])
        analyzer = SimilarityAnalyzer(manager)

        copy_paste = analyzer.detect_copy_paste([], threshold=0.90)
        self.assertEqual(len(copy_paste), 0)

    def test_uniqueness_nonexistent_source(self):
        """Test uniqueness for non-existent source"""
        manager = EmbeddingManager()
        manager.build_index([{"text": "test"}])
        analyzer = SimilarityAnalyzer(manager)

        classified_chunks = [
            {
                "chunk_id": "a_p1_c0",
                "source": "company_a.pdf",
                "text": "Test",
                "sdg_matches": []
            }
        ]

        uniqueness = analyzer.calculate_uniqueness_score(
            classified_chunks,
            "nonexistent.pdf"
        )

        # Should return 0.0 for non-existent source
        self.assertEqual(uniqueness, 0.0)


if __name__ == "__main__":
    unittest.main()
