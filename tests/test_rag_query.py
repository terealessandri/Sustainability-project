"""
Unit tests for RAG Query module
"""

import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rag_query import RAGQueryEngine, format_results
from src.embeddings import EmbeddingManager


class TestRAGQueryEngine(unittest.TestCase):
    """Test cases for RAGQueryEngine class"""

    def setUp(self):
        """Initialize engine with sample data"""
        # Create sample chunks
        self.sample_chunks = [
            {
                "chunk_id": "report_a_p1_c0",
                "source": "company_a.pdf",
                "page": 1,
                "text": "We aim to reduce carbon emissions by 50% by 2030 through renewable energy.",
                "word_count": 14
            },
            {
                "chunk_id": "report_a_p2_c0",
                "source": "company_a.pdf",
                "page": 2,
                "text": "Our governance framework ensures transparency and accountability to stakeholders.",
                "word_count": 10
            },
            {
                "chunk_id": "report_b_p1_c0",
                "source": "company_b.pdf",
                "page": 1,
                "text": "Climate action is our priority. We target carbon neutrality by 2035.",
                "word_count": 13
            },
            {
                "chunk_id": "report_b_p3_c0",
                "source": "company_b.pdf",
                "page": 3,
                "text": "Social responsibility includes fair wages and community programs.",
                "word_count": 9
            },
            {
                "chunk_id": "report_c_p1_c0",
                "source": "company_c.pdf",
                "page": 1,
                "text": "We invest in solar and wind energy to power our operations sustainably.",
                "word_count": 13
            }
        ]

        # Build embedding manager and index
        self.manager = EmbeddingManager()
        self.manager.build_index(self.sample_chunks)

        # Create RAG engine
        self.engine = RAGQueryEngine(self.manager, default_top_k=3)

    def test_initialization(self):
        """Test engine initializes correctly"""
        self.assertIsNotNone(self.engine.embedding_manager)
        self.assertEqual(self.engine.default_top_k, 3)
        self.assertIsNone(self.engine.relevance_threshold)

    def test_basic_query(self):
        """Test basic query returns results"""
        results = self.engine.query("carbon emissions reduction")

        # Should return results
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 3)  # default_top_k

        # Check result structure
        for result in results:
            self.assertIn("chunk_id", result)
            self.assertIn("source", result)
            self.assertIn("page", result)
            self.assertIn("text", result)

    def test_query_with_custom_top_k(self):
        """Test query respects custom top_k parameter"""
        results = self.engine.query("sustainability", top_k=2)
        self.assertEqual(len(results), 2)

        results = self.engine.query("sustainability", top_k=5)
        self.assertEqual(len(results), 5)

    def test_query_relevance(self):
        """Test query returns semantically relevant results"""
        # Query about carbon/climate
        results = self.engine.query("carbon emissions targets", top_k=2)

        # Top results should be about carbon/climate
        top_texts = [r["text"].lower() for r in results]
        self.assertTrue(
            any("carbon" in text or "climate" in text for text in top_texts)
        )

    def test_query_with_scores(self):
        """Test including similarity scores in results"""
        results = self.engine.query(
            "renewable energy",
            top_k=3,
            include_scores=True
        )

        # All results should have scores
        for result in results:
            self.assertIn("score", result)
            self.assertIsInstance(result["score"], float)
            self.assertGreaterEqual(result["score"], 0)

    def test_source_filtering(self):
        """Test filtering results by source"""
        # Query with source filter
        results = self.engine.query(
            "carbon emissions",
            top_k=5,
            sources=["company_a.pdf"]
        )

        # All results should be from company_a.pdf
        for result in results:
            self.assertEqual(result["source"], "company_a.pdf")

        # Test multiple source filter
        results = self.engine.query(
            "sustainability",
            top_k=5,
            sources=["company_b.pdf", "company_c.pdf"]
        )

        sources_found = set(r["source"] for r in results)
        self.assertTrue(sources_found.issubset({"company_b.pdf", "company_c.pdf"}))

    def test_query_with_context(self):
        """Test query_with_context returns structured output"""
        result = self.engine.query_with_context(
            "carbon reduction goals",
            top_k=3
        )

        # Check structure
        self.assertIn("question", result)
        self.assertIn("chunks", result)
        self.assertIn("context", result)
        self.assertIn("sources", result)
        self.assertIn("num_results", result)

        # Question should match
        self.assertEqual(result["question"], "carbon reduction goals")

        # Chunks should be list
        self.assertIsInstance(result["chunks"], list)

        # Context should be non-empty string
        self.assertIsInstance(result["context"], str)
        self.assertGreater(len(result["context"]), 0)

        # Sources should be list of unique sources
        self.assertIsInstance(result["sources"], list)

    def test_compare_sources(self):
        """Test comparing two sources on same question"""
        comparison = self.engine.compare_sources(
            "climate commitments",
            "company_a.pdf",
            "company_b.pdf",
            top_k_per_source=2
        )

        # Check structure
        self.assertIn("question", comparison)
        self.assertIn("source_a", comparison)
        self.assertIn("source_b", comparison)

        # Check source_a data
        self.assertEqual(comparison["source_a"]["name"], "company_a.pdf")
        self.assertIn("results", comparison["source_a"])
        self.assertLessEqual(len(comparison["source_a"]["results"]), 2)

        # Check source_b data
        self.assertEqual(comparison["source_b"]["name"], "company_b.pdf")
        self.assertIn("results", comparison["source_b"])
        self.assertLessEqual(len(comparison["source_b"]["results"]), 2)

        # All source_a results should be from company_a.pdf
        for result in comparison["source_a"]["results"]:
            self.assertEqual(result["source"], "company_a.pdf")

        # All source_b results should be from company_b.pdf
        for result in comparison["source_b"]["results"]:
            self.assertEqual(result["source"], "company_b.pdf")

    def test_source_coverage(self):
        """Test analyzing source coverage for a query"""
        coverage = self.engine.get_source_coverage(
            "sustainability initiatives",
            top_k=5
        )

        # Should return dict mapping sources to counts
        self.assertIsInstance(coverage, dict)

        # All keys should be source filenames
        for source in coverage.keys():
            self.assertTrue(source.endswith(".pdf"))

        # All values should be positive integers
        for count in coverage.values():
            self.assertIsInstance(count, int)
            self.assertGreater(count, 0)

        # Total counts should not exceed top_k
        total_count = sum(coverage.values())
        self.assertLessEqual(total_count, 5)

    def test_batch_query(self):
        """Test executing multiple queries at once"""
        questions = [
            "carbon emissions",
            "renewable energy",
            "governance practices"
        ]

        results = self.engine.batch_query(questions, top_k=2)

        # Should return one result per question
        self.assertEqual(len(results), len(questions))

        # Each result should have query_with_context structure
        for i, result in enumerate(results):
            self.assertEqual(result["question"], questions[i])
            self.assertIn("chunks", result)
            self.assertIn("context", result)

    def test_relevance_threshold_filtering(self):
        """Test relevance threshold filters low-quality results"""
        # Create engine with strict threshold
        strict_engine = RAGQueryEngine(
            self.manager,
            default_top_k=5,
            relevance_threshold=0.5  # Only very similar results
        )

        results = strict_engine.query("some random irrelevant text xyz123")

        # With strict threshold, might get fewer results
        # (or none if nothing is relevant enough)
        self.assertLessEqual(len(results), 5)

        # All results should be within threshold
        for result in results:
            if "score" in result:
                self.assertLessEqual(result["score"], 0.5)


class TestResultFormatting(unittest.TestCase):
    """Test result formatting utilities"""

    def test_format_results_basic(self):
        """Test basic result formatting"""
        results = [
            {
                "chunk_id": "test_p1_c0",
                "source": "test.pdf",
                "page": 1,
                "text": "This is test text for formatting.",
                "score": 0.123
            }
        ]

        output = format_results(results)

        self.assertIn("Found 1 relevant chunks", output)
        self.assertIn("test.pdf", output)
        self.assertIn("page 1", output)
        self.assertIn("This is test text", output)

    def test_format_results_with_scores(self):
        """Test formatting with scores included"""
        results = [
            {
                "chunk_id": "test_p1_c0",
                "source": "test.pdf",
                "page": 1,
                "text": "Test text",
                "score": 0.456
            }
        ]

        output = format_results(results, show_scores=True)

        self.assertIn("0.456", output)
        self.assertIn("Relevance", output)

    def test_format_results_truncation(self):
        """Test text truncation in formatting"""
        long_text = "A" * 300  # 300 characters

        results = [
            {
                "chunk_id": "test_p1_c0",
                "source": "test.pdf",
                "page": 1,
                "text": long_text
            }
        ]

        output = format_results(results, max_text_length=100)

        # Should be truncated with ellipsis
        self.assertIn("...", output)
        # Should not contain full text
        self.assertLess(len(output), len(long_text) + 100)

    def test_format_results_empty(self):
        """Test formatting empty results"""
        output = format_results([])
        self.assertEqual(output, "No results found.")

    def test_format_results_multiple(self):
        """Test formatting multiple results"""
        results = [
            {
                "chunk_id": f"test_p{i}_c0",
                "source": f"test{i}.pdf",
                "page": i,
                "text": f"Text {i}"
            }
            for i in range(1, 4)
        ]

        output = format_results(results)

        self.assertIn("Found 3 relevant chunks", output)
        self.assertIn("[1]", output)
        self.assertIn("[2]", output)
        self.assertIn("[3]", output)


class TestQueryEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def setUp(self):
        """Set up minimal engine"""
        chunks = [
            {
                "chunk_id": "test_p1_c0",
                "source": "test.pdf",
                "page": 1,
                "text": "Test document content.",
                "word_count": 3
            }
        ]
        manager = EmbeddingManager()
        manager.build_index(chunks)
        self.engine = RAGQueryEngine(manager)

    def test_empty_query(self):
        """Test querying with empty string"""
        # Should not crash, might return arbitrary results
        results = self.engine.query("", top_k=1)
        self.assertIsInstance(results, list)

    def test_query_nonexistent_source(self):
        """Test filtering by non-existent source"""
        results = self.engine.query(
            "test",
            sources=["nonexistent.pdf"]
        )

        # Should return empty list
        self.assertEqual(len(results), 0)

    def test_top_k_exceeds_index_size(self):
        """Test requesting more results than exist"""
        # Only 1 chunk in index, request 10
        results = self.engine.query("test", top_k=10)

        # FAISS may return k results (with duplicates) or fewer
        # Just ensure we get at least the chunks that exist
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 10)


if __name__ == "__main__":
    unittest.main()
