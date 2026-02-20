"""
Unit tests for Embeddings module
"""

import unittest
import sys
import os
import tempfile
import shutil
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.embeddings import EmbeddingManager


class TestEmbeddingManager(unittest.TestCase):
    """Test cases for EmbeddingManager class"""

    def setUp(self):
        """Initialize manager for each test"""
        self.manager = EmbeddingManager()

        # Sample chunks for testing
        self.sample_chunks = [
            {
                "chunk_id": "test_p1_c0",
                "source": "test.pdf",
                "page": 1,
                "text": "Climate change is a critical environmental challenge requiring immediate action.",
                "word_count": 11
            },
            {
                "chunk_id": "test_p1_c1",
                "source": "test.pdf",
                "page": 1,
                "text": "Renewable energy sources like solar and wind power are essential for sustainability.",
                "word_count": 13
            },
            {
                "chunk_id": "test_p2_c0",
                "source": "test.pdf",
                "page": 2,
                "text": "Corporate governance ensures transparent decision-making and accountability to stakeholders.",
                "word_count": 10
            },
            {
                "chunk_id": "test_p2_c1",
                "source": "test.pdf",
                "page": 2,
                "text": "Social responsibility includes fair labor practices and community engagement programs.",
                "word_count": 11
            }
        ]

    def test_initialization(self):
        """Test manager initializes with correct defaults"""
        self.assertEqual(self.manager.model_name, "all-MiniLM-L6-v2")
        self.assertEqual(self.manager.dimension, 384)
        self.assertIsNone(self.manager.model)
        self.assertIsNone(self.manager.index)
        self.assertEqual(len(self.manager.chunks), 0)

    def test_model_loading(self):
        """Test model loads successfully"""
        self.manager.load_model()
        self.assertIsNotNone(self.manager.model)

        # Model should only load once (cached)
        model_ref = self.manager.model
        self.manager.load_model()
        self.assertIs(self.manager.model, model_ref)

    def test_embed_chunks(self):
        """Test embedding generation produces correct shape"""
        embeddings = self.manager.embed_chunks(
            self.sample_chunks,
            show_progress=False
        )

        # Check shape: (num_chunks, embedding_dim)
        self.assertEqual(embeddings.shape, (len(self.sample_chunks), 384))

        # Check embeddings are normalized (for cosine similarity)
        # Note: embeddings from model aren't normalized yet
        self.assertTrue(np.all(np.isfinite(embeddings)))

    def test_build_index(self):
        """Test FAISS index builds successfully"""
        self.manager.build_index(self.sample_chunks)

        # Index should be created
        self.assertIsNotNone(self.manager.index)

        # Should contain correct number of vectors
        self.assertEqual(self.manager.index.ntotal, len(self.sample_chunks))

        # Chunks should be stored
        self.assertEqual(len(self.manager.chunks), len(self.sample_chunks))

    def test_search_functionality(self):
        """Test semantic search returns relevant results"""
        self.manager.build_index(self.sample_chunks)

        # Search for climate-related content
        results = self.manager.search("environmental sustainability", top_k=2)

        # Should return requested number of results
        self.assertEqual(len(results), 2)

        # Each result should be (chunk, score) tuple
        for chunk, score in results:
            self.assertIsInstance(chunk, dict)
            self.assertIsInstance(score, float)
            self.assertIn("chunk_id", chunk)
            self.assertIn("text", chunk)

        # Scores should be positive (L2 distance)
        scores = [score for _, score in results]
        self.assertTrue(all(s >= 0 for s in scores))

        # Results should be sorted by score (ascending = most similar first)
        self.assertEqual(scores, sorted(scores))

    def test_search_relevance(self):
        """Test search returns semantically relevant results"""
        self.manager.build_index(self.sample_chunks)

        # Query about renewable energy
        results = self.manager.search("solar and wind energy", top_k=1)
        top_chunk, _ = results[0]

        # Should return the renewable energy chunk
        self.assertIn("renewable", top_chunk["text"].lower())

    def test_search_without_index(self):
        """Test search raises error when no index built"""
        with self.assertRaises(ValueError) as context:
            self.manager.search("test query")

        self.assertIn("No index built", str(context.exception))

    def test_add_chunks(self):
        """Test incremental addition of chunks"""
        # Build initial index
        initial_chunks = self.sample_chunks[:2]
        self.manager.build_index(initial_chunks)

        initial_count = self.manager.index.ntotal

        # Add more chunks
        new_chunks = self.sample_chunks[2:]
        self.manager.add_chunks(new_chunks)

        # Index should have more vectors
        self.assertEqual(
            self.manager.index.ntotal,
            initial_count + len(new_chunks)
        )

        # Should be able to search new content
        results = self.manager.search("corporate governance", top_k=1)
        top_chunk, _ = results[0]
        self.assertIn("governance", top_chunk["text"].lower())

    def test_get_stats(self):
        """Test statistics retrieval"""
        # Before building index
        stats = self.manager.get_stats()
        self.assertIn("status", stats)

        # After building index
        self.manager.build_index(self.sample_chunks)
        stats = self.manager.get_stats()

        self.assertEqual(stats["total_chunks"], len(self.sample_chunks))
        self.assertEqual(stats["embedding_dimension"], 384)
        self.assertEqual(stats["unique_sources"], 1)  # All from test.pdf
        self.assertIn("test.pdf", stats["sources"])
        self.assertEqual(stats["model"], "all-MiniLM-L6-v2")


class TestIndexPersistence(unittest.TestCase):
    """Test saving and loading index to/from disk"""

    def setUp(self):
        """Create temporary directory for test files"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = EmbeddingManager()

        self.sample_chunks = [
            {
                "chunk_id": "test_p1_c0",
                "source": "test.pdf",
                "page": 1,
                "text": "Test text for persistence testing.",
                "word_count": 5
            }
        ]

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)

    def test_save_and_load_index(self):
        """Test index can be saved and loaded correctly"""
        # Build and save index
        self.manager.build_index(self.sample_chunks)
        original_count = self.manager.index.ntotal

        self.manager.save_index(self.temp_dir, index_name="test_index")

        # Verify files were created
        self.assertTrue(
            os.path.exists(os.path.join(self.temp_dir, "test_index.faiss"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.temp_dir, "test_index_chunks.pkl"))
        )

        # Load in new manager
        new_manager = EmbeddingManager()
        new_manager.load_index(self.temp_dir, index_name="test_index")

        # Verify loaded correctly
        self.assertEqual(new_manager.index.ntotal, original_count)
        self.assertEqual(len(new_manager.chunks), len(self.sample_chunks))
        self.assertEqual(
            new_manager.chunks[0]["chunk_id"],
            self.sample_chunks[0]["chunk_id"]
        )

    def test_load_nonexistent_index(self):
        """Test loading non-existent index raises error"""
        with self.assertRaises(FileNotFoundError):
            self.manager.load_index(self.temp_dir, index_name="nonexistent")

    def test_save_without_index(self):
        """Test saving without building index raises error"""
        with self.assertRaises(ValueError):
            self.manager.save_index(self.temp_dir)


class TestEmbeddingDimensions(unittest.TestCase):
    """Test embedding vector properties"""

    def test_embedding_consistency(self):
        """Test same text produces same embedding"""
        manager = EmbeddingManager()

        chunks = [
            {"text": "Consistent text for testing", "chunk_id": "test1"},
            {"text": "Consistent text for testing", "chunk_id": "test2"}
        ]

        embeddings = manager.embed_chunks(chunks, show_progress=False)

        # Same text should produce same embedding
        np.testing.assert_array_almost_equal(embeddings[0], embeddings[1])

    def test_different_texts_different_embeddings(self):
        """Test different texts produce different embeddings"""
        manager = EmbeddingManager()

        chunks = [
            {"text": "Climate change is serious", "chunk_id": "test1"},
            {"text": "Corporate governance matters", "chunk_id": "test2"}
        ]

        embeddings = manager.embed_chunks(chunks, show_progress=False)

        # Different texts should have different embeddings
        self.assertFalse(np.allclose(embeddings[0], embeddings[1]))


if __name__ == "__main__":
    unittest.main()
