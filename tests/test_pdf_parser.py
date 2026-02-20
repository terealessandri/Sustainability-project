"""
Unit tests for PDF Parser module
"""

import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pdf_parser import PDFParser


class TestPDFParser(unittest.TestCase):
    """Test cases for PDFParser class"""

    def setUp(self):
        """Initialize parser for each test"""
        self.parser = PDFParser(chunk_size=300, overlap=50)

    def test_initialization(self):
        """Test parser initializes with correct parameters"""
        self.assertEqual(self.parser.chunk_size, 300)
        self.assertEqual(self.parser.overlap, 50)

        # Test custom parameters
        custom_parser = PDFParser(chunk_size=500, overlap=100)
        self.assertEqual(custom_parser.chunk_size, 500)
        self.assertEqual(custom_parser.overlap, 100)

    def test_clean_text(self):
        """Test text cleaning removes artifacts and normalizes whitespace"""
        # Test whitespace normalization
        dirty_text = "This   has    excessive     spaces"
        clean = self.parser._clean_text(dirty_text)
        self.assertEqual(clean, "This has excessive spaces")

        # Test smart quote normalization
        quotes = "\u2018single\u2019 \u201cdouble\u201d"
        clean = self.parser._clean_text(quotes)
        self.assertEqual(clean, "'single' \"double\"")

        # Test dash normalization
        dashes = "em\u2014dash en\u2013dash"
        clean = self.parser._clean_text(dashes)
        self.assertEqual(clean, "em-dash en-dash")

    def test_chunking_logic(self):
        """Test text chunking with overlap"""
        # Create mock page data
        # Generate text with exactly 1000 words
        words = [f"word{i}" for i in range(1000)]
        text = " ".join(words)

        pages_data = [{
            "page": 1,
            "text": text,
            "source": "test.pdf"
        }]

        chunks = self.parser.chunk_text(pages_data)

        # Verify chunks were created
        self.assertGreater(len(chunks), 0)

        # Check first chunk structure
        first_chunk = chunks[0]
        self.assertIn("chunk_id", first_chunk)
        self.assertIn("source", first_chunk)
        self.assertIn("page", first_chunk)
        self.assertIn("text", first_chunk)
        self.assertIn("word_count", first_chunk)

        # Verify chunk size is approximately correct (±5 words tolerance)
        for chunk in chunks[:-1]:  # All except last chunk
            self.assertGreaterEqual(chunk["word_count"], 295)
            self.assertLessEqual(chunk["word_count"], 305)

        # Verify chunk IDs are unique
        chunk_ids = [c["chunk_id"] for c in chunks]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))

    def test_overlap_preservation(self):
        """Test that chunks overlap correctly"""
        # Create text with identifiable words
        words = [f"word_{i:04d}" for i in range(400)]
        text = " ".join(words)

        pages_data = [{
            "page": 1,
            "text": text,
            "source": "test.pdf"
        }]

        parser = PDFParser(chunk_size=100, overlap=20)
        chunks = parser.chunk_text(pages_data)

        # There should be multiple chunks
        self.assertGreaterEqual(len(chunks), 2)

        # Check overlap between first two chunks
        chunk1_words = chunks[0]["text"].split()
        chunk2_words = chunks[1]["text"].split()

        # Last words of chunk1 should appear in first words of chunk2
        overlap_words = chunk1_words[-20:]  # Last 20 words of chunk1
        chunk2_start = chunk2_words[:20]  # First 20 words of chunk2

        # They should share some words due to overlap
        # (exact match depends on chunking boundaries)
        self.assertTrue(any(word in chunk2_start for word in overlap_words))

    def test_small_chunks_filtered(self):
        """Test that very small chunks (<10 words) are filtered out"""
        # Create text with only 8 words
        small_text = "one two three four five six seven eight"

        pages_data = [{
            "page": 1,
            "text": small_text,
            "source": "test.pdf"
        }]

        chunks = self.parser.chunk_text(pages_data)

        # Should not create chunk with <10 words
        self.assertEqual(len(chunks), 0)

    def test_multiple_pages(self):
        """Test chunking across multiple pages"""
        pages_data = [
            {"page": 1, "text": " ".join([f"page1_word{i}" for i in range(350)]), "source": "test.pdf"},
            {"page": 2, "text": " ".join([f"page2_word{i}" for i in range(350)]), "source": "test.pdf"}
        ]

        chunks = self.parser.chunk_text(pages_data)

        # Should have chunks from both pages
        page_numbers = set(c["page"] for c in chunks)
        self.assertIn(1, page_numbers)
        self.assertIn(2, page_numbers)

        # Verify chunk IDs distinguish pages
        page1_chunks = [c for c in chunks if c["page"] == 1]
        page2_chunks = [c for c in chunks if c["page"] == 2]

        self.assertGreater(len(page1_chunks), 0)
        self.assertGreater(len(page2_chunks), 0)

        # Check chunk ID format
        self.assertTrue(page1_chunks[0]["chunk_id"].endswith("_p1_c0"))
        self.assertTrue(page2_chunks[0]["chunk_id"].endswith("_p2_c0"))

    def test_empty_pages_handling(self):
        """Test that empty pages are handled gracefully"""
        pages_data = [
            {"page": 1, "text": " ".join([f"word{i}" for i in range(350)]), "source": "test.pdf"},
            {"page": 2, "text": "", "source": "test.pdf"},  # Empty page
            {"page": 3, "text": " ".join([f"word{i}" for i in range(350)]), "source": "test.pdf"}
        ]

        chunks = self.parser.chunk_text(pages_data)

        # Should only have chunks from pages 1 and 3
        page_numbers = set(c["page"] for c in chunks)
        self.assertIn(1, page_numbers)
        self.assertIn(3, page_numbers)
        self.assertNotIn(2, page_numbers)


class TestChunkMetadata(unittest.TestCase):
    """Test chunk metadata structure"""

    def test_chunk_id_format(self):
        """Test that chunk IDs follow expected format: source_pX_cY"""
        parser = PDFParser()
        pages_data = [{
            "page": 5,
            "text": " ".join([f"word{i}" for i in range(350)]),
            "source": "my_report.pdf"
        }]

        chunks = parser.chunk_text(pages_data)
        chunk_id = chunks[0]["chunk_id"]

        # Should be: my_report_p5_c0
        self.assertTrue(chunk_id.startswith("my_report_p5_c"))
        self.assertIn("_p5_", chunk_id)
        self.assertIn("_c", chunk_id)

    def test_source_preservation(self):
        """Test that source filename is preserved in chunks"""
        parser = PDFParser()
        pages_data = [{
            "page": 1,
            "text": " ".join([f"word{i}" for i in range(350)]),
            "source": "company_esg_2023.pdf"
        }]

        chunks = parser.chunk_text(pages_data)

        for chunk in chunks:
            self.assertEqual(chunk["source"], "company_esg_2023.pdf")


if __name__ == "__main__":
    unittest.main()
