"""
PDF Parser Module
Extracts text from ESG reports and chunks into semantic segments

Uses PyMuPDF (fitz) for efficient text extraction with layout preservation.
Chunks text into ~300-word segments with 50-word overlap to maintain context.
"""

import fitz  # PyMuPDF
from typing import List, Dict
import re
import os


class PDFParser:
    """
    Extracts and chunks text from PDF documents.

    Attributes:
        chunk_size (int): Target number of words per chunk (default: 300)
        overlap (int): Number of words to overlap between chunks (default: 50)
    """

    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        """
        Initialize PDF parser with chunking parameters.

        Args:
            chunk_size: Target words per chunk (default: 300)
            overlap: Overlapping words between chunks (default: 50)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Extract text from PDF with page-level metadata.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of dicts with structure:
            [
                {
                    "page": 1,
                    "text": "Full text from page 1...",
                    "source": "filename.pdf"
                },
                ...
            ]

        Raises:
            FileNotFoundError: If PDF doesn't exist
            fitz.FileDataError: If PDF is corrupted
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        filename = os.path.basename(pdf_path)
        pages_data = []

        try:
            # Open PDF with PyMuPDF
            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")  # Extract plain text

                # Clean extracted text
                text = self._clean_text(text)

                if text.strip():  # Only include pages with actual text
                    pages_data.append({
                        "page": page_num + 1,  # 1-indexed for readability
                        "text": text,
                        "source": filename
                    })

            doc.close()

        except fitz.FileDataError as e:
            raise fitz.FileDataError(f"Corrupted or invalid PDF: {pdf_path}") from e

        return pages_data

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by normalizing whitespace and removing artifacts.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove common PDF artifacts
        text = re.sub(r'\x00', '', text)  # Null bytes
        text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)  # Control chars

        # Normalize quotes and dashes
        text = text.replace('\u2018', "'").replace('\u2019', "'")  # Smart quotes
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2013', '-').replace('\u2014', '-')  # Em/en dashes

        return text.strip()

    def chunk_text(self, pages_data: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Chunk extracted pages into overlapping segments.

        Args:
            pages_data: List of page dictionaries from extract_text_from_pdf()

        Returns:
            List of chunk dictionaries:
            [
                {
                    "chunk_id": "report_p1_c0",
                    "source": "report.pdf",
                    "page": 1,
                    "text": "Chunk text...",
                    "word_count": 305
                },
                ...
            ]
        """
        chunks = []

        for page_data in pages_data:
            text = page_data["text"]
            page_num = page_data["page"]
            source = page_data["source"]

            # Split into words
            words = text.split()

            # Create chunks with overlap
            chunk_idx = 0
            start_idx = 0

            while start_idx < len(words):
                # Extract chunk of target size
                end_idx = start_idx + self.chunk_size
                chunk_words = words[start_idx:end_idx]
                chunk_text = ' '.join(chunk_words)

                # Only add non-empty chunks with meaningful content (>10 words)
                if len(chunk_words) > 10:
                    # Generate unique chunk ID
                    source_prefix = os.path.splitext(source)[0]  # Remove .pdf extension
                    chunk_id = f"{source_prefix}_p{page_num}_c{chunk_idx}"

                    chunks.append({
                        "chunk_id": chunk_id,
                        "source": source,
                        "page": page_num,
                        "text": chunk_text,
                        "word_count": len(chunk_words)
                    })

                    chunk_idx += 1

                # Move to next chunk with overlap
                start_idx += (self.chunk_size - self.overlap)

                # Break if we've processed all words
                if end_idx >= len(words):
                    break

        return chunks

    def parse_pdf(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Complete pipeline: extract + chunk PDF in one call.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of text chunks with metadata

        Example:
            parser = PDFParser(chunk_size=300, overlap=50)
            chunks = parser.parse_pdf("esg_report.pdf")
            print(f"Extracted {len(chunks)} chunks")
        """
        pages_data = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_text(pages_data)
        return chunks


def parse_multiple_pdfs(pdf_paths: List[str], chunk_size: int = 300,
                        overlap: int = 50) -> List[Dict[str, any]]:
    """
    Convenience function to parse multiple PDFs at once.

    Args:
        pdf_paths: List of paths to PDF files
        chunk_size: Target words per chunk
        overlap: Overlapping words between chunks

    Returns:
        Combined list of chunks from all PDFs

    Example:
        chunks = parse_multiple_pdfs([
            "data/company_a_2023.pdf",
            "data/company_b_2023.pdf"
        ])
    """
    parser = PDFParser(chunk_size=chunk_size, overlap=overlap)
    all_chunks = []

    for pdf_path in pdf_paths:
        try:
            chunks = parser.parse_pdf(pdf_path)
            all_chunks.extend(chunks)
            print(f"✓ Parsed {pdf_path}: {len(chunks)} chunks")
        except Exception as e:
            print(f"✗ Failed to parse {pdf_path}: {e}")

    return all_chunks


# Demo/Testing functionality
if __name__ == "__main__":
    """
    Quick test with a sample PDF (if available)
    """
    import sys

    if len(sys.argv) > 1:
        # Test with provided PDF
        pdf_path = sys.argv[1]
        parser = PDFParser()

        print(f"Parsing: {pdf_path}")
        print("-" * 60)

        try:
            chunks = parser.parse_pdf(pdf_path)

            print(f"\nExtracted {len(chunks)} chunks")
            print(f"Chunk size: ~{parser.chunk_size} words, overlap: {parser.overlap} words")
            print("\n--- Sample Chunk ---")
            if chunks:
                sample = chunks[0]
                print(f"ID: {sample['chunk_id']}")
                print(f"Source: {sample['source']}")
                print(f"Page: {sample['page']}")
                print(f"Words: {sample['word_count']}")
                print(f"Text preview: {sample['text'][:200]}...")

        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python pdf_parser.py <path_to_pdf>")
        print("\nExample:")
        print("  python src/pdf_parser.py data/sample_reports/esg_report.pdf")
