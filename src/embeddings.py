"""
Embeddings Module
Generates sentence embeddings and manages FAISS vector store for semantic search

Uses sentence-transformers (all-MiniLM-L6-v2) for efficient 384-dimensional embeddings.
FAISS provides fast similarity search over embedded chunks.
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional
import pickle
import os


class EmbeddingManager:
    """
    Manages text embeddings and FAISS vector store for semantic search.

    Attributes:
        model_name (str): Name of sentence-transformers model
        model: Loaded SentenceTransformer model
        index: FAISS index for similarity search
        chunks: List of chunk metadata (corresponds to index vectors)
        dimension (int): Embedding vector dimension (384 for MiniLM)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding manager with specified model.

        Args:
            model_name: HuggingFace model ID (default: all-MiniLM-L6-v2)
                       384-dim, fast, good quality for semantic search
        """
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks = []
        self.dimension = 384  # MiniLM produces 384-dim vectors

    def load_model(self):
        """
        Load sentence-transformers model (lazy loading).
        Downloads model on first call (~80MB for MiniLM).
        """
        if self.model is None:
            print(f"Loading model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print(f"✓ Model loaded: {self.dimension}-dimensional embeddings")

    def embed_chunks(self, chunks: List[Dict[str, any]],
                     batch_size: int = 32,
                     show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of text chunks.

        Args:
            chunks: List of chunk dicts with 'text' field (from PDFParser)
            batch_size: Number of texts to encode at once (default: 32)
            show_progress: Show progress bar during encoding

        Returns:
            numpy array of shape (n_chunks, dimension)

        Example:
            chunks = parser.parse_pdf("report.pdf")
            embeddings = manager.embed_chunks(chunks)
        """
        self.load_model()

        # Extract text from chunks
        texts = [chunk["text"] for chunk in chunks]

        # Generate embeddings
        print(f"Embedding {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

        print(f"✓ Generated embeddings: {embeddings.shape}")
        return embeddings

    def build_index(self, chunks: List[Dict[str, any]],
                    embeddings: Optional[np.ndarray] = None) -> None:
        """
        Build FAISS index from chunks and their embeddings.

        Args:
            chunks: List of chunk dicts with metadata
            embeddings: Pre-computed embeddings (if None, will compute)

        Note:
            Uses IndexFlatL2 with L2 normalization for cosine similarity.
            Exact search (no approximation) — suitable for <100K vectors.
        """
        # Generate embeddings if not provided
        if embeddings is None:
            embeddings = self.embed_chunks(chunks)

        # Normalize vectors for cosine similarity
        # (L2 distance on normalized vectors = cosine similarity)
        faiss.normalize_L2(embeddings)

        # Create FAISS index (exact search)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)

        # Store chunk metadata
        self.chunks = chunks

        print(f"✓ FAISS index built: {self.index.ntotal} vectors")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search for most similar chunks to a query text.

        Args:
            query: Natural language search query
            top_k: Number of results to return (default: 5)

        Returns:
            List of (chunk_dict, similarity_score) tuples, sorted by relevance
            Lower score = more similar (L2 distance on normalized vectors)

        Example:
            results = manager.search("carbon emissions reduction", top_k=5)
            for chunk, score in results:
                print(f"Score: {score:.3f} | {chunk['chunk_id']}")
                print(chunk['text'][:200])
        """
        if self.index is None:
            raise ValueError("No index built. Call build_index() first.")

        self.load_model()

        # Embed query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        # Search FAISS index
        distances, indices = self.index.search(query_embedding, top_k)

        # Combine results with metadata
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):  # Valid index
                chunk = self.chunks[idx]
                results.append((chunk, float(dist)))

        return results

    def add_chunks(self, new_chunks: List[Dict[str, any]]) -> None:
        """
        Add new chunks to existing index (incremental update).

        Args:
            new_chunks: List of new chunk dicts to add

        Note:
            Useful for adding documents without rebuilding entire index.
        """
        if self.index is None:
            raise ValueError("No index exists. Use build_index() for first batch.")

        # Embed new chunks
        new_embeddings = self.embed_chunks(new_chunks, show_progress=False)
        faiss.normalize_L2(new_embeddings)

        # Add to index
        self.index.add(new_embeddings)
        self.chunks.extend(new_chunks)

        print(f"✓ Added {len(new_chunks)} chunks. Total: {self.index.ntotal}")

    def save_index(self, directory: str, index_name: str = "vector_store") -> None:
        """
        Save FAISS index and chunk metadata to disk.

        Args:
            directory: Directory to save files
            index_name: Base name for saved files (default: vector_store)

        Creates:
            {directory}/{index_name}.faiss — FAISS index
            {directory}/{index_name}_chunks.pkl — Chunk metadata
        """
        if self.index is None:
            raise ValueError("No index to save. Build index first.")

        os.makedirs(directory, exist_ok=True)

        # Save FAISS index
        index_path = os.path.join(directory, f"{index_name}.faiss")
        faiss.write_index(self.index, index_path)

        # Save chunk metadata
        chunks_path = os.path.join(directory, f"{index_name}_chunks.pkl")
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

        print(f"✓ Saved index to {directory}/")
        print(f"  - {index_name}.faiss ({self.index.ntotal} vectors)")
        print(f"  - {index_name}_chunks.pkl ({len(self.chunks)} chunks)")

    def load_index(self, directory: str, index_name: str = "vector_store") -> None:
        """
        Load FAISS index and chunk metadata from disk.

        Args:
            directory: Directory containing saved files
            index_name: Base name of saved files (default: vector_store)

        Raises:
            FileNotFoundError: If index files don't exist
        """
        index_path = os.path.join(directory, f"{index_name}.faiss")
        chunks_path = os.path.join(directory, f"{index_name}_chunks.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found: {index_path}")
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Chunks not found: {chunks_path}")

        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load chunk metadata
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        print(f"✓ Loaded index from {directory}/")
        print(f"  - {self.index.ntotal} vectors")
        print(f"  - {len(self.chunks)} chunks")

    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about the current index.

        Returns:
            Dictionary with index statistics
        """
        if self.index is None:
            return {"status": "No index built"}

        # Count unique sources
        sources = set(chunk["source"] for chunk in self.chunks)

        return {
            "total_chunks": self.index.ntotal,
            "embedding_dimension": self.dimension,
            "unique_sources": len(sources),
            "sources": list(sources),
            "model": self.model_name
        }


def build_index_from_pdfs(pdf_paths: List[str],
                          save_dir: Optional[str] = None) -> EmbeddingManager:
    """
    Convenience function: Parse PDFs and build searchable index in one call.

    Args:
        pdf_paths: List of paths to PDF files
        save_dir: Directory to save index (if None, index not saved)

    Returns:
        EmbeddingManager with built index

    Example:
        manager = build_index_from_pdfs([
            "data/company_a.pdf",
            "data/company_b.pdf"
        ], save_dir="data/processed")

        results = manager.search("climate action initiatives")
    """
    from src.pdf_parser import parse_multiple_pdfs

    # Parse all PDFs
    print("=" * 60)
    print("STEP 1: Parsing PDFs")
    print("=" * 60)
    chunks = parse_multiple_pdfs(pdf_paths)

    if not chunks:
        raise ValueError("No chunks extracted from PDFs")

    # Build index
    print("\n" + "=" * 60)
    print("STEP 2: Building Embedding Index")
    print("=" * 60)
    manager = EmbeddingManager()
    manager.build_index(chunks)

    # Save if requested
    if save_dir:
        print("\n" + "=" * 60)
        print("STEP 3: Saving Index")
        print("=" * 60)
        manager.save_index(save_dir)

    print("\n" + "=" * 60)
    print("✓ COMPLETE: Index ready for search")
    print("=" * 60)

    return manager


# Demo/Testing functionality
if __name__ == "__main__":
    """
    Quick test of embedding and search functionality
    """
    import sys

    if len(sys.argv) > 1:
        # Test with actual PDFs
        pdf_paths = sys.argv[1:]
        print(f"Building index from {len(pdf_paths)} PDF(s)...")

        try:
            manager = build_index_from_pdfs(pdf_paths)

            # Show stats
            print("\nIndex Statistics:")
            stats = manager.get_stats()
            for key, value in stats.items():
                print(f"  {key}: {value}")

            # Interactive search
            print("\n" + "=" * 60)
            print("SEARCH MODE (type 'quit' to exit)")
            print("=" * 60)

            while True:
                query = input("\nQuery: ").strip()
                if query.lower() in ["quit", "exit", "q"]:
                    break

                if query:
                    results = manager.search(query, top_k=3)
                    print(f"\nTop {len(results)} results:")
                    for i, (chunk, score) in enumerate(results, 1):
                        print(f"\n[{i}] Score: {score:.3f} | {chunk['chunk_id']}")
                        print(f"    Source: {chunk['source']} (page {chunk['page']})")
                        print(f"    Text: {chunk['text'][:150]}...")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("Usage: python src/embeddings.py <pdf1> [pdf2] [pdf3] ...")
        print("\nExample:")
        print("  python src/embeddings.py data/sample_reports/*.pdf")
        print("\nThis will:")
        print("  1. Parse all PDFs")
        print("  2. Generate embeddings")
        print("  3. Build FAISS index")
        print("  4. Launch interactive search mode")
