"""
RAG Query Module
Semantic search over document chunks for question answering

Orchestrates retrieval and presentation of relevant text chunks
from ESG reports based on natural language queries.
"""

from typing import List, Dict, Tuple, Optional
from src.embeddings import EmbeddingManager


class RAGQueryEngine:
    """
    Retrieval-Augmented Generation query engine for document search.

    Provides natural language question answering over indexed ESG reports
    by retrieving and ranking relevant text chunks.

    Attributes:
        embedding_manager: EmbeddingManager instance with built index
        default_top_k: Default number of results to return
        relevance_threshold: Minimum similarity score threshold
    """

    def __init__(self, embedding_manager: EmbeddingManager,
                 default_top_k: int = 5,
                 relevance_threshold: float = None):
        """
        Initialize RAG query engine.

        Args:
            embedding_manager: EmbeddingManager with built FAISS index
            default_top_k: Default number of chunks to retrieve (default: 5)
            relevance_threshold: Filter results above this distance (optional)
                                Lower = more similar. Typical range: 0.3-1.5
        """
        self.embedding_manager = embedding_manager
        self.default_top_k = default_top_k
        self.relevance_threshold = relevance_threshold

    def query(self, question: str,
              top_k: Optional[int] = None,
              sources: Optional[List[str]] = None,
              include_scores: bool = False) -> List[Dict]:
        """
        Query the document index with a natural language question.

        Args:
            question: Natural language query
            top_k: Number of results to return (uses default_top_k if None)
            sources: Filter results to specific sources (filenames)
            include_scores: Include similarity scores in results

        Returns:
            List of result dictionaries with structure:
            [
                {
                    "chunk_id": "report_p5_c2",
                    "source": "company_2023.pdf",
                    "page": 5,
                    "text": "We reduced emissions by 25%...",
                    "word_count": 305,
                    "score": 0.234  # Only if include_scores=True
                },
                ...
            ]

        Example:
            results = engine.query("What are the carbon reduction targets?")
            for result in results:
                print(f"{result['source']} (p{result['page']})")
                print(result['text'])
        """
        k = top_k if top_k is not None else self.default_top_k

        # Search the index
        raw_results = self.embedding_manager.search(question, top_k=k)

        # Filter by relevance threshold if set
        if self.relevance_threshold is not None:
            raw_results = [
                (chunk, score) for chunk, score in raw_results
                if score <= self.relevance_threshold
            ]

        # Filter by sources if specified
        if sources is not None:
            sources_set = set(sources)
            raw_results = [
                (chunk, score) for chunk, score in raw_results
                if chunk["source"] in sources_set
            ]

        # Format results
        results = []
        for chunk, score in raw_results:
            result = dict(chunk)  # Copy chunk data
            if include_scores:
                result["score"] = score
            results.append(result)

        return results

    def query_with_context(self, question: str,
                          top_k: Optional[int] = None,
                          sources: Optional[List[str]] = None) -> Dict:
        """
        Query and return results with aggregated context.

        Useful for feeding into LLMs or displaying structured answers.

        Args:
            question: Natural language query
            top_k: Number of results to retrieve
            sources: Filter to specific sources

        Returns:
            Dictionary with structure:
            {
                "question": "What are carbon targets?",
                "chunks": [...],  # List of result chunks
                "context": "...",  # Concatenated text from all chunks
                "sources": ["report_a.pdf", "report_b.pdf"],
                "num_results": 5
            }
        """
        results = self.query(
            question,
            top_k=top_k,
            sources=sources,
            include_scores=True
        )

        # Aggregate context
        context_parts = []
        seen_sources = set()

        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[{i}] Source: {result['source']} (page {result['page']})\n"
                f"{result['text']}\n"
            )
            seen_sources.add(result['source'])

        return {
            "question": question,
            "chunks": results,
            "context": "\n".join(context_parts),
            "sources": sorted(list(seen_sources)),
            "num_results": len(results)
        }

    def compare_sources(self, question: str,
                       source_a: str,
                       source_b: str,
                       top_k_per_source: int = 3) -> Dict:
        """
        Compare how two different sources address the same question.

        Useful for cross-company analysis (e.g., "How do Company A and B
        describe their climate commitments?").

        Args:
            question: Natural language query
            source_a: First source filename
            source_b: Second source filename
            top_k_per_source: Results to retrieve per source

        Returns:
            Dictionary with structure:
            {
                "question": "...",
                "source_a": {
                    "name": "company_a.pdf",
                    "results": [...]
                },
                "source_b": {
                    "name": "company_b.pdf",
                    "results": [...]
                }
            }
        """
        # Query each source separately
        results_a = self.query(
            question,
            top_k=top_k_per_source,
            sources=[source_a],
            include_scores=True
        )

        results_b = self.query(
            question,
            top_k=top_k_per_source,
            sources=[source_b],
            include_scores=True
        )

        return {
            "question": question,
            "source_a": {
                "name": source_a,
                "results": results_a,
                "num_results": len(results_a)
            },
            "source_b": {
                "name": source_b,
                "results": results_b,
                "num_results": len(results_b)
            }
        }

    def get_source_coverage(self, question: str, top_k: int = 10) -> Dict:
        """
        Analyze which sources contain relevant information for a query.

        Args:
            question: Natural language query
            top_k: Number of results to analyze

        Returns:
            Dictionary mapping source names to result counts:
            {
                "company_a.pdf": 6,
                "company_b.pdf": 3,
                "company_c.pdf": 1
            }
        """
        results = self.query(question, top_k=top_k)

        coverage = {}
        for result in results:
            source = result["source"]
            coverage[source] = coverage.get(source, 0) + 1

        # Sort by count (descending)
        return dict(sorted(coverage.items(), key=lambda x: x[1], reverse=True))

    def batch_query(self, questions: List[str],
                   top_k: int = 5) -> List[Dict]:
        """
        Execute multiple queries at once.

        Args:
            questions: List of natural language queries
            top_k: Results per query

        Returns:
            List of query results (one dict per question)
        """
        results = []
        for question in questions:
            query_result = self.query_with_context(question, top_k=top_k)
            results.append(query_result)

        return results


def create_query_engine(pdf_paths: List[str],
                       index_dir: Optional[str] = None,
                       **kwargs) -> RAGQueryEngine:
    """
    Convenience function: Build index from PDFs and create query engine.

    Args:
        pdf_paths: List of PDF file paths
        index_dir: Directory to save/load index (if None, not saved)
        **kwargs: Additional arguments for RAGQueryEngine

    Returns:
        Ready-to-use RAGQueryEngine instance

    Example:
        engine = create_query_engine([
            "data/company_a.pdf",
            "data/company_b.pdf"
        ])

        results = engine.query("What are carbon reduction targets?")
    """
    from src.embeddings import build_index_from_pdfs

    # Build or load index
    if index_dir:
        import os
        if os.path.exists(os.path.join(index_dir, "vector_store.faiss")):
            print(f"Loading existing index from {index_dir}...")
            manager = EmbeddingManager()
            manager.load_index(index_dir)
        else:
            manager = build_index_from_pdfs(pdf_paths, save_dir=index_dir)
    else:
        manager = build_index_from_pdfs(pdf_paths)

    return RAGQueryEngine(manager, **kwargs)


def format_results(results: List[Dict],
                  max_text_length: int = 200,
                  show_scores: bool = False) -> str:
    """
    Format query results as human-readable text.

    Args:
        results: List of result dictionaries from query()
        max_text_length: Truncate text preview to this length
        show_scores: Include similarity scores

    Returns:
        Formatted string for display
    """
    if not results:
        return "No results found."

    lines = []
    lines.append(f"Found {len(results)} relevant chunks:\n")

    for i, result in enumerate(results, 1):
        text_preview = result["text"]
        if len(text_preview) > max_text_length:
            text_preview = text_preview[:max_text_length] + "..."

        lines.append(f"[{i}] {result['source']} (page {result['page']})")

        if show_scores and "score" in result:
            lines.append(f"    Relevance: {result['score']:.3f}")

        lines.append(f"    {text_preview}\n")

    return "\n".join(lines)


# Demo/Testing functionality
if __name__ == "__main__":
    """
    Interactive RAG query interface for testing
    """
    import sys

    if len(sys.argv) > 1:
        pdf_paths = sys.argv[1:]
        print("=" * 60)
        print("RAG QUERY ENGINE — Interactive Mode")
        print("=" * 60)

        try:
            # Build index and create engine
            engine = create_query_engine(pdf_paths)

            # Show available sources
            stats = engine.embedding_manager.get_stats()
            print(f"\nIndexed sources:")
            for source in stats["sources"]:
                print(f"  - {source}")
            print(f"\nTotal chunks: {stats['total_chunks']}")

            # Interactive query loop
            print("\n" + "=" * 60)
            print("QUERY MODE (type 'quit' to exit)")
            print("Commands:")
            print("  query <text>     - Search for relevant chunks")
            print("  compare <src1> <src2> <text> - Compare two sources")
            print("  coverage <text>  - See which sources cover the topic")
            print("=" * 60)

            while True:
                user_input = input("\n> ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    break

                if not user_input:
                    continue

                # Parse command
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower() if parts else ""

                try:
                    if command == "query" and len(parts) > 1:
                        question = parts[1]
                        results = engine.query(question, top_k=5, include_scores=True)
                        print(format_results(results, show_scores=True))

                    elif command == "compare" and len(parts) > 1:
                        args = parts[1].split(maxsplit=2)
                        if len(args) >= 3:
                            src_a, src_b, question = args[0], args[1], args[2]
                            comparison = engine.compare_sources(question, src_a, src_b)

                            print(f"\nComparing: {src_a} vs {src_b}")
                            print(f"Question: {question}\n")

                            print(f"--- {src_a} ---")
                            print(format_results(comparison["source_a"]["results"]))

                            print(f"--- {src_b} ---")
                            print(format_results(comparison["source_b"]["results"]))
                        else:
                            print("Usage: compare <source1> <source2> <query>")

                    elif command == "coverage" and len(parts) > 1:
                        question = parts[1]
                        coverage = engine.get_source_coverage(question, top_k=10)
                        print(f"\nSource coverage for: '{question}'")
                        for source, count in coverage.items():
                            print(f"  {source}: {count} chunks")

                    else:
                        # Default to query
                        results = engine.query(user_input, top_k=5, include_scores=True)
                        print(format_results(results, show_scores=True))

                except Exception as e:
                    print(f"Error: {e}")

        except Exception as e:
            print(f"Failed to initialize: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("Usage: python src/rag_query.py <pdf1> [pdf2] [pdf3] ...")
        print("\nExample:")
        print("  python src/rag_query.py data/sample_reports/*.pdf")
        print("\nThis will:")
        print("  1. Parse PDFs and build index")
        print("  2. Launch interactive query interface")
        print("  3. Answer questions about the documents")
