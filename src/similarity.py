"""
Similarity Module
Cross-company comparison of SDG commitments using cosine similarity

Analyzes how companies describe the same SDG goals to detect:
- Copy-paste claims (very high similarity)
- Generic vs. unique commitments
- Language patterns across organizations
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class SimilarityAnalyzer:
    """
    Analyzes similarity between company commitments.

    Uses cosine similarity on embeddings to compare how different companies
    address the same SDGs and identify copy-paste vs. authentic language.

    Attributes:
        embedding_manager: EmbeddingManager instance (from Step 3)
        similarity_thresholds: Dict defining similarity interpretation levels
    """

    def __init__(self, embedding_manager=None):
        """
        Initialize similarity analyzer.

        Args:
            embedding_manager: EmbeddingManager with embeddings capability
        """
        self.embedding_manager = embedding_manager

        # Similarity interpretation thresholds
        self.similarity_thresholds = {
            "identical": 0.95,      # Likely copy-paste
            "very_similar": 0.85,   # Very similar phrasing
            "similar": 0.70,        # Similar concepts
            "somewhat_similar": 0.50,  # Some overlap
            "different": 0.0        # Different approaches
        }

    def compare_texts(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.

        Args:
            text1: First text to compare
            text2: Second text to compare

        Returns:
            Similarity score (0-1), where 1 = identical, 0 = completely different
        """
        if not self.embedding_manager:
            raise ValueError("EmbeddingManager required for similarity comparison")

        # Generate embeddings
        embeddings = self.embedding_manager.embed_chunks([
            {"text": text1},
            {"text": text2}
        ], show_progress=False)

        # Normalize for cosine similarity
        import faiss
        faiss.normalize_L2(embeddings)

        # Calculate cosine similarity
        similarity = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])

        return similarity

    def interpret_similarity(self, score: float) -> str:
        """
        Interpret similarity score into human-readable category.

        Args:
            score: Similarity score (0-1)

        Returns:
            Category label (identical, very_similar, similar, etc.)
        """
        if score >= self.similarity_thresholds["identical"]:
            return "identical"
        elif score >= self.similarity_thresholds["very_similar"]:
            return "very_similar"
        elif score >= self.similarity_thresholds["similar"]:
            return "similar"
        elif score >= self.similarity_thresholds["somewhat_similar"]:
            return "somewhat_similar"
        else:
            return "different"

    def compare_sources_on_sdg(self, classified_chunks: List[Dict],
                               source_a: str, source_b: str,
                               sdg_id: str) -> Dict:
        """
        Compare how two sources address a specific SDG.

        Args:
            classified_chunks: Chunks with SDG classifications
            source_a: First source filename
            source_b: Second source filename
            sdg_id: SDG to compare (e.g., "SDG 13")

        Returns:
            Comparison results:
            {
                "sdg_id": "SDG 13",
                "source_a": "company_a.pdf",
                "source_b": "company_b.pdf",
                "source_a_chunks": [...],
                "source_b_chunks": [...],
                "average_similarity": 0.73,
                "interpretation": "similar",
                "pairwise_similarities": [
                    {"chunk_a_id": "...", "chunk_b_id": "...", "score": 0.85},
                    ...
                ]
            }
        """
        # Filter chunks by source and SDG
        source_a_chunks = [
            c for c in classified_chunks
            if c.get("source") == source_a and
            any(sdg["sdg_id"] == sdg_id for sdg in c.get("sdg_matches", []))
        ]

        source_b_chunks = [
            c for c in classified_chunks
            if c.get("source") == source_b and
            any(sdg["sdg_id"] == sdg_id for sdg in c.get("sdg_matches", []))
        ]

        if not source_a_chunks or not source_b_chunks:
            return {
                "sdg_id": sdg_id,
                "source_a": source_a,
                "source_b": source_b,
                "source_a_chunks": len(source_a_chunks),
                "source_b_chunks": len(source_b_chunks),
                "average_similarity": None,
                "interpretation": "insufficient_data",
                "pairwise_similarities": []
            }

        # Calculate pairwise similarities
        pairwise_sims = []

        for chunk_a in source_a_chunks:
            for chunk_b in source_b_chunks:
                sim_score = self.compare_texts(chunk_a["text"], chunk_b["text"])

                pairwise_sims.append({
                    "chunk_a_id": chunk_a["chunk_id"],
                    "chunk_b_id": chunk_b["chunk_id"],
                    "score": sim_score,
                    "interpretation": self.interpret_similarity(sim_score)
                })

        # Calculate average similarity
        avg_similarity = np.mean([s["score"] for s in pairwise_sims])

        return {
            "sdg_id": sdg_id,
            "source_a": source_a,
            "source_b": source_b,
            "source_a_chunks": len(source_a_chunks),
            "source_b_chunks": len(source_b_chunks),
            "average_similarity": float(avg_similarity),
            "interpretation": self.interpret_similarity(avg_similarity),
            "pairwise_similarities": pairwise_sims
        }

    def compare_all_sources(self, classified_chunks: List[Dict],
                           sdg_id: str) -> Dict[Tuple[str, str], Dict]:
        """
        Compare all pairs of sources for a specific SDG.

        Args:
            classified_chunks: Chunks with SDG classifications
            sdg_id: SDG to analyze

        Returns:
            Dict mapping (source_a, source_b) → comparison results
        """
        # Get all unique sources that mention this SDG
        sources = set()
        for chunk in classified_chunks:
            if any(sdg["sdg_id"] == sdg_id for sdg in chunk.get("sdg_matches", [])):
                sources.add(chunk.get("source", "unknown"))

        sources = sorted(list(sources))

        # Compare all pairs
        comparisons = {}

        for i, source_a in enumerate(sources):
            for source_b in sources[i + 1:]:
                comparison = self.compare_sources_on_sdg(
                    classified_chunks,
                    source_a,
                    source_b,
                    sdg_id
                )
                comparisons[(source_a, source_b)] = comparison

        return comparisons

    def detect_copy_paste(self, classified_chunks: List[Dict],
                         threshold: float = 0.95) -> List[Dict]:
        """
        Detect potential copy-paste claims across sources.

        Args:
            classified_chunks: Chunks with SDG classifications
            threshold: Similarity threshold for copy-paste detection (default: 0.95)

        Returns:
            List of suspected copy-paste instances:
            [
                {
                    "source_a": "company_a.pdf",
                    "source_b": "company_b.pdf",
                    "chunk_a_id": "...",
                    "chunk_b_id": "...",
                    "similarity": 0.97,
                    "sdg_id": "SDG 13",
                    "text_preview_a": "...",
                    "text_preview_b": "..."
                }
            ]
        """
        copy_paste_instances = []

        # Group chunks by source
        by_source = defaultdict(list)
        for chunk in classified_chunks:
            by_source[chunk.get("source", "unknown")].append(chunk)

        sources = sorted(list(by_source.keys()))

        # Compare all pairs of sources
        for i, source_a in enumerate(sources):
            for source_b in sources[i + 1:]:
                chunks_a = by_source[source_a]
                chunks_b = by_source[source_b]

                # Compare each chunk pair
                for chunk_a in chunks_a:
                    for chunk_b in chunks_b:
                        # Only compare if they share SDGs
                        sdgs_a = set(s["sdg_id"] for s in chunk_a.get("sdg_matches", []))
                        sdgs_b = set(s["sdg_id"] for s in chunk_b.get("sdg_matches", []))
                        shared_sdgs = sdgs_a.intersection(sdgs_b)

                        if shared_sdgs:
                            sim = self.compare_texts(chunk_a["text"], chunk_b["text"])

                            if sim >= threshold:
                                copy_paste_instances.append({
                                    "source_a": source_a,
                                    "source_b": source_b,
                                    "chunk_a_id": chunk_a["chunk_id"],
                                    "chunk_b_id": chunk_b["chunk_id"],
                                    "similarity": sim,
                                    "shared_sdgs": sorted(list(shared_sdgs)),
                                    "text_preview_a": chunk_a["text"][:150],
                                    "text_preview_b": chunk_b["text"][:150]
                                })

        # Sort by similarity (highest first)
        copy_paste_instances.sort(key=lambda x: x["similarity"], reverse=True)

        return copy_paste_instances

    def calculate_uniqueness_score(self, classified_chunks: List[Dict],
                                   source: str) -> float:
        """
        Calculate uniqueness score for a source (0-1).

        Higher score = more unique language compared to other sources.
        Lower score = more generic/similar to others.

        Args:
            classified_chunks: Chunks with SDG classifications
            source: Source to analyze

        Returns:
            Uniqueness score (0-1), where 1 = highly unique, 0 = very generic
        """
        # Get chunks from target source
        target_chunks = [c for c in classified_chunks if c.get("source") == source]

        if not target_chunks:
            return 0.0

        # Get chunks from all other sources
        other_chunks = [c for c in classified_chunks if c.get("source") != source]

        if not other_chunks:
            return 1.0  # Only source, by definition unique

        # Fast path: use pre-computed embeddings from FAISS index via matrix multiply
        # Avoids re-embedding every chunk pair individually (O(n*m) model calls → O(n) reconstruct)
        em = self.embedding_manager
        if em is not None and em.index is not None and em.chunks:
            n = em.index.ntotal
            dim = em.dimension

            # Reconstruct all embeddings from FAISS index in one shot
            all_emb = np.zeros((n, dim), dtype=np.float32)
            for i in range(n):
                em.index.reconstruct(i, all_emb[i])

            # Map chunk_id → position in index
            id_to_pos = {c.get("chunk_id"): i for i, c in enumerate(em.chunks)}

            target_pos = [id_to_pos[c["chunk_id"]] for c in target_chunks
                          if c.get("chunk_id") in id_to_pos]
            other_pos = [id_to_pos[c["chunk_id"]] for c in other_chunks
                         if c.get("chunk_id") in id_to_pos]

            if target_pos and other_pos:
                t_emb = all_emb[target_pos]  # (n_target, dim)
                o_emb = all_emb[other_pos]   # (n_other, dim)
                # Embeddings in FAISS are L2-normalized → dot product = cosine similarity
                sim_matrix = t_emb @ o_emb.T  # (n_target, n_other)
                return float(1.0 - np.mean(sim_matrix))

        # Slow fallback: sample and compare (capped to avoid hanging)
        import random
        MAX_SAMPLES = 20
        sample_target = random.sample(target_chunks, min(MAX_SAMPLES, len(target_chunks)))
        sample_other = random.sample(other_chunks, min(MAX_SAMPLES, len(other_chunks)))

        similarities = []
        for tc in sample_target:
            sims = [self.compare_texts(tc["text"], oc["text"]) for oc in sample_other]
            similarities.append(np.mean(sims))

        return float(1.0 - np.mean(similarities))

    def format_similarity_report(self, classified_chunks: List[Dict],
                                sdg_id: Optional[str] = None) -> str:
        """
        Generate human-readable similarity analysis report.

        Args:
            classified_chunks: Chunks with SDG classifications
            sdg_id: Optional SDG to focus on (if None, overall analysis)

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("SIMILARITY ANALYSIS REPORT")
        lines.append("=" * 60)

        # Get unique sources
        sources = sorted(set(c.get("source", "unknown") for c in classified_chunks))

        if sdg_id:
            lines.append(f"\nFocus: {sdg_id}")
            comparisons = self.compare_all_sources(classified_chunks, sdg_id)

            lines.append(f"\n--- Source Comparisons for {sdg_id} ---")

            for (source_a, source_b), comp in sorted(comparisons.items()):
                if comp["average_similarity"] is not None:
                    lines.append(
                        f"\n{source_a} vs {source_b}:"
                    )
                    lines.append(
                        f"  Similarity: {comp['average_similarity']:.2f} "
                        f"({comp['interpretation']})"
                    )
                    lines.append(
                        f"  Chunks compared: {comp['source_a_chunks']} x {comp['source_b_chunks']}"
                    )

        # Copy-paste detection
        lines.append("\n" + "=" * 60)
        lines.append("COPY-PASTE DETECTION")
        lines.append("=" * 60)

        copy_paste = self.detect_copy_paste(classified_chunks, threshold=0.90)

        if copy_paste:
            lines.append(f"\nFound {len(copy_paste)} potential copy-paste instances (similarity ≥ 90%):\n")

            for i, instance in enumerate(copy_paste[:5], 1):  # Top 5
                lines.append(f"[{i}] {instance['source_a']} ↔ {instance['source_b']}")
                lines.append(f"    Similarity: {instance['similarity']:.2f}")
                lines.append(f"    SDGs: {', '.join(instance['shared_sdgs'])}")
                lines.append(f"    Preview A: {instance['text_preview_a']}...")
                lines.append(f"    Preview B: {instance['text_preview_b']}...\n")
        else:
            lines.append("\nNo copy-paste instances detected (all similarities < 90%)")

        # Uniqueness scores
        lines.append("=" * 60)
        lines.append("UNIQUENESS SCORES")
        lines.append("=" * 60)
        lines.append("\nHow unique is each company's language?\n")

        uniqueness_scores = []
        for source in sources:
            score = self.calculate_uniqueness_score(classified_chunks, source)
            uniqueness_scores.append((source, score))

        uniqueness_scores.sort(key=lambda x: x[1], reverse=True)

        for source, score in uniqueness_scores:
            interpretation = "Highly unique" if score > 0.7 else \
                           "Moderately unique" if score > 0.5 else \
                           "Generic/similar to others"

            lines.append(f"{source}: {score:.2f} ({interpretation})")

        return "\n".join(lines)


# Demo/Testing functionality
if __name__ == "__main__":
    """
    Test similarity analyzer with sample data
    """
    import sys

    if len(sys.argv) > 1:
        # Analyze PDFs
        from src.pdf_parser import parse_multiple_pdfs
        from src.sdg_classifier import SDGClassifier
        from src.embeddings import EmbeddingManager

        pdf_paths = sys.argv[1:]
        print("=" * 60)
        print("SIMILARITY ANALYZER — Cross-Company Analysis")
        print("=" * 60)

        try:
            # Parse PDFs
            print("\nStep 1: Parsing PDFs...")
            chunks = parse_multiple_pdfs(pdf_paths)

            # Classify SDGs
            print("\nStep 2: Classifying SDGs...")
            classifier = SDGClassifier()
            classified = classifier.classify_chunks(chunks, show_progress=True)

            # Build embeddings
            print("\nStep 3: Generating embeddings...")
            manager = EmbeddingManager()
            manager.build_index(classified)

            # Analyze similarity
            print("\nStep 4: Analyzing similarity...")
            analyzer = SimilarityAnalyzer(manager)

            # Generate report
            print("\n" + analyzer.format_similarity_report(classified))

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("Similarity Analyzer Demo")
        print("\nUsage: python src/similarity.py <pdf1> <pdf2> [pdf3] ...")
        print("\nThis will:")
        print("  1. Parse and classify PDFs into SDGs")
        print("  2. Generate embeddings")
        print("  3. Analyze cross-company similarity")
        print("  4. Detect potential copy-paste claims")
        print("  5. Calculate uniqueness scores")
