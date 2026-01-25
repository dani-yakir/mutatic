"""
Test KNN performance: Embeddings vs Needleman-Wunsch Alignment

Compares:
1. KNN on DNA-BERT embedded vectors (fast)
2. KNN on Needleman-Wunsch alignment distance (biologically meaningful but slow)

Needleman-Wunsch finds optimal global alignment between sequences.
"""

import pickle
import numpy as np
import time
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from typing import Dict, List, Tuple
from genome_embedder import cosine_similarity
import argparse


def load_embeddings(pkl_file: str) -> Dict[str, np.ndarray]:
    """Load pre-computed embeddings from pickle file."""
    print(f"Loading embeddings from {pkl_file}...")
    with open(pkl_file, 'rb') as f:
        embeddings = pickle.load(f)
    print(f"Loaded {len(embeddings)} embeddings")
    return embeddings


def load_sequences(fasta_file: str) -> Dict[str, str]:
    """Load sequences from FASTA file."""
    print(f"Loading sequences from {fasta_file}...")
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id] = str(record.seq)
    print(f"Loaded {len(sequences)} sequences")
    return sequences


def knn_embeddings(query_id: str, embeddings: Dict[str, np.ndarray], k: int = 10) -> Tuple[List[Tuple[str, float]], float]:
    """KNN using pre-computed embeddings."""
    start_time = time.time()
    
    query_vec = embeddings[query_id]
    similarities = []
    
    for seq_id, vec in embeddings.items():
        if seq_id != query_id:
            sim = cosine_similarity(query_vec, vec)
            similarities.append((seq_id, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    runtime = time.time() - start_time
    
    return similarities[:k], runtime


def needleman_wunsch_distance(seq1: str, seq2: str, 
                             match_score=2, mismatch_score=-1, gap_score=-1) -> float:
    """
    Compute Needleman-Wunsch alignment distance using newer Biopython API.
    
    Returns normalized distance (0 = identical, 1 = completely different).
    Lower distance = more similar.
    """
    # Create aligner with scoring parameters
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = match_score
    aligner.mismatch_score = mismatch_score
    aligner.open_gap_score = gap_score
    aligner.extend_gap_score = gap_score
    
    # Compute just the score (faster, avoids overflow)
    score = aligner.score(seq1, seq2)
    
    # Normalize by sequence length to get distance between 0 and 1
    max_possible_score = max(len(seq1), len(seq2)) * match_score
    min_possible_score = max(len(seq1), len(seq2)) * mismatch_score
    
    # Convert score to similarity (0-1), then to distance
    similarity = (score - min_possible_score) / (max_possible_score - min_possible_score)
    similarity = max(0, min(1, similarity))  # Clamp to [0,1]
    
    distance = 1 - similarity
    return distance


def knn_needleman_wunsch(query_id: str, sequences: Dict[str, str], k: int = 10) -> Tuple[List[Tuple[str, float]], float]:
    """KNN using Needleman-Wunsch alignment distance."""
    start_time = time.time()
    
    query_seq = sequences[query_id]
    distances = []
    
    for seq_id, sequence in sequences.items():
        if seq_id != query_id:
            # Compute alignment distance
            distance = needleman_wunsch_distance(query_seq, sequence)
            # Convert distance to similarity for comparison (lower distance = higher similarity)
            similarity = 1 - distance
            distances.append((seq_id, similarity))
    
    distances.sort(key=lambda x: x[1], reverse=True)
    runtime = time.time() - start_time
    
    return distances[:k], runtime


def calculate_overlap(results1: List[Tuple[str, float]], results2: List[Tuple[str, float]], k: int = 10) -> float:
    """Calculate percentage overlap between two result sets."""
    ids1 = set([seq_id for seq_id, _ in results1[:k]])
    ids2 = set([seq_id for seq_id, _ in results2[:k]])
    overlap = ids1.intersection(ids2)
    return len(overlap) / k


def main():
    parser = argparse.ArgumentParser(description='Test KNN: Embeddings vs Needleman-Wunsch')
    parser.add_argument('--embeddings', default='genome_embeddings_k3.pkl', help='Embeddings file')
    parser.add_argument('--fasta', default='sequence.fasta', help='FASTA file')
    parser.add_argument('--k', type=int, default=10, help='Number of neighbors')
    parser.add_argument('--num-tests', type=int, default=3, help='Number of test queries')
    parser.add_argument('--query-ids', nargs='+', help='Specific query IDs to test')
    parser.add_argument('--match-score', type=int, default=2, help='NW match score')
    parser.add_argument('--mismatch-score', type=int, default=-1, help='NW mismatch score')
    parser.add_argument('--gap-score', type=int, default=-1, help='NW gap score')
    
    args = parser.parse_args()
    
    print("="*80)
    print("KNN PERFORMANCE TEST: EMBEDDINGS vs NEEDLEMAN-WUNSCH")
    print("="*80)
    print(f"Testing {args.k} nearest neighbors")
    print(f"Needleman-Wunsch: match={args.match_score}, mismatch={args.mismatch_score}, gap={args.gap_score}")
    print("="*80)
    
    # Load data
    embeddings = load_embeddings(args.embeddings)
    sequences = load_sequences(args.fasta)
    
    # Get common sequence IDs
    common_ids = list(set(embeddings.keys()).intersection(set(sequences.keys())))
    print(f"Common sequences: {len(common_ids)}")
    
    # Select test queries (prefer shorter sequences for NW speed)
    if args.query_ids:
        test_ids = [qid for qid in args.query_ids if qid in common_ids]
    else:
        import random
        random.seed(42)
        # Sort by length and prefer shorter sequences for faster NW
        sorted_ids = sorted(common_ids, key=lambda x: len(sequences[x]))[:50]
        test_ids = random.sample(sorted_ids, min(args.num_tests, len(sorted_ids)))
    
    print(f"Testing {len(test_ids)} queries (selected shorter sequences for NW speed)")
    print("="*80)
    
    # Run tests
    embedding_times = []
    nw_times = []
    overlaps = []
    
    for i, query_id in enumerate(test_ids, 1):
        print(f"\n--- Test {i}: {query_id} (length: {len(sequences[query_id])} bp) ---")
        
        # Test embedding-based KNN
        emb_results, emb_time = knn_embeddings(query_id, embeddings, k=args.k)
        embedding_times.append(emb_time)
        
        # Test Needleman-Wunsch KNN
        nw_results, nw_time = knn_needleman_wunsch(query_id, sequences, k=args.k)
        nw_times.append(nw_time)
        
        # Calculate overlap
        overlap = calculate_overlap(emb_results, nw_results, k=args.k)
        overlaps.append(overlap)
        
        # Show results
        print(f"Embedding KNN: {emb_time:.4f}s")
        print(f"NW KNN: {nw_time:.4f}s")
        speedup = nw_time / emb_time if emb_time > 0 else float('inf')
        print(f"Speedup: {speedup:.1f}x")
        print(f"Overlap: {overlap*100:.1f}%")
        
        # Show top 3 results
        print(f"\nEmbedding top 3:")
        for j, (seq_id, sim) in enumerate(emb_results[:3], 1):
            marker = " <--" if seq_id in [r[0] for r in nw_results[:3]] else ""
            print(f"  {j}. {seq_id} ({sim:.3f}){marker}")
        
        print(f"NW top 3:")
        for j, (seq_id, sim) in enumerate(nw_results[:3], 1):
            marker = " <--" if seq_id in [r[0] for r in emb_results[:3]] else ""
            print(f"  {j}. {seq_id} ({sim:.3f}){marker}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    avg_emb_time = np.mean(embedding_times)
    avg_nw_time = np.mean(nw_times)
    avg_speedup = avg_nw_time / avg_emb_time if avg_emb_time > 0 else float('inf')
    avg_overlap = np.mean(overlaps)
    
    print(f"\nRuntime Performance:")
    print(f"  Embedding KNN: {avg_emb_time:.4f}s avg")
    print(f"  NW KNN: {avg_nw_time:.4f}s avg")
    print(f"  Speedup: {avg_speedup:.1f}x faster")
    
    print(f"\nResult Quality:")
    print(f"  Average overlap: {avg_overlap*100:.1f}%")
    print(f"  Overlap range: {min(overlaps)*100:.1f}% - {max(overlaps)*100:.1f}%")
    
    print(f"\nConclusion:")
    if avg_speedup > 10:
        print(f"✓ Embedding-based KNN is {avg_speedup:.1f}x FASTER")
    else:
        print(f"✗ Embedding-based KNN is {avg_speedup:.1f}x SLOWER")
    
    if avg_overlap >= 0.7:
        print(f"✓ High result agreement ({avg_overlap*100:.1f}% overlap)")
    elif avg_overlap >= 0.5:
        print(f"~ Moderate result agreement ({avg_overlap*100:.1f}% overlap)")
    else:
        print(f"✗ Low result agreement ({avg_overlap*100:.1f}% overlap)")
    
    print(f"\nNote: Needleman-Wunsch is the gold standard for sequence similarity")
    print(f"but is computationally expensive (O(n*m) per alignment).")
    print("="*80)


if __name__ == "__main__":
    main()
