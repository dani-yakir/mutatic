"""
Test KNN performance: Embeddings vs Raw Sequences

Compares:
1. KNN on DNA-BERT embedded vectors (fast)
2. KNN on raw sequences (slow baseline)

Shows runtime and result quality differences.
"""

import pickle
import numpy as np
import time
from Bio import SeqIO
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


def knn_raw_sequences(query_id: str, sequences: Dict[str, str], k: int = 10, kmer_size: int = 6) -> Tuple[List[Tuple[str, float]], float]:
    """KNN on raw sequences using k-mer frequency vectors."""
    start_time = time.time()
    
    def sequence_to_kmer_vector(seq: str) -> np.ndarray:
        """Convert sequence to k-mer frequency vector."""
        kmers = []
        for i in range(len(seq) - kmer_size + 1):
            kmer = seq[i:i + kmer_size]
            if all(base in 'ACGT' for base in kmer):
                kmers.append(kmer)
        
        # Generate all possible k-mers
        all_kmers = []
        bases = ['A', 'C', 'G', 'T']
        
        def generate_kmers(prefix, length):
            if length == 0:
                all_kmers.append(prefix)
                return
            for base in bases:
                generate_kmers(prefix + base, length - 1)
        
        generate_kmers('', kmer_size)
        
        # Count k-mers
        kmer_counts = {kmer: 0 for kmer in all_kmers}
        for kmer in kmers:
            if kmer in kmer_counts:
                kmer_counts[kmer] += 1
        
        # Convert to vector and normalize
        vector = np.array([kmer_counts[kmer] for kmer in all_kmers], dtype=float)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    # Compute query vector
    query_vec = sequence_to_kmer_vector(sequences[query_id])
    
    # Find nearest neighbors
    similarities = []
    for seq_id, sequence in sequences.items():
        if seq_id != query_id:
            seq_vec = sequence_to_kmer_vector(sequence)
            sim = np.dot(query_vec, seq_vec)
            similarities.append((seq_id, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    runtime = time.time() - start_time
    
    return similarities[:k], runtime


def calculate_overlap(results1: List[Tuple[str, float]], results2: List[Tuple[str, float]], k: int = 10) -> float:
    """Calculate percentage overlap between two result sets."""
    ids1 = set([seq_id for seq_id, _ in results1[:k]])
    ids2 = set([seq_id for seq_id, _ in results2[:k]])
    overlap = ids1.intersection(ids2)
    return len(overlap) / k


def main():
    parser = argparse.ArgumentParser(description='Test KNN performance: Embeddings vs Raw')
    parser.add_argument('--embeddings', default='genome_embeddings_k3.pkl', help='Embeddings file')
    parser.add_argument('--fasta', default='sequence.fasta', help='FASTA file')
    parser.add_argument('--k', type=int, default=10, help='Number of neighbors')
    parser.add_argument('--num-tests', type=int, default=5, help='Number of test queries')
    parser.add_argument('--query-ids', nargs='+', help='Specific query IDs to test')
    
    args = parser.parse_args()
    
    print("="*80)
    print("KNN PERFORMANCE TEST")
    print("="*80)
    print(f"Testing {args.k} nearest neighbors")
    print("="*80)
    
    # Load data
    embeddings = load_embeddings(args.embeddings)
    sequences = load_sequences(args.fasta)
    
    # Get common sequence IDs
    common_ids = list(set(embeddings.keys()).intersection(set(sequences.keys())))
    print(f"Common sequences: {len(common_ids)}")
    
    # Select test queries
    if args.query_ids:
        test_ids = [qid for qid in args.query_ids if qid in common_ids]
    else:
        import random
        random.seed(42)
        test_ids = random.sample(common_ids, min(args.num_tests, len(common_ids)))
    
    print(f"Testing {len(test_ids)} queries")
    print("="*80)
    
    # Run tests
    embedding_times = []
    raw_times = []
    overlaps = []
    
    for i, query_id in enumerate(test_ids, 1):
        print(f"\n--- Test {i}: {query_id} ---")
        
        # Test embedding-based KNN
        emb_results, emb_time = knn_embeddings(query_id, embeddings, k=args.k)
        embedding_times.append(emb_time)
        
        # Test raw sequence KNN
        raw_results, raw_time = knn_raw_sequences(query_id, sequences, k=args.k)
        raw_times.append(raw_time)
        
        # Calculate overlap
        overlap = calculate_overlap(emb_results, raw_results, k=args.k)
        overlaps.append(overlap)
        
        # Show results
        print(f"Embedding KNN: {emb_time:.4f}s")
        print(f"Raw KNN: {raw_time:.4f}s")
        print(f"Speedup: {raw_time/emb_time:.1f}x")
        print(f"Overlap: {overlap*100:.1f}%")
        
        # Show top 3 results
        print(f"\nEmbedding top 3:")
        for j, (seq_id, sim) in enumerate(emb_results[:3], 1):
            marker = " <--" if seq_id in [r[0] for r in raw_results[:3]] else ""
            print(f"  {j}. {seq_id} ({sim:.3f}){marker}")
        
        print(f"Raw top 3:")
        for j, (seq_id, sim) in enumerate(raw_results[:3], 1):
            marker = " <--" if seq_id in [r[0] for r in emb_results[:3]] else ""
            print(f"  {j}. {seq_id} ({sim:.3f}){marker}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    avg_emb_time = np.mean(embedding_times)
    avg_raw_time = np.mean(raw_times)
    avg_speedup = avg_raw_time / avg_emb_time
    avg_overlap = np.mean(overlaps)
    
    print(f"\nRuntime Performance:")
    print(f"  Embedding KNN: {avg_emb_time:.4f}s avg")
    print(f"  Raw KNN: {avg_raw_time:.4f}s avg")
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
    
    print("="*80)


if __name__ == "__main__":
    main()
