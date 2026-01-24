from genome_embedder import GenomeEmbedder, find_most_similar, cosine_similarity
import numpy as np
import pickle
from Bio import SeqIO
import argparse

def main():
    parser = argparse.ArgumentParser(description='Process genome sequences and create embeddings')
    parser.add_argument('--fasta', default='sequence.fasta', help='Input FASTA file')
    parser.add_argument('--output', default='genome_embeddings.pkl', help='Output embeddings file')
    parser.add_argument('--k', type=int, default=6, help='K-mer size')
    parser.add_argument('--stride', type=int, default=3, help='K-mer stride')
    parser.add_argument('--test-only', action='store_true', help='Test with first 10 sequences only')
    
    args = parser.parse_args()
    
    print("="*70)
    print("GENOME SEQUENCE EMBEDDING PIPELINE")
    print("="*70)
    print(f"Input FASTA: {args.fasta}")
    print(f"Output file: {args.output}")
    print(f"K-mer size: {args.k}")
    print(f"Stride: {args.stride}")
    print("="*70)
    
    # Count sequences
    print("\nCounting sequences...")
    total_seqs = sum(1 for _ in SeqIO.parse(args.fasta, "fasta"))
    print(f"Total sequences in file: {total_seqs}")
    
    if args.test_only:
        print("\n⚠️  TEST MODE: Processing only first 10 sequences")
        total_seqs = min(10, total_seqs)
    
    # Initialize embedder
    print("\nInitializing GenomeEmbedder...")
    embedder = GenomeEmbedder(k=args.k, stride=args.stride)
    
    # Process sequences
    print(f"\nProcessing {total_seqs} sequences...")
    
    if args.test_only:
        # Manual processing for test mode
        embeddings = {}
        for i, record in enumerate(SeqIO.parse(args.fasta, "fasta")):
            if i >= 10:
                break
            print(f"  [{i+1}/10] {record.id} ({len(record.seq)} bp)")
            embedding = embedder.sequence_to_vector(str(record.seq))
            embeddings[record.id] = embedding
    else:
        # Full processing with progress bar
        embeddings = embedder.embed_fasta(args.fasta, normalize=True, show_progress=True)
    
    # Save embeddings
    print(f"\nSaving embeddings to {args.output}...")
    embedder.save_embeddings(embeddings, args.output)
    
    # Statistics
    print("\n" + "="*70)
    print("EMBEDDING STATISTICS")
    print("="*70)
    print(f"Total sequences embedded: {len(embeddings)}")
    
    embedding_dim = list(embeddings.values())[0].shape[0]
    print(f"Embedding dimension: {embedding_dim}")
    
    # Sample similarity check
    print("\n" + "="*70)
    print("SAMPLE SIMILARITY CHECK")
    print("="*70)
    
    seq_ids = list(embeddings.keys())
    query_id = seq_ids[0]
    print(f"\nQuery sequence: {query_id}")
    print(f"Finding top 5 most similar sequences...\n")
    
    similar = find_most_similar(query_id, embeddings, top_k=5)
    for rank, (seq_id, similarity) in enumerate(similar, 1):
        print(f"  {rank}. {seq_id}")
        print(f"     Similarity: {similarity:.6f}")
    
    # Compute average pairwise similarity (sample)
    print("\n" + "="*70)
    print("PAIRWISE SIMILARITY STATISTICS (sample of 100 pairs)")
    print("="*70)
    
    sample_size = min(100, len(seq_ids))
    similarities = []
    
    for i in range(sample_size):
        for j in range(i+1, min(i+10, sample_size)):  # Compare each to next 10
            sim = cosine_similarity(embeddings[seq_ids[i]], embeddings[seq_ids[j]])
            similarities.append(sim)
    
    if similarities:
        print(f"Sample pairs analyzed: {len(similarities)}")
        print(f"Mean similarity: {np.mean(similarities):.6f}")
        print(f"Std similarity: {np.std(similarities):.6f}")
        print(f"Min similarity: {np.min(similarities):.6f}")
        print(f"Max similarity: {np.max(similarities):.6f}")
    
    print("\n" + "="*70)
    print("✓ PROCESSING COMPLETE")
    print("="*70)
    print(f"\nEmbeddings saved to: {args.output}")
    print(f"Use these embeddings for ANN/KNN search with libraries like FAISS, Annoy, or ScaNN")
    print("\nNext steps:")
    print("  1. Load embeddings: embeddings = pickle.load(open('genome_embeddings.pkl', 'rb'))")
    print("  2. Build FAISS index for fast similarity search")
    print("  3. Query for similar genomes")

if __name__ == "__main__":
    main()
