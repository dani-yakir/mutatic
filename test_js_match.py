"""
Test that JavaScript implementation matches Python implementation.
Run a quick test and output results for comparison.
"""

import pickle
import numpy as np
from Bio import SeqIO
from genome_embedder import cosine_similarity
from test_needleman_wunsch_knn import needleman_wunsch_distance
import json

# Load data
print("Loading data...")
with open('genome_embeddings_k3.pkl', 'rb') as f:
    embeddings = pickle.load(f)

sequences = {}
for record in SeqIO.parse('sequence.fasta', 'fasta'):
    sequences[record.id] = str(record.seq)

# Get common IDs
common_ids = list(set(embeddings.keys()).intersection(set(sequences.keys())))
print(f"Total sequences: {len(common_ids)}")

# Pick a test query (first sequence)
query_id = common_ids[0]
query_vec = embeddings[query_id]
query_seq = sequences[query_id]

print(f"\nTest Query: {query_id}")
print(f"Sequence length: {len(query_seq)} bp")
print(f"Embedding shape: {query_vec.shape}")

# Test Vector KNN (cosine similarity)
print("\n" + "="*80)
print("VECTOR KNN (Cosine Similarity)")
print("="*80)

vector_results = []
for seq_id in common_ids[:10]:  # Test with first 10 for speed
    if seq_id == query_id:
        continue
    sim = cosine_similarity(query_vec, embeddings[seq_id])
    vector_results.append((seq_id, sim))

vector_results.sort(key=lambda x: x[1], reverse=True)
print("\nTop 5 results:")
for i, (seq_id, sim) in enumerate(vector_results[:5], 1):
    print(f"{i}. {seq_id}: {sim:.6f}")

# Test Needleman-Wunsch
print("\n" + "="*80)
print("NEEDLEMAN-WUNSCH")
print("="*80)

nw_results = []
for seq_id in common_ids[:10]:  # Test with first 10 for speed
    if seq_id == query_id:
        continue
    distance = needleman_wunsch_distance(query_seq, sequences[seq_id])
    similarity = 1 - distance  # Convert distance to similarity
    nw_results.append((seq_id, similarity))

nw_results.sort(key=lambda x: x[1], reverse=True)
print("\nTop 5 results:")
for i, (seq_id, sim) in enumerate(nw_results[:5], 1):
    print(f"{i}. {seq_id}: {sim:.6f}")

# Export test data for JS verification
test_data = {
    "query_id": query_id,
    "query_sequence": query_seq,
    "query_embedding": query_vec.tolist(),
    "test_sequences": []
}

for seq_id in common_ids[:10]:
    if seq_id != query_id:
        test_data["test_sequences"].append({
            "id": seq_id,
            "sequence": sequences[seq_id],
            "embedding": embeddings[seq_id].tolist(),
            "vector_similarity": float(cosine_similarity(query_vec, embeddings[seq_id])),
            "nw_similarity": float(1 - needleman_wunsch_distance(query_seq, sequences[seq_id]))
        })

with open('web/data/test_verification.json', 'w') as f:
    json.dump(test_data, f, indent=2)

print("\n" + "="*80)
print("Test data exported to web/data/test_verification.json")
print("You can now compare these results with the JavaScript implementation")
print("="*80)
