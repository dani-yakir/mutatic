# Mutatic - Genome Similarity Comparison

A k-mer based genome embedding system for efficient similarity comparison and ANN/KNN search across large genome databases.

## Overview

This project converts DNA sequences into fixed-dimensional vectors using k-mer embeddings with DNA-BERT. The resulting vectors can be used for:
- Fast similarity search across thousands of genomes
- Approximate Nearest Neighbor (ANN) search with FAISS/Annoy
- Phylogenetic analysis
- Genome clustering

## Strategy

1. **K-mer Extraction**: Break each genome into overlapping k-mers (default k=6, stride=3)
2. **Embedding**: Embed each k-mer using DNA-BERT (768-dimensional vectors)
3. **Averaging**: Average all k-mer embeddings to create a single genome vector
4. **Normalization**: L2-normalize for cosine similarity comparison

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Test with sample sequences (first 10)
```bash
python process_all_sequences.py --test-only
```

### Process all sequences
```bash
python process_all_sequences.py --fasta sequence.fasta --output genome_embeddings.pkl
```

### Custom parameters
```bash
python process_all_sequences.py --k 7 --stride 4 --fasta sequence.fasta
```

## Usage

### Basic embedding
```python
from genome_embedder import GenomeEmbedder

# Initialize
embedder = GenomeEmbedder(k=6, stride=3)

# Embed a single sequence
sequence = "ATCGATCGATCG..."
vector = embedder.sequence_to_vector(sequence)

# Embed all sequences in FASTA file
embeddings = embedder.embed_fasta("sequence.fasta")
embedder.save_embeddings(embeddings, "embeddings.pkl")
```

### Find similar sequences
```python
from genome_embedder import find_most_similar, GenomeEmbedder
import pickle

# Load embeddings
with open("genome_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

# Find top 10 most similar to a query
similar = find_most_similar("OP120847.1", embeddings, top_k=10)

for seq_id, similarity in similar:
    print(f"{seq_id}: {similarity:.4f}")
```

## Files

- `genome_embedder.py` - Core embedding pipeline
- `process_all_sequences.py` - Batch processing script
- `test.py` - Original DNA-BERT test
- `sequence.fasta` - Input genome sequences (1,861 sequences)
- `requirements.txt` - Python dependencies

## Parameters

- `k`: K-mer size (default: 6)
  - Smaller k: More general, faster, less specific
  - Larger k: More specific, slower, sparser
  
- `stride`: Step size for k-mer extraction (default: 3)
  - stride=k: Non-overlapping k-mers (faster)
  - stride<k: Overlapping k-mers (more information)

## Next Steps: ANN/KNN Search

Once embeddings are generated, use FAISS for fast similarity search:

```python
import faiss
import numpy as np
import pickle

# Load embeddings
with open("genome_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

# Convert to matrix
ids = list(embeddings.keys())
vectors = np.array([embeddings[id] for id in ids]).astype('float32')

# Build FAISS index
dimension = vectors.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
index.add(vectors)

# Search for k nearest neighbors
k = 10
query_vector = vectors[0:1]  # First sequence
distances, indices = index.search(query_vector, k)

print("Top 10 similar sequences:")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"{i+1}. {ids[idx]}: similarity = {dist:.4f}")
```

## Data

Current dataset: 1,861 *Cladocopium* and *Symbiodinium* species sequences from NCBI
- Ribosomal RNA genes and ITS regions
- Sequence length: 260-730 bp
- Contains IUPAC ambiguity codes (handled by skipping ambiguous k-mers)
