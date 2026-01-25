from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from Bio import SeqIO
import pickle
from tqdm import tqdm

class GenomeEmbedder:
    """
    Convert genome sequences to fixed-dimensional vectors using whole-sequence embeddings.
    Suitable for ANN/KNN search across large genome databases.
    """
    
    def __init__(self, model_name="zhihan1996/DNA_bert_3", k=3, stride=1, device=None, use_whole_sequence=True):
        """
        Initialize the genome embedder.
        
        Args:
            model_name: HuggingFace model for DNA embeddings
            k: K-mer size (default 3) - MUST match model vocabulary (DNA-BERT uses 3-mers)
            stride: Step size for k-mer extraction (default 1 for overlapping k-mers)
            device: torch device (cuda/cpu), auto-detected if None
            use_whole_sequence: If True, embed entire sequence (recommended). If False, use old k-mer averaging.
        """
        print(f"Loading DNA-BERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        
        self.k = k
        self.stride = stride
        self.use_whole_sequence = use_whole_sequence
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
        print(f"Embedding mode: {'WHOLE SEQUENCE' if use_whole_sequence else 'K-mer averaging'}")
        print(f"K-mer size: {k}, Stride: {stride}")
    
    def clean_sequence(self, sequence: str) -> str:
        """
        Clean sequence by converting to uppercase and handling ambiguity codes.
        For now, we'll filter out k-mers with ambiguous bases.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            Cleaned sequence string
        """
        return sequence.upper().strip()
    
    def extract_kmers(self, sequence: str, skip_ambiguous=True) -> List[str]:
        """
        Extract k-mers from a DNA sequence using sliding window.
        
        Args:
            sequence: DNA sequence string
            skip_ambiguous: If True, skip k-mers containing ambiguous bases (N, R, Y, etc.)
            
        Returns:
            List of k-mer strings
        """
        sequence = self.clean_sequence(sequence)
        kmers = []
        
        valid_bases = set('ACGT')
        
        for i in range(0, len(sequence) - self.k + 1, self.stride):
            kmer = sequence[i:i + self.k]
            
            if skip_ambiguous:
                # Only include k-mers with valid DNA bases
                if all(base in valid_bases for base in kmer):
                    kmers.append(kmer)
            else:
                kmers.append(kmer)
        
        return kmers
    
    def embed_kmer(self, kmer: str) -> np.ndarray:
        """
        Embed a single k-mer using DNA-BERT.
        
        Args:
            kmer: K-mer string
            
        Returns:
            Embedding vector as numpy array
        """
        tokens = self.tokenizer(kmer, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.bert(**tokens)
        
        # Use CLS token embedding as k-mer representation
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return cls_embedding.squeeze()
    
    def embed_kmers_batch(self, kmers: List[str], batch_size=32) -> np.ndarray:
        """
        Embed multiple k-mers in batches for efficiency.
        
        Args:
            kmers: List of k-mer strings
            batch_size: Number of k-mers to process at once
            
        Returns:
            Array of embeddings, shape (n_kmers, embedding_dim)
        """
        if not kmers:
            return np.array([])
        
        embeddings = []
        
        for i in range(0, len(kmers), batch_size):
            batch = kmers[i:i + batch_size]
            
            # Tokenize batch
            tokens = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.bert(**tokens)
            
            # Extract CLS embeddings
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def sequence_to_vector(self, sequence: str, normalize=True) -> np.ndarray:
        """
        Convert a genome sequence to a single fixed-dimensional vector.
        
        Strategy (use_whole_sequence=True): Tokenize entire sequence -> Mean pool all token embeddings
        Strategy (use_whole_sequence=False): Extract k-mers -> Embed each -> Average embeddings
        
        Args:
            sequence: DNA sequence string
            normalize: If True, L2-normalize the final vector (recommended for cosine similarity)
            
        Returns:
            Fixed-dimensional embedding vector
        """
        if self.use_whole_sequence:
            return self._embed_whole_sequence(sequence, normalize)
        else:
            return self._embed_kmers_averaged(sequence, normalize)
    
    def _embed_whole_sequence(self, sequence: str, normalize=True) -> np.ndarray:
        """
        Embed entire sequence at once using DNA-BERT.
        This preserves positional information and sequence context.
        """
        sequence = self.clean_sequence(sequence)
        
        # DNA-BERT requires k-mer formatted input (space-separated k-mers)
        # Convert sequence to k-mer string
        kmers = []
        valid_bases = set('ACGT')
        for i in range(0, len(sequence) - self.k + 1, self.stride):
            kmer = sequence[i:i + self.k]
            # Only include k-mers with valid DNA bases
            if all(base in valid_bases for base in kmer):
                kmers.append(kmer)
        
        if not kmers:
            # Return zero vector if no valid k-mers
            return np.zeros(768)
        
        # Join k-mers with spaces (DNA-BERT format)
        kmer_sequence = ' '.join(kmers)
        
        # Tokenize the k-mer sequence
        tokens = self.tokenizer(
            kmer_sequence,
            return_tensors="pt",
            truncation=True,
            max_length=512,  # BERT models typically have max length
            padding=False
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.bert(**tokens)
        
        # Get all token embeddings (not just CLS)
        # Shape: (1, seq_len, hidden_dim)
        token_embeddings = outputs.last_hidden_state
        
        # Mean pooling over all tokens (excluding padding if any)
        # This preserves information from the entire sequence
        if 'attention_mask' in tokens:
            mask = tokens['attention_mask'].unsqueeze(-1).float()
            masked_embeddings = token_embeddings * mask
            sum_embeddings = masked_embeddings.sum(dim=1)
            sum_mask = mask.sum(dim=1)
            avg_embedding = (sum_embeddings / sum_mask).cpu().numpy().squeeze()
        else:
            avg_embedding = token_embeddings.mean(dim=1).cpu().numpy().squeeze()
        
        # Normalize if requested
        if normalize:
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm
        
        return avg_embedding
    
    def _embed_kmers_averaged(self, sequence: str, normalize=True) -> np.ndarray:
        """
        OLD METHOD: Extract k-mers -> Embed each -> Average embeddings.
        This loses positional information.
        """
        # Extract k-mers
        kmers = self.extract_kmers(sequence)
        
        if not kmers:
            # Return zero vector if no valid k-mers
            embedding_dim = 768  # DNA-BERT embedding dimension
            return np.zeros(embedding_dim)
        
        # Embed k-mers in batches
        kmer_embeddings = self.embed_kmers_batch(kmers)
        
        # Average embeddings
        avg_embedding = np.mean(kmer_embeddings, axis=0)
        
        # Normalize if requested
        if normalize:
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm
        
        return avg_embedding
    
    def embed_fasta(self, fasta_file: str, normalize=True, show_progress=True) -> Dict[str, np.ndarray]:
        """
        Embed all sequences in a FASTA file.
        
        Args:
            fasta_file: Path to FASTA file
            normalize: If True, L2-normalize vectors
            show_progress: Show progress bar
            
        Returns:
            Dictionary mapping sequence IDs to embedding vectors
        """
        embeddings = {}
        
        # Count sequences first for progress bar
        seq_count = sum(1 for _ in SeqIO.parse(fasta_file, "fasta"))
        
        iterator = SeqIO.parse(fasta_file, "fasta")
        if show_progress:
            iterator = tqdm(iterator, total=seq_count, desc="Embedding sequences")
        
        for record in iterator:
            seq_id = record.id
            sequence = str(record.seq)
            
            embedding = self.sequence_to_vector(sequence, normalize=normalize)
            embeddings[seq_id] = embedding
        
        return embeddings
    
    def save_embeddings(self, embeddings: Dict[str, np.ndarray], output_file: str):
        """Save embeddings to file."""
        with open(output_file, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Embeddings saved to {output_file}")
    
    def load_embeddings(self, input_file: str) -> Dict[str, np.ndarray]:
        """Load embeddings from file."""
        with open(input_file, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Loaded {len(embeddings)} embeddings from {input_file}")
        return embeddings


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def find_most_similar(query_id: str, embeddings: Dict[str, np.ndarray], top_k=10) -> List[Tuple[str, float]]:
    """
    Find the most similar sequences to a query sequence.
    
    Args:
        query_id: ID of query sequence
        embeddings: Dictionary of all embeddings
        top_k: Number of top results to return
        
    Returns:
        List of (sequence_id, similarity_score) tuples, sorted by similarity
    """
    if query_id not in embeddings:
        raise ValueError(f"Query ID {query_id} not found in embeddings")
    
    query_vec = embeddings[query_id]
    similarities = []
    
    for seq_id, vec in embeddings.items():
        if seq_id != query_id:  # Skip self
            sim = cosine_similarity(query_vec, vec)
            similarities.append((seq_id, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


if __name__ == "__main__":
    # Example usage
    print("Genome Embedder - K-mer based sequence comparison")
    print("=" * 60)
    
    # Initialize embedder
    embedder = GenomeEmbedder(k=6, stride=3)
    
    # Test with a few sequences
    print("\nTesting with sample sequences from sequence.fasta...")
    
    # Load first few sequences
    sample_seqs = {}
    for i, record in enumerate(SeqIO.parse("sequence.fasta", "fasta")):
        if i >= 5:  # Just test with 5 sequences
            break
        sample_seqs[record.id] = str(record.seq)
        print(f"{i+1}. {record.id}: {len(record.seq)} bp")
    
    # Embed sequences
    print("\nEmbedding sequences...")
    sample_embeddings = {}
    for seq_id, sequence in sample_seqs.items():
        embedding = embedder.sequence_to_vector(sequence)
        sample_embeddings[seq_id] = embedding
        print(f"  {seq_id}: embedding shape {embedding.shape}")
    
    # Compare first sequence to others
    query_id = list(sample_embeddings.keys())[0]
    print(f"\nFinding sequences similar to {query_id}:")
    
    similar = find_most_similar(query_id, sample_embeddings, top_k=4)
    for rank, (seq_id, similarity) in enumerate(similar, 1):
        print(f"  {rank}. {seq_id}: similarity = {similarity:.4f}")
