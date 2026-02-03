import pickle
import json
import numpy as np
from Bio import SeqIO

def convert_pickle_to_json(pickle_file, fasta_file, json_file):
    """Convert pickle embeddings and FASTA sequences to JSON format for web use."""
    
    # Load pickle data
    with open(pickle_file, 'rb') as f:
        embeddings = pickle.load(f)
    
    # Load FASTA sequences
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id] = str(record.seq)
    
    # Convert to web-friendly format
    web_data = {
        "sequences": []
    }
    
    for seq_id, embedding in embeddings.items():
        # Convert numpy array to list for JSON serialization
        if isinstance(embedding, np.ndarray):
            embedding_list = embedding.tolist()
        else:
            embedding_list = list(embedding) if hasattr(embedding, '__iter__') else [embedding]
        
        # Get actual sequence if available
        sequence_str = sequences.get(seq_id, "")
        
        web_data["sequences"].append({
            "id": seq_id,
            "embedding": embedding_list,
            "sequence": sequence_str,
            "length": len(sequence_str),
            "dimension": len(embedding_list)
        })
    
    # Add metadata
    web_data["metadata"] = {
        "total_sequences": len(web_data["sequences"]),
        "embedding_dimension": web_data["sequences"][0]["dimension"] if web_data["sequences"] else 0,
        "created_from": pickle_file,
        "fasta_file": fasta_file
    }
    
    # Save as JSON
    with open(json_file, 'w') as f:
        json.dump(web_data, f, indent=2)
    
    print(f"Converted {web_data['metadata']['total_sequences']} sequences from {pickle_file} to {json_file}")
    print(f"Embedding dimension: {web_data['metadata']['embedding_dimension']}")
    print(f"Sequences with actual DNA data: {sum(1 for s in web_data['sequences'] if s['sequence'])}")
    
    return web_data

if __name__ == "__main__":
    # Convert the pickle file
    convert_pickle_to_json("genome_embeddings_k3.pkl", "sequence.fasta", "web/data/sequences.json")
