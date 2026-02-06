from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
from datetime import datetime
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from genome_embedder import cosine_similarity

app = Flask(__name__, static_folder='../web', static_url_path='')
CORS(app)

# Directory for storing test results
RESULTS_DIR = Path(__file__).parent / 'test_results'
RESULTS_DIR.mkdir(exist_ok=True)

# Load sequence data ONCE at startup and cache in memory
DATA_PATH = Path(__file__).parent.parent / 'web' / 'data' / 'sequences.json'

print(f"Loading sequence data from {DATA_PATH}...")
with open(DATA_PATH, 'r') as f:
    SEQUENCE_DATA = json.load(f)
print(f"✓ Loaded {len(SEQUENCE_DATA['sequences'])} sequences into memory")

def get_sequence_data():
    """Get cached sequence data (already loaded in memory)"""
    return SEQUENCE_DATA

def needleman_wunsch_similarity(seq1, seq2):
    """
    Optimized Needleman-Wunsch similarity using pure Python lists.
    Faster than numpy for small matrices due to less overhead.
    """
    m, n = len(seq1), len(seq2)
    
    match_score = 2
    mismatch_penalty = -1
    gap_penalty = -1
    
    # Use only two rows instead of full matrix for memory efficiency
    prev_row = [j * gap_penalty for j in range(n + 1)]
    curr_row = [0] * (n + 1)
    
    # Fill matrix row by row
    for i in range(1, m + 1):
        curr_row[0] = i * gap_penalty
        seq1_char = seq1[i-1]  # Cache character lookup
        
        for j in range(1, n + 1):
            # Inline the match/mismatch check for speed
            if seq1_char == seq2[j-1]:
                match = prev_row[j-1] + match_score
            else:
                match = prev_row[j-1] + mismatch_penalty
            
            delete = prev_row[j] + gap_penalty
            insert = curr_row[j-1] + gap_penalty
            
            # Inline max for speed
            if match >= delete:
                if match >= insert:
                    curr_row[j] = match
                else:
                    curr_row[j] = insert
            else:
                if delete >= insert:
                    curr_row[j] = delete
                else:
                    curr_row[j] = insert
        
        # Swap rows
        prev_row, curr_row = curr_row, prev_row
    
    # Normalize to 0-1 range
    max_score = min(m, n) * match_score
    min_score = max(m, n) * gap_penalty
    normalized = (prev_row[n] - min_score) / (max_score - min_score)
    
    return max(0.0, min(1.0, normalized))

@app.route('/')
def index():
    """Serve the menu page as default"""
    return send_from_directory(app.static_folder, 'menu.html')

@app.route('/api/sequences', methods=['GET'])
def get_sequences():
    """Get all sequence data"""
    return jsonify(SEQUENCE_DATA)

@app.route('/api/run-test', methods=['POST'])
def run_test():
    """Run a statistical test"""
    try:
        data = request.json
        num_sequences = data.get('num_sequences', 10)
        k_value = data.get('k_value', 10)
        
        print(f"\n=== Starting test: {num_sequences} sequences, K={k_value} ===")
        
        # Use cached sequence data (already in memory)
        sequences = SEQUENCE_DATA['sequences']
        
        # Select random sequences
        import random
        import time
        random_indices = random.sample(range(len(sequences)), min(num_sequences, len(sequences)))
        
        print(f"Selected random indices: {random_indices}")
        
        results = {}
        
        for idx in random_indices:
            query_seq = sequences[idx]
            query_id = query_seq['id']
            query_embedding = query_seq['embedding']
            query_sequence = query_seq['sequence']
            
            print(f"\nProcessing {query_id}...")
            
            # Vector KNN
            vec_start = time.time()
            vec_similarities = []
            print("  Running Vector KNN...")
            for i, seq in tqdm(enumerate(sequences), total=len(sequences), desc="  Vector KNN", leave=False):
                if i != idx:
                    sim = cosine_similarity(query_embedding, seq['embedding'])
                    vec_similarities.append({
                        'id': seq['id'],
                        'similarity': float(sim),
                        'index': i
                    })
            vec_similarities.sort(key=lambda x: x['similarity'], reverse=True)
            vec_knn = vec_similarities[:k_value]
            vec_time = time.time() - vec_start
            
            # Needleman-Wunsch KNN
            nw_start = time.time()
            nw_similarities = []
            print("  Running Needleman-Wunsch...")
            
            # Debug: Check sequence lengths
            query_len = len(query_sequence)
            print(f"    Query sequence length: {query_len}")
            
            nw_times = []
            for i, seq in tqdm(enumerate(sequences), total=len(sequences), desc="  NW", leave=False):
                if i != idx:
                    nw_single_start = time.time()
                    sim = needleman_wunsch_similarity(query_sequence, seq['sequence'])
                    nw_single_time = time.time() - nw_single_start
                    nw_times.append(nw_single_time)
                    
                    nw_similarities.append({
                        'id': seq['id'],
                        'similarity': float(sim),
                        'index': i
                    })
                    
                    # Log first few comparisons for debugging
                    if i < 3:
                        print(f"    Comparison {i}: {len(seq['sequence'])} bases, {nw_single_time:.3f}s")
            
            nw_similarities.sort(key=lambda x: x['similarity'], reverse=True)
            nw_knn = nw_similarities[:k_value]
            nw_time = time.time() - nw_start
            
            avg_nw_time = sum(nw_times) / len(nw_times) if nw_times else 0
            print(f"    Average NW time per comparison: {avg_nw_time:.3f}s")
            
            print(f"  ✓ Vector KNN: {vec_time:.2f}s, NW: {nw_time:.2f}s")
            
            results[query_id] = {
                'vec_knn': {
                    'runtime': vec_time,
                    'returned_knn': [{'id': r['id'], 'similarity': r['similarity']} for r in vec_knn]
                },
                'nw_knn': {
                    'runtime': nw_time,
                    'returned_knn': [{'id': r['id'], 'similarity': r['similarity']} for r in nw_knn]
                }
            }
        
        # Create response with header
        response = {
            'header': {
                'timestamp': datetime.now().isoformat(),
                'num_sequences': num_sequences,
                'requested_knn': k_value
            },
            'seqs': results
        }
        
        print(f"=== Test complete! Processed {len(results)} sequences ===\n")
        return jsonify(response)
        
    except Exception as e:
        print(f"ERROR in run_test: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-test', methods=['POST'])
def save_test():
    """Save test results to file"""
    try:
        data = request.json
        print(f"Received save request with data keys: {data.keys()}")
        
        timestamp = data['header']['timestamp'].replace(':', '-').replace('.', '-')
        filename = f"genome_test_{timestamp}.json"
        filepath = RESULTS_DIR / filename
        
        print(f"Saving to: {filepath}")
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Successfully saved {filename}")
        return jsonify({'success': True, 'filename': filename})
    except Exception as e:
        print(f"Error saving test: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/list-tests', methods=['GET'])
def list_tests():
    """List all saved test results"""
    files = []
    for filepath in RESULTS_DIR.glob('genome_test_*.json'):
        files.append({
            'filename': filepath.name,
            'timestamp': filepath.stat().st_mtime
        })
    files.sort(key=lambda x: x['timestamp'], reverse=True)
    return jsonify(files)

@app.route('/api/load-test/<filename>', methods=['GET'])
def load_test(filename):
    """Load a specific test result"""
    filepath = RESULTS_DIR / filename
    if not filepath.exists():
        return jsonify({'error': 'File not found'}), 404
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return jsonify(data)

@app.route('/api/compare-single', methods=['POST'])
def compare_single():
    """Run comparison for a single sequence"""
    try:
        data = request.json
        sequence_id = data.get('sequence_id')
        k_value = data.get('k_value', 10)
        
        print(f"\n=== Single comparison for {sequence_id}, K={k_value} ===")
        
        # Use cached sequence data (already in memory)
        sequences = SEQUENCE_DATA['sequences']
        
        # Find query sequence
        query_seq = next((s for s in sequences if s['id'] == sequence_id), None)
        if not query_seq:
            return jsonify({'error': 'Sequence not found'}), 404
        
        query_idx = sequences.index(query_seq)
        query_embedding = query_seq['embedding']
        query_sequence = query_seq['sequence']
        
        # Vector KNN
        import time
        vec_start = time.time()
        vec_similarities = []
        print("Running Vector KNN...")
        for i, seq in tqdm(enumerate(sequences), total=len(sequences), desc="Vector KNN", leave=False):
            if i != query_idx:
                sim = cosine_similarity(query_embedding, seq['embedding'])
                vec_similarities.append({
                    'id': seq['id'],
                    'similarity': float(sim),
                    'index': i
                })
        vec_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        vec_results = vec_similarities[:k_value]
        vec_time = time.time() - vec_start
        print(f"Vector KNN: {vec_time:.2f}s")
        
        # Needleman-Wunsch KNN
        nw_start = time.time()
        nw_similarities = []
        print("Running Needleman-Wunsch...")
        for i, seq in tqdm(enumerate(sequences), total=len(sequences), desc="NW", leave=False):
            if i != query_idx:
                sim = needleman_wunsch_similarity(query_sequence, seq['sequence'])
                nw_similarities.append({
                    'id': seq['id'],
                    'similarity': float(sim),
                    'index': i
                })
        nw_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        nw_results = nw_similarities[:k_value]
        nw_time = time.time() - nw_start
        print(f"NW: {nw_time:.2f}s")
        print(f"=== Comparison complete! ===\n")
        
        return jsonify({
            'vectorResults': vec_results,
            'nwResults': nw_results,
            'vectorTime': vec_time,
            'nwTime': nw_time
        })
    except Exception as e:
        print(f"Error in compare_single: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
