# Genome Sequence Comparison - Backend Server

Python Flask backend for handling genome sequence comparisons and test result storage.

## Setup

1. Install Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

## Running the Server

Start the Flask backend server:
```bash
python app.py
```

The server will run on `http://localhost:5000`

## API Endpoints

### GET /api/sequences
Returns all sequence data with embeddings.

### POST /api/run-test
Run a statistical test on random sequences.

**Request body:**
```json
{
  "num_sequences": 10,
  "k_value": 10
}
```

**Response:**
```json
{
  "header": {
    "timestamp": "2026-02-06T14:28:00.000Z",
    "num_sequences": 10,
    "requested_knn": 10
  },
  "seqs": {
    "sequence_id": {
      "vec_knn": {
        "runtime": 0.123,
        "returned_knn": [...]
      },
      "nw_knn": {
        "runtime": 2.456,
        "returned_knn": [...]
      }
    }
  }
}
```

### POST /api/save-test
Save test results to server storage.

### GET /api/list-tests
List all saved test results.

### GET /api/load-test/<filename>
Load a specific test result file.

### POST /api/compare-single
Run comparison for a single sequence.

**Request body:**
```json
{
  "sequence_id": "LC889810.1",
  "k_value": 10
}
```

## Test Results Storage

Test results are automatically saved in `backend/test_results/` directory.

## Frontend Integration

The frontend automatically connects to the backend at `http://localhost:5000`. Make sure the backend is running before accessing the web interface.
