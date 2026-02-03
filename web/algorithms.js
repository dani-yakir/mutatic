// Needleman-Wunsch Algorithm Implementation
class NeedlemanWunsch {
    constructor(matchScore = 2, mismatchPenalty = -1, gapPenalty = -1) {
        this.matchScore = matchScore;
        this.mismatchPenalty = mismatchPenalty;
        this.gapPenalty = gapPenalty;
    }

    // Calculate alignment score between two sequences
    align(seq1, seq2) {
        const m = seq1.length;
        const n = seq2.length;
        
        // Create scoring matrix
        const score = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0));
        
        // Initialize first row and column
        for (let i = 0; i <= m; i++) {
            score[i][0] = i * this.gapPenalty;
        }
        for (let j = 0; j <= n; j++) {
            score[0][j] = j * this.gapPenalty;
        }
        
        // Fill the matrix
        for (let i = 1; i <= m; i++) {
            for (let j = 1; j <= n; j++) {
                const match = score[i-1][j-1] + (seq1[i-1] === seq2[j-1] ? this.matchScore : this.mismatchPenalty);
                const deleteScore = score[i-1][j] + this.gapPenalty;
                const insertScore = score[i][j-1] + this.gapPenalty;
                
                score[i][j] = Math.max(match, deleteScore, insertScore);
            }
        }
        
        return score[m][n];
    }

    // Convert alignment score to similarity (0-1 range) - matches Python implementation
    getSimilarity(seq1, seq2) {
        // Python uses max length for both bounds
        const maxLen = Math.max(seq1.length, seq2.length);
        const maxPossibleScore = maxLen * this.matchScore;
        const minPossibleScore = maxLen * this.mismatchPenalty;
        
        const score = this.align(seq1, seq2);
        
        // Normalize to 0-1 similarity (higher = more similar)
        const similarity = (score - minPossibleScore) / (maxPossibleScore - minPossibleScore);
        const clampedSimilarity = Math.max(0, Math.min(1, similarity));
        
        return clampedSimilarity;
    }

    // Get the actual alignment with gaps for visualization
    getAlignment(seq1, seq2) {
        const m = seq1.length;
        const n = seq2.length;
        
        // Create scoring matrix
        const score = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0));
        
        // Initialize first row and column
        for (let i = 0; i <= m; i++) {
            score[i][0] = i * this.gapPenalty;
        }
        for (let j = 0; j <= n; j++) {
            score[0][j] = j * this.gapPenalty;
        }
        
        // Fill the matrix
        for (let i = 1; i <= m; i++) {
            for (let j = 1; j <= n; j++) {
                const match = score[i-1][j-1] + (seq1[i-1] === seq2[j-1] ? this.matchScore : this.mismatchPenalty);
                const deleteScore = score[i-1][j] + this.gapPenalty;
                const insertScore = score[i][j-1] + this.gapPenalty;
                
                score[i][j] = Math.max(match, deleteScore, insertScore);
            }
        }
        
        // Traceback to get alignment
        let aligned1 = '';
        let aligned2 = '';
        let i = m;
        let j = n;
        
        while (i > 0 || j > 0) {
            if (i > 0 && j > 0 && score[i][j] === score[i-1][j-1] + (seq1[i-1] === seq2[j-1] ? this.matchScore : this.mismatchPenalty)) {
                // Match or mismatch
                aligned1 = seq1[i-1] + aligned1;
                aligned2 = seq2[j-1] + aligned2;
                i--;
                j--;
            } else if (i > 0 && score[i][j] === score[i-1][j] + this.gapPenalty) {
                // Deletion (gap in seq2)
                aligned1 = seq1[i-1] + aligned1;
                aligned2 = '-' + aligned2;
                i--;
            } else {
                // Insertion (gap in seq1)
                aligned1 = '-' + aligned1;
                aligned2 = seq2[j-1] + aligned2;
                j--;
            }
        }
        
        return {
            seq1: aligned1,
            seq2: aligned2,
            score: score[m][n]
        };
    }
}

// Vector KNN Implementation
class VectorKNN {
    constructor() {
        this.sequences = [];
        this.embeddings = [];
    }

    // Load sequence data
    loadData(sequenceData) {
        this.sequences = sequenceData.sequences;
        this.embeddings = this.sequences.map(seq => seq.embedding);
    }

    // Calculate Euclidean distance between two vectors
    euclideanDistance(vec1, vec2) {
        let sum = 0;
        for (let i = 0; i < vec1.length; i++) {
            const diff = vec1[i] - vec2[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    // Calculate cosine similarity between two vectors
    cosineSimilarity(vec1, vec2) {
        let dotProduct = 0;
        let norm1 = 0;
        let norm2 = 0;
        
        for (let i = 0; i < vec1.length; i++) {
            dotProduct += vec1[i] * vec2[i];
            norm1 += vec1[i] * vec1[i];
            norm2 += vec2[i] * vec2[i];
        }
        
        norm1 = Math.sqrt(norm1);
        norm2 = Math.sqrt(norm2);
        
        if (norm1 === 0 || norm2 === 0) return 0;
        return dotProduct / (norm1 * norm2);
    }

    // Find k nearest neighbors using cosine similarity (matches Python implementation)
    findKNN(queryIndex, k = 10) {
        const queryVector = this.embeddings[queryIndex];
        const similarities = [];
        
        for (let i = 0; i < this.embeddings.length; i++) {
            if (i === queryIndex) continue;
            
            // Use cosine similarity (higher = more similar)
            const similarity = this.cosineSimilarity(queryVector, this.embeddings[i]);
            
            similarities.push({
                index: i,
                id: this.sequences[i].id,
                similarity: similarity
            });
        }
        
        // Sort by similarity (descending - highest first)
        similarities.sort((a, b) => b.similarity - a.similarity);
        return similarities.slice(0, k);
    }
}

// Performance Monitor
class PerformanceMonitor {
    constructor() {
        this.startTime = 0;
        this.elapsedTime = 0;
        this.isRunning = false;
        this.updateInterval = null;
    }

    start() {
        this.startTime = performance.now();
        this.isRunning = true;
        this.updateInterval = setInterval(() => {
            this.elapsedTime = (performance.now() - this.startTime) / 1000;
        }, 50);
    }

    stop() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        this.isRunning = false;
        this.elapsedTime = (performance.now() - this.startTime) / 1000;
        return this.elapsedTime;
    }

    getCurrentTime() {
        return this.elapsedTime;
    }
}

// Utility functions
function formatTime(seconds) {
    if (seconds < 0.001) return '0.00s';
    if (seconds < 1) return (seconds * 1000).toFixed(2) + 'ms';
    return seconds.toFixed(3) + 's';
}

function formatDistance(distance) {
    return distance.toFixed(4);
}

// Export for use in main app
window.Algorithms = {
    NeedlemanWunsch,
    VectorKNN,
    PerformanceMonitor,
    formatTime,
    formatDistance
};
