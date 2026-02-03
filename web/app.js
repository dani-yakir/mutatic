// Main Application Controller
class GenomeComparisonApp {
    constructor() {
        this.sequenceData = null;
        this.selectedSequence = null;
        this.kValue = 10;
        this.nwAlgorithm = new Algorithms.NeedlemanWunsch();
        this.vectorKNN = new Algorithms.VectorKNN();
        this.nwMonitor = new Algorithms.PerformanceMonitor();
        this.vectorMonitor = new Algorithms.PerformanceMonitor();
        
        this.init();
    }

    async init() {
        await this.loadSequenceData();
        this.setupEventListeners();
        this.createFloatingCards();
        this.showScreen('selection-screen');
    }

    async loadSequenceData() {
        try {
            const response = await fetch('data/sequences.json');
            this.sequenceData = await response.json();
            this.vectorKNN.loadData(this.sequenceData);
            console.log(`Loaded ${this.sequenceData.metadata.total_sequences} sequences`);
        } catch (error) {
            console.error('Error loading sequence data:', error);
            this.showError('Failed to load sequence data');
        }
    }

    setupEventListeners() {
        // Sequence card selection
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('sequence-card')) {
                this.selectSequence(e.target);
            }
        });

        // Compare button
        document.getElementById('compare-btn').addEventListener('click', () => {
            this.startComparison();
        });

        // Restart button
        document.getElementById('restart-btn').addEventListener('click', () => {
            this.restart();
        });

        // K value input
        document.getElementById('k-value').addEventListener('change', (e) => {
            this.kValue = parseInt(e.target.value) || 10;
        });
    }

    createFloatingCards() {
        const container = document.getElementById('sequence-cards');
        container.innerHTML = '';

        // Create floating cards for a subset of sequences (for performance)
        const displayCount = Math.min(50, this.sequenceData.sequences.length);
        const indices = this.getRandomIndices(displayCount, this.sequenceData.sequences.length);

        indices.forEach((index, i) => {
            const sequence = this.sequenceData.sequences[index];
            const card = document.createElement('div');
            card.className = 'sequence-card';
            card.textContent = sequence.id;
            card.dataset.index = index;
            
            // Random floating animation
            this.animateCard(card, i);
            
            container.appendChild(card);
        });
    }

    getRandomIndices(count, max) {
        const indices = [];
        while (indices.length < count) {
            const index = Math.floor(Math.random() * max);
            if (!indices.includes(index)) {
                indices.push(index);
            }
        }
        return indices;
    }

    animateCard(card, index) {
        const container = document.getElementById('sequence-cards');
        const containerRect = container.getBoundingClientRect();
        
        // Random position within container
        const x = Math.random() * (containerRect.width - 150);
        const y = Math.random() * (containerRect.height - 50);
        
        card.style.left = x + 'px';
        card.style.top = y + 'px';
        
        // Random animation delay and duration
        card.style.animationDelay = (Math.random() * 5) + 's';
        card.style.animationDuration = (8 + Math.random() * 4) + 's';
    }

    selectSequence(card) {
        // Remove previous selection
        document.querySelectorAll('.sequence-card.selected').forEach(c => {
            c.classList.remove('selected');
        });

        // Select new card
        card.classList.add('selected');
        this.selectedSequence = {
            index: parseInt(card.dataset.index),
            id: card.textContent,
            data: this.sequenceData.sequences[parseInt(card.dataset.index)]
        };

        // Show selection panel
        const panel = document.getElementById('selection-panel');
        panel.classList.remove('hidden');
        document.getElementById('selected-id').textContent = this.selectedSequence.id;
    }

    async startComparison() {
        if (!this.selectedSequence) {
            this.showError('Please select a sequence first');
            return;
        }

        // Immediately show comparison screen
        this.showScreen('comparison-screen');
        
        // Run comparison asynchronously (don't await here to avoid freezing)
        this.runComparison();
    }

    async runComparison() {
        const queryIndex = this.selectedSequence.index;
        const querySequence = this.sequenceData.sequences[queryIndex].sequence;

        // Reset progress bars
        this.resetProgressBars();

        // Start both algorithms
        const vectorPromise = this.runVectorKNN(queryIndex);
        const nwPromise = this.runNeedlemanWunsch(querySequence, queryIndex);

        // Wait for both to complete
        const [vectorResults, nwResults] = await Promise.all([vectorPromise, nwPromise]);

        // Store results for later display
        this.comparisonResults = { 
            vectorResults, 
            nwResults,
            vectorTime: this.vectorMonitor.elapsedTime,
            nwTime: this.nwMonitor.elapsedTime
        };
        
        // Show "View Results" button instead of auto-transitioning
        this.showViewResultsButton();
    }

    async runVectorKNN(queryIndex) {
        const statusEl = document.getElementById('vector-status');
        const progressEl = document.getElementById('vector-progress');
        const timeEl = document.getElementById('vector-time');

        statusEl.textContent = 'Running Vector KNN...';
        this.vectorMonitor.start();

        // Update progress in real-time
        const progressInterval = setInterval(() => {
            const progress = Math.min(95, (this.vectorMonitor.getCurrentTime() / 0.5) * 100);
            progressEl.style.width = progress + '%';
            timeEl.textContent = Algorithms.formatTime(this.vectorMonitor.getCurrentTime());
        }, 50);

        try {
            // Use setTimeout to allow UI to update
            const results = await new Promise((resolve) => {
                setTimeout(() => {
                    const res = this.vectorKNN.findKNN(queryIndex, this.kValue);
                    resolve(res);
                }, 10);
            });
            
            clearInterval(progressInterval);
            
            progressEl.style.width = '100%';
            const totalTime = this.vectorMonitor.stop();
            timeEl.textContent = Algorithms.formatTime(totalTime);
            statusEl.textContent = `Completed in ${Algorithms.formatTime(totalTime)}`;
            
            return results;
        } catch (error) {
            clearInterval(progressInterval);
            statusEl.textContent = 'Error!';
            throw error;
        }
    }

    async runNeedlemanWunsch(querySequence, queryIndex) {
        const statusEl = document.getElementById('nw-status');
        const progressEl = document.getElementById('nw-progress');
        const timeEl = document.getElementById('nw-time');

        statusEl.textContent = 'Running Needleman-Wunsch...';
        this.nwMonitor.start();

        // Calculate progress based on sequences processed
        const totalSequences = this.sequenceData.sequences.length;
        let processedCount = 0;

        const progressInterval = setInterval(() => {
            const progress = Math.min(95, (processedCount / totalSequences) * 100);
            progressEl.style.width = progress + '%';
            timeEl.textContent = Algorithms.formatTime(this.nwMonitor.getCurrentTime());
        }, 100);

        try {
            const similarities = [];
            
            // Process in chunks to allow UI updates
            const chunkSize = 50;
            for (let i = 0; i < this.sequenceData.sequences.length; i++) {
                if (i === queryIndex) continue;
                
                const otherSequence = this.sequenceData.sequences[i].sequence;
                const similarity = this.nwAlgorithm.getSimilarity(querySequence, otherSequence);
                
                similarities.push({
                    index: i,
                    id: this.sequenceData.sequences[i].id,
                    similarity: similarity
                });
                
                processedCount++;
                
                // Yield to UI every chunkSize iterations
                if (processedCount % chunkSize === 0) {
                    await new Promise(resolve => setTimeout(resolve, 0));
                }
            }

            clearInterval(progressInterval);
            
            // Sort by similarity (descending - highest first)
            similarities.sort((a, b) => b.similarity - a.similarity);
            const results = similarities.slice(0, this.kValue);
            
            progressEl.style.width = '100%';
            const totalTime = this.nwMonitor.stop();
            timeEl.textContent = Algorithms.formatTime(totalTime);
            statusEl.textContent = `Completed in ${Algorithms.formatTime(totalTime)}`;
            
            return results;
        } catch (error) {
            clearInterval(progressInterval);
            statusEl.textContent = 'Error!';
            throw error;
        }
    }

    getSequenceById(sequenceId) {
        const seqData = this.sequenceData.sequences.find(s => s.id === sequenceId);
        return seqData ? seqData.sequence : '';
    }

    showViewResultsButton() {
        // Show a button to manually transition to results
        const comparisonScreen = document.getElementById('comparison-screen');
        let viewResultsBtn = document.getElementById('view-results-btn');
        
        if (!viewResultsBtn) {
            viewResultsBtn = document.createElement('button');
            viewResultsBtn.id = 'view-results-btn';
            viewResultsBtn.className = 'view-results-btn';
            viewResultsBtn.textContent = 'VIEW RESULTS';
            viewResultsBtn.addEventListener('click', () => {
                this.showResultsScreen();
            });
            comparisonScreen.querySelector('.comparison-container').appendChild(viewResultsBtn);
        }
        
        viewResultsBtn.style.display = 'block';
    }
    
    showResultsScreen() {
        this.showScreen('results-screen');
        this.displayResults(
            this.comparisonResults.vectorResults, 
            this.comparisonResults.nwResults,
            this.comparisonResults.vectorTime,
            this.comparisonResults.nwTime
        );
    }

    displayResults(vectorResults, nwResults, vectorTime, nwTime) {
        // Display query sequence
        const querySeq = this.sequenceData.sequences[this.selectedSequence.index];
        document.getElementById('query-id').textContent = this.selectedSequence.id;
        document.getElementById('query-sequence-data').textContent = querySeq.sequence;

        // Display runtimes
        document.getElementById('vector-runtime').textContent = `Runtime: ${Algorithms.formatTime(vectorTime)}`;
        document.getElementById('nw-runtime').textContent = `Runtime: ${Algorithms.formatTime(nwTime)}`;

        // Find common results
        const vectorIds = new Set(vectorResults.map(r => r.id));
        const commonIds = nwResults.filter(r => vectorIds.has(r.id)).map(r => r.id);

        // Display vector results
        const vectorContainer = document.getElementById('vector-results');
        vectorContainer.innerHTML = '';
        vectorResults.forEach((result, idx) => {
            const item = this.createResultItem(result, commonIds.includes(result.id), idx + 1, querySeq.sequence, 'vector');
            vectorContainer.appendChild(item);
        });

        // Display NW results
        const nwContainer = document.getElementById('nw-results');
        nwContainer.innerHTML = '';
        nwResults.forEach((result, idx) => {
            const item = this.createResultItem(result, commonIds.includes(result.id), idx + 1, querySeq.sequence, 'nw');
            nwContainer.appendChild(item);
        });
    }

    createResultItem(result, isCommon, rank, querySequence, algorithmType) {
        const item = document.createElement('div');
        item.className = 'result-item' + (isCommon ? ' common' : '');
        const similarity = result.similarity || 0;
        
        // Get the compared sequence
        const comparedSeq = this.sequenceData.sequences.find(s => s.id === result.id);
        
        if (!comparedSeq) {
            console.error(`Sequence not found for ID: ${result.id}`);
        }
        
        // Create unique IDs by prefixing with algorithm type
        const uniqueQueryId = `${algorithmType}-query-${result.id}`;
        const uniqueComparedId = `${algorithmType}-compared-${result.id}`;
        
        item.innerHTML = `
            <div class="result-header">
                <span class="result-rank">#${rank}</span>
                <span class="result-id">${result.id}</span>
                <span class="result-similarity">${similarity.toFixed(4)}</span>
                <span class="expand-icon">▼</span>
            </div>
            <div class="result-details" style="display: none;">
                <div class="sequence-comparison">
                    <div class="sequence-label">Query Sequence:</div>
                    <div class="sequence-display" id="${uniqueQueryId}"></div>
                    <div class="sequence-label">Compared Sequence (${result.id}):</div>
                    <div class="sequence-display" id="${uniqueComparedId}"></div>
                </div>
            </div>
        `;
        
        // Add click handler to expand/collapse
        const header = item.querySelector('.result-header');
        const details = item.querySelector('.result-details');
        const icon = item.querySelector('.expand-icon');
        
        header.style.cursor = 'pointer';
        header.addEventListener('click', () => {
            const isExpanded = details.style.display !== 'none';
            
            if (isExpanded) {
                details.style.display = 'none';
                icon.textContent = '▼';
                item.classList.remove('expanded');
            } else {
                details.style.display = 'block';
                icon.textContent = '▲';
                item.classList.add('expanded');
                
                // Perform alignment and display diff
                if (comparedSeq && comparedSeq.sequence) {
                    this.displaySequenceDiff(querySequence, comparedSeq.sequence, uniqueQueryId, uniqueComparedId);
                } else {
                    console.error(`No sequence data for ${result.id}`);
                    document.getElementById(uniqueQueryId).textContent = 'Sequence data not available';
                    document.getElementById(uniqueComparedId).textContent = 'Sequence data not available';
                }
            }
        });
        
        return item;
    }

    showScreen(screenId) {
        document.querySelectorAll('.screen').forEach(screen => {
            screen.classList.remove('active');
        });
        document.getElementById(screenId).classList.add('active');
    }

    resetProgressBars() {
        document.querySelectorAll('.progress-bar').forEach(bar => {
            bar.style.width = '0%';
        });
        document.querySelectorAll('.progress-time').forEach(time => {
            time.textContent = '0.00s';
        });
        document.querySelectorAll('.status').forEach(status => {
            status.textContent = 'Ready...';
        });
    }

    restart() {
        this.selectedSequence = null;
        this.comparisonResults = null;
        document.getElementById('selection-panel').classList.add('hidden');
        document.querySelectorAll('.sequence-card.selected').forEach(card => {
            card.classList.remove('selected');
        });
        
        // Hide view results button if it exists
        const viewResultsBtn = document.getElementById('view-results-btn');
        if (viewResultsBtn) {
            viewResultsBtn.style.display = 'none';
        }
        
        this.showScreen('selection-screen');
    }

    displaySequenceDiff(querySeq, comparedSeq, queryDisplayId, comparedDisplayId) {
        const queryDisplay = document.getElementById(queryDisplayId);
        const comparedDisplay = document.getElementById(comparedDisplayId);
        
        if (!queryDisplay || !comparedDisplay) {
            console.error(`Display elements not found:`, queryDisplayId, comparedDisplayId);
            return;
        }
        
        if (!querySeq || !comparedSeq) {
            queryDisplay.textContent = 'Query sequence not available';
            comparedDisplay.textContent = 'Compared sequence not available';
            console.error('Missing sequence data for', queryDisplayId);
            return;
        }
        
        try {
            // Perform Needleman-Wunsch alignment to get the actual alignment
            const alignment = this.nwAlgorithm.getAlignment(querySeq, comparedSeq);
            
            if (!alignment || !alignment.seq1 || !alignment.seq2) {
                throw new Error('Alignment failed');
            }
            
            // Display aligned sequences with diff highlighting
            const queryHTML = this.formatAlignedSequence(alignment.seq1, alignment.seq2, true);
            const comparedHTML = this.formatAlignedSequence(alignment.seq2, alignment.seq1, false);
            
            queryDisplay.innerHTML = queryHTML;
            comparedDisplay.innerHTML = comparedHTML;
            
            // Force parent container to stay visible
            const resultDetails = queryDisplay.closest('.result-details');
            if (resultDetails) {
                resultDetails.style.display = 'block';
            }
        } catch (error) {
            console.error(`Error aligning sequences:`, error);
            queryDisplay.textContent = querySeq.substring(0, 200) + (querySeq.length > 200 ? '...' : '');
            comparedDisplay.textContent = comparedSeq.substring(0, 200) + (comparedSeq.length > 200 ? '...' : '');
        }
    }
    
    formatAlignedSequence(seq, otherSeq, isQuery) {
        let html = '';
        for (let i = 0; i < seq.length; i++) {
            const char = seq[i];
            const otherChar = otherSeq[i];
            
            if (char === '-') {
                // Gap in this sequence (insertion in other)
                html += `<span class="diff-gap">${char}</span>`;
            } else if (otherChar === '-') {
                // Gap in other sequence (deletion)
                html += `<span class="diff-deletion">${char}</span>`;
            } else if (char !== otherChar) {
                // Mismatch
                html += `<span class="diff-mismatch">${char}</span>`;
            } else {
                // Match
                html += `<span class="diff-match">${char}</span>`;
            }
        }
        return html;
    }

    showError(message) {
        console.error(message);
        // You could add a proper error display here
        alert(message);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new GenomeComparisonApp();
});
