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
        this.lastTestData = null;
    }

    async init() {
        await this.loadSequenceData();
        this.setupEventListeners();
        this.createFloatingCards();
    }

    async loadSequenceData() {
        try {
            const response = await fetch('http://localhost:5000/api/sequences');
            this.sequenceData = await response.json();
            this.vectorKNN.loadData(this.sequenceData);
            console.log(`Loaded ${this.sequenceData.metadata.total_sequences} sequences`);
        } catch (error) {
            console.error('Error loading sequence data:', error);
            this.showError('Failed to load sequence data. Make sure the backend server is running.');
        }
    }

    setupEventListeners() {
        // Helper function to safely add event listener
        const addListener = (id, event, handler) => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener(event, handler);
            }
        };

        // Menu navigation
        addListener('single-comparison-btn', 'click', () => {
            this.showScreen('selection-screen');
        });

        addListener('run-test-btn-menu', 'click', () => {
            this.resetTestRunnerScreen();
            this.showScreen('test-runner-screen');
        });

        addListener('load-test-btn-menu', 'click', () => {
            this.showScreen('test-viewer-screen');
        });

        // Back to menu buttons
        addListener('back-to-menu-btn', 'click', () => {
            this.showScreen('menu-screen');
        });

        addListener('back-to-menu-from-runner', 'click', () => {
            this.showScreen('menu-screen');
        });

        addListener('back-to-menu-from-viewer', 'click', () => {
            this.showScreen('menu-screen');
        });

        // Sequence card selection
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('sequence-card')) {
                this.selectSequence(e.target);
            }
        });

        // Compare button
        addListener('compare-btn', 'click', () => {
            this.startComparison();
        });

        // Restart button
        addListener('restart-btn', 'click', () => {
            this.showScreen('menu-screen');
        });

        // Test runner
        addListener('start-test-btn', 'click', () => {
            this.startStatisticalTest();
        });

        addListener('view-test-results-btn', 'click', () => {
            if (this.lastTestData) {
                this.displayTestInViewer(this.lastTestData);
                this.showScreen('test-viewer-screen');
            }
        });

        // Test viewer
        addListener('load-test-input', 'change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.loadTestFromFile(file);
            }
        });

        // K value input
        const kValueEl = document.getElementById('k-value');
        if (kValueEl) {
            kValueEl.addEventListener('change', (e) => {
                this.kValue = parseInt(e.target.value) || 10;
            });
        }
    }

    createFloatingCards() {
        const container = document.getElementById('sequence-cards');
        if (!container) return; // Container doesn't exist on this page
        
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
        const sequenceId = this.selectedSequence.id;
        console.log('Starting comparison for sequence:', sequenceId);

        // Reset progress bars
        this.resetProgressBars();

        try {
            // Run Vector KNN (fast)
            console.log('Running Vector KNN...');
            const vectorResult = await this.runVectorKNN(sequenceId, this.kValue, true);
            console.log('Vector KNN completed:', vectorResult);
            
            // Run Needleman-Wunsch (slow, with progress)
            console.log('Running Needleman-Wunsch...');
            const nwResult = await this.runNeedlemanWunsch(sequenceId, this.kValue, true);
            console.log('Needleman-Wunsch completed:', nwResult);

            // Store results for later display
            this.comparisonResults = { 
                vectorResults: vectorResult.results, 
                nwResults: nwResult.results,
                vectorTime: vectorResult.time,
                nwTime: nwResult.time
            };
            
            console.log('Comparison complete, showing view results button');
            // Show "View Results" button instead of auto-transitioning
            this.showViewResultsButton();
        } catch (error) {
            console.error('Error running comparison:', error);
            document.getElementById('vector-status').textContent = 'Error!';
            document.getElementById('nw-status').textContent = 'Error!';
            alert('Failed to run comparison: ' + error.message);
        }
    }

    async runVectorKNN(sequenceIdOrIndex, kValue = this.kValue, showProgress = true) {
        // Find the index if a sequence ID was provided
        let queryIndex;
        if (typeof sequenceIdOrIndex === 'string') {
            queryIndex = this.sequenceData.sequences.findIndex(s => s.id === sequenceIdOrIndex);
        } else {
            queryIndex = sequenceIdOrIndex;
        }

        const statusEl = document.getElementById('vector-status');
        const progressEl = document.getElementById('vector-progress');
        const timeEl = document.getElementById('vector-time');

        if (showProgress) {
            statusEl.textContent = 'Running Vector KNN...';
        }
        this.vectorMonitor.start();

        let progressInterval;
        if (showProgress) {
            // Update progress in real-time
            progressInterval = setInterval(() => {
                const progress = Math.min(95, (this.vectorMonitor.getCurrentTime() / 0.5) * 100);
                progressEl.style.width = progress + '%';
                timeEl.textContent = Algorithms.formatTime(this.vectorMonitor.getCurrentTime());
            }, 50);
        }

        try {
            // Use setTimeout to allow UI to update
            const results = await new Promise((resolve) => {
                setTimeout(() => {
                    const res = this.vectorKNN.findKNN(queryIndex, kValue);
                    resolve(res);
                }, 10);
            });
            
            if (progressInterval) clearInterval(progressInterval);
            
            const totalTime = this.vectorMonitor.stop();
            
            if (showProgress) {
                progressEl.style.width = '100%';
                timeEl.textContent = Algorithms.formatTime(totalTime);
                statusEl.textContent = `Completed in ${Algorithms.formatTime(totalTime)}`;
            }
            
            return { results, time: totalTime };
        } catch (error) {
            if (progressInterval) clearInterval(progressInterval);
            if (showProgress) statusEl.textContent = 'Error!';
            throw error;
        }
    }

    async runNeedlemanWunsch(sequenceIdOrIndex, kValue = this.kValue, showProgress = true) {
        // Find the index and sequence if a sequence ID was provided
        let queryIndex, querySequence;
        if (typeof sequenceIdOrIndex === 'string') {
            queryIndex = this.sequenceData.sequences.findIndex(s => s.id === sequenceIdOrIndex);
            querySequence = this.sequenceData.sequences[queryIndex].sequence;
        } else {
            queryIndex = sequenceIdOrIndex;
            querySequence = this.sequenceData.sequences[queryIndex].sequence;
        }

        const statusEl = document.getElementById('nw-status');
        const progressEl = document.getElementById('nw-progress');
        const timeEl = document.getElementById('nw-time');

        if (showProgress) {
            statusEl.textContent = 'Running Needleman-Wunsch...';
        }
        this.nwMonitor.start();

        // Calculate progress based on sequences processed
        const totalSequences = this.sequenceData.sequences.length;
        let processedCount = 0;

        let progressInterval;
        if (showProgress) {
            progressInterval = setInterval(() => {
                const progress = Math.min(95, (processedCount / totalSequences) * 100);
                progressEl.style.width = progress + '%';
                timeEl.textContent = Algorithms.formatTime(this.nwMonitor.getCurrentTime());
            }, 100);
        }

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
            const results = similarities.slice(0, kValue);
            
            if (progressInterval) clearInterval(progressInterval);
            
            const totalTime = this.nwMonitor.stop();
            
            if (showProgress) {
                progressEl.style.width = '100%';
                timeEl.textContent = Algorithms.formatTime(totalTime);
                statusEl.textContent = `Completed in ${Algorithms.formatTime(totalTime)}`;
            }
            
            return { results, time: totalTime };
        } catch (error) {
            if (progressInterval) clearInterval(progressInterval);
            if (showProgress) statusEl.textContent = 'Error!';
            throw error;
        }
    }

    getSequenceById(sequenceId) {
        const seqData = this.sequenceData.sequences.find(s => s.id === sequenceId);
        return seqData ? seqData.sequence : '';
    }

    showViewResultsButton() {
        // Show a button to manually transition to results
        console.log('showViewResultsButton called');
        const viewResultsBtn = document.getElementById('view-results-btn');
        console.log('view-results-btn element:', viewResultsBtn);
        
        if (viewResultsBtn) {
            console.log('Removing hidden class from button');
            viewResultsBtn.classList.remove('hidden');
            console.log('Button classes after removal:', viewResultsBtn.className);
        } else {
            console.error('view-results-btn element not found!');
        }
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
        document.getElementById('query-id').textContent = `Query Sequence: ${this.selectedSequence.id}`;

        // Display runtimes
        document.getElementById('vector-runtime').textContent = `Runtime: ${Algorithms.formatTime(vectorTime)}`;
        document.getElementById('nw-runtime').textContent = `Runtime: ${Algorithms.formatTime(nwTime)}`;

        // Calculate and display agreement
        const agreement = this.calculateAgreement(vectorResults, nwResults);
        document.getElementById('agreement-percentage').textContent = `Agreement: ${agreement.toFixed(1)}%`;

        // Find common results
        const vectorIds = new Set(vectorResults.map(r => r.id));
        const nwIds = new Set(nwResults.map(r => r.id));
        const commonIds = [...vectorIds].filter(id => nwIds.has(id));

        // Display vector results
        const vectorContainer = document.getElementById('vector-results');
        vectorContainer.innerHTML = '';
        vectorResults.forEach((result, idx) => {
            const item = this.createDetailResultItem(result, commonIds.includes(result.id), idx + 1, querySeq, 'vector');
            vectorContainer.appendChild(item);
        });

        // Display NW results
        const nwContainer = document.getElementById('nw-results');
        nwContainer.innerHTML = '';
        nwResults.forEach((result, idx) => {
            const item = this.createDetailResultItem(result, commonIds.includes(result.id), idx + 1, querySeq, 'nw');
            nwContainer.appendChild(item);
        });
    }

    calculateAgreement(vectorResults, nwResults) {
        const vectorIds = new Set(vectorResults.map(r => r.id));
        const nwIds = new Set(nwResults.map(r => r.id));
        
        let commonCount = 0;
        for (const id of vectorIds) {
            if (nwIds.has(id)) {
                commonCount++;
            }
        }
        
        return (commonCount / vectorResults.length) * 100;
    }

    createDetailResultItem(result, isCommon, rank, querySeqData, algorithmType) {
        const item = document.createElement('div');
        item.className = 'detail-result-item' + (isCommon ? ' common' : '');
        const similarity = result.similarity || 0;
        
        // Get the compared sequence
        const comparedSeq = this.sequenceData.sequences.find(s => s.id === result.id);
        
        // Create unique IDs by prefixing with algorithm type
        const uniqueQueryId = `${algorithmType}-detail-query-${result.id}`;
        const uniqueComparedId = `${algorithmType}-detail-compared-${result.id}`;
        
        item.innerHTML = `
            <div class="detail-result-header">
                <span class="detail-result-rank">#${rank}</span>
                <span class="detail-result-id">${result.id}</span>
                <span class="detail-result-similarity">${similarity.toFixed(4)}</span>
                <span class="detail-expand-icon">▼</span>
            </div>
            <div class="detail-result-details" style="display: none;">
                <div class="sequence-comparison">
                    <div class="sequence-label">Query Sequence (${querySeqData.id}):</div>
                    <div class="sequence-display" id="${uniqueQueryId}">Loading...</div>
                    <div class="sequence-label">Compared Sequence (${result.id}):</div>
                    <div class="sequence-display" id="${uniqueComparedId}">Loading...</div>
                </div>
            </div>
        `;
        
        // Add click handler to expand/collapse
        const header = item.querySelector('.detail-result-header');
        const details = item.querySelector('.detail-result-details');
        const icon = item.querySelector('.detail-expand-icon');
        
        header.style.cursor = 'pointer';
        header.addEventListener('click', () => {
            const isExpanded = details.style.display !== 'none';
            
            if (isExpanded) {
                details.style.display = 'none';
                icon.textContent = '▼';
            } else {
                details.style.display = 'block';
                icon.textContent = '▲';
                
                // Perform alignment and display diff
                if (comparedSeq && comparedSeq.sequence) {
                    this.displaySequenceDiff(querySeqData.sequence, comparedSeq.sequence, uniqueQueryId, uniqueComparedId);
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

    resetTestRunnerScreen() {
        // Reset test runner to initial state
        const configSection = document.querySelector('.test-config');
        if (configSection) {
            configSection.style.display = 'block';
        }
        document.getElementById('test-progress').classList.add('hidden');
        document.getElementById('test-complete').classList.add('hidden');
        document.getElementById('test-progress-bar').style.width = '0%';
    }

    async startStatisticalTest() {
        const numSequences = parseInt(document.getElementById('test-num-sequences').value) || 10;
        const kValue = parseInt(document.getElementById('test-k-value').value) || 10;

        // Show progress section
        document.querySelector('.test-config').style.display = 'none';
        document.getElementById('test-progress').classList.remove('hidden');
        document.getElementById('test-complete').classList.add('hidden');

        // Update UI
        document.getElementById('total-tests').textContent = numSequences;
        document.getElementById('current-test').textContent = '0';
        document.getElementById('current-sequence-id').textContent = 'Starting...';
        document.getElementById('test-status').textContent = 'Running algorithms in parallel...';
        
        // Reset progress bars
        document.getElementById('vec-progress-bar').style.width = '0%';
        document.getElementById('nw-progress-bar').style.width = '0%';
        document.getElementById('vec-status').textContent = 'Ready...';
        document.getElementById('nw-status').textContent = 'Ready...';

        try {
            // Select random sequences
            const sequences = this.sequenceData.sequences;
            const randomIndices = [];
            const usedIndices = new Set();
            while (randomIndices.length < Math.min(numSequences, sequences.length)) {
                const idx = Math.floor(Math.random() * sequences.length);
                if (!usedIndices.has(idx)) {
                    randomIndices.push(idx);
                    usedIndices.add(idx);
                }
            }

            const results = {};
            
            // Process each sequence
            for (let i = 0; i < randomIndices.length; i++) {
                const idx = randomIndices[i];
                const querySeq = sequences[idx];
                
                document.getElementById('current-test').textContent = i + 1;
                document.getElementById('current-sequence-id').textContent = querySeq.id;
                
                // Run Vector KNN and Needleman-Wunsch in parallel
                const [vecResult, nwResult] = await Promise.all([
                    this.runVectorKNNForTest(idx, kValue, i, randomIndices.length),
                    this.runNeedlemanWunschForTest(idx, kValue, i, randomIndices.length)
                ]);
                
                // Store results
                results[querySeq.id] = {
                    vec_knn: {
                        runtime: vecResult.time,
                        returned_knn: vecResult.results.map(r => ({
                            id: r.id,
                            similarity: r.similarity
                        }))
                    },
                    nw_knn: {
                        runtime: nwResult.time,
                        returned_knn: nwResult.results.map(r => ({
                            id: r.id,
                            similarity: r.similarity
                        }))
                    }
                };
            }

            // Create test data in proper JSON format
            const testData = {
                header: {
                    timestamp: new Date().toISOString(),
                    num_sequences: numSequences,
                    requested_knn: kValue
                },
                seqs: results
            };

            document.getElementById('test-status').textContent = 'Test complete! Saving results...';

            // Save to server
            const saveResponse = await fetch('http://localhost:5000/api/save-test', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(testData)
            });

            if (!saveResponse.ok) {
                console.error('Failed to save to server:', await saveResponse.text());
                document.getElementById('test-status').textContent = 'Test complete! (Warning: Failed to save to server)';
            } else {
                const saveResult = await saveResponse.json();
                console.log('Saved to server:', saveResult.filename);
                document.getElementById('test-status').textContent = `Test complete! Saved as ${saveResult.filename}`;
            }

            // Show completion
            document.getElementById('test-progress').classList.add('hidden');
            document.getElementById('test-complete').classList.remove('hidden');
        } catch (error) {
            console.error('Error running test:', error);
            document.getElementById('test-status').textContent = 'Error: ' + error.message;
            alert('Failed to run test: ' + error.message);
        }
    }

    async runVectorKNNForTest(queryIndex, kValue, currentTest, totalTests) {
        const startTime = performance.now();
        const querySeq = this.sequenceData.sequences[queryIndex];
        const queryEmbedding = querySeq.embedding;
        
        document.getElementById('vec-status').textContent = `Processing ${currentTest + 1}/${totalTests}...`;
        
        const similarities = [];
        for (let i = 0; i < this.sequenceData.sequences.length; i++) {
            if (i !== queryIndex) {
                const seq = this.sequenceData.sequences[i];
                const sim = this.vectorKNN.cosineSimilarity(queryEmbedding, seq.embedding);
                similarities.push({
                    id: seq.id,
                    similarity: sim,
                    index: i
                });
            }
        }
        
        similarities.sort((a, b) => b.similarity - a.similarity);
        const results = similarities.slice(0, kValue);
        const time = (performance.now() - startTime) / 1000;
        
        // Update progress bar
        const progress = ((currentTest + 1) / totalTests) * 100;
        document.getElementById('vec-progress-bar').style.width = progress + '%';
        document.getElementById('vec-status').textContent = `Complete: ${time.toFixed(3)}s`;
        
        return { results, time };
    }

    async runNeedlemanWunschForTest(queryIndex, kValue, currentTest, totalTests) {
        const startTime = performance.now();
        const querySeq = this.sequenceData.sequences[queryIndex];
        const querySequence = querySeq.sequence;
        
        document.getElementById('nw-status').textContent = `Processing ${currentTest + 1}/${totalTests}...`;
        
        const similarities = [];
        for (let i = 0; i < this.sequenceData.sequences.length; i++) {
            if (i !== queryIndex) {
                const seq = this.sequenceData.sequences[i];
                const sim = this.nwAlgorithm.getSimilarity(querySequence, seq.sequence);
                similarities.push({
                    id: seq.id,
                    similarity: sim,
                    index: i
                });
            }
            
            // Update progress periodically
            if (i % 100 === 0) {
                const innerProgress = (i / this.sequenceData.sequences.length) * 100;
                const totalProgress = ((currentTest + innerProgress / 100) / totalTests) * 100;
                document.getElementById('nw-progress-bar').style.width = totalProgress + '%';
                await new Promise(resolve => setTimeout(resolve, 0)); // Allow UI update
            }
        }
        
        similarities.sort((a, b) => b.similarity - a.similarity);
        const results = similarities.slice(0, kValue);
        const time = (performance.now() - startTime) / 1000;
        
        // Update progress bar
        const progress = ((currentTest + 1) / totalTests) * 100;
        document.getElementById('nw-progress-bar').style.width = progress + '%';
        document.getElementById('nw-status').textContent = `Complete: ${time.toFixed(3)}s`;
        
        return { results, time };
    }


    saveCompleteTestData(testResults, numSequences, kValue) {
        const timestamp = new Date().toISOString();
        
        // Build sequences object with structure: {seq_id: {vec_knn: {...}, nw_knn: {...}}}
        const sequences = {};
        testResults.forEach(result => {
            sequences[result.sequenceId] = {
                vec_knn: {
                    runtime: result.vectorTime,
                    returned_knn: result.vectorKNN.map(r => ({
                        id: r.id,
                        similarity: r.similarity
                    }))
                },
                nw_knn: {
                    runtime: result.nwTime,
                    returned_knn: result.nwKNN.map(r => ({
                        id: r.id,
                        similarity: r.similarity
                    }))
                }
            };
        });

        const exportData = {
            header: {
                timestamp: timestamp,
                num_sequences: numSequences,
                requested_knn: kValue
            },
            seqs: sequences
        };

        // Create downloadable JSON file
        const dataStr = JSON.stringify(exportData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        
        const link = document.createElement('a');
        link.href = url;
        const filename = `genome_test_${timestamp.replace(/[:.]/g, '-')}.json`;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);

        console.log(`Complete test data saved to ${filename}`);
        
        // Return in format compatible with viewer (convert back)
        return {
            metadata: {
                timestamp: timestamp,
                numSequencesTested: numSequences,
                kValue: kValue
            },
            testResults: testResults
        };
    }

    loadTestFromFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const data = JSON.parse(e.target.result);
                
                // Convert new JSON structure to internal format
                const convertedData = this.convertLoadedData(data);
                this.displayTestInViewer(convertedData);
                console.log('Test data loaded successfully');
            } catch (error) {
                console.error('Error loading test data:', error);
                alert('Failed to load test data. Please check the file format.');
            }
        };
        reader.readAsText(file);
    }

    convertLoadedData(data) {
        // Convert from new structure {header: {...}, seqs: {...}} to internal format
        const testResults = [];
        
        Object.keys(data.seqs).forEach(sequenceId => {
            const seqData = data.seqs[sequenceId];
            
            // Calculate agreement
            const vecIds = new Set(seqData.vec_knn.returned_knn.map(r => r.id));
            const nwIds = new Set(seqData.nw_knn.returned_knn.map(r => r.id));
            let commonCount = 0;
            for (const id of vecIds) {
                if (nwIds.has(id)) commonCount++;
            }
            const agreement = (commonCount / seqData.vec_knn.returned_knn.length) * 100;
            
            testResults.push({
                sequenceId: sequenceId,
                sequenceIndex: 0, // Not stored in new format
                agreement: agreement,
                vectorTime: seqData.vec_knn.runtime,
                nwTime: seqData.nw_knn.runtime,
                vectorKNN: seqData.vec_knn.returned_knn,
                nwKNN: seqData.nw_knn.returned_knn
            });
        });
        
        // Calculate distribution
        const distribution = new Array(10).fill(0);
        testResults.forEach(result => {
            const bucket = Math.min(9, Math.floor(result.agreement / 10));
            distribution[bucket]++;
        });
        
        const avgAgreement = testResults.reduce((sum, r) => sum + r.agreement, 0) / testResults.length;
        
        return {
            metadata: {
                timestamp: data.header.timestamp,
                numSequencesTested: data.header.num_sequences,
                kValue: data.header.requested_knn,
                averageAgreement: avgAgreement
            },
            distribution: distribution,
            testResults: testResults
        };
    }

    displayTestInViewer(testData) {
        // Show metadata
        document.getElementById('test-metadata').classList.remove('hidden');
        document.getElementById('test-data-display').classList.remove('hidden');

        const date = new Date(testData.metadata.timestamp);
        document.getElementById('test-date').textContent = date.toLocaleString();
        document.getElementById('test-num-sequences-display').textContent = testData.metadata.numSequencesTested;
        document.getElementById('test-k-value-display').textContent = testData.metadata.kValue;
        document.getElementById('test-avg-agreement').textContent = `${testData.metadata.averageAgreement.toFixed(1)}%`;

        // Display distribution chart
        this.displayDistributionChart(testData.distribution, 'viewer-distribution-bars');

        // Display individual results
        const resultsContainer = document.getElementById('viewer-individual-results');
        resultsContainer.innerHTML = '';

        testData.testResults.forEach(result => {
            const card = document.createElement('div');
            card.className = 'test-result-card';

            card.innerHTML = `
                <div class="test-card-id">${result.sequenceId}</div>
                <div class="test-card-agreement">${result.agreement.toFixed(1)}% Agreement</div>
                <div style="font-size: 0.9rem; margin-top: 10px; opacity: 0.8;">
                    Vector KNN: ${result.vectorTime.toFixed(3)}s<br>
                    NW: ${result.nwTime.toFixed(3)}s
                </div>
            `;

            card.addEventListener('click', () => {
                this.showDetailedComparison(result, testData.metadata.kValue);
            });

            resultsContainer.appendChild(card);
        });
    }

    displayDistributionChart(distribution, containerId) {
        const maxCount = Math.max(...distribution, 1);
        const barsContainer = document.getElementById(containerId);
        barsContainer.innerHTML = '';

        distribution.forEach((count, idx) => {
            const barContainer = document.createElement('div');
            barContainer.className = 'bar-container';

            const barCount = document.createElement('div');
            barCount.className = 'bar-count';
            barCount.textContent = count;

            const bar = document.createElement('div');
            bar.className = 'bar';
            
            const maxHeight = 180;
            const height = count === 0 ? 0 : Math.max(20, (count / maxCount) * maxHeight);
            bar.style.height = `${height}px`;

            const label = document.createElement('div');
            label.className = 'bar-label';
            label.textContent = `${idx * 10}-${(idx + 1) * 10}%`;

            barContainer.appendChild(barCount);
            barContainer.appendChild(bar);
            barContainer.appendChild(label);
            barsContainer.appendChild(barContainer);
        });
    }

    showDetailedComparison(testResult, kValue) {
        // Set up for displaying detailed comparison
        this.selectedSequence = {
            id: testResult.sequenceId,
            index: testResult.sequenceIndex
        };

        // Convert back to expected format
        this.comparisonResults = {
            vectorResults: testResult.vectorKNN,
            nwResults: testResult.nwKNN,
            vectorTime: testResult.vectorTime,
            nwTime: testResult.nwTime
        };

        this.kValue = kValue;
        this.showResultsScreen();
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new GenomeComparisonApp();
});
