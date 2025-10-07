// app.js
class MovieRecommendationApp {
    constructor() {
        this.interactions = [];
        this.items = new Map();
        this.users = new Map();
        this.userRatings = new Map();
        this.userTopRated = new Map();
        
        this.userIndexMap = new Map();
        this.itemIndexMap = new Map();
        this.reverseUserIndex = new Map();
        this.reverseItemIndex = new Map();
        
        this.simpleModel = null;
        this.deepModel = null;
        
        this.isDataLoaded = false;
        this.isTraining = false;
        
        this.lossHistory = { simple: [], deep: [] };
        this.currentEpoch = 0;
        
        this.initEventListeners();
        this.updateStatus('Ready to load data. Choose a loading method above.', 'status');
    }

    initEventListeners() {
        document.getElementById('loadData').addEventListener('click', () => this.loadDataFromFiles());
        document.getElementById('loadSampleData').addEventListener('click', () => this.loadSampleData());
        document.getElementById('uploadData').addEventListener('click', () => this.loadFromUpload());
        document.getElementById('trainModel').addEventListener('click', () => this.trainModel());
        document.getElementById('testModel').addEventListener('click', () => this.testModel());
    }

    updateStatus(message, className = 'status') {
        const statusEl = document.getElementById('status');
        statusEl.textContent = message;
        statusEl.className = `status ${className}`;
    }

    updateProgress(percent, message = '') {
        const progressBar = document.getElementById('trainingProgress');
        const progressFill = document.getElementById('progressFill');
        
        if (percent > 0) {
            progressBar.style.display = 'block';
            progressFill.style.width = `${percent}%`;
        } else {
            progressBar.style.display = 'none';
        }
        
        if (message) {
            this.updateStatus(message, 'loading');
        }
    }

    async tryMultiplePaths() {
        const paths = [
            'data/u.data',
            'data/adata',
            './data/u.data',
            './data/adata',
            'u.data',
            'adata'
        ];
        
        for (const path of paths) {
            try {
                const response = await fetch(path);
                if (response.ok) {
                    console.log(`Found file at: ${path}`);
                    return { path, data: await response.text() };
                }
            } catch (error) {
                console.log(`Failed to load from ${path}`);
            }
        }
        return null;
    }

    async loadDataFromFiles() {
        try {
            this.updateStatus('Searching for data files...', 'loading');
            
            // Try multiple possible file paths and names
            const interactionsResult = await this.tryMultiplePaths();
            if (!interactionsResult) {
                throw new Error('Could not find u.data or adata file. Tried: data/u.data, data/adata, ./data/u.data, ./data/adata, u.data, adata');
            }

            // Try to find items file
            const itemsPaths = [
                'data/u.item',
                'data/utkem', 
                './data/u.item',
                './data/utkem',
                'u.item',
                'utkem'
            ];
            
            let itemsResult = null;
            for (const path of itemsPaths) {
                try {
                    const response = await fetch(path);
                    if (response.ok) {
                        console.log(`Found file at: ${path}`);
                        itemsResult = { path, data: await response.text() };
                        break;
                    }
                } catch (error) {
                    continue;
                }
            }

            if (!itemsResult) {
                throw new Error('Could not find u.item or utkem file. Tried: data/u.item, data/utkem, ./data/u.item, ./data/utkem, u.item, utkem');
            }

            this.updateStatus(`Found files: ${interactionsResult.path} and ${itemsResult.path}\nParsing data...`, 'loading');
            
            this.parseInteractions(interactionsResult.data);
            this.parseItems(itemsResult.data);
            this.buildUserItemMappings();
            this.calculateUserTopRated();
            
            this.isDataLoaded = true;
            document.getElementById('trainModel').disabled = false;
            this.updateStatus(`✅ Data loaded successfully!\n- Users: ${this.users.size}\n- Items: ${this.items.size}\n- Interactions: ${this.interactions.length}\n- Files: ${interactionsResult.path}, ${itemsResult.path}`, 'success');
            
        } catch (error) {
            this.updateStatus(`❌ Error: ${error.message}\n\nTry using "Load Sample Data" or upload files manually.`, 'error');
            console.error('Data loading error:', error);
        }
    }

    async loadFromUpload() {
        try {
            const uDataFile = document.getElementById('uDataFile').files[0];
            const uItemFile = document.getElementById('uItemFile').files[0];
            
            if (!uDataFile || !uItemFile) {
                throw new Error('Please select both u.data and u.item files');
            }
            
            this.updateStatus('Reading uploaded files...', 'loading');
            
            const interactionsText = await this.readFileAsText(uDataFile);
            const itemsText = await this.readFileAsText(uItemFile);
            
            this.parseInteractions(interactionsText);
            this.parseItems(itemsText);
            this.buildUserItemMappings();
            this.calculateUserTopRated();
            
            this.isDataLoaded = true;
            document.getElementById('trainModel').disabled = false;
            this.updateStatus(`✅ Data loaded successfully from uploaded files!\n- Users: ${this.users.size}\n- Items: ${this.items.size}\n- Interactions: ${this.interactions.length}`, 'success');
            
        } catch (error) {
            this.updateStatus(`❌ Upload error: ${error.message}`, 'error');
            console.error('Upload error:', error);
        }
    }

    readFileAsText(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = e => resolve(e.target.result);
            reader.onerror = reject;
            reader.readAsText(file);
        });
    }

    loadSampleData() {
        try {
            this.updateStatus('Generating sample MovieLens-like data...', 'loading');
            
            // Create sample data structure
            this.generateSampleData();
            this.buildUserItemMappings();
            this.calculateUserTopRated();
            
            this.isDataLoaded = true;
            document.getElementById('trainModel').disabled = false;
            this.updateStatus(`✅ Sample data generated successfully!\n- Users: ${this.users.size}\n- Items: ${this.items.size}\n- Interactions: ${this.interactions.length}\nUsing synthetic data for demonstration.`, 'success');
            
        } catch (error) {
            this.updateStatus(`❌ Sample data error: ${error.message}`, 'error');
            console.error('Sample data error:', error);
        }
    }

    generateSampleData() {
        // Generate sample movies
        const sampleMovies = [
            { id: 1, title: "Toy Story (1995)", year: 1995 },
            { id: 2, title: "Jumanji (1995)", year: 1995 },
            { id: 3, title: "Grumpier Old Men (1995)", year: 1995 },
            { id: 4, title: "Waiting to Exhale (1995)", year: 1995 },
            { id: 5, title: "Father of the Bride Part II (1995)", year: 1995 },
            { id: 6, title: "Heat (1995)", year: 1995 },
            { id: 7, title: "Sabrina (1995)", year: 1995 },
            { id: 8, title: "Tom and Huck (1995)", year: 1995 },
            { id: 9, title: "Sudden Death (1995)", year: 1995 },
            { id: 10, title: "GoldenEye (1995)", year: 1995 },
            { id: 11, title: "The American President (1995)", year: 1995 },
            { id: 12, title: "Dracula: Dead and Loving It (1995)", year: 1995 },
            { id: 13, title: "Balto (1995)", year: 1995 },
            { id: 14, title: "Nixon (1995)", year: 1995 },
            { id: 15, title: "Cutthroat Island (1995)", year: 1995 }
        ];

        // Create items map
        this.items.clear();
        sampleMovies.forEach(movie => {
            this.items.set(movie.id, {
                title: movie.title.replace(/\s*\(\d{4}\)$/, ''),
                year: movie.year,
                genres: new Array(19).fill(0).map(() => Math.random() > 0.7 ? 1 : 0)
            });
        });

        // Generate sample interactions
        this.interactions = [];
        for (let userId = 1; userId <= 100; userId++) {
            const numRatings = Math.floor(Math.random() * 20) + 10; // 10-30 ratings per user
            const ratedItems = new Set();
            
            for (let i = 0; i < numRatings; i++) {
                let itemId;
                do {
                    itemId = Math.floor(Math.random() * sampleMovies.length) + 1;
                } while (ratedItems.has(itemId));
                
                ratedItems.add(itemId);
                
                const rating = Math.floor(Math.random() * 4) + 1; // 1-5 rating
                const timestamp = Date.now() - Math.floor(Math.random() * 1000000000);
                
                this.interactions.push({
                    userId,
                    itemId,
                    rating,
                    timestamp
                });
            }
        }
    }

    parseInteractions(data) {
        const lines = data.trim().split('\n');
        this.interactions = lines.map((line, index) => {
            try {
                // Try tab separation first, then fallback to any whitespace
                const parts = line.split(/\t/);
                if (parts.length < 3) {
                    const altParts = line.split(/\s+/);
                    if (altParts.length >= 3) {
                        return this.createInteraction(altParts, index);
                    }
                    return null;
                }
                return this.createInteraction(parts, index);
            } catch (error) {
                console.warn(`Error parsing line ${index}: ${line}`, error);
                return null;
            }
        }).filter(interaction => interaction !== null);
        
        console.log(`Parsed ${this.interactions.length} interactions`);
    }

    createInteraction(parts, index) {
        const userId = parseInt(parts[0]);
        const itemId = parseInt(parts[1]);
        const rating = parseInt(parts[2]);
        const timestamp = parts[3] ? parseInt(parts[3]) : Date.now();
        
        if (isNaN(userId) || isNaN(itemId) || isNaN(rating)) {
            console.warn(`Skipping invalid line ${index}: ${parts.join(', ')}`);
            return null;
        }
        
        return { userId, itemId, rating, timestamp };
    }

    parseItems(data) {
        const lines = data.trim().split('\n');
        this.items.clear();
        
        lines.forEach((line, index) => {
            try {
                // Try pipe separation first (original format), then fallback to tab
                let parts = line.split('|');
                if (parts.length < 2) {
                    parts = line.split('\t');
                }
                
                if (parts.length >= 2) {
                    const itemId = parseInt(parts[0]);
                    if (isNaN(itemId)) {
                        console.warn(`Skipping item with invalid ID on line ${index}: ${parts[0]}`);
                        return;
                    }
                    
                    const title = parts[1].trim();
                    
                    // Extract year from title (format: "Title (YYYY)")
                    const yearMatch = title.match(/\((\d{4})\)$/);
                    const year = yearMatch ? parseInt(yearMatch[1]) : null;
                    
                    // Parse genres if available (last 19 fields in original format)
                    let genres = new Array(19).fill(0);
                    if (parts.length >= 24) {
                        genres = parts.slice(5, 24).map(x => parseInt(x) || 0);
                    }
                    
                    this.items.set(itemId, {
                        title: title.replace(/\s*\(\d{4}\)$/, ''), // Remove year from title
                        year,
                        genres
                    });
                }
            } catch (error) {
                console.warn(`Error parsing item line ${index}:`, error);
            }
        });
        
        console.log(`Parsed ${this.items.size} items`);
    }

    buildUserItemMappings() {
        // Reset all mappings
        this.userIndexMap.clear();
        this.itemIndexMap.clear();
        this.reverseUserIndex.clear();
        this.reverseItemIndex.clear();
        this.users.clear();
        this.userRatings.clear();

        // Create user mappings
        const uniqueUsers = [...new Set(this.interactions.map(i => i.userId))].sort((a, b) => a - b);
        uniqueUsers.forEach((userId, index) => {
            this.userIndexMap.set(userId, index);
            this.reverseUserIndex.set(index, userId);
            this.users.set(userId, { id: userId, index });
        });

        // Create item mappings
        const uniqueItems = [...new Set(this.interactions.map(i => i.itemId))].sort((a, b) => a - b);
        uniqueItems.forEach((itemId, index) => {
            this.itemIndexMap.set(itemId, index);
            this.reverseItemIndex.set(index, itemId);
        });

        // Build user ratings
        this.interactions.forEach(interaction => {
            const userId = interaction.userId;
            if (!this.userRatings.has(userId)) {
                this.userRatings.set(userId, []);
            }
            this.userRatings.get(userId).push({
                itemId: interaction.itemId,
                rating: interaction.rating,
                timestamp: interaction.timestamp
            });
        });

        console.log(`Built mappings: ${this.users.size} users, ${this.items.size} items`);
    }

    calculateUserTopRated() {
        this.userTopRated.clear();
        for (const [userId, ratings] of this.userRatings.entries()) {
            // Sort by rating (descending), then by timestamp (descending for recent)
            const sorted = ratings.sort((a, b) => {
                if (b.rating !== a.rating) return b.rating - a.rating;
                return b.timestamp - a.timestamp;
            });
            this.userTopRated.set(userId, sorted.slice(0, 10));
        }
    }

    async trainModel() {
        if (!this.isDataLoaded || this.isTraining) return;
        
        this.isTraining = true;
        document.getElementById('trainModel').disabled = true;
        document.getElementById('testModel').disabled = true;
        this.updateStatus('Starting model training...\nThis may take 2-5 minutes depending on your browser.', 'loading');
        
        const config = {
            epochs: 10,
            batchSize: 256,
            embeddingDim: 32,
            learningRate: 0.001,
            maxInteractions: Math.min(50000, this.interactions.length)
        };

        // Use subset of data if too large
        const trainingInteractions = this.interactions.slice(0, config.maxInteractions);
        
        // Initialize models
        this.simpleModel = new TwoTowerModel(
            this.users.size,
            this.items.size,
            config.embeddingDim,
            false // simple model
        );
        
        this.deepModel = new TwoTowerModel(
            this.users.size,
            this.items.size,
            config.embeddingDim,
            true, // deep model
            this.items
        );

        // Prepare training data
        const userIndices = trainingInteractions.map(i => this.userIndexMap.get(i.userId));
        const itemIndices = trainingInteractions.map(i => this.itemIndexMap.get(i.itemId));
        
        this.lossHistory = { simple: [], deep: [] };
        this.currentEpoch = 0;
        
        this.initLossChart();
        
        try {
            // Train both models
            await this.trainModelIteration(this.simpleModel, userIndices, itemIndices, config, 'simple');
            await this.trainModelIteration(this.deepModel, userIndices, itemIndices, config, 'deep');
            
            this.isTraining = false;
            document.getElementById('testModel').disabled = false;
            this.updateStatus('✅ Training completed! Click "Test Recommendation" to see results.', 'success');
            
            // Visualize embeddings
            await this.visualizeEmbeddings();
        } catch (error) {
            this.isTraining = false;
            document.getElementById('trainModel').disabled = false;
            this.updateStatus(`❌ Training failed: ${error.message}`, 'error');
            console.error('Training error:', error);
        }
    }

    async trainModelIteration(model, userIndices, itemIndices, config, modelType) {
        const totalBatches = Math.ceil(userIndices.length / config.batchSize);
        
        for (let epoch = 0; epoch < config.epochs; epoch++) {
            this.currentEpoch = epoch;
            let epochLoss = 0;
            let batchCount = 0;
            
            // Calculate progress
            const overallProgress = ((modelType === 'simple' ? epoch : epoch + config.epochs) / (config.epochs * 2)) * 100;
            this.updateProgress(overallProgress, `Training ${modelType} model... Epoch ${epoch + 1}/${config.epochs}`);
            
            // Shuffle data
            const shuffledIndices = tf.util.createShuffledIndices(userIndices.length);
            const shuffledUsers = shuffledIndices.map(i => userIndices[i]);
            const shuffledItems = shuffledIndices.map(i => itemIndices[i]);
            
            for (let batch = 0; batch < totalBatches; batch++) {
                const start = batch * config.batchSize;
                const end = Math.min(start + config.batchSize, userIndices.length);
                
                const batchUsers = shuffledUsers.slice(start, end);
                const batchItems = shuffledItems.slice(start, end);
                
                try {
                    const loss = await model.trainStep(batchUsers, batchItems);
                    epochLoss += loss;
                    batchCount++;
                    
                    // Update loss chart every few batches
                    if (batch % 5 === 0) {
                        this.updateLossChart(modelType, loss, epoch * totalBatches + batch);
                        await tf.nextFrame(); // Allow UI updates
                    }
                } catch (error) {
                    console.error(`Training error in ${modelType} model, batch ${batch}:`, error);
                }
            }
            
            const avgLoss = batchCount > 0 ? epochLoss / batchCount : 0;
            this.lossHistory[modelType].push(avgLoss);
            console.log(`${modelType} Model - Epoch ${epoch + 1}/${config.epochs}, Loss: ${avgLoss.toFixed(4)}`);
        }
    }

    initLossChart() {
        const canvas = document.getElementById('lossChart');
        const ctx = canvas.getContext('2d');
        
        // Clear canvas
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw initial axes
        ctx.strokeStyle = '#ccc';
        ctx.beginPath();
        ctx.moveTo(50, 50);
        ctx.lineTo(50, canvas.height - 50);
        ctx.lineTo(canvas.width - 50, canvas.height - 50);
        ctx.stroke();
        
        ctx.fillStyle = 'black';
        ctx.font = '12px Arial';
        ctx.fillText('Loss', 10, canvas.height / 2);
        ctx.fillText('Batch', canvas.width / 2, canvas.height - 10);
        ctx.fillText('Simple Model (blue)', canvas.width - 150, 30);
        ctx.fillStyle = 'red';
        ctx.fillText('Deep Model (red)', canvas.width - 150, 50);
    }

    updateLossChart(modelType, loss, batchIndex) {
        const canvas = document.getElementById('lossChart');
        const ctx = canvas.getContext('2d');
        
        const x = 50 + (batchIndex % 200) * (canvas.width - 100) / 200;
        const y = canvas.height - 50 - Math.min(loss * 50, canvas.height - 100);
        
        ctx.fillStyle = modelType === 'simple' ? 'blue' : 'red';
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, 2 * Math.PI);
        ctx.fill();
    }

    async visualizeEmbeddings() {
        try {
            this.updateStatus('Visualizing item embeddings...', 'loading');
            
            // Use deep model for visualization
            const itemEmbeddings = this.deepModel.getItemEmbeddings();
            const sampleSize = Math.min(200, this.items.size);
            
            // Sample random items
            const sampleIndices = [];
            for (let i = 0; i < sampleSize; i++) {
                sampleIndices.push(Math.floor(Math.random() * this.items.size));
            }
            
            const sampleEmbeddings = tf.gather(itemEmbeddings, sampleIndices);
            const embeddings2D = await this.computePCA(sampleEmbeddings, 2);
            
            this.drawEmbeddingChart(embeddings2D, sampleIndices);
            
            // Clean up
            sampleEmbeddings.dispose();
        } catch (error) {
            console.error('Error visualizing embeddings:', error);
        }
    }

    async computePCA(embeddings, components = 2) {
        return tf.tidy(() => {
            const embeddingArray = embeddings.arraySync();
            const n = embeddingArray.length;
            const dim = embeddingArray[0].length;
            
            // Center the data
            const mean = new Array(dim).fill(0);
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < dim; j++) {
                    mean[j] += embeddingArray[i][j];
                }
            }
            for (let j = 0; j < dim; j++) {
                mean[j] /= n;
            }
            
            const centered = embeddingArray.map(row => 
                row.map((val, j) => val - mean[j])
            );
            
            // Simple projection using first two dimensions
            const projected = centered.map(row => [row[0], row[1]]);
            
            return projected;
        });
    }

    drawEmbeddingChart(embeddings2D, indices) {
        const canvas = document.getElementById('embeddingChart');
        const ctx = canvas.getContext('2d');
        
        // Clear canvas
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Find bounds
        const xValues = embeddings2D.map(p => p[0]);
        const yValues = embeddings2D.map(p => p[1]);
        const xMin = Math.min(...xValues);
        const xMax = Math.max(...xValues);
        const yMin = Math.min(...yValues);
        const yMax = Math.max(...yValues);
        
        // Scale to canvas
        const scaleX = (x) => 50 + (x - xMin) * (canvas.width - 100) / (xMax - xMin || 1);
        const scaleY = (y) => canvas.height - 50 - (y - yMin) * (canvas.height - 100) / (yMax - yMin || 1);
        
        // Draw points
        ctx.fillStyle = 'blue';
        embeddings2D.forEach((point, i) => {
            const x = scaleX(point[0]);
            const y = scaleY(point[1]);
            
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fill();
        });
        
        // Add title
        ctx.fillStyle = 'black';
        ctx.font = '14px Arial';
        ctx.fillText('Item Embeddings Visualization (PCA projection)', canvas.width / 2 - 100, 20);
    }

    async testModel() {
        if (!this.simpleModel || !this.deepModel) {
            this.updateStatus('Please train models first!', 'error');
            return;
        }

        this.updateStatus('Generating recommendations...', 'loading');

        // Find users with at least 10 ratings
        const qualifiedUsers = [...this.userRatings.entries()]
            .filter(([_, ratings]) => ratings.length >= 10)
            .map(([userId]) => userId);

        if (qualifiedUsers.length === 0) {
            this.updateStatus('No qualified users found (need users with ≥10 ratings)', 'error');
            return;
        }

        // Select random qualified user
        const randomUser = qualifiedUsers[Math.floor(Math.random() * qualifiedUsers.length)];
        const userIndex = this.userIndexMap.get(randomUser);
        
        // Get user's rated items
        const ratedItems = new Set(this.userRatings.get(randomUser).map(r => r.itemId));
        const ratedItemIndices = [...ratedItems].map(itemId => this.itemIndexMap.get(itemId));

        // Display user info
        document.getElementById('userInfo').innerHTML = 
            `<h3>Selected User: ${randomUser} (${this.userRatings.get(randomUser).length} ratings)</h3>`;

        // Get recommendations from both models
        const simpleRecs = await this.simpleModel.getRecommendations(userIndex, ratedItemIndices, 10);
        const deepRecs = await this.deepModel.getRecommendations(userIndex, ratedItemIndices, 10);

        // Display results
        this.displayResults(randomUser, simpleRecs, deepRecs);
        document.getElementById('resultsContainer').style.display = 'flex';
        
        this.updateStatus('✅ Recommendations generated successfully!', 'success');
    }

    displayResults(userId, simpleRecs, deepRecs) {
        // Display historical ratings
        const historicalRatings = this.userTopRated.get(userId) || [];
        const historicalTable = document.getElementById('historicalTable').querySelector('tbody');
        historicalTable.innerHTML = '';
        
        historicalRatings.slice(0, 10).forEach((rating, index) => {
            const item = this.items.get(rating.itemId);
            const row = historicalTable.insertRow();
            row.insertCell(0).textContent = index + 1;
            row.insertCell(1).textContent = item ? `${item.title} (${item.year || 'N/A'})` : `Item ${rating.itemId}`;
            row.insertCell(2).textContent = rating.rating;
        });

        // Display simple model recommendations
        const simpleTable = document.getElementById('simpleRecommendationsTable').querySelector('tbody');
        simpleTable.innerHTML = '';
        
        simpleRecs.forEach((rec, index) => {
            const itemId = this.reverseItemIndex.get(rec.itemIndex);
            const item = this.items.get(itemId);
            const row = simpleTable.insertRow();
            row.insertCell(0).textContent = index + 1;
            row.insertCell(1).textContent = item ? `${item.title} (${item.year || 'N/A'})` : `Item ${itemId}`;
            row.insertCell(2).textContent = rec.score.toFixed(4);
        });

        // Display deep model recommendations
        const deepTable = document.getElementById('deepRecommendationsTable').querySelector('tbody');
        deepTable.innerHTML = '';
        
        deepRecs.forEach((rec, index) => {
            const itemId = this.reverseItemIndex.get(rec.itemIndex);
            const item = this.items.get(itemId);
            const row = deepTable.insertRow();
            row.insertCell(0).textContent = index + 1;
            row.insertCell(1).textContent = item ? `${item.title} (${item.year || 'N/A'})` : `Item ${itemId}`;
            row.insertCell(2).textContent = rec.score.toFixed(4);
        });
    }
}

// Initialize app when page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new MovieRecommendationApp();
    console.log('Movie Recommendation App initialized');
});
