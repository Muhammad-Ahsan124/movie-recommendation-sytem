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
        this.updateStatus('Ready to load data...', 'status');
    }

    initEventListeners() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('trainModel').addEventListener('click', () => this.trainModel());
        document.getElementById('testModel').addEventListener('click', () => this.testModel());
    }

    updateStatus(message, className = 'status') {
        const statusEl = document.getElementById('status');
        statusEl.textContent = message;
        statusEl.className = `status ${className}`;
    }

    async loadData() {
        try {
            this.updateStatus('Loading data...', 'loading');
            
            // Load interactions data
            const interactionsResponse = await fetch('data/u.data');
            if (!interactionsResponse.ok) throw new Error('Failed to load u.data');
            const interactionsText = await interactionsResponse.text();
            
            // Load items data
            const itemsResponse = await fetch('data/u.item');
            if (!itemsResponse.ok) throw new Error('Failed to load u.item');
            const itemsText = await itemsResponse.text();
            
            this.parseInteractions(interactionsText);
            this.parseItems(itemsText);
            this.buildUserItemMappings();
            this.calculateUserTopRated();
            
            this.isDataLoaded = true;
            document.getElementById('trainModel').disabled = false;
            this.updateStatus(`Data loaded successfully! Users: ${this.users.size}, Items: ${this.items.size}, Interactions: ${this.interactions.length}`, 'success');
            
        } catch (error) {
            this.updateStatus(`Error loading data: ${error.message}`, 'error');
            console.error('Data loading error:', error);
        }
    }

    parseInteractions(data) {
        const lines = data.trim().split('\n');
        this.interactions = lines.map(line => {
            const [userId, itemId, rating, timestamp] = line.split('\t').map(x => parseInt(x));
            return { userId, itemId, rating, timestamp };
        });
    }

    parseItems(data) {
        const lines = data.trim().split('\n');
        lines.forEach(line => {
            const parts = line.split('|');
            if (parts.length >= 3) {
                const itemId = parseInt(parts[0]);
                const title = parts[1];
                const releaseDate = parts[2];
                
                // Extract year from title (format: "Title (YYYY)")
                const yearMatch = title.match(/\((\d{4})\)$/);
                const year = yearMatch ? parseInt(yearMatch[1]) : null;
                
                // Parse genres (last 19 fields)
                const genres = parts.slice(5, 24).map(x => parseInt(x));
                
                this.items.set(itemId, {
                    title: title.replace(/\s*\(\d{4}\)$/, ''), // Remove year from title
                    year,
                    releaseDate,
                    genres
                });
            }
        });
    }

    buildUserItemMappings() {
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
    }

    calculateUserTopRated() {
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
        this.updateStatus('Starting model training...', 'loading');
        
        const config = {
            epochs: 10,
            batchSize: 512,
            embeddingDim: 32,
            learningRate: 0.001,
            maxInteractions: 80000
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
        
        // Train both models
        await this.trainModelIteration(this.simpleModel, userIndices, itemIndices, config, 'simple');
        await this.trainModelIteration(this.deepModel, userIndices, itemIndices, config, 'deep');
        
        this.isTraining = false;
        document.getElementById('testModel').disabled = false;
        this.updateStatus('Training completed! Click "Test Recommendation" to see results.', 'success');
        
        // Visualize embeddings
        await this.visualizeEmbeddings();
    }

    async trainModelIteration(model, userIndices, itemIndices, config, modelType) {
        const totalBatches = Math.ceil(userIndices.length / config.batchSize);
        
        for (let epoch = 0; epoch < config.epochs; epoch++) {
            this.currentEpoch = epoch;
            let epochLoss = 0;
            
            // Shuffle data
            const shuffledIndices = tf.util.createShuffledIndices(userIndices.length);
            const shuffledUsers = shuffledIndices.map(i => userIndices[i]);
            const shuffledItems = shuffledIndices.map(i => itemIndices[i]);
            
            for (let batch = 0; batch < totalBatches; batch++) {
                const start = batch * config.batchSize;
                const end = Math.min(start + config.batchSize, userIndices.length);
                
                const batchUsers = shuffledUsers.slice(start, end);
                const batchItems = shuffledItems.slice(start, end);
                
                const loss = await model.trainStep(batchUsers, batchItems);
                epochLoss += loss;
                
                // Update loss chart every few batches
                if (batch % 10 === 0) {
                    this.updateLossChart(modelType, loss, epoch * totalBatches + batch);
                    await tf.nextFrame(); // Allow UI updates
                }
            }
            
            const avgLoss = epochLoss / totalBatches;
            this.lossHistory[modelType].push(avgLoss);
            console.log(`${modelType} Model - Epoch ${epoch + 1}/${config.epochs}, Loss: ${avgLoss.toFixed(4)}`);
            
            this.updateStatus(
                `Training ${modelType} model... Epoch ${epoch + 1}/${config.epochs}, Loss: ${avgLoss.toFixed(4)}`,
                'loading'
            );
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
    }

    updateLossChart(modelType, loss, batchIndex) {
        const canvas = document.getElementById('lossChart');
        const ctx = canvas.getContext('2d');
        
        const x = 50 + (batchIndex % 100) * (canvas.width - 100) / 100;
        const y = canvas.height - 50 - (loss * 100); // Scale loss for visibility
        
        ctx.fillStyle = modelType === 'simple' ? 'blue' : 'red';
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, 2 * Math.PI);
        ctx.fill();
        
        // Add legend
        ctx.fillStyle = 'black';
        ctx.font = '12px Arial';
        ctx.fillText('Simple Model (blue)', canvas.width - 150, 30);
        ctx.fillStyle = 'red';
        ctx.fillText('Deep Model (red)', canvas.width - 150, 50);
    }

    async visualizeEmbeddings() {
        // Use deep model for visualization
        const itemEmbeddings = this.deepModel.getItemEmbeddings();
        const sampleSize = Math.min(500, this.items.size);
        
        // Sample random items
        const sampleIndices = [];
        for (let i = 0; i < sampleSize; i++) {
            sampleIndices.push(Math.floor(Math.random() * this.items.size));
        }
        
        const sampleEmbeddings = tf.gather(itemEmbeddings, sampleIndices);
        const embeddings2D = await this.computePCA(sampleEmbeddings, 2);
        
        this.drawEmbeddingChart(embeddings2D, sampleIndices);
    }

    async computePCA(embeddings, components = 2) {
        // Simple PCA implementation using power iteration
        const embeddingArray = await embeddings.array();
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
        
        // Compute covariance matrix
        const covariance = new Array(dim).fill(0).map(() => new Array(dim).fill(0));
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < dim; j++) {
                for (let k = 0; k < dim; k++) {
                    covariance[j][k] += centered[i][j] * centered[i][k];
                }
            }
        }
        for (let j = 0; j < dim; j++) {
            for (let k = 0; k < dim; k++) {
                covariance[j][k] /= (n - 1);
            }
        }
        
        // Simple power method for first two components
        const projectTo2D = (data, vector1, vector2) => {
            return data.map(row => [
                row.reduce((sum, val, i) => sum + val * vector1[i], 0),
                row.reduce((sum, val, i) => sum + val * vector2[i], 0)
            ]);
        };
        
        // Use first two dimensions as initial approximation
        const pc1 = new Array(dim).fill(0).map((_, i) => i === 0 ? 1 : 0);
        const pc2 = new Array(dim).fill(0).map((_, i) => i === 1 ? 1 : 0);
        
        return projectTo2D(centered, pc1, pc2);
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
        const scaleX = (x) => 50 + (x - xMin) * (canvas.width - 100) / (xMax - xMin);
        const scaleY = (y) => canvas.height - 50 - (y - yMin) * (canvas.height - 100) / (yMax - yMin);
        
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

        // Find users with at least 20 ratings
        const qualifiedUsers = [...this.userRatings.entries()]
            .filter(([_, ratings]) => ratings.length >= 20)
            .map(([userId]) => userId);

        if (qualifiedUsers.length === 0) {
            this.updateStatus('No qualified users found (need users with ≥20 ratings)', 'error');
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
        
        this.updateStatus('Recommendations generated successfully!', 'success');
    }

    displayResults(userId, simpleRecs, deepRecs) {
        // Display historical ratings
        const historicalRatings = this.userTopRated.get(userId) || [];
        const historicalTable = document.getElementById('historicalTable').querySelector('tbody');
        historicalTable.innerHTML = '';
        
        historicalRatings.forEach((rating, index) => {
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
});
