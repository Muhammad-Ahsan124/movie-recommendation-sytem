// two-tower.js
class TwoTowerModel {
    constructor(numUsers, numItems, embeddingDim, useDeepLearning = false, items = null) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embeddingDim = embeddingDim;
        this.useDeepLearning = useDeepLearning;
        this.items = items;
        
        // Initialize optimizer
        this.optimizer = tf.train.adam(0.001);
        
        // Initialize model parameters
        this.initializeModel();
    }

    initializeModel() {
        if (this.useDeepLearning) {
            this.initializeDeepModel();
        } else {
            this.initializeSimpleModel();
        }
    }

    initializeSimpleModel() {
        // Simple model: direct embedding lookup
        const initializer = tf.initializers.glorotNormal();
        
        this.userEmbedding = tf.variable(
            initializer.apply([this.numUsers, this.embeddingDim]),
            true,
            'userEmbedding'
        );
        
        this.itemEmbedding = tf.variable(
            initializer.apply([this.numItems, this.embeddingDim]),
            true,
            'itemEmbedding'
        );
    }

    initializeDeepModel() {
        // Deep model: MLP towers with hidden layers
        const initializer = tf.initializers.glorotNormal();
        
        // User tower: user_id -> embedding -> hidden layer -> output
        this.userEmbedding = tf.variable(
            initializer.apply([this.numUsers, this.embeddingDim]),
            true,
            'userEmbedding'
        );
        
        // User MLP layers
        this.userHiddenWeights = tf.variable(
            initializer.apply([this.embeddingDim, 64]),
            true,
            'userHiddenWeights'
        );
        this.userHiddenBias = tf.variable(
            tf.zeros([64]),
            true,
            'userHiddenBias'
        );
        this.userOutputWeights = tf.variable(
            initializer.apply([64, this.embeddingDim]),
            true,
            'userOutputWeights'
        );
        this.userOutputBias = tf.variable(
            tf.zeros([this.embeddingDim]),
            true,
            'userOutputBias'
        );

        // Item tower with genre features
        const itemInputDim = this.embeddingDim + (this.items ? 19 : 0); // embedding + genres
        
        this.itemEmbedding = tf.variable(
            initializer.apply([this.numItems, this.embeddingDim]),
            true,
            'itemEmbedding'
        );
        
        // Item MLP layers
        this.itemHiddenWeights = tf.variable(
            initializer.apply([itemInputDim, 64]),
            true,
            'itemHiddenWeights'
        );
        this.itemHiddenBias = tf.variable(
            tf.zeros([64]),
            true,
            'itemHiddenBias'
        );
        this.itemOutputWeights = tf.variable(
            initializer.apply([64, this.embeddingDim]),
            true,
            'itemOutputWeights'
        );
        this.itemOutputBias = tf.variable(
            tf.zeros([this.embeddingDim]),
            true,
            'itemOutputBias'
        );
    }

    userForward(userIndices) {
        if (this.useDeepLearning) {
            return this.userForwardDeep(userIndices);
        } else {
            return this.userForwardSimple(userIndices);
        }
    }

    userForwardSimple(userIndices) {
        // Simple: direct embedding lookup
        return tf.gather(this.userEmbedding, userIndices);
    }

    userForwardDeep(userIndices) {
        // Deep: embedding -> hidden layer -> output
        const userEmb = tf.gather(this.userEmbedding, userIndices);
        
        // MLP: hidden layer with ReLU
        const hidden = tf.relu(
            tf.add(tf.matMul(userEmb, this.userHiddenWeights), this.userHiddenBias)
        );
        
        // Output layer (no activation for embeddings)
        const output = tf.add(tf.matMul(hidden, this.userOutputWeights), this.userOutputBias);
        
        return output;
    }

    itemForward(itemIndices) {
        if (this.useDeepLearning) {
            return this.itemForwardDeep(itemIndices);
        } else {
            return this.itemForwardSimple(itemIndices);
        }
    }

    itemForwardSimple(itemIndices) {
        // Simple: direct embedding lookup
        return tf.gather(this.itemEmbedding, itemIndices);
    }

    itemForwardDeep(itemIndices) {
        // Deep: embedding + genre features -> hidden layer -> output
        const itemEmb = tf.gather(this.itemEmbedding, itemIndices);
        
        // Add genre features if available
        let itemFeatures = itemEmb;
        if (this.items) {
            const genreFeatures = this.getGenreFeatures(itemIndices);
            itemFeatures = tf.concat([itemEmb, genreFeatures], 1);
        }
        
        // MLP: hidden layer with ReLU
        const hidden = tf.relu(
            tf.add(tf.matMul(itemFeatures, this.itemHiddenWeights), this.itemHiddenBias)
        );
        
        // Output layer (no activation for embeddings)
        const output = tf.add(tf.matMul(hidden, this.itemOutputWeights), this.itemOutputBias);
        
        return output;
    }

    getGenreFeatures(itemIndices) {
        // Convert item indices back to original IDs and get genre features
        const genreArray = [];
        for (let i = 0; i < itemIndices.length; i++) {
            const itemIndex = itemIndices[i];
            // This would need the reverse mapping from app.js - we'll simulate for now
            const genres = new Array(19).fill(0); // Default: no genres
            genreArray.push(genres);
        }
        return tf.tensor2d(genreArray);
    }

    score(userEmbeddings, itemEmbeddings) {
        // Dot product between user and item embeddings
        return tf.sum(tf.mul(userEmbeddings, itemEmbeddings), 1, true);
    }

    computeLoss(userIndices, positiveItemIndices) {
        return tf.tidy(() => {
            const userEmb = this.userForward(userIndices);
            const posItemEmb = this.itemForward(positiveItemIndices);
            
            // In-batch negative sampling: use all other items in batch as negatives
            const scores = tf.matMul(userEmb, posItemEmb, false, true);
            
            // Labels: diagonal is positive, others are negative
            const batchSize = userIndices.length;
            const labels = tf.oneHot(tf.range(0, batchSize), batchSize);
            
            // Softmax cross-entropy loss
            const loss = tf.losses.softmaxCrossEntropy(labels, scores);
            
            return loss;
        });
    }

    async trainStep(userIndices, itemIndices) {
        return tf.tidy(() => {
            const lossFn = () => this.computeLoss(userIndices, itemIndices);
            const loss = this.optimizer.minimize(lossFn, true);
            return loss.dataSync()[0];
        });
    }

    getItemEmbeddings() {
        if (this.useDeepLearning) {
            // For deep model, we need to forward pass through the MLP
            const allItemIndices = Array.from({length: this.numItems}, (_, i) => i);
            return this.itemForward(allItemIndices);
        } else {
            return this.itemEmbedding;
        }
    }

    async getRecommendations(userIndex, excludeItemIndices, topK = 10) {
        return tf.tidy(() => {
            // Get user embedding
            const userEmb = this.userForward([userIndex]);
            
            // Get all item embeddings
            const allItemIndices = Array.from({length: this.numItems}, (_, i) => i);
            const itemEmbs = this.itemForward(allItemIndices);
            
            // Compute scores
            const scores = tf.matMul(userEmb, itemEmbs, false, true);
            const scoresArray = scores.dataSync();
            
            // Create recommendations, excluding specified items
            const recommendations = [];
            for (let i = 0; i < scoresArray.length; i++) {
                if (!excludeItemIndices.includes(i)) {
                    recommendations.push({
                        itemIndex: i,
                        score: scoresArray[i]
                    });
                }
            }
            
            // Sort by score descending and take topK
            recommendations.sort((a, b) => b.score - a.score);
            return recommendations.slice(0, topK);
        });
    }

    dispose() {
        if (this.userEmbedding) this.userEmbedding.dispose();
        if (this.itemEmbedding) this.itemEmbedding.dispose();
        
        if (this.useDeepLearning) {
            if (this.userHiddenWeights) this.userHiddenWeights.dispose();
            if (this.userHiddenBias) this.userHiddenBias.dispose();
            if (this.userOutputWeights) this.userOutputWeights.dispose();
            if (this.userOutputBias) this.userOutputBias.dispose();
            if (this.itemHiddenWeights) this.itemHiddenWeights.dispose();
            if (this.itemHiddenBias) this.itemHiddenBias.dispose();
            if (this.itemOutputWeights) this.itemOutputWeights.dispose();
            if (this.itemOutputBias) this.itemOutputBias.dispose();
        }
    }
}
