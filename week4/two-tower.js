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
        
        console.log(`Initialized ${useDeepLearning ? 'Deep' : 'Simple'} Two-Tower Model`);
        console.log(`- Users: ${numUsers}, Items: ${numItems}, Embedding Dim: ${embeddingDim}`);
    }

    initializeModel() {
        // Clear any existing variables
        this.dispose();
        
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
        this.itemEmbedding = tf.variable(
            initializer.apply([this.numItems, this.embeddingDim]),
            true,
            'itemEmbedding'
        );
        
        // Item MLP layers
        this.itemHiddenWeights = tf.variable(
            initializer.apply([this.embeddingDim + (this.items ? 19 : 0), 64]),
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
        return tf.tidy(() => {
            if (this.useDeepLearning) {
                return this.userForwardDeep(userIndices);
            } else {
                return this.userForwardSimple(userIndices);
            }
        });
    }

    userForwardSimple(userIndices) {
        return tf.gather(this.userEmbedding, userIndices);
    }

    userForwardDeep(userIndices) {
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
        return tf.tidy(() => {
            if (this.useDeepLearning) {
                return this.itemForwardDeep(itemIndices);
            } else {
                return this.itemForwardSimple(itemIndices);
            }
        });
    }

    itemForwardSimple(itemIndices) {
        return tf.gather(this.itemEmbedding, itemIndices);
    }

    itemForwardDeep(itemIndices) {
        const itemEmb = tf.gather(this.itemEmbedding, itemIndices);
        
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
        // Create genre features tensor
        const genreArray = [];
        const itemIndicesArray = Array.isArray(itemIndices) ? itemIndices : Array.from({length: itemIndices.shape[0]}, (_, i) => i);
        
        for (let i = 0; i < itemIndicesArray.length; i++) {
            const itemIndex = itemIndicesArray[i];
            // Get the original item ID from the index
            const originalItemId = Array.from(this.items.keys())[itemIndex];
            const item = this.items.get(originalItemId);
            
            if (item && item.genres) {
                genreArray.push(item.genres);
            } else {
                // Default: no genres
                genreArray.push(new Array(19).fill(0));
            }
        }
        
        return tf.tensor2d(genreArray);
    }

    score(userEmbeddings, itemEmbeddings) {
        return tf.tidy(() => {
            // Dot product between user and item embeddings
            return tf.sum(tf.mul(userEmbeddings, itemEmbeddings), 1, true);
        });
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
            return loss ? loss.dataSync()[0] : 0;
        });
    }

    getItemEmbeddings() {
        if (this.useDeepLearning) {
            // For deep model, forward pass through the MLP for all items
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
        const tensors = [
            this.userEmbedding, this.itemEmbedding,
            this.userHiddenWeights, this.userHiddenBias, this.userOutputWeights, this.userOutputBias,
            this.itemHiddenWeights, this.itemHiddenBias, this.itemOutputWeights, this.itemOutputBias
        ];
        
        tensors.forEach(tensor => {
            if (tensor && !tensor.isDisposed) {
                tensor.dispose();
            }
        });
    }
}
