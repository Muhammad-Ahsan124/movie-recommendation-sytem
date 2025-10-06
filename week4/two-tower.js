two-tower.js
/* Two-Tower models implemented in TensorFlow.js
   - Classic TwoTowerModel: separate user & item embedding tables, dot-product scoring.
   - Optionally a Deep (MLP) variant: small MLP on top of embeddings and item genres.
   - Training supports two loss modes:
       * in-batch sampled softmax: logits = U @ I_pos^T, labels diagonal
       * bpr pairwise: -log sigmoid(s_pos - s_neg)
   - The class exposes:
       constructor(numUsers, numItems, embDim, opts)
       userForward(userIdxTensor) -> tensor [B, D]
       itemForward(itemIdxTensor) -> tensor [B, D]
       getUserEmbedding(idx) -> array
       getItemEmbedding(idx) -> array
       dispose()
   - Note: The Deep MLP variant uses itemGenres provided (as array of arrays) to create an item features matrix.
*/

/* global tf */

/**
 * TwoTowerModel
 * @param {number} numUsers
 * @param {number} numItems
 * @param {number} embDim
 * @param {Object} opts - {useMLP: bool, itemGenres: array of [gflags], hiddenUnits: number, dropout: number}
 */
class TwoTowerModel {
  constructor(numUsers, numItems, embDim=32, opts={}) {
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.embDim = embDim;
    this.useMLP = Boolean(opts.useMLP);
    this.hiddenUnits = opts.hiddenUnits || Math.max(64, embDim*2);
    this.dropout = opts.dropout || 0.0;

    // Embedding tables as variables (trainable)
    // Using tf.variable allows us to optimize with grads
    this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05), true, 'userEmbedding');
    this.itemEmbedding = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05), true, 'itemEmbedding');

    // If deep MLP variant requested, build small MLP weights
    this.trainableVars = [this.userEmbedding, this.itemEmbedding];
    if (this.useMLP) {
      // We'll transform user embedding through an MLP and item embedding concatenated with genres through an MLP
      // Setup weights for user MLP: EmbDim -> hidden -> embDim (projection)
      this.userW1 = tf.variable(tf.randomNormal([embDim, this.hiddenUnits], 0, 0.1));
      this.userB1 = tf.variable(tf.zeros([this.hiddenUnits]));
      this.userW2 = tf.variable(tf.randomNormal([this.hiddenUnits, embDim], 0, 0.1));
      this.userB2 = tf.variable(tf.zeros([embDim]));

      // For items we optionally take genres feature vector (if provided in opts)
      this.itemGenres = opts.itemGenres || null; // array[numItems][genreCount]
      this.genreCount = (this.itemGenres && this.itemGenres.length>0) ? this.itemGenres[0].length : 0;

      // We'll create a genre matrix variable (but keep it non-trainable for simplicity)
      if (this.genreCount > 0) {
        // Create tensor2d shaped [numItems, genreCount] to be used in itemForward
        this.itemGenresTensor = tf.tensor2d(this.itemGenres, [numItems, this.genreCount], 'float32');
      } else {
        this.itemGenresTensor = null;
      }

      // Item MLP: (embDim + genreCount) -> hidden -> embDim
      const itemInputDim = embDim + this.genreCount;
      this.itemW1 = tf.variable(tf.randomNormal([itemInputDim, this.hiddenUnits], 0, 0.1));
      this.itemB1 = tf.variable(tf.zeros([this.hiddenUnits]));
      this.itemW2 = tf.variable(tf.randomNormal([this.hiddenUnits, embDim], 0, 0.1));
      this.itemB2 = tf.variable(tf.zeros([embDim]));

      // Add to trainable vars
      this.trainableVars.push(this.userW1, this.userB1, this.userW2, this.userB2,
                             this.itemW1, this.itemB1, this.itemW2, this.itemB2);
      // Note: itemGenresTensor is not trainable (features)
    }
  }

  // Gather user embeddings for a batch of integer indices (int32 tensor)
  userForward(userIdxTensor) {
    // userIdxTensor: [B] int32
    // returns: [B, embDim]
    return tf.tidy(() => {
      const emb = tf.gather(this.userEmbedding, userIdxTensor);
      if (!this.useMLP) return emb;
      // MLP: emb -> hidden -> embDim projection (nonlinear)
      const h1 = emb.matMul(this.userW1).add(this.userB1).relu();
      const drop = this.dropout > 0 ? tf.dropout(h1, this.dropout) : h1;
      const proj = drop.matMul(this.userW2).add(this.userB2); // [B, embDim]
      return proj;
    });
  }

  // Gather item embeddings for a batch of indices. If MLP enabled, concatenate genres and pass through MLP.
  itemForward(itemIdxTensor) {
    // itemIdxTensor: [B] int32
    return tf.tidy(() => {
      const emb = tf.gather(this.itemEmbedding, itemIdxTensor); // [B, embDim]
      if (!this.useMLP) return emb;
      if (this.itemGenresTensor) {
        // gather genre rows
        const genres = tf.gather(this.itemGenresTensor, itemIdxTensor); // [B, genreCount]
        const concat = emb.concat(genres, 1); // [B, embDim+genreCount]
        const h1 = concat.matMul(this.itemW1).add(this.itemB1).relu();
        const drop = this.dropout > 0 ? tf.dropout(h1, this.dropout) : h1;
        const proj = drop.matMul(this.itemW2).add(this.itemB2); // [B, embDim]
        return proj;
      } else {
        // If no genres available, still transform embedding
        const h1 = emb.matMul(this.itemW1).add(this.itemB1).relu();
        const drop = this.dropout > 0 ? tf.dropout(h1, this.dropout) : h1;
        const proj = drop.matMul(this.itemW2).add(this.itemB2);
        return proj;
      }
    });
  }

  // Score: dot product along last dim
  score(userEmb, itemEmb) {
    // userEmb: [B, D], itemEmb: [B, D] or itemEmb: [N, D] for cross scoring
    return tf.tidy(() => {
      // If itemEmb is [N,D] and userEmb is [B,D], compute matMul accordingly
      if (userEmb.shape.length === 2 && itemEmb.shape.length === 2 && userEmb.shape[0] === itemEmb.shape[0]) {
        // elementwise dot -> vector
        return tf.sum(tf.mul(userEmb, itemEmb), -1);
      } else if (userEmb.shape.length === 2 && itemEmb.shape.length === 2 && userEmb.shape[0] !== itemEmb.shape[0]) {
        // compute U @ I^T -> [B, N]
        return tf.matMul(userEmb, itemEmb, false, true);
      } else if (userEmb.shape.length === 1 && itemEmb.shape.length === 2) {
        // single user vector -> matmul to [N]
        const u = userEmb.reshape([1, userEmb.shape[0]]);
        return tf.matMul(u, itemEmb, false, true).reshape([itemEmb.shape[0]]);
      } else {
        // fallback elementwise
        return tf.sum(tf.mul(userEmb, itemEmb), -1);
      }
    });
  }

  // Return array of user's embedding (synchronous via arraySync)
  getUserEmbedding(idx) {
    const t = tf.tensor1d([idx], 'int32');
    const emb = this.userForward(t);
    const arr = emb.arraySync()[0];
    emb.dispose(); t.dispose();
    return arr;
  }

  // Return array for item embedding (synchronous)
  getItemEmbedding(idx) {
    const t = tf.tensor1d([idx], 'int32');
    const emb = this.itemForward(t);
    const arr = emb.arraySync()[0];
    emb.dispose(); t.dispose();
    return arr;
  }

  // Expose trainable variable list for optimizer
  get trainableVars() {
    if (!this._trainableVars) {
      // fallback: list variables manually
      const out = [this.userEmbedding, this.itemEmbedding];
      if (this.useMLP) {
        out.push(this.userW1, this.userB1, this.userW2, this.userB2,
                 this.itemW1, this.itemB1, this.itemW2, this.itemB2);
      }
      this._trainableVars = out;
    }
    return this._trainableVars;
  }

  // Clean up
  dispose() {
    try {
      this.userEmbedding.dispose();
      this.itemEmbedding.dispose();
      if (this.useMLP) {
        this.userW1.dispose(); this.userB1.dispose();
        this.userW2.dispose(); this.userB2.dispose();
        this.itemW1.dispose(); this.itemB1.dispose();
        this.itemW2.dispose(); this.itemB2.dispose();
        if (this.itemGenresTensor) this.itemGenresTensor.dispose();
      }
    } catch (e) {
      console.warn('Error disposing TwoTowerModel tensors', e);
    }
  }
}

// Expose globally
if (typeof window !== 'undefined') window.TwoTowerModel = TwoTowerModel;
if (typeof module !== 'undefined') module.exports = TwoTowerModel;
