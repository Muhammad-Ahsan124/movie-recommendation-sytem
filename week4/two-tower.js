// two-tower.js
// Implements TwoTowerModel (embedding tables + in-batch softmax/BPR training) and DeepRecModel (embeddings + MLP taking optional genre and user features).

class TwoTowerModel {
  constructor(numUsers, numItems, embDim=32) {
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.embDim = embDim;

    // Use unique variable names with timestamp to prevent conflicts
    const timestamp = Date.now();
    this.userEmbed = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05), true, `userEmbed_${timestamp}`);
    this.itemEmbed = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05), true, `itemEmbed_${timestamp}`);
  }

  userEmbedLookup(uIdxArr) {
    return tf.tidy(() => {
      const ids = tf.tensor1d(uIdxArr, 'int32');
      return tf.gather(this.userEmbed, ids);
    });
  }
  itemEmbedLookup(iIdxArr) {
    return tf.tidy(() => {
      const ids = tf.tensor1d(iIdxArr, 'int32');
      return tf.gather(this.itemEmbed, ids);
    });
  }

  async trainStepInBatch(uIdxArr, posIdxArr, optimizer, useBPR=false) {
    const varList = [this.userEmbed, this.itemEmbed];
    const lossScalar = await optimizer.minimize(() => {
      return tf.tidy(() => {
        const U = this.userEmbedLookup(uIdxArr);
        const P = this.itemEmbedLookup(posIdxArr);
        const logits = tf.matMul(U, P, false, true);
        if (!useBPR) {
          const B = uIdxArr.length;
          const labels = tf.oneHot(tf.range(0, B, 1, 'int32'), B);
          const losses = tf.losses.softmaxCrossEntropy(labels, logits, undefined, 'none');
          return losses.mean();
        } else {
          const B = uIdxArr.length;
          const negIdxArr = posIdxArr.map((_,i) => posIdxArr[(i+1)%B]);
          const N = this.itemEmbedLookup(negIdxArr);
          const sPos = tf.sum(tf.mul(U, P), 1);
          const sNeg = tf.sum(tf.mul(U, N), 1);
          const diff = sPos.sub(sNeg);
          const loss = tf.neg(tf.log(tf.sigmoid(diff).add(1e-9))).mean();
          return loss;
        }
      });
    }, true, varList);

    const val = (await lossScalar.data())[0];
    lossScalar.dispose();
    return val;
  }

  async getItemEmbeddings(indexArr) {
    return tf.tidy(() => this.itemEmbedLookup(indexArr));
  }

  async getUserEmbedding(uIdx) {
    return tf.tidy(() => this.userEmbed.gather(tf.tensor1d([uIdx], 'int32')).squeeze());
  }

  async scoreAllItems(userEmbTensor) {
    return tf.tidy(() => {
      let u = userEmbTensor;
      if (u.rank === 1) u = u.expandDims(0);
      const s = tf.matMul(u, this.itemEmbed, false, true).squeeze();
      return s;
    });
  }

  dispose() {
    if (this.userEmbed) {
      this.userEmbed.dispose();
    }
    if (this.itemEmbed) {
      this.itemEmbed.dispose();
    }
  }
}

class DeepRecModel {
  constructor(config) {
    this.numUsers = config.numUsers;
    this.numItems = config.numItems;
    this.embDim = config.embDim || 32;
    this.useGenres = !!config.useGenres;
    this.useUserFeat = !!config.useUserFeat;
    this.userFeatArray = config.userFeatArray || null;
    this.itemGenresArray = config.itemGenresArray || null;
    this.genreDim = config.genreDim || 0;

    // Use unique variable names with timestamp to prevent conflicts
    const timestamp = Date.now();
    this.userEmbed = tf.variable(tf.randomNormal([this.numUsers, this.embDim], 0, 0.05), true, `dl_userEmbed_${timestamp}`);
    this.itemEmbed = tf.variable(tf.randomNormal([this.numItems, this.embDim], 0, 0.05), true, `dl_itemEmbed_${timestamp}`);

    this.userFeatDim = (this.useUserFeat && this.userFeatArray && this.userFeatArray.length>0) ? this.userFeatArray[0].length : 0;

    // Input dimension calculation: user_emb + item_emb + genres + user_features
    this.inputDim = this.embDim + this.embDim + (this.useGenres ? this.genreDim : 0) + (this.useUserFeat ? this.userFeatDim : 0);
    
    // MLP layers with unique names - FIXED: Create proper layer variables
    this.dense1Weights = tf.variable(
      tf.randomNormal([this.inputDim, Math.max(64, this.inputDim)], 0, 0.05), 
      true, 
      `dense1_weights_${timestamp}`
    );
    this.dense1Bias = tf.variable(
      tf.zeros([Math.max(64, this.inputDim)]), 
      true, 
      `dense1_bias_${timestamp}`
    );
    
    this.dense2Weights = tf.variable(
      tf.randomNormal([Math.max(64, this.inputDim), 32], 0, 0.05), 
      true, 
      `dense2_weights_${timestamp}`
    );
    this.dense2Bias = tf.variable(
      tf.zeros([32]), 
      true, 
      `dense2_bias_${timestamp}`
    );
    
    this.outWeights = tf.variable(
      tf.randomNormal([32, 1], 0, 0.05), 
      true, 
      `out_weights_${timestamp}`
    );
    this.outBias = tf.variable(
      tf.zeros([1]), 
      true, 
      `out_bias_${timestamp}`
    );
  }

  userEmbedLookup(uIdxArr) {
    return tf.tidy(() => tf.gather(this.userEmbed, tf.tensor1d(uIdxArr,'int32')));
  }
  itemEmbedLookup(iIdxArr) {
    return tf.tidy(() => tf.gather(this.itemEmbed, tf.tensor1d(iIdxArr,'int32')));
  }

  itemGenreLookup(iIdxArr) {
    if (!this.useGenres || !this.itemGenresArray || this.genreDim===0) {
      return tf.zeros([iIdxArr.length, 0]);
    }
    const arr = iIdxArr.map(i => {
      return this.itemGenresArray[i] || new Array(this.genreDim).fill(0);
    });
    return tf.tensor2d(arr, [arr.length, this.genreDim]);
  }

  userFeatureLookup(uIdxArr) {
    if (!this.useUserFeat || !this.userFeatArray) return tf.zeros([uIdxArr.length, 0]);
    const arr = uIdxArr.map(u => this.userFeatArray[u] || new Array(this.userFeatDim).fill(0));
    return tf.tensor2d(arr, [arr.length, this.userFeatDim]);
  }

  // FIXED: Proper MLP forward pass with manual layers
  mlpForward(input) {
    return tf.tidy(() => {
      const h1 = tf.relu(tf.add(tf.matMul(input, this.dense1Weights), this.dense1Bias));
      const h2 = tf.relu(tf.add(tf.matMul(h1, this.dense2Weights), this.dense2Bias));
      const out = tf.add(tf.matMul(h2, this.outWeights), this.outBias);
      return out;
    });
  }

  async trainStep(uIdxArr, posIdxArr, optimizer, useBPR=false) {
    // FIXED: Collect all trainable variables properly
    const trainableVars = [
      this.userEmbed, 
      this.itemEmbed,
      this.dense1Weights,
      this.dense1Bias,
      this.dense2Weights, 
      this.dense2Bias,
      this.outWeights,
      this.outBias
    ];

    const that = this;
    const lossScalar = await optimizer.minimize(() => {
      return tf.tidy(() => {
        const B = uIdxArr.length;
        const Uembed = that.userEmbedLookup(uIdxArr);        // [B, embDim]
        const Pembed = that.itemEmbedLookup(posIdxArr);      // [B, embDim]
        const genreTensor = that.itemGenreLookup(posIdxArr); // [B, genreDim]
        const userFeatTensor = that.userFeatureLookup(uIdxArr); // [B, userFeatDim]

        // Build MLP input by concatenating all features
        const inputParts = [Uembed, Pembed];
        if (that.useGenres && that.genreDim > 0) inputParts.push(genreTensor);
        if (that.useUserFeat && that.userFeatDim > 0) inputParts.push(userFeatTensor);
        
        const input = tf.concat(inputParts, 1); // [B, inputDim]

        // Forward pass through MLP
        const mlpOut = that.mlpForward(input).reshape([B]); // [B]

        if (!useBPR) {
          // In-batch softmax loss
          const logits = tf.matMul(Uembed, Pembed, false, true); // [B, B]
          const labels = tf.oneHot(tf.range(0, B, 1, 'int32'), B);
          const losses = tf.losses.softmaxCrossEntropy(labels, logits, undefined, 'none');
          return losses.mean();
        } else {
          // BPR pairwise loss
          const negIdxArr = posIdxArr.map((_,i) => posIdxArr[(i+1)%B]);
          const Nembed = that.itemEmbedLookup(negIdxArr);
          const sPos = tf.sum(tf.mul(Uembed, Pembed), 1);
          const sNeg = tf.sum(tf.mul(Uembed, Nembed), 1);
          const diff = sPos.sub(sNeg);
          const loss = tf.neg(tf.log(tf.sigmoid(diff).add(1e-9))).mean();
          return loss;
        }
      });
    }, true, trainableVars);

    const val = (await lossScalar.data())[0];
    lossScalar.dispose();
    return val;
  }

  async getUserEmbedding(uIdx) {
    return tf.tidy(() => this.userEmbed.gather(tf.tensor1d([uIdx], 'int32')).squeeze());
  }

  async scoreAllItems(userEmbTensor) {
    return tf.tidy(() => {
      let u = userEmbTensor;
      if (u.rank === 1) u = u.expandDims(0); // [1, embDim]
      
      const itemEmbMatrix = this.itemEmbed; // [numItems, embDim]
      
      // Dot product scores (baseline two-tower)
      const dotScores = tf.matMul(u, itemEmbMatrix, false, true).squeeze(); // [numItems]

      // MLP-enhanced scores if using additional features
      if ((this.useGenres && this.genreDim > 0) || (this.useUserFeat && this.userFeatDim > 0)) {
        // Prepare repeated user embedding for all items
        const uRepeated = u.tile([this.numItems, 1]); // [numItems, embDim]
        
        // Prepare genre features for all items
        let genreMat = tf.zeros([this.numItems, 0]);
        if (this.useGenres && this.genreDim > 0 && this.itemGenresArray) {
          genreMat = tf.tensor2d(this.itemGenresArray, [this.numItems, this.genreDim]);
        }
        
        // Prepare user features repeated for all items  
        let userFeatMat = tf.zeros([this.numItems, 0]);
        if (this.useUserFeat && this.userFeatDim > 0 && this.userFeatArray) {
          // For scoring, we use the features of the single user repeated for all items
          const singleUserFeat = this.userFeatArray[0] || new Array(this.userFeatDim).fill(0);
          userFeatMat = tf.tile(tf.tensor2d([singleUserFeat], [1, this.userFeatDim]), [this.numItems, 1]);
        }

        // Build MLP input
        const inputParts = [uRepeated, itemEmbMatrix];
        if (this.useGenres && this.genreDim > 0) inputParts.push(genreMat);
        if (this.useUserFeat && this.userFeatDim > 0) inputParts.push(userFeatMat);
        
        const input = tf.concat(inputParts, 1); // [numItems, inputDim]

        // MLP forward pass
        const mlpScores = this.mlpForward(input).reshape([this.numItems]); // [numItems]

        // Combine dot product and MLP scores
        const combined = dotScores.add(mlpScores);
        return combined;
      } else {
        // Just use dot product scores
        return dotScores;
      }
    });
  }

  dispose() {
    const vars = [
      this.userEmbed, this.itemEmbed,
      this.dense1Weights, this.dense1Bias,
      this.dense2Weights, this.dense2Bias,
      this.outWeights, this.outBias
    ];
    
    vars.forEach(v => {
      if (v) v.dispose();
    });
  }
}
