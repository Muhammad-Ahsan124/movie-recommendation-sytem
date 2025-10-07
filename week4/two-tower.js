// two-tower.js
// Implements TwoTowerModel (embedding tables + in-batch softmax/BPR training) and DeepRecModel (embeddings + MLP taking optional genre and user features).
// Note: This file is unchanged in core behavior but is included here to ensure you have the full project files.

class TwoTowerModel {
  constructor(numUsers, numItems, embDim=32) {
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.embDim = embDim;

    this.userEmbed = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05), true, 'userEmbed');
    this.itemEmbed = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05), true, 'itemEmbed');
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
}

class DeepRecModel {
  constructor(config) {
    this.numUsers = config.numUsers;
    this.numItems = config.numItems;
    this.embDim = config.embDim || 32;
    this.useGenres = !!config.useGenres;
    this.useUserFeat = !!config.useUserFeat;
    this.itemMeta = config.itemMeta || new Map();
    this.userFeatArray = config.userFeatArray || null;

    this.userEmbed = tf.variable(tf.randomNormal([this.numUsers, this.embDim], 0, 0.05), true, 'dl_userEmbed');
    this.itemEmbed = tf.variable(tf.randomNormal([this.numItems, this.embDim], 0, 0.05), true, 'dl_itemEmbed');

    this.genreDim = 0;
    for (const v of this.itemMeta.values()) { if (v.genres && v.genres.length) { this.genreDim = v.genres.length; break; } }
    if (!this.genreDim && this.useGenres) this.genreDim = 19;

    this.userFeatDim = (this.useUserFeat && this.userFeatArray && this.userFeatArray.length>0) ? this.userFeatArray[0].length : 0;

    this.inputDim = this.embDim + this.embDim + (this.useGenres ? this.genreDim : 0) + (this.useUserFeat ? this.userFeatDim : 0);
    this.dense1 = tf.layers.dense({units: Math.max(64, this.inputDim*2), activation: 'relu', useBias: true});
    this.dense2 = tf.layers.dense({units: 32, activation: 'relu', useBias: true});
    this.outDense = tf.layers.dense({units: 1, activation: null, useBias: true});

    const dummy = tf.zeros([1, this.inputDim]);
    this.dense1.apply(dummy);
    this.dense2.apply(this.dense1.apply(dummy));
    this.outDense.apply(this.dense2.apply(this.dense1.apply(dummy)));
    dummy.dispose();
  }

  userEmbedLookup(uIdxArr) {
    return tf.tidy(() => tf.gather(this.userEmbed, tf.tensor1d(uIdxArr,'int32')));
  }
  itemEmbedLookup(iIdxArr) {
    return tf.tidy(() => tf.gather(this.itemEmbed, tf.tensor1d(iIdxArr,'int32')));
  }

  itemGenreLookup(iIdxArr) {
    if (!this.useGenres || this.genreDim===0) {
      return tf.zeros([iIdxArr.length, 0]);
    }
    const arr = iIdxArr.map(i => {
      return this._genreLookupForInternalIndex(i);
    });
    return tf.tensor2d(arr, [arr.length, this.genreDim]);
  }

  userFeatureLookup(uIdxArr) {
    if (!this.useUserFeat || !this.userFeatArray) return tf.zeros([uIdxArr.length, 0]);
    const arr = uIdxArr.map(u => this.userFeatArray[u] || new Array(this.userFeatDim).fill(0));
    return tf.tensor2d(arr, [arr.length, this.userFeatDim]);
  }

  async trainStep(uIdxArr, posIdxArr, optimizer, useBPR=false) {
    const trainableVars = [
      this.userEmbed, this.itemEmbed,
      ...this.dense1.trainableWeights.map(w => w.val),
      ...this.dense2.trainableWeights.map(w => w.val),
      ...this.outDense.trainableWeights.map(w => w.val)
    ];

    const that = this;
    const lossTensor = await optimizer.minimize(() => {
      return tf.tidy(() => {
        const B = uIdxArr.length;
        const Uembed = that.userEmbedLookup(uIdxArr);
        const Pembed = that.itemEmbedLookup(posIdxArr);
        const genreTensor = that.itemGenreLookup(posIdxArr);
        const userFeatTensor = that.userFeatureLookup(uIdxArr);

        let input = tf.concat([Uembed, Pembed], 1);
        if (that.useGenres && that.genreDim>0) input = tf.concat([input, genreTensor], 1);
        if (that.useUserFeat && that.userFeatDim>0) input = tf.concat([input, userFeatTensor], 1);

        const h1 = that.dense1.apply(input);
        const h2 = that.dense2.apply(h1);
        const out = that.outDense.apply(h2).reshape([B]);

        if (!useBPR) {
          const logits = tf.matMul(Uembed, Pembed, false, true);
          const labels = tf.oneHot(tf.range(0, B, 1, 'int32'), B);
          const losses = tf.losses.softmaxCrossEntropy(labels, logits, undefined, 'none');
          return losses.mean();
        } else {
          const negIdxArr = posIdxArr.map((_,i)=>posIdxArr[(i+1)%B]);
          const Nembed = that.itemEmbedLookup(negIdxArr);
          const sPos = tf.sum(tf.mul(Uembed, Pembed), 1);
          const sNeg = tf.sum(tf.mul(Uembed, Nembed), 1);
          const diff = sPos.sub(sNeg);
          const loss = tf.neg(tf.log(tf.sigmoid(diff).add(1e-9))).mean();
          return loss;
        }
      });
    }, true, trainableVars);

    const val = (await lossTensor.data())[0];
    lossTensor.dispose();
    return val;
  }

  setInternalItemGenres(arrayOfGenreArrays) {
    this._internalIdxToGenre = arrayOfGenreArrays;
    this._genreLookupForInternalIndex = (i) => {
      if (this._internalIdxToGenre && this._internalIdxToGenre[i]) return this._internalIdxToGenre[i];
      return new Array(this.genreDim).fill(0);
    };
  }

  async getUserEmbedding(uIdx) {
    return tf.tidy(() => this.userEmbed.gather(tf.tensor1d([uIdx], 'int32')).squeeze());
  }

  async scoreAllItems(userEmbTensor) {
    return tf.tidy(() => {
      let u = userEmbTensor;
      if (u.rank === 1) u = u.expandDims(0);
      const itemEmbMatrix = this.itemEmbed;
      const dotScores = tf.matMul(u, itemEmbMatrix, false, true).squeeze();

      let mlpScores = null;
      if (this.useGenres || this.useUserFeat) {
        let genreMat = null;
        if (this.genreDim>0 && this._internalIdxToGenre) genreMat = tf.tensor2d(this._internalIdxToGenre, [this._internalIdxToGenre.length, this.genreDim]);
        else genreMat = tf.zeros([this.numItems, 0]);

        let userFeatMat = null;
        if (this.userFeatDim>0 && this.userFeatArray) userFeatMat = tf.tensor2d(this.userFeatArray, [this.userFeatArray.length, this.userFeatDim]);
        else userFeatMat = tf.zeros([this.numItems, 0]);

        const uRepeated = u.tile([this.numItems,1]);
        const input = tf.concat([uRepeated, itemEmbMatrix, (this.genreDim>0?genreMat:tf.zeros([this.numItems,0])), (this.userFeatDim>0?userFeatMat:tf.zeros([this.numItems,0]))], 1);
        const h1 = this.dense1.apply(input);
        const h2 = this.dense2.apply(h1);
        mlpScores = this.outDense.apply(h2).reshape([this.numItems]);
        genreMat.dispose(); userFeatMat.dispose(); uRepeated.dispose(); input.dispose(); h1.dispose(); h2.dispose();
      }

      if (mlpScores) {
        const combined = dotScores.add(mlpScores);
        dotScores.dispose(); if (mlpScores) mlpScores.dispose();
        return combined;
      } else {
        return dotScores;
      }
    });
  }
}
