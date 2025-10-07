// two-tower.js
class TwoTowerModel {
  constructor(numUsers, numItems, embDim, useBPR = false) {
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.embDim = embDim;
    this.useBPR = useBPR;
    this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05));
    this.itemEmbedding = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05));
    this.optimizer = tf.train.adam(0.001);
  }

  userForward(userIdx) { return tf.gather(this.userEmbedding, userIdx); }
  itemForward(itemIdx) { return tf.gather(this.itemEmbedding, itemIdx); }

  score(uEmb, iEmb) {
    return tf.sum(tf.mul(uEmb, iEmb), -1); // dot product
  }

  trainStep(userIdx, posIdx) {
    return this.optimizer.minimize(() => {
      const uEmb = this.userForward(userIdx);
      const iEmb = this.itemForward(posIdx);
      const logits = tf.matMul(uEmb, this.itemEmbedding, false, true); // in-batch negatives
      const labels = tf.range(0, userIdx.shape[0], 1, 'int32');
      const loss = this.useBPR
        ? this.bprLoss(uEmb, iEmb)
        : tf.losses.softmaxCrossEntropy(tf.oneHot(labels, logits.shape[1]), logits);
      return loss;
    }, true);
  }

  bprLoss(uEmb, iPos) {
    const batchSize = uEmb.shape[0];
    const negIdx = tf.randomUniform([batchSize], 0, this.numItems, 'int32');
    const iNeg = this.itemForward(negIdx);
    const posScore = this.score(uEmb, iPos);
    const negScore = this.score(uEmb, iNeg);
    return tf.neg(tf.mean(tf.logSigmoid(tf.sub(posScore, negScore))));
  }

  getUserEmbedding(uIdx) { return this.userForward(uIdx); }

  getScoresForAllItems(uEmb) {
    return tf.matMul(uEmb, this.itemEmbedding, false, true);
  }
}

// Deep MLP Model
class DeepMLPModel {
  constructor(numUsers, numItems, embDim, userFeatDim, itemFeatDim) {
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.embDim = embDim;

    this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05));
    this.itemEmbedding = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05));

    // Project auxiliary features to embDim first
    this.userDense = tf.layers.dense({ units: embDim, activation: 'relu' });
    this.itemDense = tf.layers.dense({ units: embDim, activation: 'relu' });

    // MLP head
    this.hidden = tf.layers.dense({ units: embDim, activation: 'relu' });
    this.outDense = tf.layers.dense({ units: 1 });

    this.optimizer = tf.train.adam(0.001);
  }

  forwardUser(userIdx, userFeat) {
    const uEmb = tf.gather(this.userEmbedding, userIdx);
    const uFeatProj = this.userDense.apply(userFeat);
    return tf.add(uEmb, uFeatProj);
  }

  forwardItem(itemIdx, itemFeat) {
    const iEmb = tf.gather(this.itemEmbedding, itemIdx);
    const iFeatProj = this.itemDense.apply(itemFeat);
    return tf.add(iEmb, iFeatProj);
  }

  trainStep(userIdx, posIdx, userFeat, itemFeat) {
    return this.optimizer.minimize(() => {
      const uVec = this.forwardUser(userIdx, userFeat);
      const iVec = this.forwardItem(posIdx, itemFeat);
      const score = this.hidden.apply(tf.concat([uVec, iVec], -1));
      const logits = this.outDense.apply(score);
      const labels = tf.onesLike(logits);
      const loss = tf.losses.sigmoidCrossEntropy(labels, logits);
      return loss;
    }, true);
  }

  getScoresForAllItems(uEmb, itemFeatMat) {
    const uExp = uEmb.expandDims(1);
    const tiled = tf.tile(uExp, [1, this.numItems, 1]);
    const itemFeatProj = this.itemDense.apply(itemFeatMat);
    const concat = tf.concat([tiled, itemFeatProj.expandDims(0).squeeze()], -1);
    const h = this.hidden.apply(concat);
    return this.outDense.apply(h).squeeze();
  }
}
