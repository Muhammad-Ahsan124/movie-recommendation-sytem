class TwoTowerModel {
  constructor(numUsers, numItems, embDim) {
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.embDim = embDim;

    // Two embedding tables
    this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05));
    this.itemEmbedding = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05));
  }

  userForward(uIdxTensor) {
    return tf.gather(this.userEmbedding, uIdxTensor);
  }

  itemForward(iIdxTensor) {
    return tf.gather(this.itemEmbedding, iIdxTensor);
  }

  score(uEmb, iEmb) {
    return tf.matMul(uEmb, iEmb, false, true);
  }

  async trainBatch(uIdxs, iIdxs, optimizer) {
    const u = tf.tensor1d(uIdxs, 'int32');
    const i = tf.tensor1d(iIdxs, 'int32');
    const lossFn = () => {
      const uEmb = this.userForward(u);
      const iEmb = this.itemForward(i);
      const logits = this.score(uEmb, iEmb);
      const labels = tf.tensor1d([...Array(uIdxs.length).keys()], 'int32');
      const loss = tf.losses.softmaxCrossEntropy(
        tf.oneHot(labels, iIdxs.length), logits
      ).mean();
      return loss;
    };
    const { value, grads } = tf.variableGrads(lossFn);
    optimizer.applyGradients(grads);
    const lossVal = value.dataSync()[0];
    tf.dispose([u, i, value, grads]);
    return lossVal;
  }

  getUserEmbedding(uTensor) {
    return this.userForward(uTensor);
  }
}
