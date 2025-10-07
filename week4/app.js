async function buildTwoTower(userCount, itemCount, embeddingDim, bprLoss) {
  const userInput = tf.input({shape:[1], name:'userInput'});
  const itemInput = tf.input({shape:[1], name:'itemInput'});

  const userEmbed = tf.layers.embedding({inputDim:userCount+1, outputDim:embeddingDim}).apply(userInput);
  const itemEmbed = tf.layers.embedding({inputDim:itemCount+1, outputDim:embeddingDim}).apply(itemInput);

  const userFlat = tf.layers.flatten().apply(userEmbed);
  const itemFlat = tf.layers.flatten().apply(itemEmbed);

  const dot = tf.layers.dot({axes:1, normalize:true}).apply([userFlat, itemFlat]);
  const out = tf.layers.dense({units:1, activation:'sigmoid'}).apply(dot);

  const model = tf.model({inputs:[userInput,itemInput], outputs:out});
  model.compile({optimizer:tf.train.adam(0.001),
                 loss:bprLoss?'hinge':'binaryCrossentropy'});
  return model;
}
