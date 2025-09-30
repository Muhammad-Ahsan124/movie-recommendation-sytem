// script.js
// Model, training and UI logic for Matrix Factorization recommender using TensorFlow.js

// Exposed global model variable
var model = null;

// Simple helper to update status/result text
function updateStatus(msg, isError=false) {
  const el = document.getElementById('result');
  if (!el) return;
  el.textContent = msg;
  el.style.color = isError ? '#b91c1c' : '';
}

// Populate user and movie dropdowns after data loaded
function populateDropdowns() {
  const userSelect = document.getElementById('user-select');
  const movieSelect = document.getElementById('movie-select');
  if (!userSelect || !movieSelect) return;

  // Populate users: movieLens uses 1..numUsers
  userSelect.innerHTML = '';
  for (let u = 1; u <= numUsers; u++) {
    const opt = document.createElement('option');
    opt.value = u;
    opt.textContent = `User ${u}`;
    userSelect.appendChild(opt);
  }

  // Populate movies (use titles)
  movieSelect.innerHTML = '';
  // Sort by id ascending for stable display
  const movieIds = Object.keys(items).map(k => parseInt(k,10)).sort((a,b)=>a-b);
  movieIds.forEach(id => {
    const opt = document.createElement('option');
    opt.value = id;
    const title = items[id] ? items[id].title : `Movie ${id}`;
    opt.textContent = `${id} — ${title}`;
    movieSelect.appendChild(opt);
  });
}

/**
 * createModel(numUsers, numMovies, latentDim)
 *
 * Matrix factorization with user and item embeddings and optional biases.
 *
 * Inputs:
 *  - userInput: scalar integer (user id)
 *  - movieInput: scalar integer (movie id)
 *
 * Embeddings:
 *  - userEmbedding: (numUsers+1, latentDim)
 *  - movieEmbedding: (numMovies+1, latentDim)
 *  - userBias: (numUsers+1, 1)
 *  - movieBias: (numMovies+1, 1)
 *
 * Prediction: dot(userVec, movieVec) + userBias + movieBias + globalBias
 */
function createModel(numUsersArg, numMoviesArg, latentDim=32) {
  const numUsersLocal = numUsersArg;
  const numMoviesLocal = numMoviesArg;

  // Inputs (integer ids)
  const userInput = tf.input({ shape: [1], dtype: 'int32', name: 'userInput' });
  const movieInput = tf.input({ shape: [1], dtype: 'int32', name: 'movieInput' });

  // Embedding layers ( +1 to allow ids == num to be handled properly; MovieLens ids start at 1 )
  const userEmbeddingLayer = tf.layers.embedding({
    inputDim: numUsersLocal + 1,
    outputDim: latentDim,
    embeddingsInitializer: 'glorotUniform',
    name: 'userEmbedding'
  });

  const movieEmbeddingLayer = tf.layers.embedding({
    inputDim: numMoviesLocal + 1,
    outputDim: latentDim,
    embeddingsInitializer: 'glorotUniform',
    name: 'movieEmbedding'
  });

  // Bias embeddings
  const userBiasLayer = tf.layers.embedding({
    inputDim: numUsersLocal + 1,
    outputDim: 1,
    embeddingsInitializer: 'zeros',
    name: 'userBias'
  });

  const movieBiasLayer = tf.layers.embedding({
    inputDim: numMoviesLocal + 1,
    outputDim: 1,
    embeddingsInitializer: 'zeros',
    name: 'movieBias'
  });

  // Apply embeddings
  // result shapes: [batch, 1, latentDim] for embeddings and [batch,1,1] for biases
  const userVec = userEmbeddingLayer.apply(userInput); // shape: [batch,1,latentDim]
  const movieVec = movieEmbeddingLayer.apply(movieInput); // shape: [batch,1,latentDim]

  const userBias = userBiasLayer.apply(userInput); // [batch,1,1]
  const movieBias = movieBiasLayer.apply(movieInput); // [batch,1,1]

  // Flatten embeddings and biases to shape [batch, latentDim] and [batch,1]
  const userVecFlat = tf.layers.flatten().apply(userVec);
  const movieVecFlat = tf.layers.flatten().apply(movieVec);
  const userBiasFlat = tf.layers.flatten().apply(userBias);
  const movieBiasFlat = tf.layers.flatten().apply(movieBias);

  // Dot product of user and movie vectors -> [batch, 1] (tf.layers.dot returns shape [batch,1])
  const dot = tf.layers.dot({ axes: -1 }).apply([userVecFlat, movieVecFlat]);

  // Global bias - a trainable scalar variable
  const globalBias = tf.variable(tf.scalar(0.0), true, 'globalBias');

  // Sum dot + userBias + movieBias + globalBias
  // We need to create layers that can add tensors. Use tf.layers.add in combination with constant for globalBias.
  // Convert globalBias scalar to a layer by creating a small custom layer that adds it (we'll use tf.layers.add with a lambda)
  // Simpler: use tf.layers.add to combine dot + userBias + movieBias, then create a final Dense layer with bias initialized to globalBias.
  // We'll add them: sum1 = dot + userBias + movieBias
  const sum1 = tf.layers.add().apply([dot, userBiasFlat, movieBiasFlat]); // [batch,1]

  // Optionally pass through activation (linear)
  // To include global bias as trainable bias we can use a Dense layer with units=1 and useBias=true but kernelInitializer zeros and bias initializer from globalBias
  const out = tf.layers.dense({
    units: 1,
    useBias: true,
    kernelInitializer: 'zeros',
    biasInitializer: 'zeros',
    activation: 'linear',
    name: 'predictionDense'
  }).apply(sum1);

  // Build model
  const mfModel = tf.model({
    inputs: [userInput, movieInput],
    outputs: out,
    name: 'matrixFactorizationModel'
  });

  return mfModel;
}

/**
 * trainModel()
 * Trains the model on the parsed ratings arrays.
 * Uses userIdArray, itemIdArray, ratingValueArray from data.js
 */
async function trainModel() {
  try {
    updateStatus('Preparing model and training data...');

    // choose latent dim (smaller for speed in-browser)
    const latentDim = 32;

    // Create model
    model = createModel(numUsers, numMovies, latentDim);

    // Compile
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'meanSquaredError'
    });

    // Prepare tensors
    // Convert to shape [n,1]
    const usersTensor = tf.tensor2d(userIdArray, [userIdArray.length, 1], 'int32');
    const itemsTensor = tf.tensor2d(itemIdArray, [itemIdArray.length, 1], 'int32');
    const ratingsTensor = tf.tensor2d(ratingValueArray, [ratingValueArray.length, 1], 'float32');

    updateStatus('Starting training (this may take a bit)...');

    // Fit model with callbacks to update UI on epoch end
    const epochs = 8;
    const batchSize = 64;
    await model.fit([usersTensor, itemsTensor], ratingsTensor, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          updateStatus(`Training... Epoch ${epoch + 1}/${epochs} — loss: ${logs.loss.toFixed(4)}`);
          // allow UI to update
          await tf.nextFrame();
        }
      }
    });

    // Dispose tensors used for training
    usersTensor.dispose();
    itemsTensor.dispose();
    ratingsTensor.dispose();

    updateStatus('Training complete — model is ready. Select a user & movie, then click Predict Rating.');
  } catch (err) {
    console.error(err);
    updateStatus('Training failed: ' + (err && err.message ? err.message : err), true);
  }
}

/**
 * predictRating()
 * Triggered by the UI button to predict rating for the selected user & movie.
 */
async function predictRating() {
  try {
    if (!model) {
      updateStatus('Model not ready yet. Please wait for training to complete.', true);
      return;
    }

    const userSelect = document.getElementById('user-select');
    const movieSelect = document.getElementById('movie-select');
    if (!userSelect || !movieSelect) return;

    const userId = parseInt(userSelect.value, 10);
    const movieId = parseInt(movieSelect.value, 10);

    if (Number.isNaN(userId) || Number.isNaN(movieId)) {
      updateStatus('Invalid user or movie selection.', true);
      return;
    }

    updateStatus('Predicting...');

    // Create input tensors of shape [1,1] with dtype int32
    const u = tf.tensor2d([userId], [1, 1], 'int32');
    const m = tf.tensor2d([movieId], [1, 1], 'int32');

    // Predict (model output shape [1,1])
    let pred = model.predict([u, m]);

    // pred may be tensor or array; if array, take first
    if (Array.isArray(pred)) pred = pred[0];

    const predData = await pred.data();
    const rawValue = predData[0];

    // The model was trained on raw rating scale (1-5). Clamp for safety.
    const rating = Math.min(5, Math.max(1, rawValue));

    // show result with movie title if available
    const title = (items && items[movieId] && items[movieId].title) ? items[movieId].title : `Movie ${movieId}`;

    updateStatus(`Predicted rating for User ${userId} → "${title}": ${rating.toFixed(2)} (raw: ${rawValue.toFixed(3)})`);

    // dispose tensors
    u.dispose();
    m.dispose();
    if (pred && pred.dispose) pred.dispose();

  } catch (err) {
    console.error(err);
    updateStatus('Prediction failed: ' + (err && err.message ? err.message : err), true);
  }
}

// On load: fetch data, populate UI, train model
window.onload = async function() {
  try {
    updateStatus('Initializing — fetching dataset...');
    await loadData();           // loadData is defined in data.js
    populateDropdowns();        // populate selects
    // Train model (async). Training updates the UI as it proceeds.
    trainModel();
  } catch (err) {
    console.error(err);
    updateStatus('Initialization failed: ' + (err && err.message ? err.message : err), true);
  }
};
