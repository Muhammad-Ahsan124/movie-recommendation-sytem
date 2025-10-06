app.js
/* Main application glue for the Two-Tower TF.js demo
   - loads MovieLens 100K u.data and u.item from data/
   - prepares data structures and index mapping
   - orchestrates training, charting, projection, and testing
*/

/* global TwoTowerModel, tf */

(async () => {
  // DOM elements
  const btnLoad = document.getElementById('btnLoad');
  const btnTrain = document.getElementById('btnTrain');
  const btnTest = document.getElementById('btnTest');
  const status = document.getElementById('status');
  const lossCanvas = document.getElementById('lossCanvas');
  const projCanvas = document.getElementById('projCanvas');
  const resultsTables = document.getElementById('resultsTables');
  const hoverTitle = document.getElementById('hoverTitle');

  // Config inputs
  const maxInteractIn = document.getElementById('maxInteract');
  const embDimIn = document.getElementById('embDim');
  const epochsIn = document.getElementById('epochs');
  const batchSizeIn = document.getElementById('batchSize');
  const lrIn = document.getElementById('lr');
  const lossChoice = document.getElementById('lossChoice');
  const useMLP = document.getElementById('useMLP');

  // Internal state
  let interactions = []; // {userId, itemId, rating, ts}
  let items = new Map(); // itemId -> {title, year, genres: [0/1 flags], rawGenres}
  let userToRated = new Map(); // userId -> [{itemId, rating, ts}]
  let userIndex = new Map(), itemIndex = new Map();
  let indexToUser = [], indexToItem = [];
  let numUsers = 0, numItems = 0;
  let model = null, modelMLP = null;
  let lossHistory = [];
  let drawLossHandle = null;
  let embeddingsProjection = null;

  // Chart helpers (simple)
  const lossCtx = lossCanvas.getContext('2d');
  function clearCanvas(ctx, canvas) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }
  function drawLoss() {
    const w = lossCanvas.width = lossCanvas.clientWidth * devicePixelRatio;
    const h = lossCanvas.height = lossCanvas.clientHeight * devicePixelRatio;
    clearCanvas(lossCtx, lossCanvas);
    lossCtx.save();
    lossCtx.scale(devicePixelRatio, devicePixelRatio);
    lossCtx.lineWidth = 2;
    lossCtx.strokeStyle = "#2b6cb0";
    const maxPts = Math.min(lossHistory.length, 1000);
    if (maxPts < 2) { lossCtx.restore(); return; }
    const display = lossHistory.slice(-maxPts);
    const pad = 6;
    const cw = lossCanvas.clientWidth;
    const ch = lossCanvas.clientHeight;
    const maxVal = Math.max(...display);
    const minVal = Math.min(...display);
    lossCtx.beginPath();
    for (let i=0;i<display.length;i++){
      const x = pad + (i/(display.length-1))*(cw-2*pad);
      const y = pad + ((display[i]-minVal)/(maxVal-minVal+1e-8))*(ch-2*pad);
      const yInv = ch - y;
      if (i===0) lossCtx.moveTo(x,yInv); else lossCtx.lineTo(x,yInv);
    }
    lossCtx.stroke();
    // labels
    lossCtx.fillStyle = "#666";
    lossCtx.font = "12px sans-serif";
    lossCtx.fillText(`loss (last ${display.length})`, 8, 14);
    lossCtx.restore();
  }

  // Utility: shuffle array in-place (Fisher-Yates)
  function shuffle(array) {
    for (let i=array.length-1;i>0;i--) {
      const j = Math.floor(Math.random()*(i+1));
      [array[i], array[j]] = [array[j], array[i]];
    }
  }

  // Load data parsing
  async function loadData() {
    status.textContent = "Fetching data/u.item ...";
    const itemResp = await fetch('data/u.item');
    if (!itemResp.ok) throw new Error('Failed to fetch data/u.item');
    const itemTxt = await itemResp.text();

    status.textContent = "Parsing u.item ...";
    // u.item format: item_id|title|release_date|video_release_date|IMDb_URL|genre flags...
    const itemLines = itemTxt.split('\n').map(l=>l.trim()).filter(l=>l);
    items.clear();
    for (const line of itemLines) {
      const parts = line.split('|');
      const id = parseInt(parts[0]);
      const titleRaw = parts[1] || "";
      let year = null;
      const titleMatch = titleRaw.match(/(.*)\s+\((\d{4})\)\s*$/);
      let title = titleRaw;
      if (titleMatch) { title = titleMatch[1].trim(); year = parseInt(titleMatch[2]); }
      // Genres are last 19 boolean flags in MovieLens 100K (0/1)
      const genreFlags = parts.slice(5).map(x => parseInt(x || "0"));
      items.set(id, { title, year, genres: genreFlags, rawTitle: titleRaw });
    }

    status.textContent = "Fetching data/u.data ...";
    const dataResp = await fetch('data/u.data');
    if (!dataResp.ok) throw new Error('Failed to fetch data/u.data');
    const dataTxt = await dataResp.text();

    status.textContent = "Parsing u.data ...";
    const dataLines = dataTxt.split('\n').map(l=>l.trim()).filter(l=>l);
    interactions = [];
    for (const line of dataLines) {
      // u.data: user_id \t item_id \t rating \t timestamp
      const p = line.split('\t');
      if (p.length < 4) continue;
      const userId = parseInt(p[0]);
      const itemId = parseInt(p[1]);
      const rating = parseInt(p[2]);
      const ts = parseInt(p[3]);
      if (!items.has(itemId)) continue; // safeguard
      interactions.push({userId, itemId, rating, ts});
    }

    // Optionally limit interactions for memory
    const maxInteract = Math.max(1000, parseInt(maxInteractIn.value));
    if (interactions.length > maxInteract) {
      shuffle(interactions);
      interactions = interactions.slice(0, maxInteract);
    }

    // Build user->ratings
    userToRated.clear();
    for (const it of interactions) {
      if (!userToRated.has(it.userId)) userToRated.set(it.userId, []);
      userToRated.get(it.userId).push({ itemId: it.itemId, rating: it.rating, ts: it.ts });
    }
    // sort user rated lists for quick retrieval (rating desc, ts desc)
    for (const [u, arr] of userToRated.entries()) {
      arr.sort((a,b)=> {
        if (b.rating !== a.rating) return b.rating - a.rating;
        return b.ts - a.ts;
      });
    }

    // Build indexers (0-based)
    userIndex.clear(); itemIndex.clear();
    indexToUser = []; indexToItem = [];
    const userIds = Array.from(userToRated.keys()).sort((a,b)=>a-b);
    let uid = 0;
    for (const u of userIds) { userIndex.set(u, uid); indexToUser[uid]=u; uid++; }
    numUsers = uid;

    // items present in interactions only
    const presentItemsSet = new Set(interactions.map(x=>x.itemId));
    const itemIds = Array.from(presentItemsSet).sort((a,b)=>a-b);
    let iid = 0;
    for (const it of itemIds) { itemIndex.set(it, iid); indexToItem[iid]=it; iid++; }
    numItems = iid;

    // Create dense arrays for genres aligned to itemIndex
    const itemGenresMatrix = [];
    for (let i=0;i<numItems;i++) {
      const originalItemId = indexToItem[i];
      const meta = items.get(originalItemId);
      itemGenresMatrix.push(meta.genres || Array(19).fill(0));
    }

    // Rebuild interactions with indices
    const interactionsIndexed = interactions.map(it => ({
      u: userIndex.get(it.userId),
      i: itemIndex.get(it.itemId),
      rating: it.rating,
      ts: it.ts
    }));

    // Update internal references
    interactions = interactionsIndexed;
    status.textContent = `Loaded ${interactions.length} interactions, ${numUsers} users, ${numItems} items.`;
    btnTrain.disabled = false;
    btnTest.disabled = true;
    lossHistory = [];
    drawLoss();

    // Attach to global for model building
    window._mlData = {
      interactions,
      items,
      itemGenresMatrix,
      userIndex,
      itemIndex,
      indexToUser,
      indexToItem,
      numUsers,
      numItems,
    };
  }

  // Build dataset batches for training: yields {usersTensor, posItemsTensor, negItemsTensor?}
  function* batchGenerator(interactions, batchSize, negativeSamples=1) {
    const idxs = [...Array(interactions.length).keys()];
    shuffle(idxs);
    for (let start=0; start < idxs.length; start += batchSize) {
      const slice = idxs.slice(start, start+batchSize);
      const users = new Int32Array(slice.length);
      const pos = new Int32Array(slice.length);
      for (let i=0;i<slice.length;i++){
        users[i] = interactions[slice[i]].u;
        pos[i] = interactions[slice[i]].i;
      }
      // For BPR we sample negatives
      let negs = null;
      if (negativeSamples > 0) {
        negs = new Int32Array(slice.length * negativeSamples);
        for (let i=0;i<slice.length;i++){
          for (let k=0;k<negativeSamples;k++){
            negs[i*negativeSamples + k] = Math.floor(Math.random()*window._mlData.numItems);
          }
        }
      }
      yield { users, pos, negs };
    }
  }

  // Train orchestration
  async function trainClick() {
    if (!window._mlData) return;
    // read config
    const embDim = Math.max(8, parseInt(embDimIn.value));
    const epochs = Math.max(1, parseInt(epochsIn.value));
    const batchSize = Math.max(8, parseInt(batchSizeIn.value));
    const lr = parseFloat(lrIn.value) || 0.01;
    const lossMode = lossChoice.value;
    const useDeep = useMLP.checked;

    status.textContent = 'Initializing models...';
    // Dispose old models
    if (model) model.dispose();
    if (modelMLP) modelMLP.dispose();

    const { numUsers, numItems, itemGenresMatrix } = window._mlData;

    // Create two models:
    // 1) classic two-tower embedding dot-product
    model = new TwoTowerModel(numUsers, numItems, embDim, { useMLP:false });
    // 2) deep MLP two-tower (user & item MLPs using genres)
    modelMLP = new TwoTowerModel(numUsers, numItems, embDim, { useMLP:true, itemGenres: itemGenresMatrix });

    // optimizer
    const optimizer = tf.train.adam(lr);

    // Prepare interactions
    const inter = window._mlData.interactions;

    status.textContent = `Training (emb=${embDim}) — epochs: ${epochs}, batch: ${batchSize}, loss: ${lossMode}, MLP: ${useDeep ? 'ON' : 'OFF'}.`;
    lossHistory = [];
    drawLoss();

    // training loop
    for (let epoch=0; epoch<epochs; epoch++) {
      status.textContent = `Epoch ${epoch+1}/${epochs} — training...`;
      // generator
      const gen = batchGenerator(inter, batchSize, lossMode==='bpr' ? 1 : 0);
      let batchIdx = 0;
      for (const batch of gen) {
        // Closure for single step
        const usersArr = batch.users;
        const posArr = batch.pos;
        const negArr = batch.negs;

        // In-batch softmax: we will compute logits = U @ Ipos^T (batch x batch)
        // For MLP we use modelMLP; for classic use model
        const trainModel = useDeep ? modelMLP : model;

        const lossVal = await trainStep(trainModel, optimizer, usersArr, posArr, negArr, lossMode);
        const lossScalar = lossVal.dataSync()[0];
        lossHistory.push(lossScalar);
        if (batchIdx % 10 === 0) drawLoss();
        batchIdx++;
        // keep UI responsive
        await tf.nextFrame();
      }
      // epoch end
      drawLoss();
      status.textContent = `Finished epoch ${epoch+1}/${epochs}`;
      await tf.nextFrame();
    }

    status.textContent = 'Training complete. Computing projections...';
    btnTest.disabled = false;

    // compute item embeddings sample and project to 2D
    await computeProjection(useDeep ? modelMLP : model);
    status.textContent = 'Ready. Click Test to evaluate a random user (with ≥20 ratings).';
  }

  // Single training step: returns scalar loss tensor
  async function trainStep(modelObj, optimizer, usersArr, posArr, negArr, lossMode) {
    // usersArr and posArr are Int32Array
    return tf.tidy(() => {
      const users = tf.tensor1d(usersArr, 'int32');
      const pos = tf.tensor1d(posArr, 'int32');

      let lossTensor = tf.scalar(0);

      if (lossMode === 'inbatch') {
        // in-batch softmax:
        // U: (B, D), Ipos: (B, D) -> logits: (B, B) = U @ Ipos^T
        lossTensor = optimizer.minimize(() => {
          const U = modelObj.userForward(users); // [B,D]
          const Ipos = modelObj.itemForward(pos); // [B,D]
          const logits = tf.matMul(U, Ipos, false, true); // [B,B]
          const labels = tf.range(0, logits.shape[0], 1, 'int32'); // diagonal indexes
          const xent = tf.losses.sparseSoftmaxCrossEntropy(labels, logits);
          return xent;
        }, true, modelObj.trainableVars);
      } else if (lossMode === 'bpr') {
        // pairwise BPR: sample negs per positive
        const neg = tf.tensor1d(negArr, 'int32'); // shape [B]
        lossTensor = optimizer.minimize(() => {
          const U = modelObj.userForward(users); // [B,D]
          const Ipos = modelObj.itemForward(pos); // [B,D]
          const Ineg = modelObj.itemForward(neg); // [B,D]
          const sPos = tf.sum(tf.mul(U, Ipos), -1); // [B]
          const sNeg = tf.sum(tf.mul(U, Ineg), -1);
          // loss = -log sigmoid(sPos - sNeg)
          const diff = tf.sub(sPos, sNeg);
          const lossVec = tf.neg(tf.log(tf.sigmoid(diff).add(1e-8)));
          return tf.mean(lossVec);
        }, true, modelObj.trainableVars);
      } else {
        throw new Error('Unknown loss mode');
      }
      return lossTensor;
    });
  }

  // Compute item embedding projection using SVD (tf.linalg.svd). Sample up to 1000 items to display.
  async function computeProjection(modelObj) {
    // sample items
    const maxSample = 1000;
    const N = Math.min(window._mlData.numItems, maxSample);
    const perm = [...Array(window._mlData.numItems).keys()];
    shuffle(perm);
    const sampleIdx = perm.slice(0,N);
    // get embeddings
    const idxTensor = tf.tensor1d(sampleIdx, 'int32');
    const emb = tf.tidy(()=> modelObj.itemForward(idxTensor).arraySync()); // returns [N, D] (synchronous array)
    idxTensor.dispose();

    // center and compute SVD via tfjs
    const embTensor = tf.tensor2d(emb); // [N, D]
    const mean = embTensor.mean(0, true);
    const centered = embTensor.sub(mean);
    const svd = await tf.linalg.svd(centered, true); // returns {u, s, v}
    const U = svd[0]; // [N,N] or [N, min(N,D)] depending on backend
    const S = svd[1]; // vector
    const V = svd[2]; // [D,D]
    // Project to first 2 principal components using V (PCs are columns of V)
    const V2 = V.slice([0,0],[V.shape[0], 2]); // [D,2]
    const coords = centered.matMul(V2).arraySync(); // [N,2]
    embTensor.dispose(); mean.dispose(); centered.dispose(); svd[0].dispose(); svd[1].dispose(); svd[2].dispose();

    // map sampleIdx -> coords and titles
    embeddingsProjection = { sampleIdx, coords };
    drawProjection();
  }

  // Draw scatter on projCanvas with hover
  function drawProjection() {
    if (!embeddingsProjection) return;
    const { sampleIdx, coords } = embeddingsProjection;
    const w = projCanvas.width = projCanvas.clientWidth * devicePixelRatio;
    const h = projCanvas.height = projCanvas.clientHeight * devicePixelRatio;
    const ctx = projCanvas.getContext('2d');
    clearCanvas(ctx, projCanvas);
    ctx.save();
    ctx.scale(devicePixelRatio, devicePixelRatio);
    const cw = projCanvas.clientWidth;
    const ch = projCanvas.clientHeight;
    // find bounds
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of coords) { if (p[0]<minX) minX=p[0]; if (p[0]>maxX) maxX=p[0]; if (p[1]<minY) minY=p[1]; if (p[1]>maxY) maxY=p[1]; }
    const pad = 10;
    function toCanvas(x,y){
      const xN = (x - minX) / (maxX - minX + 1e-9);
      const yN = (y - minY) / (maxY - minY + 1e-9);
      return [pad + xN*(cw - 2*pad), ch - (pad + yN*(ch - 2*pad))];
    }
    // draw points
    ctx.fillStyle = "#2b6cb0";
    for (let i=0;i<coords.length;i++){
      const [x,y] = toCanvas(coords[i][0], coords[i][1]);
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI*2);
      ctx.fill();
    }
    ctx.restore();

    // hover handling
    projCanvas.onmousemove = (ev) => {
      const rect = projCanvas.getBoundingClientRect();
      const x = (ev.clientX - rect.left);
      const y = (ev.clientY - rect.top);
      // find nearest
      let best = -1; let bestDist = 1e9;
      for (let i=0;i<coords.length;i++){
        const [cx, cy] = toCanvas(coords[i][0], coords[i][1]);
        const dx = cx - x; const dy = cy - y;
        const d = dx*dx + dy*dy;
        if (d < bestDist) { best = i; bestDist = d; }
      }
      if (best >= 0 && bestDist < 100) {
        const idx = embeddingsProjection.sampleIdx[best];
        const itemId = window._mlData.indexToItem[idx];
        const meta = window._mlData.items.get(itemId);
        hoverTitle.style.left = (ev.pageX + 12) + 'px';
        hoverTitle.style.top = (ev.pageY + 12) + 'px';
        hoverTitle.textContent = `${meta.title} (${meta.year || 'n/a'})`;
        hoverTitle.style.display = 'block';
      } else {
        hoverTitle.style.display = 'none';
      }
    };
    projCanvas.onmouseleave = () => { hoverTitle.style.display = 'none'; };
  }

  // Test: pick random user with >=20 ratings, show left historical top10 and right recommendations
  async function testClick() {
    if (!window._mlData) return;
    const minRatings = 20;
    const usersWithCounts = [];
    for (let ui=0; ui<window._mlData.numUsers; ui++) {
      const origU = window._mlData.indexToUser[ui];
      const cnt = (window._mlData.userIndex.has(origU)) ? (window._mlData.items ? (window._mlData.userIndex.has(origU) && window._mlData.userIndex.size>=0 ? (window._mlData.userIndex.has(origU) ? 0:0) : 0) : 0) : 0;
      // easier: count from userToRated stored earlier (original ids)
    }
    // more direct: iterate original userToRated map
    const candidateOriginal = [];
    for (const [origU, arr] of window._mlData.items ? window._mlData.items : []) {} // no-op (safety)
    // We'll use userToRated stored earlier in app scope via window._mlData
    const candidates = [];
    const mapOriginalToRatings = new Map();
    for (const [origU, arr] of window._mlData.items ? window._mlData.items : []) {} // noop
    // userToRated (original ids) exists in top-level userToRated variable in closure? Not global. Rebuild quickly:
    // Recreate mapping from interactions to original userId from indexToUser arrays
    // We have indexToUser mapping: indexToUser[idx] => original userId
    // We can count interactions
    const counts = new Array(window._mlData.numUsers).fill(0);
    for (const it of window._mlData.interactions) counts[it.u]++;
    const eligible = [];
    for (let u=0; u<counts.length; u++) if (counts[u] >= minRatings) eligible.push(u);
    if (eligible.length === 0) {
      status.textContent = `No user with ≥${minRatings} ratings in the loaded subset. Reduce min or load more interactions.`;
      return;
    }
    const chosen = eligible[Math.floor(Math.random()*eligible.length)];
    const origUserId = window._mlData.indexToUser[chosen];

    status.textContent = `Testing for user index ${chosen} (orig id ${origUserId}) — computing recommendations...`;

    // Build set of items user rated (to exclude)
    const userRatedSet = new Set();
    // find all interactions with u==chosen:
    const ratedList = [];
    for (const it of window._mlData.interactions) {
      if (it.u === chosen) {
        userRatedSet.add(it.i);
        const origItemId = window._mlData.indexToItem[it.i];
        const meta = window._mlData.items.get(origItemId);
        ratedList.push({ itemIndex: it.i, rating: it.rating, ts: it.ts, title: meta.title, year: meta.year });
      }
    }
    // sort user's historical by rating desc then ts desc
    ratedList.sort((a,b)=> (b.rating - a.rating) || (b.ts - a.ts));
    const historicalTop10 = ratedList.slice(0,10);

    // Inference: compute user embedding and score against all items in batches to avoid OOM
    // We'll compute for both classic model and MLP model (if available)
    const k = 10;
    const classicTop = await recommendForUser(model, chosen, userRatedSet, k);
    const deepTop = await recommendForUser(modelMLP, chosen, userRatedSet, k);

    // Build table: 3 columns: Historical | Classic Retrieval | Deep (MLP)
    resultsTables.innerHTML = '';
    const leftCol = document.createElement('div'); leftCol.className='col';
    const midCol = document.createElement('div'); midCol.className='col';
    const rightCol = document.createElement('div'); rightCol.className='col';
    // historical
    const histTbl = document.createElement('table'); histTbl.className='results';
    const histHdr = histTbl.insertRow();
    histHdr.insertCell().outerHTML = '<th>Historical Top-10 (by rating, then recency)</th>';
    for (const h of historicalTop10) {
      const row = histTbl.insertRow();
      const c = row.insertCell();
      c.innerHTML = `<strong>${h.title}</strong> ${h.year ? '('+h.year+')':''} <div style="color:#666; font-size:12px;">rating: ${h.rating}, ts:${h.ts}</div>`;
    }
    leftCol.appendChild(histTbl);

    // classic
    const classicTbl = document.createElement('table'); classicTbl.className='results';
    const ch = classicTbl.insertRow();
    ch.insertCell().outerHTML = '<th>Classic Two-Tower (embedding dot-product)</th>';
    for (const c of classicTop) {
      const row = classicTbl.insertRow();
      const cell = row.insertCell();
      const meta = window._mlData.items.get(window._mlData.indexToItem[c.index]);
      cell.innerHTML = `<strong>${meta.title}</strong> ${meta.year? '('+meta.year+')':''} <div style="color:#666; font-size:12px;">score: ${c.score.toFixed(4)}</div>`;
    }
    midCol.appendChild(classicTbl);

    // deep
    const deepTbl = document.createElement('table'); deepTbl.className='results';
    const dh = deepTbl.insertRow();
    dh.insertCell().outerHTML = '<th>Deep Two-Tower (MLP with genres)</th>';
    for (const d of deepTop) {
      const row = deepTbl.insertRow();
      const cell = row.insertCell();
      const meta = window._mlData.items.get(window._mlData.indexToItem[d.index]);
      cell.innerHTML = `<strong>${meta.title}</strong> ${meta.year? '('+meta.year+')':''} <div style="color:#666; font-size:12px;">score: ${d.score.toFixed(4)}</div>`;
    }
    rightCol.appendChild(deepTbl);

    resultsTables.appendChild(leftCol);
    resultsTables.appendChild(midCol);
    resultsTables.appendChild(rightCol);

    status.textContent = `Test completed for user ${origUserId}.`;
  }

  // Recommend top-k using modelObj (TwoTowerModel)
  async function recommendForUser(modelObj, userIdx, userRatedSet, k=10) {
    // compute user embedding
    const uTensor = tf.tensor1d([userIdx], 'int32');
    const uEmb = modelObj.userForward(uTensor); // [1,D]
    // Now compute dot products against all items in batches
    const BATCH = 1024;
    const numItems = window._mlData.numItems;
    let results = [];
    for (let start=0; start<numItems; start += BATCH) {
      const end = Math.min(numItems, start+BATCH);
      const idxs = new Int32Array(end-start);
      for (let i=start;i<end;i++) idxs[i-start] = i;
      const itemIdxTensor = tf.tensor1d(idxs, 'int32');
      const itemEmb = modelObj.itemForward(itemIdxTensor); // [b,D]
      const logits = tf.matMul(uEmb, itemEmb, false, true).reshape([end-start]); // [b]
      const scores = await logits.array();
      for (let i=0;i<scores.length;i++) {
        const globalIndex = start + i;
        if (userRatedSet.has(globalIndex)) continue; // exclude seen
        results.push({ index: globalIndex, score: scores[i] });
      }
      itemIdxTensor.dispose();
      itemEmb.dispose();
      logits.dispose();
      await tf.nextFrame();
    }
    uTensor.dispose(); uEmb.dispose();
    // sort top-k
    results.sort((a,b)=>b.score - a.score);
    return results.slice(0,k);
  }

  // Wire buttons
  btnLoad.addEventListener('click', async () => {
    try {
      btnLoad.disabled = true;
      status.textContent = 'Loading...';
      await loadData();
    } catch (e) {
      console.error(e);
      status.textContent = 'Error loading data: ' + e.message;
      btnLoad.disabled = false;
    }
  });

  btnTrain.addEventListener('click', async () => {
    btnTrain.disabled = true;
    try {
      await trainClick();
    } catch (e) {
      console.error(e);
      status.textContent = 'Training error: ' + e.message;
      btnTrain.disabled = false;
    }
  });

  btnTest.addEventListener('click', async () => {
    btnTest.disabled = true;
    try {
      await testClick();
    } catch (e) {
      console.error(e);
      status.textContent = 'Test error: ' + e.message;
    } finally {
      btnTest.disabled = false;
    }
  });

  // Initialize small placeholder
  status.textContent = 'Ready. Click "Load Data" to start.';

})();

