// app.js
// Main application logic: data loading, training loop, UI, plotting, PCA projection, and test rendering.
// Depends on two-tower.js which exposes TwoTowerModel and DeepRecModel classes.

(async () => {
  // DOM elements
  const btnLoad = document.getElementById('btnLoad');
  const btnTrain = document.getElementById('btnTrain');
  const btnTest = document.getElementById('btnTest');
  const status = document.getElementById('status');
  const lossCanvas = document.getElementById('lossCanvas');
  const projCanvas = document.getElementById('projCanvas');
  const progressDiv = document.getElementById('progress');
  const tableArea = document.getElementById('tableArea');
  const tooltip = document.getElementById('tooltip');

  const inputEmbDim = document.getElementById('inputEmbDim');
  const inputEpochs = document.getElementById('inputEpochs');
  const inputBatch = document.getElementById('inputBatch');
  const inputMaxInt = document.getElementById('inputMaxInt');

  const optBPR = document.getElementById('optBPR');
  const optUseGenres = document.getElementById('optUseGenres');
  const optUseUserFeat = document.getElementById('optUseUserFeat');
  const optIncludeDL = document.getElementById('optIncludeDL');

  // Canvas contexts
  const lossCtx = lossCanvas.getContext('2d');
  const projCtx = projCanvas.getContext('2d');

  // App state
  let interactions = []; // {userId, itemId, rating, ts}
  let items = new Map(); // itemId -> {title, year, genres: [0/1..]}
  let usersMap = new Map(); // userId -> [interactions...]
  let userIndex = new Map(); // original userId -> 0-based idx
  let itemIndex = new Map(); // original itemId -> 0-based idx
  let indexUser = []; // reverse
  let indexItem = [];
  let numUsers = 0, numItems = 0;

  let twoTower = null;
  let deepModel = null;

  let itemEmbeddingSample2D = []; // for plotting: {x,y,title,idx}

  // Helpers: update status
  function setStatus(s) { status.innerText = s; }
  function setProgress(s) { progressDiv.innerText = s; }

  function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

  // Simple canvas line plot (loss)
  function plotLoss(lossHistory) {
    const ctx = lossCtx;
    const w = lossCanvas.width;
    const h = lossCanvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#fff';
    ctx.fillRect(0,0,w,h);

    if (lossHistory.length === 0) {
      ctx.fillStyle = '#666';
      ctx.fillText('No loss yet', 8, 20);
      return;
    }
    const pad = 32;
    const max = Math.max(...lossHistory);
    const min = Math.min(...lossHistory);
    ctx.strokeStyle = '#2b7';
    ctx.beginPath();
    for (let i=0;i<lossHistory.length;i++) {
      const x = pad + (i/(lossHistory.length-1))*(w-2*pad);
      const v = lossHistory[i];
      const y = h - pad - ((v-min)/(max-min+1e-9))*(h-2*pad);
      if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.stroke();
    ctx.fillStyle = '#333';
    ctx.fillText(`loss (min ${min.toFixed(4)} max ${max.toFixed(4)})`, 8, 12);
  }

  // PCA via power iteration to get top-2 components (approx). Works on tf.Tensor2d (n x d).
  async function computePCA2D(tensorX, iterations=30) {
    return tf.tidy(() => {
      // center
      const mean = tensorX.mean(0);
      let X = tensorX.sub(mean); // [n,d]
      // covariance approx: C = X^T X / n
      const n = X.shape[0];
      const C = X.transpose().matMul(X).div(n); // [d,d]

      // power iteration for first eigenvector
      function powerIter(Cmat) {
        let v = tf.randomNormal([Cmat.shape[1], 1]);
        for (let i=0;i<iterations;i++) {
          v = Cmat.matMul(v);
          const norm = v.norm();
          v = v.div(norm.add(1e-9));
        }
        const eigenvec = v.reshape([Cmat.shape[1]]);
        const eigenval = tf.squeeze(tf.matMul(tf.matMul(eigenvec.expandDims(0), Cmat), eigenvec.expandDims(1)));
        return {eigenvec, eigenval};
      }

      const {eigenvec: v1, eigenval: l1} = powerIter(C);
      // deflate
      const v1col = v1.reshape([v1.shape[0],1]);
      const C2 = C.sub(v1col.matMul(v1col.transpose()).mul(l1));

      const {eigenvec: v2} = powerIter(C2);

      // project X onto [v1, v2]
      const P = tf.stack([v1, v2], 1); // [d,2]
      const proj = X.matMul(P); // [n,2]
      return {proj, mean, pcs: P};
    });
  }

  // Draw projection scatter and set up tooltip hover
  function drawProjection(points) {
    const ctx = projCtx;
    const W = projCanvas.width, H = projCanvas.height;
    ctx.clearRect(0,0,W,H);
    ctx.fillStyle = '#fff';
    ctx.fillRect(0,0,W,H);
    if (!points || points.length===0) {
      ctx.fillStyle = '#666'; ctx.fillText('No projection', 16, 24); return;
    }
    // scale to canvas
    const xs = points.map(p=>p.x), ys = points.map(p=>p.y);
    const minx = Math.min(...xs), maxx = Math.max(...xs);
    const miny = Math.min(...ys), maxy = Math.max(...ys);
    const pad = 24;
    for (let p of points) {
      const cx = pad + ((p.x - minx)/(maxx - minx + 1e-9))*(W-2*pad);
      const cy = pad + ((p.y - miny)/(maxy - miny + 1e-9))*(H-2*pad);
      p.canvasX = cx; p.canvasY = cy;
      // draw point
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, Math.PI*2);
      ctx.fillStyle = '#1565c0';
      ctx.fill();
    }
    // hover handling
    projCanvas.onmousemove = (ev) => {
      const r = projCanvas.getBoundingClientRect();
      const mx = ev.clientX - r.left, my = ev.clientY - r.top;
      let nearest = null, nd = 20;
      for (let p of points) {
        const d = Math.hypot(p.canvasX - mx, p.canvasY - my);
        if (d < nd) { nearest = p; nd = d; }
      }
      if (nearest) {
        tooltip.style.display = 'block';
        tooltip.style.left = (ev.clientX + 10) + 'px';
        tooltip.style.top = (ev.clientY + 10) + 'px';
        tooltip.innerText = `${nearest.title}`;
      } else {
        tooltip.style.display = 'none';
      }
    };
    projCanvas.onmouseleave = () => tooltip.style.display = 'none';
  }

  // Parse u.item: itemId|title|release_date|...|genreFlags(19)
  function parseItemLine(line) {
    // split by pipe - title can include parentheses but not pipes
    const parts = line.split('|');
    if (parts.length < 6) return null;
    const id = parseInt(parts[0], 10);
    const rawTitle = parts[1];
    // try to extract year from title like "Toy Story (1995)"
    let year = null;
    const m = rawTitle.match(/\((\d{4})\)$/);
    if (m) year = parseInt(m[1], 10);
    // genres begin at index 5 onwards
    const genreFlags = parts.slice(5).map(x => parseInt(x || '0', 10));
    return {id, title: rawTitle, year, genres: genreFlags};
  }

  // Parse u.data: userId \t itemId \t rating \t timestamp
  function parseDataLine(line) {
    const parts = line.trim().split('\t');
    if (parts.length < 4) return null;
    return {
      userId: parseInt(parts[0],10),
      itemId: parseInt(parts[1],10),
      rating: parseInt(parts[2],10),
      ts: parseInt(parts[3],10)
    };
  }

  // Build mappings and filtered interactions
  function buildIndexing(maxInteractions) {
    // filter interactions by rating or timestamp if needed already done
    // build unique sets
    const uSet = new Set(), iSet = new Set();
    const interactionsTrim = interactions.slice(0, maxInteractions);
    for (const it of interactionsTrim) { uSet.add(it.userId); iSet.add(it.itemId); }
    indexUser = Array.from(uSet).sort((a,b)=>a-b);
    indexItem = Array.from(iSet).sort((a,b)=>a-b);
    userIndex = new Map(indexUser.map((v,i)=>[v,i]));
    itemIndex = new Map(indexItem.map((v,i)=>[v,i]));
    numUsers = indexUser.length; numItems = indexItem.length;

    // build usersMap: index -> list of {itemIdx, rating, ts}
    usersMap = new Map();
    for (const it of interactionsTrim) {
      const u0 = userIndex.get(it.userId);
      const i0 = itemIndex.get(it.itemId);
      if (u0==null || i0==null) continue; // safety
      if (!usersMap.has(u0)) usersMap.set(u0, []);
      usersMap.get(u0).push({itemIdx: i0, rating: it.rating, ts: it.ts});
    }
    // sort each user's interactions by recency desc
    for (const [u, arr] of usersMap) arr.sort((a,b)=>b.ts - a.ts);
  }

  // Build ranked top-10 historically for a given internal user idx:
  function getUserTopRatedTitles(uIdx, limit=10) {
    const arr = usersMap.get(uIdx) || [];
    // sort by rating desc then recency (ts desc)
    const copy = arr.slice().sort((a,b) => {
      if (b.rating !== a.rating) return b.rating - a.rating;
      return b.ts - a.ts;
    });
    const top = copy.slice(0, limit).map(x => {
      const origId = indexItem[x.itemIdx];
      const it = items.get(origId);
      return {title: it ? it.title : String(origId), itemIdx: x.itemIdx};
    });
    return top;
  }

  // Create dataset batches as arrays of [userIdx, posItemIdx]
  function buildPosPairs() {
    const pairs = [];
    for (const [uIdx, arr] of usersMap) {
      for (const r of arr) pairs.push([uIdx, r.itemIdx]);
    }
    return pairs;
  }

  // ========== UI Actions ==========

  btnLoad.onclick = async () => {
    setStatus('loading data...');
    try {
      const maxInt = parseInt(inputMaxInt.value,10) || 80000;
      // fetch files
      const [rawData, rawItem] = await Promise.all([
        fetch('data/u.data').then(r => r.ok ? r.text() : Promise.reject('u.data not found')),
        fetch('data/u.item').then(r => r.ok ? r.text() : Promise.reject('u.item not found'))
      ]);
      // parse items
      items.clear();
      rawItem.split('\n').forEach(line => {
        if (!line.trim()) return;
        const parsed = parseItemLine(line);
        if (!parsed) return;
        items.set(parsed.id, {title: parsed.title, year: parsed.year, genres: parsed.genres});
      });
      // parse interactions
      interactions = [];
      rawData.split('\n').forEach(line => {
        if (!line.trim()) return;
        const p = parseDataLine(line);
        if (!p) return;
        interactions.push(p);
      });
      // sort interactions by timestamp (optionally)
      interactions.sort((a,b)=>a.ts - b.ts);

      buildIndexing(maxInt);

      setStatus(`loaded: interactions=${interactions.length} users=${numUsers} items=${numItems}`);
      btnTrain.disabled = false;
      btnTest.disabled = true;
    } catch (e) {
      setStatus('error loading data: ' + String(e));
      console.error(e);
    }
  };

  // Training loop
  btnTrain.onclick = async () => {
    try {
      btnTrain.disabled = true;
      btnLoad.disabled = true;
      btnTest.disabled = true;
      setStatus('initializing models...');
      const embDim = parseInt(inputEmbDim.value,10) || 32;
      const epochs = parseInt(inputEpochs.value,10) || 5;
      const batchSize = parseInt(inputBatch.value,10) || 128;
      const maxInt = parseInt(inputMaxInt.value,10) || 80000;
      const useBPR = optBPR.checked;
      const useGenres = optUseGenres.checked;
      const useUserFeat = optUseUserFeat.checked;
      const includeDL = optIncludeDL.checked;

      // rebuild indexing (in case user changed maxInteractions)
      buildIndexing(maxInt);

      // create models
      twoTower = new TwoTowerModel(numUsers, numItems, embDim);
      if (includeDL) {
        // synthesize simple user features: avg rating and count (normalized)
        // Build a userFeatMatrix as Map userIdx -> [avgRating, countNorm]
        const userFeat = new Array(numUsers).fill(0).map(()=>[0,0]);
        for (let u=0; u<numUsers; u++) {
          const arr = usersMap.get(u) || [];
          if (arr.length===0) userFeat[u] = [0,0];
          else {
            const avg = arr.reduce((s,x)=>s+x.rating,0)/arr.length;
            userFeat[u] = [avg/5.0, Math.log1p(arr.length)/Math.log(1+50)];
          }
        }
        deepModel = new DeepRecModel({
          numUsers, numItems, embDim,
          useGenres, useUserFeat,
          itemMeta: items,
          userFeatArray: userFeat
        });
      } else {
        deepModel = null;
      }

      // Prepare pairs
      const pairs = buildPosPairs(); // [ [uIdx, itemIdx], ... ]
      // shuffle
      for (let i=pairs.length-1;i>0;i--) {
        const j = Math.floor(Math.random()*(i+1));
        [pairs[i], pairs[j]] = [pairs[j], pairs[i]];
      }

      const optimizer = tf.train.adam(0.001);
      const lossHistory = [];
      plotLoss(lossHistory);

      // Training loop: for each epoch, iterate batches
      for (let e=0;e<epochs;e++) {
        setProgress(`Epoch ${e+1}/${epochs}`);
        let batchLossAccum = 0, batchCount=0;
        for (let start=0; start<pairs.length; start += batchSize) {
          const batch = pairs.slice(start, start+batchSize);
          if (batch.length < 2) continue;
          // build tensors
          const uBatchArr = batch.map(p=>p[0]);
          const posBatchArr = batch.map(p=>p[1]);

          // Run two-tower training step
          const lossVal = await twoTower.trainStepInBatch(uBatchArr, posBatchArr, optimizer, useBPR);
          // optionally train deep model (with same batch pairs)
          let dlLoss = 0;
          if (deepModel) {
            dlLoss = await deepModel.trainStep(uBatchArr, posBatchArr, optimizer, useBPR);
          }

          const combined = lossVal + (dlLoss || 0);
          lossHistory.push(combined);
          batchLossAccum += combined; batchCount++;

          if (lossHistory.length % 10 === 0) plotLoss(lossHistory);

          // progress UI
          if (start % (batchSize*50) === 0) {
            setStatus(`epoch ${e+1}/${epochs} - processed ${(start+batchSize)}/${pairs.length} pairs`);
            await sleep(5); // yield to UI
          }
        }
        setStatus(`finished epoch ${e+1}/${epochs} avgLoss=${(batchLossAccum/batchCount).toFixed(4)}`);
      }

      plotLoss(lossHistory);
      setStatus('training complete — computing item projection...');
      // compute item embeddings (sample up to 1000)
      const sampleN = Math.min(1000, numItems);
      const step = Math.max(1, Math.floor(numItems / sampleN));
      const sampleIdxs = [];
      for (let i=0;i<numItems;i+=step) sampleIdxs.push(i);
      const itemEmbTensor = await twoTower.getItemEmbeddings(sampleIdxs); // [n,d]
      // PCA
      const {proj} = await computePCA2D(itemEmbTensor);
      const projArr = await proj.array();
      itemEmbeddingSample2D = sampleIdxs.map((origIdx, i) => {
        const origId = indexItem[origIdx];
        const it = items.get(origId);
        return {x: projArr[i][0], y: projArr[i][1], title: it ? it.title : String(origId), idx: origIdx};
      });
      drawProjection(itemEmbeddingSample2D);

      setStatus('done. You can now Test a random user.');
      btnTest.disabled = false;
      btnLoad.disabled = false;
    } catch (err) {
      console.error(err);
      setStatus('training error: ' + String(err));
      btnLoad.disabled = false;
      btnTrain.disabled = false;
      btnTest.disabled = true;
    }
  };

  // Test: pick random user with >= 20 ratings
  btnTest.onclick = async () => {
    try {
      btnTest.disabled = true; btnTrain.disabled = true; btnLoad.disabled = true;
      setStatus('testing...');
      // find eligible users
      const eligible = [];
      for (const [u, arr] of usersMap) if (arr.length >= 20) eligible.push(u);
      if (eligible.length === 0) { setStatus('no user with >=20 ratings'); return; }
      const uIdx = eligible[Math.floor(Math.random()*eligible.length)];
      const topHist = getUserTopRatedTitles(uIdx, 10);

      // compute recommendations from two-tower
      const userEmb = await twoTower.getUserEmbedding(uIdx); // tf.Tensor1d
      const scoresTensor = await twoTower.scoreAllItems(userEmb);
      const scores = await scoresTensor.array();
      // mask items the user already rated
      const rated = new Set((usersMap.get(uIdx) || []).map(x=>x.itemIdx));
      const pairs = scores.map((s,i)=>({i,s})).filter(p=>!rated.has(p.i));
      pairs.sort((a,b)=>b.s - a.s);
      const topRec = pairs.slice(0,10).map(p => {
        const origId = indexItem[p.i];
        const it = items.get(origId);
        return {title: it ? it.title : String(origId), itemIdx: p.i};
      });

      // Deep model recommendations if available
      let topRecDL = [];
      if (deepModel) {
        const userEmbDL = await deepModel.getUserEmbedding(uIdx);
        const scoresDL = await deepModel.scoreAllItems(userEmbDL);
        const sDL = await scoresDL.array();
        const pairsDL = sDL.map((s,i)=>({i,s})).filter(p=>!rated.has(p.i));
        pairsDL.sort((a,b)=>b.s - a.s);
        topRecDL = pairsDL.slice(0,10).map(p => {
          const origId = indexItem[p.i];
          const it = items.get(origId);
          return {title: it ? it.title : String(origId), itemIdx: p.i};
        });
      }

      // render side-by-side table: left historical; middle two-tower rec; right deep rec (if available)
      let html = '<div class="side-table"><div class="panel"><b>Top-10 Historically Rated</b><ol>';
      for (const t of topHist) html += `<li>${escapeHtml(t.title)}</li>`;
      html += '</ol></div>';

      html += '<div class="panel"><b>Two-Tower Top-10 (no seen)</b><ol>';
      for (const t of topRec) html += `<li>${escapeHtml(t.title)}</li>`;
      html += '</ol></div>';

      if (deepModel) {
        html += '<div class="panel"><b>Deep (MLP) Top-10</b><ol>';
        for (const t of topRecDL) html += `<li>${escapeHtml(t.title)}</li>`;
        html += '</ol></div>';
      }

      html += '</div>';
      tableArea.innerHTML = html;
      setStatus('test complete');
      btnTest.disabled = false; btnTrain.disabled = false; btnLoad.disabled = false;
    } catch (err) {
      console.error(err);
      setStatus('test error: ' + String(err));
      btnTest.disabled = false; btnTrain.disabled = false; btnLoad.disabled = false;
    }
  };

  // Escape HTML utility
  function escapeHtml(text) {
    return (text+'').replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
  }

  // ========== initialization ==========
  setStatus('ready');

})();
