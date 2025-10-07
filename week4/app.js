// app.js
let data = [], items = new Map();
let userToRated = new Map();
let userIndexer = new Map(), itemIndexer = new Map();
let idxToUser = [], idxToItem = [];
let model, deepModel;
let userFeatures, itemFeatures;

async function fetchFile(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path} not found`);
  return await r.text();
}

async function loadData() {
  const status = document.getElementById('status');
  try {
    const udata = await fetchFile('data/u.data');
    const uitem = await fetchFile('data/u.item');
    const lines = udata.trim().split('\n');
    const itemLines = uitem.trim().split('\n');
    itemLines.forEach(l => {
      const p = l.split('|');
      const id = parseInt(p[0]);
      const title = p[1];
      const genres = p.slice(-19).map(x => parseInt(x));
      items.set(id, { title, genres });
    });

    data = lines.map(l => {
      const [u, i, r, t] = l.split('\t').map(Number);
      return { u, i, r, t };
    });

    // Build mappings
    const users = [...new Set(data.map(x => x.u))];
    const itemIds = [...items.keys()];
    users.forEach((u, j) => { userIndexer.set(u, j); idxToUser[j] = u; });
    itemIds.forEach((i, j) => { itemIndexer.set(i, j); idxToItem[j] = i; });

    // Build user->rated
    userToRated = new Map();
    for (const d of data) {
      if (!userToRated.has(d.u)) userToRated.set(d.u, []);
      userToRated.get(d.u).push(d);
    }

    status.textContent = `Loaded ${data.length} interactions, ${users.length} users, ${itemIds.length} items`;
  } catch (e) {
    status.textContent = 'Error loading data: ' + e.message;
  }
}
document.getElementById('loadBtn').onclick = loadData;

async function trainModel() {
  const embDim = parseInt(document.getElementById('embDim').value);
  const epochs = parseInt(document.getElementById('epochs').value);
  const batchSize = parseInt(document.getElementById('batch').value);
  const maxInt = parseInt(document.getElementById('maxInt').value);
  const useBPR = document.getElementById('useBPR').checked;
  const useGenres = document.getElementById('useGenres').checked;
  const useUserFeat = document.getElementById('useUserFeat').checked;
  const useDeep = document.getElementById('useDeep').checked;
  const status = document.getElementById('status');
  const lossCanvas = document.getElementById('lossChart');
  const ctx = lossCanvas.getContext('2d');
  ctx.clearRect(0,0,lossCanvas.width,lossCanvas.height);

  const numUsers = userIndexer.size;
  const numItems = itemIndexer.size;
  const interactions = data.slice(0, maxInt);
  model = new TwoTowerModel(numUsers, numItems, embDim, useBPR);

  // prepare optional features
  if (useDeep) {
    // synthesize user features (mean rating count, avg rating)
    const uFeat = [];
    for (let i=0; i<numUsers; i++) {
      const uid = idxToUser[i];
      const rated = userToRated.get(uid) || [];
      const avgR = rated.length ? rated.reduce((a,b)=>a+b.r,0)/rated.length : 0;
      uFeat.push([rated.length/50.0, avgR/5.0]);
    }
    userFeatures = tf.tensor2d(uFeat);
    const itemFeat = [];
    for (let i=0; i<numItems; i++) {
      const id = idxToItem[i];
      const g = items.get(id)?.genres || new Array(19).fill(0);
      itemFeat.push(g);
    }
    itemFeatures = tf.tensor2d(itemFeat);
    deepModel = new DeepMLPModel(numUsers, numItems, embDim, 2, 19);
  }

  const losses = [];
  for (let ep=0; ep<epochs; ep++) {
    tf.util.shuffle(interactions);
    for (let i=0; i<interactions.length; i+=batchSize) {
      const batch = interactions.slice(i,i+batchSize);
      const uIdx = tf.tensor1d(batch.map(b=>userIndexer.get(b.u)),'int32');
      const iIdx = tf.tensor1d(batch.map(b=>itemIndexer.get(b.i)),'int32');
      const loss = model.trainStep(uIdx, iIdx);
      if (useDeep) deepModel.trainStep(uIdx, iIdx,
          tf.gather(userFeatures, uIdx), tf.gather(itemFeatures, iIdx));
      const lossVal = await loss.data();
      losses.push(lossVal[0]);
      if (i % (batchSize*10) === 0) drawLoss(ctx, losses);
      tf.dispose([uIdx,iIdx,loss]);
    }
    status.textContent = `Epoch ${ep+1}/${epochs}`;
  }
  drawLoss(ctx, losses);
  status.textContent = 'Training complete';
  drawEmbeddingProjection();
}
document.getElementById('trainBtn').onclick = trainModel;

function drawLoss(ctx, arr) {
  ctx.clearRect(0,0,400,200);
  ctx.beginPath();
  arr.forEach((v,i)=>{
    const x=i*400/arr.length, y=200 - (v*50);
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  });
  ctx.stroke();
}

function drawEmbeddingProjection() {
  const proj = document.getElementById('projCanvas');
  const ctx = proj.getContext('2d');
  ctx.clearRect(0,0,proj.width,proj.height);
  const W = model.itemEmbedding.dataSync();
  const mat = tf.tensor2d(W, [model.numItems, model.embDim]);
  const mean = mat.mean(0);
  const centered = mat.sub(mean);
  const cov = tf.matMul(centered.transpose(), centered);
  const [vals, vecs] = tf.linalg.eigh(cov);
  const top2 = vecs.slice([0, model.embDim-2],[model.embDim,2]);
  const proj2D = tf.matMul(centered, top2);
  const coords = proj2D.arraySync();
  coords.slice(0,1000).forEach((c,i)=>{
    const x = (c[0]*20)+300, y=(c[1]*20)+200;
    ctx.fillRect(x,y,2,2);
  });
}

async function testModel() {
  const resultsDiv = document.getElementById('resultsTable');
  const users = [...userToRated.keys()].filter(u=>userToRated.get(u).length>=20);
  const uid = users[Math.floor(Math.random()*users.length)];
  const rated = userToRated.get(uid).sort((a,b)=>b.r-a.r).slice(0,10);
  const uIdx = tf.tensor1d([userIndexer.get(uid)],'int32');
  const uEmb = model.getUserEmbedding(uIdx);
  const scores = model.getScoresForAllItems(uEmb).arraySync()[0];
  const ratedIds = new Set(rated.map(x=>x.i));
  const recs = [...scores.map((s,i)=>({i:idxToItem[i],s}))].filter(x=>!ratedIds.has(x.i))
    .sort((a,b)=>b.s-a.s).slice(0,10);

  let deepRecs = [];
  if (deepModel) {
    const userFeat = tf.gather(userFeatures, uIdx);
    const itemFeatMat = itemFeatures;
    const deepScores = deepModel.getScoresForAllItems(userFeat, itemFeatMat).arraySync()[0];
    deepRecs = [...deepScores.map((s,i)=>({i:idxToItem[i],s}))].filter(x=>!ratedIds.has(x.i))
      .sort((a,b)=>b.s-a.s).slice(0,10);
  }

  const ratedTitles = rated.map(x=>items.get(x.i)?.title||x.i);
  const recTitles = recs.map(x=>items.get(x.i)?.title||x.i);
  const deepTitles = deepRecs.map(x=>items.get(x.i)?.title||x.i);

  let html = `<h4>User ${uid} — Comparison</h4><table><tr><th>Top-10 Rated</th><th>Two-Tower Recs</th><th>Deep (MLP) Recs</th></tr>`;
  for(let i=0;i<10;i++){
    html+=`<tr><td>${ratedTitles[i]||''}</td><td>${recTitles[i]||''}</td><td>${deepTitles[i]||''}</td></tr>`;
  }
  html+='</table>';
  resultsDiv.innerHTML = html;
}
document.getElementById('testBtn').onclick = testModel;
