let interactions = [];
let items = new Map();
let userToItems = new Map();
let userIdx = new Map(), itemIdx = new Map(), revUserIdx = [], revItemIdx = [];
let model;

const statusDiv = document.getElementById('status');
const lossCanvas = document.getElementById('lossChart');
const embedCanvas = document.getElementById('embedCanvas');
const resultDiv = document.getElementById('results');

document.getElementById('loadDataBtn').onclick = loadData;
document.getElementById('trainBtn').onclick = trainModel;
document.getElementById('testBtn').onclick = testModel;

// === LOAD DATA ===
async function loadData() {
  try {
    statusDiv.textContent = 'Loading data...';
    const [udataRes, uitemRes] = await Promise.all([
      fetch('data/u.data'),
      fetch('data/u.item')
    ]);

    if (!udataRes.ok || !uitemRes.ok) {
      statusDiv.textContent = '❌ Failed to fetch data files. Check paths under /data/.';
      return;
    }

    const udataTxt = await udataRes.text();
    const uitemTxt = await uitemRes.text();

    // Parse u.item
    items.clear();
    uitemTxt.split('\n').forEach(line => {
      if (!line.trim()) return;
      const parts = line.split('|');
      const id = parseInt(parts[0]);
      const title = parts[1]?.trim();
      if (id && title) items.set(id, { title });
    });

    // Parse u.data
    interactions = [];
    udataTxt.split('\n').forEach(line => {
      if (!line.trim()) return;
      const [u, i, r, ts] = line.split('\t').map(x => parseInt(x));
      if (!u || !i) return;
      interactions.push({ userId: u, itemId: i, rating: r, ts });
    });

    // Limit data
    const maxInt = parseInt(document.getElementById('maxInteractions').value);
    interactions = interactions.slice(0, maxInt);

    // Build indexers
    const users = [...new Set(interactions.map(x => x.userId))];
    const itemsAll = [...new Set(interactions.map(x => x.itemId))];
    userIdx.clear(); itemIdx.clear(); revUserIdx = []; revItemIdx = [];

    users.forEach((u, i) => { userIdx.set(u, i); revUserIdx[i] = u; });
    itemsAll.forEach((i, j) => { itemIdx.set(i, j); revItemIdx[j] = i; });

    // Build user→items
    userToItems.clear();
    for (const x of interactions) {
      if (!userToItems.has(x.userId)) userToItems.set(x.userId, []);
      userToItems.get(x.userId).push(x);
    }

    statusDiv.textContent = `✅ Loaded ${interactions.length} interactions, ${users.length} users, ${itemsAll.length} items.`;
    document.getElementById('trainBtn').disabled = false;
  } catch (e) {
    console.error(e);
    statusDiv.textContent = '❌ Error loading data: ' + e.message;
  }
}

// === TRAIN MODEL ===
async function trainModel() {
  const embDim = parseInt(document.getElementById('embDim').value);
  const epochs = parseInt(document.getElementById('epochs').value);
  const batchSize = parseInt(document.getElementById('batchSize').value);
  const lr = parseFloat(document.getElementById('lr').value);

  const numUsers = userIdx.size;
  const numItems = itemIdx.size;

  model = new TwoTowerModel(numUsers, numItems, embDim, false);
  const optimizer = tf.train.adam(lr);

  const ctx = lossCanvas.getContext('2d');
  ctx.clearRect(0,0,lossCanvas.width,lossCanvas.height);
  ctx.beginPath();
  let batchCount = 0;

  for (let epoch = 0; epoch < epochs; epoch++) {
    tf.util.shuffle(interactions);
    for (let start = 0; start < interactions.length; start += batchSize) {
      const batch = interactions.slice(start, start + batchSize);
      const uIdxs = batch.map(x => userIdx.get(x.userId));
      const iIdxs = batch.map(x => itemIdx.get(x.itemId));

      const loss = await model.trainBatch(uIdxs, iIdxs, optimizer);
      batchCount++;

      if (batchCount % 10 === 0) {
        const x = batchCount / 5;
        const y = 200 - Math.min(200, loss * 200);
        ctx.lineTo(x, y);
        ctx.stroke();
      }
    }
    statusDiv.textContent = `Epoch ${epoch+1}/${epochs} done.`;
    await tf.nextFrame();
  }

  statusDiv.textContent = '✅ Training complete!';
  document.getElementById('testBtn').disabled = false;

  drawEmbeddingProjection();
}

// === TEST ===
async function testModel() {
  const eligibleUsers = [...userToItems.keys()].filter(u => userToItems.get(u).length >= 20);
  const userId = eligibleUsers[Math.floor(Math.random() * eligibleUsers.length)];
  const userIdxVal = userIdx.get(userId);

  const hist = userToItems.get(userId)
    .sort((a,b) => b.rating - a.rating || b.ts - a.ts)
    .slice(0,10)
    .map(x => items.get(x.itemId)?.title || ('#'+x.itemId));

  const userEmb = model.getUserEmbedding(tf.tensor1d([userIdxVal],'int32'));
  const allItemEmb = model.itemEmbedding;
  const scores = tf.matMul(userEmb, allItemEmb, false, true).dataSync();

  const ratedSet = new Set(userToItems.get(userId).map(x => x.itemId));
  const ranked = scores
    .map((s, i) => ({ idx: i, s }))
    .filter(x => !ratedSet.has(revItemIdx[x.idx]))
    .sort((a,b) => b.s - a.s)
    .slice(0,10)
    .map(x => items.get(revItemIdx[x.idx])?.title || ('#'+revItemIdx[x.idx]));

  renderComparison(hist, ranked);
}

function renderComparison(hist, recs) {
  let html = '<table><tr><th>Top-10 Rated</th><th>Top-10 Recommended</th></tr>';
  for (let i=0;i<10;i++) {
    html += `<tr><td>${hist[i]||''}</td><td>${recs[i]||''}</td></tr>`;
  }
  html += '</table>';
  resultDiv.innerHTML = html;
}

// === EMBEDDING VISUALIZATION ===
function drawEmbeddingProjection() {
  const ctx = embedCanvas.getContext('2d');
  ctx.clearRect(0,0,embedCanvas.width,embedCanvas.height);
  const emb = model.itemEmbedding.dataSync();
  const n = model.itemEmbedding.shape[0];
  const dim = model.itemEmbedding.shape[1];

  const xs = [], ys = [];
  // Quick PCA approximation: project on first two dims
  for (let i=0;i<n;i++) {
    xs.push(emb[i*dim]);
    ys.push(emb[i*dim+1]);
  }
  const minX=Math.min(...xs), maxX=Math.max(...xs);
  const minY=Math.min(...ys), maxY=Math.max(...ys);
  ctx.fillStyle = 'steelblue';
  for (let i=0;i<xs.length;i+=Math.floor(n/1000)+1) {
    const x = ((xs[i]-minX)/(maxX-minX))*embedCanvas.width;
    const y = ((ys[i]-minY)/(maxY-minY))*embedCanvas.height;
    ctx.fillRect(x, y, 2, 2);
  }
}
