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

async function loadData() {
  try {
    statusDiv.textContent = 'Loading data...';
    const [udataRes, uitemRes] = await Promise.all([
      fetch('data/u.data'),
      fetch('data/u.item')
    ]);

    if (!udataRes.ok || !uitemRes.ok) {
      statusDiv.textContent = '❌ Failed to fetch data files. Check /data folder.';
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
      const [u, i, r, ts] = line.split('\t').map(Number);
      if (!u || !i) return;
      interactions.push({ userId: u, itemId: i, rating: r, ts });
    });

    const maxInt = parseInt(document.getElementById('maxInteractions').value);
    interactions = interactions.slice(0, maxInt);

    // Index maps
    const users = [...new Set(interactions.map(x => x.userId))];
    const itemSet = [...new Set(interactions.map(x => x.itemId))];
    userIdx.clear(); itemIdx.clear(); revUserIdx = []; revItemIdx = [];
    users.forEach((u, i) => { userIdx.set(u, i); revUserIdx[i] = u; });
    itemSet.forEach((i, j) => { itemIdx.set(i, j); revItemIdx[j] = i; });

    // User → items
    userToItems.clear();
    for (const x of interactions) {
      if (!userToItems.has(x.userId)) userToItems.set(x.userId, []);
      userToItems.get(x.userId).push(x);
    }

    statusDiv.textContent = `✅ Loaded ${interactions.length} interactions, ${users.length} users, ${itemSet.length} items.`;
    document.getElementById('trainBtn').disabled = false;
  } catch (e) {
    console.error(e);
    statusDiv.textContent = '❌ Error loading data: ' + e.message;
  }
}

// === TRAIN ===
async function trainModel() {
  const embDim = parseInt(document.getElementById('embDim').value);
  const epochs = parseInt(document.getElementById('epochs').value);
  const batchSize = parseInt(document.getElementById('batchSize').value);
  const lr = parseFloat(document.getElementById('lr').value);

  const numUsers = userIdx.size;
  const numItems = itemIdx.size;

  model = new TwoTowerModel(numUsers, numItems, embDim);
  const optimizer = tf.train.adam(lr);

  const ctx = lossCanvas.getContext('2d');
  ctx.clearRect(0,0,lossCanvas.width,lossCanvas.height);
  ctx.beginPath();
  let step = 0;

  for (let epoch = 0; epoch < epochs; epoch++) {
    tf.util.shuffle(interactions);
    for (let start = 0; start < interactions.length; start += batchSize) {
      const batch = interactions.slice(start, start + batchSize);
      const uIdxs = batch.map(x => userIdx.get(x.userId));
      const iIdxs = batch.map(x => itemIdx.get(x.itemId));

      const lossVal = await model.trainBatch(uIdxs, iIdxs, optimizer);
      step++;

      if (step % 10 === 0) {
        const x = step / 5;
        const y = 200 - Math.min(200, lossVal * 200);
        ctx.lineTo(x, y);
        ctx.stroke();
      }
    }
    statusDiv.textContent = `Epoch ${epoch+1}/${epochs} completed`;
    await tf.nextFrame();
  }

  statusDiv.textContent = '✅ Training complete!';
  document.getElementById('testBtn').disabled = false;

  drawEmbeddingProjection();
}

// === TEST ===
async function testModel() {
  if (!model) return;

  const eligible = [...userToItems.keys()].filter(u => userToItems.get(u).length >= 20);
  if (eligible.length === 0) {
    statusDiv.textContent = '⚠️ No users with ≥20 ratings.';
    return;
  }
  const userId = eligible[Math.floor(Math.random() * eligible.length)];
  const hist = userToItems.get(userId)
    .sort((a,b) => b.rating - a.rating || b.ts - a.ts)
    .slice(0,10)
    .map(x => items.get(x.itemId)?.title || ('#'+x.itemId));

  const uIdxVal = userIdx.get(userId);
  const userEmb = model.getUserEmbedding(tf.tensor1d([uIdxVal],'int32'));
  const allItemEmb = model.itemEmbedding;
  const scores = tf.matMul(userEmb, allItemEmb, false, true).dataSync();

  const rated = new Set(userToItems.get(userId).map(x => x.itemId));
  const ranked = scores.map((s,i)=>({i,s}))
    .filter(o=>!rated.has(revItemIdx[o.i]))
    .sort((a,b)=>b.s-a.s)
    .slice(0,10)
    .map(o=>items.get(revItemIdx[o.i])?.title||('#'+revItemIdx[o.i]));

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

  const emb = model.itemEmbedding.arraySync(); // shape [n, embDim]
  const n = emb.length;
  const dim = emb[0].length;

  // Simple PCA on covariance
  const means = Array(dim).fill(0);
  for (let i=0;i<n;i++) for (let d=0;d<dim;d++) means[d]+=emb[i][d];
  for (let d=0;d<dim;d++) means[d]/=n;
  const X = emb.map(row => row.map((v,d)=>v-means[d]));
  const cov = tf.matMul(tf.tensor2d(X), tf.tensor2d(X), true, false).div(n);
  const { eigVals, eigVecs } = tf.linalg.eigh(cov);
  const idxs = eigVals.arraySync().map((v,i)=>[v,i]).sort((a,b)=>b[0]-a[0]).map(x=>x[1]);
  const W = tf.gather(eigVecs, tf.tensor1d(idxs.slice(0,2),'int32'), 1);
  const proj = tf.matMul(tf.tensor2d(X), W).arraySync();

  const xs = proj.map(p=>p[0]), ys = proj.map(p=>p[1]);
  const minX=Math.min(...xs), maxX=Math.max(...xs);
  const minY=Math.min(...ys), maxY=Math.max(...ys);
  ctx.fillStyle='steelblue';
  for (let i=0;i<n;i+=Math.floor(n/1000)+1) {
    const x=((xs[i]-minX)/(maxX-minX))*embedCanvas.width;
    const y=((ys[i]-minY)/(maxY-minY))*embedCanvas.height;
    ctx.fillRect(x,y,2,2);
  }
}
