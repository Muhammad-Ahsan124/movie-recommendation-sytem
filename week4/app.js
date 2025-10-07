async function buildDeepModel(userCount,itemCount,embeddingDim,useGenres,useSynthUser){
  const userInput=tf.input({shape:[1],name:'userInput'});
  const itemInput=tf.input({shape:[1],name:'itemInput'});

  const userEmb=tf.layers.embedding({inputDim:userCount+1,outputDim:embeddingDim}).apply(userInput);
  const itemEmb=tf.layers.embedding({inputDim:itemCount+1,outputDim:embeddingDim}).apply(itemInput);
  const userFlat=tf.layers.flatten().apply(userEmb);
  const itemFlat=tf.layers.flatten().apply(itemEmb);

  let genreInput,genreFeat,userSynthInput,userSynthFeat;

  if(useGenres){
    genreInput=tf.input({shape:[19],name:'genreInput'});
    const genreDense=tf.layers.dense({units:embeddingDim,activation:'relu'}).apply(genreInput);
    genreFeat=genreDense;
  }

  if(useSynthUser){
    userSynthInput=tf.input({shape:[2],name:'userSynthInput'});
    const userDense=tf.layers.dense({units:embeddingDim,activation:'relu'}).apply(userSynthInput);
    userSynthFeat=userDense;
  }

  let userConcat=userFlat;
  if(useSynthUser) userConcat=tf.layers.concatenate().apply([userFlat,userSynthFeat]);
  userConcat=tf.layers.dense({units:embeddingDim,activation:'relu'}).apply(userConcat);

  let itemConcat=itemFlat;
  if(useGenres) itemConcat=tf.layers.concatenate().apply([itemFlat,genreFeat]);
  itemConcat=tf.layers.dense({units:embeddingDim,activation:'relu'}).apply(itemConcat);

  const dot=tf.layers.dot({axes:1,normalize:true}).apply([userConcat,itemConcat]);
  const output=tf.layers.dense({units:1,activation:'sigmoid'}).apply(dot);

  const inputs=[userInput,itemInput];
  if(useGenres) inputs.push(genreInput);
  if(useSynthUser) inputs.push(userSynthInput);

  const model=tf.model({inputs,outputs:output});
  model.compile({optimizer:tf.train.adam(0.001),loss:'binaryCrossentropy'});
  return model;
}

// ------------------- Training logic -------------------
document.getElementById('trainBtn').onclick=async()=>{
  document.getElementById('errorMsg').textContent='';
  const useGenres=document.getElementById('useGenres').checked;
  const useSynth=document.getElementById('useSynth').checked;
  const useMLP=document.getElementById('useMLP').checked;
  const useBPR=document.getElementById('useBPR').checked;
  const emb=parseInt(document.getElementById('embDim').value);
  const epochs=parseInt(document.getElementById('epochs').value);

  // Simple demo synthetic data
  const users=1000,items=1700;
  const userIds=tf.tensor1d([...Array(5000)].map(()=>Math.floor(Math.random()*users)),'int32');
  const itemIds=tf.tensor1d([...Array(5000)].map(()=>Math.floor(Math.random()*items)),'int32');
  const labels=tf.tensor1d([...Array(5000)].map(()=>Math.random()>0.5?1:0),'float32');

  const model=useMLP
    ?await buildDeepModel(users,items,emb,useGenres,useSynth)
    :await buildTwoTower(users,items,emb,useBPR);

  await model.fit([userIds,itemIds],labels,{
    epochs,batchSize:128,
    callbacks:{onEpochEnd:(e,l)=>console.log(`epoch ${e}:`,l.loss)}
  });

  alert('✅ Training complete — no concat2D error!');
};
