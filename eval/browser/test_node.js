// Offline correctness test of worker.js against a random-weight Mixtral + toy BPE tokenizer.
global.self = undefined;
const E = require('./engine.patched.js') /* extract the engine-src script from /playground/ and add the router hook line, see README */; global.PlaygroundEngine = E;
const EV = require('./worker.js');
function rnd(seed){ let s=seed; return ()=>{ s=(s*1103515245+12345)&0x7fffffff; return (s/0x7fffffff-0.5)*0.2; }; }
const r = rnd(7);
const cfg = {vocab_size: 300, hidden_size: 24, num_layers: 2, num_heads: 6, num_kv_heads: 2, intermediate_size: 16, num_experts: 4, top_k: 2, max_seq_len: 160, rope_theta: 1e6, norm_eps: 1e-5, attention_mode: 'global', tie_embeddings: false};
const H=cfg.hidden_size, I=cfg.intermediate_size, kv=H/cfg.num_heads*cfg.num_kv_heads, V=cfg.vocab_size;
const shapes = {'embed.weight':[V,H],'lm_head.weight':[V,H],'norm.weight':[H]};
for (let l=0;l<cfg.num_layers;l++){ const p=`layers.${l}.`; Object.assign(shapes,{[p+'attn_norm.weight']:[H],[p+'ffn_norm.weight']:[H],[p+'attn.q_proj.weight']:[H,H],[p+'attn.k_proj.weight']:[kv,H],[p+'attn.v_proj.weight']:[kv,H],[p+'attn.o_proj.weight']:[H,H],[p+'moe.gate.weight']:[cfg.num_experts,H]}); for(let e=0;e<4;e++){const q=`${p}moe.experts.${e}.`; shapes[q+'w1.weight']=[I,H]; shapes[q+'w3.weight']=[I,H]; shapes[q+'w2.weight']=[H,I];}}
const header={}; let off=0; for (const [k,sh] of Object.entries(shapes)){ const n=sh.reduce((a,b)=>a*b,1); header[k]={dtype:'F32',shape:sh,data_offsets:[off,off+4*n]}; off+=4*n; }
const hb=Buffer.from(JSON.stringify(header)); const buf=Buffer.alloc(8+hb.length+off); buf.writeBigUInt64LE(BigInt(hb.length),0); hb.copy(buf,8);
const f=new Float32Array(buf.buffer, buf.byteOffset+8+hb.length, off/4); for(let i=0;i<f.length;i++) f[i]=r(); 
for (const [k,sh] of Object.entries(shapes)) if (k.endsWith('norm.weight')) { const [a]=header[k].data_offsets; for(let i=0;i<sh[0];i++) f[a/4+i]=1; }
const tensors = E.parseSafetensors(buf.buffer.slice(buf.byteOffset, buf.byteOffset+buf.length));
const model = new E.MixtralModel(cfg, tensors);
// toy byte-level BPE: 256 byte tokens + specials
const BE = (()=>{ const bs=[]; for(let i=33;i<=126;i++)bs.push(i); for(let i=161;i<=172;i++)bs.push(i); for(let i=174;i<=255;i++)bs.push(i); const cs=bs.slice(); let n=0; for(let b=0;b<256;b++) if(!bs.includes(b)){bs.push(b);cs.push(256+n);n++;} const m={}; bs.forEach((b,i)=>m[b]=String.fromCharCode(cs[i])); return m; })();
const vocab={}; for(let b=0;b<256;b++) vocab[BE[b]]=b; 
const tokJson={model:{type:'BPE',vocab,merges:[],unk_token:null},added_tokens:[{id:296,content:'<pad>',special:true},{id:297,content:'<unk>',special:true},{id:298,content:'<bos>',special:true},{id:299,content:'<eos>',special:true}],pre_tokenizer:{type:'ByteLevel'}};
const tok = new E.ByteLevelBPE(tokJson);
const m = {tok, model, kind:'mixtral', cfg};
// 1) MCQ KV-rewind == fresh scoring of prompt+option
const item={q:'Which nerve supplies the deltoid?', opts:['Axillary nerve','Radial','Ulnar nerve palsy','Median'], cop:0};
const res = EV.mcqItem(m, item);
const bos=tok.tokenToId('<bos>'); const hdr='### System:\nProvide accurate educational medical information, and acknowledge uncertainty.\n\n### User:\n'+item.q+'\n\n### Assistant:\n';
let maxDiff=0; const sums=[];
for (const o of item.opts){ const oi=tok.encode(o); const longest=Math.max(...item.opts.map(x=>tok.encode(x).length)); const p=[bos].concat(tok.encode(hdr)).slice(-Math.max(8,128-longest)); const s=EV.scoreSeq(m, p.concat(oi), p.length); sums.push(-s.nll); }
// recompute via rewind path numbers
EV.setRoute(null);
const rw = (()=>{ const longest=Math.max(...item.opts.map(x=>tok.encode(x).length)); const p=[bos].concat(tok.encode(hdr)).slice(-Math.max(8,128-longest)); model.reset(); let last; for(let i=0;i<p.length;i++) last=model.step(p[i], i===p.length-1); last=Float32Array.from(last); const P=model.pos; return item.opts.map(o=>{ const oi=tok.encode(o); model.pos=P; let lp=EV.logSoftmaxAt(last,oi[0]); for(let j=0;j<oi.length-1;j++) lp+=EV.logSoftmaxAt(model.step(oi[j],true),oi[j+1]); return lp; }); })();
rw.forEach((v,i)=>maxDiff=Math.max(maxDiff,Math.abs(v-sums[i])));
console.log('MCQ rewind vs fresh: max |diff| =', maxDiff.toExponential(2), '| predNorm', res.predNorm, 'predSum', res.predSum);
// 2) scoreSeq: NLL equals sum of per-step -logp computed independently
const ids=tok.encode('Hypertension is a major risk factor for stroke and heart disease.').slice(0,40);
const s1=EV.scoreSeq(m, ids, 1); let manual=0; model.reset(); for(let i=0;i<ids.length-1;i++){ const lg=model.step(ids[i],true); let mx=-Infinity; for(const x of lg) mx=Math.max(mx,x); let z=0; for(const x of lg) z+=Math.exp(x-mx); manual += -(lg[ids[i+1]]-mx-Math.log(z)); }
console.log('scoreSeq nll', s1.nll.toFixed(6), 'manual', manual.toFixed(6), 'n', s1.n, '(expected', ids.length-1, ')');
// 3) sftExample mirrors tokenize_example: budget 129, answer truncation keeps eos, prompt from right
const longA='x'.repeat(500); const ex=EV.sftExample(m,'What is anemia?', longA);
console.log('sft long answer: total', ex.ids.length, 'scoreFrom', ex.scoreFrom, 'last=eos', ex.ids[ex.ids.length-1]===299, 'questionCut', ex.questionCut);
const ex2=EV.sftExample(m,'What is anemia?', 'Low hemoglobin.');
console.log('sft short answer: total', ex2.ids.length, 'scoreFrom', ex2.scoreFrom, 'first=bos', ex2.ids[0]===298, 'questionCut', ex2.questionCut);
// 4) routing counts: 2 selections per token per layer
const rt=EV.newRoute(m); EV.setRoute(rt); EV.scoreSeq(m, ids, 1); const tot=rt.map(r=>r.reduce((a,b)=>a+b,0)); console.log('route totals per layer', tot, 'expected', 2*(ids.length-1));
// 5) greedy stays within 128 positions
EV.setRoute(null); const g=EV.greedy(m, 'y'.repeat(400), 60, true); console.log('greedy prompt', g.promptTokens, 'gen', g.tokens, 'final pos', model.pos);
