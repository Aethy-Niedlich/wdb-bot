import { createRequire } from 'node:module';
import fs from 'node:fs';
const require = createRequire(import.meta.url);

globalThis.window = globalThis; globalThis.self = globalThis;
try { Object.defineProperty(globalThis,'navigator',{value:{userAgent:'node'},configurable:true}); } catch {}
globalThis.document = { createElement: () => ({ getContext: () => null, style:{}, width:0, height:0 }),
                        body:{ appendChild(){} }, getElementById: () => null };
globalThis.requestAnimationFrame = cb => setTimeout(() => cb(Date.now()), 16);
globalThis.cancelAnimationFrame = id => clearTimeout(id);
const rdr = { width:0,height:0,resize(){},render(){},destroy(){},view:{},stage:{children:[]},forceRendererSize(){},backgroundColor:0 };
globalThis.PIXI = { autoDetectRenderer: () => ({...rdr}), settings:{}, utils:{},
  Container: class { addChild(){} },
  Graphics: class { beginFill(){} lineStyle(){} drawRect(){} beginTextureFill(){} },
  Text: class { constructor(){ this.resolution=1; this.width=100; this.height=20; this.x=0; this.y=0; this.alpha=1; this.visible=true; } },
  filters:{ AdjustmentFilter:class{}, BlurFilter:class{}, FXAAFilter:class{} },
  resources:{ SVGResource: class { load(){ return Promise.resolve(); } } } };
globalThis.AudioContext = class { constructor(){ this.destination={}; } createGain(){ return {connect(){},gain:{value:0}}; } };
globalThis.peerjs = { peerjs:{ Peer: class{} } };
globalThis.Howl = class { constructor(){} play(){} volume(){return this} stereo(){return this} on(){} };
globalThis.GameResources = { soundStrings:{capIncrease:'',capDecrease:'',discDisc:'',platBounce:''}, svgStrings:{} };
globalThis.dcodeIO = { PSON:{ StaticPair: class { constructor(d){this.dict=d} encode(){return{toBase64:()=>''}} decode(){return{}} } } };

{ const src = fs.readFileSync(new URL('./SafeTrig.js', import.meta.url), 'utf8');
  (0, eval)(src + '\n;globalThis.SafeTrig = SafeTrig;'); }

let amd = null;
globalThis.define = (...a) => { const f = a[a.length-1];
  if (typeof f !== 'function') { amd = f; return; }
  const deps = a.find(x => Array.isArray(x)) || []; const ex = {}, mo = { exports: ex };
  let v = deps.map(d => d==='exports'?ex : d==='module'?mo : globalThis[d]);
  if (!deps.length && f.length) v = [ex, mo].slice(0, f.length);
  const r = f(...v); amd = (r != null) ? r : (Object.keys(ex).length ? ex : mo.exports); };
globalThis.define.amd = true;
let box2d = require('./Box2DModuleGlMod.js');
const isB2 = x => x && typeof x === 'object' && (x.Common || x.Dynamics);
box2d = isB2(box2d) ? box2d : isB2(globalThis.Box2D) ? globalThis.Box2D : isB2(amd) ? amd : amd;
const bonk = (await import('./bonk-core.mjs')).createBonkCore(box2d, { thumbScale: 1 });

const K = Number(process.argv[2] || 8);
const T = Number(process.argv[3] || 500);
const REPLAY_PATH = process.argv[4] || new URL('./replays/321746.json', import.meta.url).pathname;

function decodeInputs(b64){
  const buf = Buffer.from(b64,'base64'); let off=0;
  const count = buf.readUInt16BE(off); off+=2; const byTick=[];
  for(let i=0;i<count;i++){
    const slot=buf.readUInt16BE(off); off+=2;
    const tick=buf.readUInt32BE(off); off+=4;
    const f=buf.readUInt8(off); off+=1;
    (byTick[tick] ||= [])[slot]={left:!!(f&1),right:!!(f&2),up:!!(f&4),down:!!(f&8),action:!!(f&16),action2:!!(f&32)};
  }
  return byTick;
}

const replay = JSON.parse(fs.readFileSync(REPLAY_PATH,'utf8'));
const byTick = decodeInputs(replay.inputs);
const step = bonk.Physics.prototype.step;
const FPS = 30, SUB = 1;

const envs = [];
for(let k=0; k<K; k++){
  const held = [];
  const at = t => { if(byTick[t]) for(const key in byTick[t]) held[key]=byTick[t][key]; return held; };
  envs.push({
    state: JSON.parse(JSON.stringify(replay.startingState)),
    at, t: 0, atk: replay.adminInputs?.[0] || null
  });
}

function makeObs(state){
  const obs = [];
  for(let s=0; s<state.discs.length; s++){
    const d = state.discs[s];
    if(d){ obs.push(d.x, d.y, d.xv, d.yv, d.a, d.av, d.a1a); }
  }
  const bb = state.physics.bodies.find(b => b && b.s && b.s.n === 'DeathBall');
  if(bb){ obs.push(bb.p[0], bb.p[1], bb.lv[0], bb.lv[1], bb.a, bb.av); }
  return new Float32Array(obs);
}

const t0 = performance.now();
const snap = JSON.stringify(envs[0].state);
const snapTime = performance.now() - t0;
const t1 = performance.now();
JSON.parse(snap);
const restoreTime = performance.now() - t1;

const latencies = [];
const t2 = performance.now();
for(let t=0; t<T; t++){
  const stepStart = performance.now();
  for(let k=0; k<K; k++){
    const env = envs[k];
    if(env.state.fte === 0) continue;
    const inp = env.at(env.t);
    env.state = step.call(bonk.Physics, env.state, inp, env.atk, FPS, replay.gameSettings, SUB);
    env.t++;
    const obs = makeObs(env.state);
  }
  const stepTime = performance.now() - stepStart;
  latencies.push(stepTime);
}
const totalTime = performance.now() - t2;

latencies.sort((a,b)=>a-b);
const p50 = latencies[Math.floor(latencies.length*0.5)];
const p99 = latencies[Math.floor(latencies.length*0.99)];
const stepsPerSec = (T * K) / (totalTime / 1000);
const cores = Number(process.env.CORES || 1);
const perCore = stepsPerSec / cores;

console.log(`K=${K} T=${T} total=${totalTime.toFixed(0)}ms`);
console.log(`  steps/s: ${stepsPerSec.toFixed(0)} | per-core (@${cores}): ${perCore.toFixed(0)}`);
console.log(`  p50: ${p50.toFixed(2)}ms | p99: ${p99.toFixed(2)}ms`);
console.log(`  snap: ${snapTime.toFixed(2)}ms | restore: ${restoreTime.toFixed(2)}ms (~${(restoreTime/33.3).toFixed(1)} ticks)`);
console.log(`  obs stub: ${makeObs(envs[0].state).length} floats`);
