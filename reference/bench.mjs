import { createRequire } from 'node:module';
import fs from 'fs';
const require = createRequire(import.meta.url);

globalThis.window = globalThis; globalThis.self = globalThis;
try { Object.defineProperty(globalThis, 'navigator', { value: { userAgent: 'node' }, configurable: true }); } catch {}
globalThis.document = { createElement: () => ({ getContext: () => null, style: {}, width:0, height:0 }), body:{appendChild(){}}, getElementById: () => null };
globalThis.requestAnimationFrame = cb => setTimeout(() => cb(Date.now()), 16);
globalThis.cancelAnimationFrame = id => clearTimeout(id);

const rdr = { width:0,height:0,resize(){},render(){},destroy(){},view:{},stage:{children:[]},forceRendererSize(){},backgroundColor:0 };
globalThis.PIXI = { autoDetectRenderer: () => ({...rdr}), settings:{}, utils:{},
  Container: class { addChild(){} }, Graphics: class { beginFill(){} lineStyle(){} drawRect(){} beginTextureFill(){} },
  Text: class { resolution=1; width=100; height=20; x=0; y=0; alpha=1; visible=true },
  filters:{AdjustmentFilter:class{},BlurFilter:class{},FXAAFilter:class{}},
  resources:{SVGResource:class{load(){return Promise.resolve()}}} };
globalThis.AudioContext = class { constructor(){this.destination={}} createGain(){return{connect(){},gain:{value:0}}} };
globalThis.peerjs = { peerjs:{ Peer: class{} } };
globalThis.Howl = class { constructor(){} play(){} volume(){return this} stereo(){return this} on(){} };
globalThis.GameResources = { soundStrings:{capIncrease:'',capDecrease:'',discDisc:'',platBounce:''}, svgStrings:{} };
globalThis.dcodeIO = { PSON:{ StaticPair: class { constructor(d){this.dict=d} encode(){return{toBase64:()=>''}} decode(){return{}} } } };

// ---- SafeTrig shim (TEMPORARY: verify against real impl for determinism) ----
const trig = { safeCos:Math.cos, safeSin:Math.sin, safeTan:Math.tan, safeATan2:Math.atan2,
               safeACos:Math.acos, safeASin:Math.asin, safeSqrt:Math.sqrt, safeAbs:Math.abs };
const seen = new Set();
globalThis.SafeTrig = new Proxy(trig, { get(t,p){
  if (p in t) return t[p];
  if (typeof p === 'string' && p.startsWith('safe')) {
    const k = p.slice(4).toLowerCase();
    if (!seen.has(p)) { seen.add(p); console.log('[SafeTrig fallback]', p, '->', k); }
    if (typeof Math[k] === 'function') return Math[k].bind(Math);
  }
  return undefined;
}});

let amd = null;
globalThis.define = (...a) => { const f=a[a.length-1];
  if (typeof f!=='function'){amd=f;return}
  const deps=a.find(x=>Array.isArray(x))||[]; const ex={},mo={exports:ex};
  let v=deps.map(d=>d==='exports'?ex:d==='module'?mo:globalThis[d]);
  if(!deps.length&&f.length) v=[ex,mo].slice(0,f.length);
  const r=f(...v); amd=(r!=null)?r:(Object.keys(ex).length?ex:mo.exports); };
globalThis.define.amd = true;
let box2d = require('./Box2DModuleGlMod.js');
const ok = x => x && typeof x==='object' && (x.Common||x.Dynamics);
box2d = ok(box2d)?box2d:ok(globalThis.Box2D)?globalThis.Box2D:ok(amd)?amd:amd;

const core = await import('./bonk-core.mjs');
const bonk = core.createBonkCore(box2d, { thumbScale: 1 });
const replay = JSON.parse(fs.readFileSync('/workspaces/wdb-bot/replay.json','utf8'));

// ---- JS port of halfUnserialize (verified format) ----
function decodeInputs(b64) {
  const buf = Buffer.from(b64,'base64'); let off = 0;
  const count = buf.readUInt16BE(off); off += 2;
  const byTick = [];
  for (let i=0;i<count;i++){
    const slot=buf.readUInt16BE(off); off+=2;   // PLAYER SLOT (UShort)
    const tick=buf.readUInt32BE(off); off+=4;   // TICK (Uint)
    const f=buf.readUInt8(off); off+=1;
    (byTick[tick] ||= [])[slot] = { left:!!(f&1), right:!!(f&2), up:!!(f&4),
                                    down:!!(f&8), action:!!(f&16), action2:!!(f&32) };
  }
  const slots=[...new Set(byTick.filter(Boolean).flatMap(a=>Object.keys(a).map(Number)))].sort((a,b)=>a-b);
  console.log('events:', count, '| slots:', slots, '| ticks with input:', byTick.filter(Boolean).length, '| max tick:', byTick.length-1);
  const per={}; for(const a of byTick) if(a) for(const k in a) per[k]=(per[k]||0)+1;
  console.log('events per slot:', per);
  return byTick;
}
const byTick = decodeInputs(replay.inputs);
const cur = [];
const inputsAt = t => { if (byTick[t]) for (const s in byTick[t]) cur[s] = byTick[t][s]; return cur; };

// ---- step: (state, inputs, adminInputs, fps, gameSettings, substeps, s, mapProvider) ----
const step = bonk.Physics.prototype.step;
const REPS = 20, N = byTick.length;
let ticks = 0;
const t0 = performance.now();
for (let r = 0; r < REPS; r++) {
  let state = JSON.parse(JSON.stringify(replay.startingState));
  const c = [];
  const at = t => { if (byTick[t]) for (const k in byTick[t]) c[k] = byTick[t][k]; return c; };
  for (let t = 0; t < N; t++) {
    state = step.call(bonk.Physics, state, at(t), replay.adminInputs?.[t] || null,
                      60, replay.gameSettings, 1, 0, null);
    ticks++;
    if (!state || !state.discs) break;
  }
}
const dt = (performance.now() - t0) / 1000;
const rate = ticks / dt;
console.log('ticks:', ticks, '| sec:', dt.toFixed(3), '| ticks/sec:', Math.round(rate));
console.log('10M steps, 1 core (h):', (1e7/rate/3600).toFixed(2), '| 8 cores (min):', (1e7/rate/8/60).toFixed(1));
