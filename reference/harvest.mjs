import { createRequire } from 'node:module';
import fs from 'node:fs';
const require = createRequire(import.meta.url);

// ---------- browser shims ----------
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

// ---------- real SafeTrig (bonk.io/js/SafeTrig.js) ----------
{ const src = fs.readFileSync(new URL('./SafeTrig.js', import.meta.url), 'utf8');
  (0, eval)(src + '\n;globalThis.SafeTrig = SafeTrig;'); }

// ---------- AMD shim + box2dweb ----------
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

// ---------- args: fps, substeps, replay path, source ----------
const FPS = Number(process.argv[2] || 30);
const SUB = Number(process.argv[3] || 1);
const REPLAY_PATH = process.argv[4] || new URL('../replay.json', import.meta.url).pathname;
const TAG = REPLAY_PATH.split('/').pop().replace(/\.json$/, '');
// Q2: provenance tag. Default inference from path; explicit arg overrides.
const SOURCE = process.argv[5] || (REPLAY_PATH.includes('/own/') ? 'own' : 'reference');
const replay = JSON.parse(fs.readFileSync(REPLAY_PATH, 'utf8'));

// ---------- decoder (verified against the game's halfUnserialize) ----------
function decodeInputs(b64) {
  const buf = Buffer.from(b64, 'base64'); let off = 0;
  const count = buf.readUInt16BE(off); off += 2; const byTick = [];
  for (let i = 0; i < count; i++) {
    const slot = buf.readUInt16BE(off); off += 2;
    const tick = buf.readUInt32BE(off); off += 4;
    const f = buf.readUInt8(off); off += 1;
    (byTick[tick] ||= [])[slot] = { left:!!(f&1), right:!!(f&2), up:!!(f&4),
                                    down:!!(f&8), action:!!(f&16), action2:!!(f&32) };
  }
  return { byTick, count };
}
const { byTick, count } = decodeInputs(replay.inputs);

// ---------- locate the death ball ----------
const bodies0 = replay.startingState.physics.bodies, fx0 = replay.startingState.physics.fixtures;
let ballIdx = bodies0.findIndex(b => b && b.s && b.s.n === 'DeathBall');
if (ballIdx < 0) ballIdx = bodies0.findIndex(b => b && b.s && b.s.type === 'd' &&
  (b.fx || []).some(fi => fx0[fi] && fx0[fi].d === true));

// ---------- rollout ----------
const step = bonk.Physics.prototype.step, N = byTick.length;
const slots = replay.startingState.discs.map((d,i) => d ? i : null).filter(v => v !== null);
const demos = {}; for (const s of slots) demos[s] = [];
let state = replay.startingState, stop = 'end-of-clip', unlabelled = 0;
// Q1: anchor — createNewState sets ftu=60 (o flag), 5 (r flag), 90 (neither, our resets).
const ftu0 = state.ftu;
const held = [];
const at = t => { if (byTick[t]) for (const k in byTick[t]) held[k] = byTick[t][k]; return held; };

for (let t = 0; t < N; t++) {
  if (state.fte === 0) { stop = `banner at tick ${t}`; break; }
  const prevFte = state.fte;
  const inp = at(t);
  const bb = ballIdx >= 0 ? state.physics.bodies[ballIdx] : null;
  const ball = bb ? { x:bb.p[0], y:bb.p[1], vx:bb.lv[0], vy:bb.lv[1], a:bb.a, av:bb.av } : null;
  for (const s of slots) {
    const d = state.discs[s]; if (!d) continue;
    const act = inp[s] || null; if (!act) unlabelled++;
    demos[s].push({
      t,
      rl: state.rl,
      ftu: state.ftu,           // Q1: input state's ftu; mask iff > -1
      team: d.team,             // Q2: ego team for score context
      obs: { x:d.x, y:d.y, vx:d.xv, vy:d.yv, a:d.a, av:d.av, a1a:d.a1a,
             lhid:d.lhid, lht:d.lht, swing:d.swing||null },
      act, ball,
      scores: state.scores,
      capZones: state.capZones,
      others: state.discs.map((o,i) => (o && i !== s)
              ? { i, x:o.x, y:o.y, vx:o.xv, vy:o.yv, team:o.team }
              : null).filter(Boolean),
    });
  }
  state = step.call(bonk.Physics, state, inp, replay.adminInputs?.[t] || null, FPS, replay.gameSettings, SUB);
  if (!state || !state.discs) { stop = `state ended at tick ${t}`; break; }
  if (prevFte === -1 && state.fte >= 0) { stop = `round decided at tick ${t}`; break; }
}

fs.mkdirSync(new URL('./demos/', import.meta.url), { recursive: true });
let total = 0;
for (const s of slots) {
  fs.writeFileSync(new URL(`./demos/${TAG}_p${s}.jsonl`, import.meta.url),
                   demos[s].map(r => JSON.stringify(r)).join('\n'));
  total += demos[s].length;
}
const dm = stop.match(/tick (\d+)/);
const decision = dm ? +dm[1] : null;
fs.writeFileSync(new URL(`./demos/${TAG}.meta.json`, import.meta.url), JSON.stringify({
  tag: TAG,
  mid: replay.mid,
  source: SOURCE,             // Q2: 'own' | 'reference'
  rl_start: replay.startingState.rl,
  ftu0: ftu0,                 // Q1: sanity anchor (90 = clean reset, 60/5 = o/r flags)
  ticks: N,
  stop,
  decision_tick: decision,
  round_length: decision != null ? replay.startingState.rl + decision + 1 : null,
  frames: Object.fromEntries(slots.map(s => [s, demos[s].length]))
}, null, 1));
console.log(`${TAG} | mid=${replay.mid} source=${SOURCE} fps=${FPS} sub=${SUB} | events=${count} ticks=${N} ball=${ballIdx} ftu0=${ftu0} | stop: ${stop}`);
console.log(`  frames: ${slots.map(s => `${s}:${demos[s].length}`).join(' ')} | total ${total} | unlabelled ${unlabelled}`);