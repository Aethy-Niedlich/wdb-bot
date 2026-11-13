import { createRequire } from 'node:module';
import fs from 'fs';
const require = createRequire(import.meta.url);

// ---- globals (same as harness.mjs) ----
globalThis.window = globalThis; globalThis.self = globalThis;
try { Object.defineProperty(globalThis, 'navigator', { value: { userAgent: 'node' }, configurable: true }); } catch {}
globalThis.document = { createElement: () => ({ getContext: () => null, style: {}, width: 0, height: 0 }), body: { appendChild(){} } };
globalThis.requestAnimationFrame = cb => setTimeout(() => cb(Date.now()), 16);
globalThis.cancelAnimationFrame = id => clearTimeout(id);

const rdr = { width:0, height:0, resize(){}, render(){}, destroy(){}, view:{}, stage:{children:[]}, forceRendererSize(){}, backgroundColor:0 };
globalThis.PIXI = { 
  autoDetectRenderer: () => ({ ...rdr }), 
  settings:{}, 
  utils:{},
  Container: class { addChild(){} },
  Graphics: class { beginFill(){}; lineStyle(){}; drawRect(){}; beginTextureFill(){} },
  Text: class { resolution=1; width=100; height=20; x=0; y=0; alpha=1; visible=true },
  filters: { AdjustmentFilter: class {}, BlurFilter: class {}, FXAAFilter: class {} },
  resources: { SVGResource: class { load(){return Promise.resolve()} } }
};
globalThis.AudioContext = class { constructor(){ this.destination={}; } createGain(){ return { connect(){}, gain:{value:0} }; } };
globalThis.peerjs = { peerjs: { Peer: class {} } };
globalThis.Howl = class { constructor(){} play(){} volume(){return this} stereo(){return this} on(){} };
globalThis.GameResources = { soundStrings: { capIncrease: '', capDecrease: '', discDisc: '', platBounce: '' }, svgStrings: {} };
globalThis.dcodeIO = { PSON: { StaticPair: class { constructor(d){this.dict=d} encode(){return {toBase64:() => ''}} decode(){return {}} } } };

let amd = null;
globalThis.define = (...a) => {
  const f = a[a.length-1];
  if (typeof f !== 'function') { amd = f; return; }
  const deps = a.find(x => Array.isArray(x)) || [];
  const ex = {}; const mo = { exports: ex };
  let v = deps.map(d => d==='exports'?ex : d==='module'?mo : globalThis[d]);
  if (!deps.length && f.length) v = [ex, mo].slice(0, f.length);
  const r = f(...v);
  amd = (r != null) ? r : (Object.keys(ex).length ? ex : mo.exports);
};
globalThis.define.amd = true;
let box2d = require('./Box2DModuleGlMod.js');
const ok = x => x && typeof x==='object' && (x.Common||x.Dynamics);
box2d = ok(box2d) ? box2d : ok(globalThis.Box2D) ? globalThis.Box2D : ok(amd) ? amd : amd;

const core = await import('./bonk-core.mjs');
const bonk = core.createBonkCore(box2d, { thumbScale: 1 });

// ---- PROBE 1: Function signatures (first 800 chars) ----
console.log('\n=== createNewState signature ===');
console.log(bonk.Physics.createNewState.toString().slice(0, 800));
console.log('\n=== step signature ===');
console.log(bonk.Physics.prototype.step.toString().slice(0, 800));

// ---- PROBE 2: Test decodeInputs on our replay ----
const replay = JSON.parse(fs.readFileSync('/workspaces/wdb-bot/WDB_live_map.json', 'utf8'));
console.log('\n=== decodeInputs test ===');
console.log('decodeInputs type:', typeof bonk.game.decodeInputs);
try {
  const decoded = bonk.game.decodeInputs(replay.inputs);
  console.log('decoded type:', typeof decoded, Array.isArray(decoded));
  if (Array.isArray(decoded)) {
    console.log('decoded length:', decoded.length);
    console.log('first non-null slot:', decoded.findIndex(x => x !== null));
    const firstSlot = decoded.find(x => x !== null);
    console.log('first slot keys:', firstSlot ? Object.keys(firstSlot[0] || firstSlot) : 'empty');
  }
} catch (e) {
  console.log('decodeInputs error:', e.message);
}

// ---- PROBE 3: createNewState with minimal args ----
console.log('\n=== createNewState minimal test ===');
try {
  const map = replay.startingState;
  const players = replay.playerArray.filter(p => p !== null);
  const seed = replay.startingState.seed || 0;
  console.log('map.physics.ppm:', map.physics?.ppm);
  console.log('players count:', players.length);
  console.log('seed:', seed);
  
  // Try calling with (players, map, seed)
  const state = bonk.Physics.createNewState(players, map, seed, false, false, replay.gameSettings, false);
  console.log('state created:', typeof state);
  console.log('state keys:', Object.keys(state).slice(0, 20));
  console.log('state.discs length:', state.discs?.length);
  console.log('state.discs[14] (you):', state.discs?.[14]);
} catch (e) {
  console.log('createNewState error:', e.message);
  console.log('stack:', e.stack?.slice(0, 400));
}
