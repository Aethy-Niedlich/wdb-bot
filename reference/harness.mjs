import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);

// ---- browser-ish globals ----
globalThis.window = globalThis; globalThis.self = globalThis;
try { Object.defineProperty(globalThis, 'navigator', { value: { userAgent: 'node' }, configurable: true }); } catch {}
globalThis.document = { createElement: () => ({ getContext: () => null, style: {}, width: 0, height: 0 }), body: { appendChild(){} } };
globalThis.requestAnimationFrame = cb => setTimeout(() => cb(Date.now()), 16);
globalThis.cancelAnimationFrame = id => clearTimeout(id);

// ---- PIXI + audio stubs ----
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

// ---- Howl (audio) stub ----
globalThis.Howl = class {
  constructor(opts) { this.src = opts.src; this.volume = opts.volume; this.loop = opts.loop; }
  play() {}
  volume(v) { return this; }
  stereo(v) { return this; }
  on(evt, cb) {}
};
globalThis.GameResources = {
  soundStrings: { capIncrease: '', capDecrease: '', discDisc: '', platBounce: '' },
  svgStrings: {}
};

// ---- dcodeIO.PSON stub ----
globalThis.dcodeIO = {
  PSON: {
    StaticPair: class {
      constructor(dict) { this.dict = dict; }
      encode(obj) { return { toBase64: () => '' }; }
      decode(data) { return {}; }
    }
  }
};

// ---- AMD shim, then load box2dweb ----
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
console.log('box2d:', ok(box2d), '| b2Vec2:', typeof box2d?.Common?.Math?.b2Vec2);

// ---- load bonk-core and instantiate ----
const core = await import('./bonk-core.mjs');
console.log('core exports:', Object.keys(core));
const bonk = core.createBonkCore(box2d, { thumbScale: 1 });
console.log('bonk keys:', Object.keys(bonk));
console.log('Physics methods:', Object.getOwnPropertyNames(bonk.Physics.prototype || {}));

// ---- ENTRY POINT PROBE ----
console.log('\n--- ENTRY POINTS ---');
console.log('Physics statics:', Object.getOwnPropertyNames(bonk.Physics).filter(k => typeof bonk.Physics[k] === 'function'));
console.log('game keys:', Object.keys(bonk.game));
console.log('Physics.createNewState?', typeof bonk.Physics.createNewState);
console.log('game.createNewState?', typeof bonk.game.createNewState);
console.log('Physics.prototype.step params:', bonk.Physics.prototype.step.length);
