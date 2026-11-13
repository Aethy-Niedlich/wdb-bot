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
export const bonk = (await import('./bonk-core.mjs')).createBonkCore(box2d, { thumbScale: 1 });
