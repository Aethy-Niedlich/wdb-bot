#!/usr/bin/env node
// reference/bridge_worker.mjs

import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
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

const step = bonk.Physics.prototype.step;
const FPS = 30, SUB = 1;

// ---------- shared: input decoder + mask helpers ----------
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

function maskToAct(m){
  return { left:!!(m&1), right:!!(m&2), up:!!(m&4),
           down:!!(m&8), action:!!(m&16), action2:!!(m&32) };
}

// ---------- mode dispatch ----------
if (process.argv[2] === '--serve') {
  await serveMode();
} else {
  probeMode();
}

// ---------- probe mode (unchanged) ----------
function probeMode(){
  const K = Number(process.argv[2] || 8);
  const T = Number(process.argv[3] || 500);
  const REPLAY_PATH = process.argv[4] || new URL('./replays/321746.json', import.meta.url).pathname;

  const replay = JSON.parse(fs.readFileSync(REPLAY_PATH,'utf8'));
  const byTick = decodeInputs(replay.inputs);

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
}

// ---------- serve mode ----------
async function serveMode(){
  const rest = process.argv.slice(3);
  let K = 8, vocabPath = null;
  for (let i = 0; i < rest.length; i++) {
    if (rest[i] === '--vocab') vocabPath = rest[++i];
    else if (rest[i] === '--K') K = Number(rest[++i]);
    else if (!isNaN(Number(rest[i])) && rest[i] !== '') K = Number(rest[i]);
  }
  if (!Number.isFinite(K) || K < 1 || K > 256) throw new Error(`bad K: ${K}`);
  
  const VOCAB = vocabPath
    ? JSON.parse(fs.readFileSync(vocabPath, 'utf8')).vocab
    : Array.from({ length: 32 }, (_, i) => i);
  if (!Array.isArray(VOCAB) || VOCAB.length > 32)
    throw new Error(`bad vocab: length=${VOCAB?.length}`);

  const OBS_DIM = 13;   // TODO: obs.py schema (122)
  const N_ACTIONS = 32;

  // Load RAW map from disk (respects double-scaling warning)
  const RAW_MAP_PATH = path.resolve(process.cwd(), 'WDB_live_map.json');
  let RAW_MAP = null;
  if (fs.existsSync(RAW_MAP_PATH)) {
    RAW_MAP = JSON.parse(fs.readFileSync(RAW_MAP_PATH, 'utf8'));
  }

  const TAG_STEP = 0x01, TAG_RESET = 0x02, TAG_OBS = 0x03, TAG_STATS = 0x04;

  function encodeFrame(tag, payload) {
    const out = Buffer.allocUnsafe(5 + payload.length);
    out.writeUInt32LE(1 + payload.length, 0);
    out.writeUInt8(tag, 4);
    payload.copy(out, 5);
    return out;
  }

  class FrameReader {
    constructor(stream) {
      this.stream = stream;
      this.buf = Buffer.alloc(1 << 20);
      this.len = 0;
      this.queue = [];
      this.waiters = [];
      this.closed = false;
      stream.on('data', c => this._onData(c));
      stream.on('end', () => { this.closed = true; this._drain(); });
    }
    _onData(chunk) {
      while (this.buf.length < this.len + chunk.length) {
        const nb = Buffer.alloc(this.buf.length * 2);
        this.buf.copy(nb, 0, 0, this.len);
        this.buf = nb;
      }
      chunk.copy(this.buf, this.len);
      this.len += chunk.length;
      this._drain();
    }
    _drain() {
      while (this.len >= 4) {
        const total = this.buf.readUInt32LE(0);
        if (this.len < 4 + total) break;
        const tag = this.buf.readUInt8(4);
        const payload = Buffer.from(this.buf.subarray(5, 4 + total));
        this.buf.copy(this.buf, 0, 4 + total, this.len);
        this.len -= 4 + total;
        const f = { tag, payload };
        if (this.waiters.length) this.waiters.shift()(f);
        else this.queue.push(f);
      }
      if (this.closed && this.len === 0) {
        while (this.waiters.length) this.waiters.shift()(null);
      }
    }
    read() {
      if (this.queue.length) return Promise.resolve(this.queue.shift());
      if (this.closed && this.len === 0) return Promise.resolve(null);
      return new Promise(r => this.waiters.push(r));
    }
  }

  const reader = new FrameReader(process.stdin);
  const envs = new Array(K).fill(null);
  const roundStartCache = new Map();

  const obsBuf   = new Float32Array(K * OBS_DIM);
  const rewBuf   = new Float32Array(K);
  const doneBuf  = new Uint8Array(K);
  const truncBuf = new Uint8Array(K);
  const rlBuf    = new Int32Array(K);

  const meta = {
    obs_dim: OBS_DIM, K, schema: 'stub.v0',
    n_actions: N_ACTIONS, vocab_len: VOCAB.length, vocab_version: 1,
  };
  process.stdout.write(encodeFrame(TAG_STATS, Buffer.from(JSON.stringify(meta), 'utf8')));

  let batches = 0;
  let engineNs = 0n, outNs = 0n;
  let lastReport = process.hrtime.bigint();

  function fillStubObs(i) {
    const base = i * OBS_DIM;
    for (let j = 0; j < OBS_DIM; j++) obsBuf[base + j] = 0;
  }

  function emitObs() {
    const bodyLen = obsBuf.byteLength + rewBuf.byteLength + K + K + rlBuf.byteLength;
    const out = Buffer.allocUnsafe(5 + bodyLen);
    out.writeUInt32LE(1 + bodyLen, 0);
    out.writeUInt8(TAG_OBS, 4);
    let o = 5;
    Buffer.from(obsBuf.buffer).copy(out, o); o += obsBuf.byteLength;
    Buffer.from(rewBuf.buffer).copy(out, o); o += rewBuf.byteLength;
    Buffer.from(doneBuf.buffer).copy(out, o); o += K;
    Buffer.from(truncBuf.buffer).copy(out, o); o += K;
    Buffer.from(rlBuf.buffer).copy(out, o);
    process.stdout.write(out);
  }

  function stepOne(i, actionId) {
    const env = envs[i];
    if (!env) { doneBuf[i] = 1; rewBuf[i] = 0; rlBuf[i] = 0; fillStubObs(i); return; }
    const prev = env.state;
    
    if (!prev || prev.fte === 0) {
      doneBuf[i] = 1; rewBuf[i] = 0; rlBuf[i] = (prev?.rl ?? 0) | 0; fillStubObs(i); return;
    }
    
    const mask = VOCAB[actionId] | 0;
    const inp = new Array(prev.discs.length);
    if (env.egoSlot >= 0 && env.egoSlot < inp.length) inp[env.egoSlot] = maskToAct(mask);
    
    const next = step.call(bonk.Physics, prev, inp, null, FPS, env.gameSettings, SUB);
    if (!next || !next.discs) { 
      doneBuf[i] = 1; rewBuf[i] = 0; fillStubObs(i); return; 
    }
    
    env.state = next;
    
    // Qwen required termination logic:
    const prevFte = prev.fte;
    const nextFte = next.fte;
    
    // Truncation: fte goes from -1 to >=0 (goal tick kept)
    const isTrunc = (prevFte === -1 && nextFte >= 0) ? 1 : 0;
    // Termination: fte === 0 (safety net)
    const isDone  = (nextFte === 0) ? 1 : 0;

    rewBuf[i]   = 0;                    // TODO: deriveReward (swing.b === ballIdx)
    doneBuf[i]  = isDone;
    truncBuf[i] = isTrunc;
    rlBuf[i]    = next.rl | 0;
    fillStubObs(i);                     // TODO: real obs.py
  }

  function buildEnvFromReset(desc) {
    if (desc.kind === 'replay') {
      const replay = JSON.parse(fs.readFileSync(desc.path, 'utf8'));
      let state = JSON.parse(JSON.stringify(replay.startingState));
      const byTick = decodeInputs(replay.inputs);
      const held = [];
      const at = t => { if(byTick[t]) for(const k in byTick[t]) held[k]=byTick[t][k]; return held; };
      const targetTick = desc.tick ?? 0;
      for (let t = 0; t < targetTick; t++) {
        if (state.fte === 0) break;
        state = step.call(bonk.Physics, state, at(t), replay.adminInputs?.[t] || null, FPS, replay.gameSettings, SUB);
        if (!state || !state.discs) break;
      }
      const egoSlot = desc.ego_slot ?? state.discs.findIndex(d => d !== null);
      return { state, egoSlot, gameSettings: replay.gameSettings };
    } 
    
    if (desc.kind === 'round_start') {
      if (!RAW_MAP) throw new Error("WDB_live_map.json not found in cwd");
      const mapId = desc.map_id ?? RAW_MAP.m?.dbid ?? 80622;
      const seed = desc.seed ?? 0;
      const cacheKey = `${mapId}_${seed}`;

      if (roundStartCache.has(cacheKey)) {
        const cached = roundStartCache.get(cacheKey);
        const egoSlot = desc.ego_slot ?? cached.discs.findIndex(d => d !== null && d.team === desc.ego_team);
        return { state: cached, egoSlot, gameSettings: cached.gameSettings };
      }

      const players = desc.players ?? [{id: 0, team: 2}, {id: 1, team: 3}];
      const settings = desc.settings ?? { tea: true, mo: 'sp', bal: 1, gt: 1, wl: 1, q: 1, map: RAW_MAP };
      
      let s = bonk.Physics.createNewState(players, RAW_MAP, seed, false, null, settings, false);
      
      for (let k = 0; k < 91; k++) {
        if (!s || s.fte === 0) break;
        s = step.call(bonk.Physics, s, [], null, FPS, settings, SUB);
      }
      
      s.gameSettings = settings; 
      roundStartCache.set(cacheKey, s);
      
      const egoSlot = desc.ego_slot ?? s.discs.findIndex(d => d !== null && d.team === desc.ego_team);
      return { state: s, egoSlot, gameSettings: settings };
    }

    throw new Error(`reset kind not implemented: ${desc.kind}`);
  }

  function handleReset(payload) {
    const flags = payload.subarray(0, K);
    let off = K;
    for (let i = 0; i < K; i++) {
      if (!flags[i]) continue;
      const jsonLen = payload.readUInt32LE(off); off += 4;
      const jsonBytes = payload.subarray(off, off + jsonLen); off += jsonLen;
      const desc = JSON.parse(jsonBytes.toString('utf8'));
      envs[i] = buildEnvFromReset(desc);
      rewBuf[i] = 0; doneBuf[i] = 0; truncBuf[i] = 0;
      rlBuf[i] = (envs[i]?.state?.rl ?? 0) | 0;
      fillStubObs(i);
    }
    emitObs();
  }

  while (true) {
    const frame = await reader.read();
    if (!frame) break;
    if (frame.tag === TAG_STEP) {
      const actions = new Int32Array(
        frame.payload.buffer, frame.payload.byteOffset, frame.payload.byteLength >> 2
      );
      if (actions.length !== K)
        throw new Error(`STEP payload has ${actions.length} actions, K=${K}`);
      const t0 = process.hrtime.bigint();
      for (let i = 0; i < K; i++) stepOne(i, actions[i]);
      engineNs += process.hrtime.bigint() - t0;
      const t1 = process.hrtime.bigint();
      emitObs();
      outNs += process.hrtime.bigint() - t1;
      batches += 1;
    } else if (frame.tag === TAG_RESET) {
      handleReset(frame.payload);
      batches += 1;
    } else {
      process.stderr.write(`bridge_worker: unknown tag 0x${frame.tag.toString(16)}\n`);
    }
    const now = process.hrtime.bigint();
    if (Number(now - lastReport) > 1_000_000_000 && batches > 0) {
      const stats = {
        batches,
        engine_ms_per_batch: Number(engineNs) / 1e6 / batches,
        out_ms_per_batch:    Number(outNs)   / 1e6 / batches,
        rss_kb: process.memoryUsage().rss >> 10,
      };
      process.stdout.write(encodeFrame(TAG_STATS, Buffer.from(JSON.stringify(stats), 'utf8')));
      batches = 0; engineNs = 0n; outNs = 0n;
      lastReport = now;
    }
  }
}