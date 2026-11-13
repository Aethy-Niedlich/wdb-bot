import fs from 'node:fs';
import crypto from 'node:crypto';
import { bonk } from './env.mjs';

const FPS = 30, SUB = 1, T = 100, K = 100;
const REPLAY_PATH = process.argv[2] || new URL('./replays/321746.json', import.meta.url).pathname;
const OTHER_PATH  = process.argv[3] || new URL('./replays/bonkverse-replay-6-355011.json', import.meta.url).pathname;

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
const step = bonk.Physics.prototype.step;
const hashState = s => crypto.createHash('sha256').update(JSON.stringify({
  d: s.discs.map(d=>d?[d.x,d.y,d.xv,d.yv,d.a,d.av,d.a1a]:null),
  b: s.physics.bodies.map(b=>b?[b.p,b.lv,b.a,b.av]:null)
})).digest('hex').slice(0,16);

function makeRunner(byTick){ const held=[]; return t => { if(byTick[t]) for(const k in byTick[t]) held[k]=byTick[t][k]; return held; }; }

function runReplay(replay, fromT, nTicks, snapshot){
  const at = makeRunner(decodeInputs(replay.inputs));
  for(let t=0;t<fromT;t++) at(t);                 // rebuild held inputs without stepping
  let state = snapshot || JSON.parse(JSON.stringify(replay.startingState));
  const hashes=[];
  for(let t=fromT;t<fromT+nTicks;t++){
    if(state.fte===0) break;
    const prev=state.fte;
    hashes.push(hashState(state));
    state = step.call(bonk.Physics, state, at(t), replay.adminInputs?.[t]||null, FPS, replay.gameSettings, SUB);
    if(!state||!state.discs) break;
    if(prev===-1 && state.fte>=0) break;
  }
  return { hashes, state };
}

const replay = JSON.parse(fs.readFileSync(REPLAY_PATH,'utf8'));
const other  = JSON.parse(fs.readFileSync(OTHER_PATH,'utf8'));

const a = runReplay(replay,0,T+K), b = runReplay(replay,0,T+K);
console.log('GATE2 determinism:',
  (a.hashes.length===b.hashes.length && a.hashes.every((h,i)=>h===b.hashes[i])) ? 'PASS':'FAIL',
  `(${a.hashes.length} ticks)`);

const mid  = runReplay(replay,0,T).state;
const snap = JSON.parse(JSON.stringify(mid));
const resumed = runReplay(replay,T,K,snap);
const cont    = runReplay(replay,0,T+K);
const tail = cont.hashes.slice(T, T+resumed.hashes.length);
console.log('GATE3 snapshot round trip:',
  (resumed.hashes.length>0 && tail.length===resumed.hashes.length && resumed.hashes.every((h,i)=>h===tail[i])) ? 'PASS':'FAIL',
  `(${resumed.hashes.length} resumed ticks)`);

runReplay(other,0,50);                            // pollute module state with a different match
const polluted = runReplay(replay,0,T+K);
console.log('GATE4 purity:',
  (polluted.hashes.length===a.hashes.length && polluted.hashes.every((h,i)=>h===a.hashes[i])) ? 'PASS':'FAIL');
