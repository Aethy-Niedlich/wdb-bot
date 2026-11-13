global.window = global;
global.self = global;
global.document = { createElement: () => ({ getContext: () => null, style: {} }) };
global.navigator = { userAgent: 'node' };

// --- AMD shim: capture whatever define() receives ---
let amdResult = null;
global.define = function (...args) {
  const factory = args[args.length - 1];
  if (typeof factory !== 'function') { amdResult = factory; return; }
  const deps = args.find(a => Array.isArray(a)) || [];
  const exportsObj = {};
  const modObj = { exports: exportsObj };
  let vals = deps.map(d => d === 'exports' ? exportsObj : d === 'module' ? modObj : d === 'require' ? require : global[d]);
  if (deps.length === 0 && factory.length > 0) vals = [exportsObj, require, modObj].slice(0, factory.length);
  const ret = factory(...vals);
  amdResult = (ret !== undefined && ret !== null) ? ret : (Object.keys(exportsObj).length ? exportsObj : modObj.exports);
};
global.define.amd = true;

let m = null;
try { m = require('./Box2DModuleGlMod.js'); }
catch (e) { console.log('require threw:', e.message); }

const looksLikeBox2D = x => x && typeof x === 'object' && (x.Common || x.Dynamics || x.Collision);
const cands = [m, global.Box2D, amdResult, global.Box2DModuleGlMod];
m = cands.find(looksLikeBox2D) || cands.find(x => typeof x === 'function') || amdResult || m;

console.log('type:', typeof m);
const get = (o, p) => p.split('.').reduce((a, k) => (a == null ? a : a[k]), o);
const finish = b => {
  console.log('keys:', Object.keys(b || {}).slice(0, 30));
  for (const p of ['Common.Math.b2Vec2', 'Common.Math.b2Math', 'Collision.Shapes.b2CircleShape'])
    console.log(p, '->', typeof get(b, p));
};
if (typeof m === 'function') Promise.resolve(m({})).then(finish).catch(e => console.log('factory err:', e.message));
else finish(m);
