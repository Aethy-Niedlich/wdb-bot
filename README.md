# wdb-bot

**wdb-bot is a work-in-progress PyTorch Reinforcement Learning and Imitation
Learning bot for the Weird Death Ball (WDB) map in Bonk.io.** It is currently
an offline training and simulation prototype: the repository does not contain
a live Bonk.io client integration or a path for controlling a player in an
online match.

The main training path is Python PPO over a vector of local environments. A
native Rust extension supplies the environment simulation. Python also has
behavioral-cloning helpers, while the JavaScript files under `reference/`
support replay inspection, game-physics experiments, and simulator diagnostics.
Those pieces are not yet integrated into one end-to-end imitation-learning
pipeline.

## Status and build note

The checked-in Rust sources are archived as
[`rust/src/lib.rs (ARCHIVED).txt`](rust/src/lib.rs%20%28ARCHIVED%29.txt) and
[`rust/src/physics.rs (ARCHIVED).txt`](rust/src/physics.rs%20%28ARCHIVED%29.txt),
not at the `.rs` paths expected by Cargo. As a result, a clean native build
currently fails (`rust/Cargo.toml` expects `rust/src/lib.rs`). There is a
prebuilt extension at
[`python/wdb/wdb_engine.cpython-311-x86_64-linux-gnu.so`](python/wdb/wdb_engine.cpython-311-x86_64-linux-gnu.so);
it is usable with CPython 3.11 on x86-64 Linux, but it is not a portable
replacement for buildable Rust source.

## Architecture

```text
Python training loop (PyTorch)
  ├─ ActorCritic policy, PPO updates, checkpointing
  └─ VecEnv: observations, normalization, episode handling, opponents
       └─ wdb.wdb_engine.RustEnv (PyO3 extension)
            └─ map loading, physics step, rewards, terminal state

JavaScript reference tools
  └─ Bonk.io/Box2D assets, replay experiments, trace and demo inspection
```

### Python and Rust

- [`python/wdb/main.py`](python/wdb/main.py) runs PPO training. The current
  entry point uses 16 environments, stage 0 (player versus the ball, without
  player 2), 128-step rollouts, and 50 updates. It selects CUDA when available
  and otherwise uses CPU. These values are currently set in the entry point,
  rather than exposed as command-line options.
- [`python/wdb/policy.py`](python/wdb/policy.py) defines a shared multilayer
  PyTorch actor-critic network. Its four categorical action heads represent
  movement (3 choices), jump (3), heavy (2), and grapple (2); a critic predicts
  state value. The action vector is `[move, jump, heavy, grapple]`, with each
  component represented by its head's class index.
- [`python/wdb/vec_env.py`](python/wdb/vec_env.py) manages multiple native
  environments, builds 37-element observations, updates running observation
  statistics, records episode outcomes, and resets completed episodes. It
  reads the map from [`WDB_live_map.json`](WDB_live_map.json). In stages with
  player 2 enabled, it can select random, no-op, or scripted-chaser actions;
  stage 0, used by the current trainer, supplies a no-op player-2 action.
- The Rust `RustEnv` extension is imported as `wdb.wdb_engine`. Python loads
  the map JSON, calls `reset` and `step`, and receives a state dictionary plus
  reward and done flag. The Rust implementation parses map shapes into walls,
  advances the simulation, and reports shaping/terminal reward components,
  player and ball state, grapple information, and termination/truncation
  fields. The archived Rust implementation is a custom simulation, not a
  guarantee of exact parity with the live game.
- [`python/wdb/imitation.py`](python/wdb/imitation.py) contains JSONL replay
  loading and a behavioral-cloning training helper. Although `main.py` accepts
  a `--replay` option, it currently does not use it to run behavioral cloning.
  The helper expects rows with `obs` and `action` arrays; harvested game replay
  records are not automatically converted to the current normalized
  37-feature observation format.
- [`python/wdb/league.py`](python/wdb/league.py) provides policy snapshots,
  but the current training loop does not use the league to select opponents.

The trainer saves a checkpoint named `latest.pt` in its **current working
directory**. It contains model weights, observation dimension/version metadata,
and observation-normalization statistics. Run it from the repository root as
shown below to write `./latest.pt` rather than overwriting the tracked
[`python/latest.pt`](python/latest.pt) checkpoint.

### JavaScript and reference data

The JavaScript assets are for offline reference work, not the Python
environment's runtime:

- [`reference/env.mjs`](reference/env.mjs) loads the Bonk core and Box2D code
  with browser API shims for Node.js. The large bundled
  [`reference/bonk-core.mjs`](reference/bonk-core.mjs) and
  [`reference/Box2DModuleGlMod.js`](reference/Box2DModuleGlMod.js), along with
  [`reference/SafeTrig.js`](reference/SafeTrig.js), provide the basis for
  replay and physics experiments.
- [`reference/harvest.mjs`](reference/harvest.mjs) steps a replay and writes
  per-player JSONL frames and metadata under `reference/demos/`.
  [`reference/gates.mjs`](reference/gates.mjs) checks replay determinism,
  snapshot/resume behavior, and whether stepping one replay affects another.
  [`reference/bench.mjs`](reference/bench.mjs) benchmarks replay stepping;
  `harness.mjs` and the `probe_*.mjs`/`probe_*.js` scripts inspect the core and
  its APIs.
- [`scripts/record_reference.js`](scripts/record_reference.js) generates
  synthetic traces for the archived Rust parity workflow. It implements a
  separate simplified simulation; it is not an authoritative Bonk.io replay
  or a production training-data exporter.
- Root-level files such as [`alpha2s.js`](alpha2s.js),
  [`Box2DModuleGIMod.js`](Box2DModuleGIMod.js), and
  [`replay.json`](replay.json), as well as the contents of `reference/`, are
  game/reference bundles or data rather than Python package entry points.

## Setup and run

### Run with the bundled extension

This route applies only to **CPython 3.11 on x86-64 Linux**, matching the
checked-in extension. You will also need a PyTorch installation compatible
with your machine; CPU works, and CUDA is optional.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy torch

# From the repository root; the map path is resolved from the package.
PYTHONPATH=python python -m wdb.main
```

Training prints update/episode statistics and saves `latest.pt` in the
repository root. That path is ignored by Git. To use the bundled checkpoint
with the baseline evaluator, run:

```bash
PYTHONPATH=python python python/eval_baseline.py
```

The evaluator loads `latest.pt` from the current directory. The supplied
baseline script reports evaluation statistics but is still experimental.

### Build the extension from source

The intended build uses Python 3.11+, a Rust stable toolchain, and maturin.
Once active Rust source files are present at `rust/src/lib.rs` and
`rust/src/physics.rs`, the repository's packaging configuration can build and
install the extension with:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy torch maturin
python -m pip install ./python
```

**This source-build procedure does not work with the current checkout** until
the archived Rust sources are restored to the paths expected by Cargo. The
bundled binary only covers the CPython 3.11 Linux x86-64 ABI; other Python
versions and platforms need a successful native build.

## Reference tools

Node.js is used for the JavaScript reference scripts; CI specifies Node 20.
There is no JavaScript package manifest or npm dependency-install step.

```bash
# Generate synthetic reference traces (writes reference/frames.jsonl).
node scripts/record_reference.js

# Harvest replay frames (optional args: fps, substeps, replay JSON path).
node reference/harvest.mjs 30 1 reference/replays/321746.json

# Run the replay determinism/snapshot checks with the default fixtures.
node reference/gates.mjs

# Benchmark replay stepping (optional args: environment count, ticks, replay).
node reference/bench.mjs 8 500 reference/replays/321746.json
```

Harvested frames are exploratory data and are not directly consumable by the
current Python behavioral-cloning helper without preprocessing.

## Tests and known gaps

Python tests are under [`python/tests/`](python/tests) and the additional
diagnostic scripts are directly under `python/`. Rust/extension tests and
parity checks are configured in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).
The current checkout is not in a state where that clean-source CI path can
pass: Cargo cannot find `rust/src/lib.rs`, and
[`python/tests/test_parity.py`](python/tests/test_parity.py) calls a
`step_json` extension method that the archived `RustEnv` source does not
provide. The prebuilt extension does provide the `reset`, `load_map`, and
`step` methods used by the current vector environment.

Other current limitations:

- The training loop is an experimental stage-0 PPO run, not a finished
  competitive or online bot.
- The `--replay` flag and league opponent pool are not wired into the training
  loop; behavioral cloning is helper-level functionality only.
- JavaScript replay frames, synthetic Rust parity traces, and Python's
  normalized observations have different schemas and should not be treated as
  interchangeable datasets.
