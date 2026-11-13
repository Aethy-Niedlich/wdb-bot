# wdb-bot

> **Status:** Early Work in Progress (WIP) — Offline RL/IL training prototype. Please bear with me while it's still learning!

An **Imitation Learning (IL)** and **Reinforcement Learning (RL)** bot built in PyTorch for the **Weird Death Ball (WDB)** map in [Bonk.io](https://bonk.io). 

Currently, `wdb-bot` focuses on offline simulation and policy training. It does not yet include live browser integration or online match control.

---

## Architecture

The project pairs a **PyTorch PPO training loop** with a **Rust PyO3 physics extension** for fast vectorized rollouts, alongside **Node.js reference tools** for inspecting Bonk.io replays and Box2D mechanics.

```text
Python Training Loop (PyTorch)
 ├── ActorCritic Policy & PPO Updates (python/wdb/policy.py, main.py)
 └── VecEnv: 37-dim observations, normalization, opponents (python/wdb/vec_env.py)
      └── wdb.wdb_engine.RustEnv (PyO3 Native Extension)
           └── Map loading (WDB_live_map.json), custom physics step, rewards

JavaScript Reference Tools (Node.js)
 └── Bonk.io / Box2D assets, replay harvesting, determinism & parity checks
```

### Core Components

* **Training Loop ([`python/wdb/main.py`](python/wdb/main.py)):** Runs PPO across 16 parallel environments (currently hardcoded to Stage 0: 1 player vs. ball, 128-step rollouts, 50 updates) using CUDA if available or CPU otherwise.
* **Policy Network ([`python/wdb/policy.py`](python/wdb/policy.py)):** Shared multilayer actor-critic network outputting four categorical action heads (`[move, jump, heavy, grapple]` with dimensions `[3, 3, 2, 2]`) plus a state-value critic.
* **Vector Environment ([`python/wdb/vec_env.py`](python/wdb/vec_env.py)):** Loads [`WDB_live_map.json`](WDB_live_map.json), builds normalized 37-feature observations, tracks episode stats, and manages Player 2 behavior (no-op in Stage 0; random or scripted chaser in later stages).
* **Rust Simulation (`wdb.wdb_engine`):** Custom Box2D-inspired physics engine exposed via PyO3 that handles wall collisions, grapple mechanics, and shaping/terminal rewards.
* **Imitation & League Helpers ([`imitation.py`](python/wdb/imitation.py), [`league.py`](python/wdb/league.py)):** Scaffolding for behavioral cloning from JSONL replays and self-play policy snapshots (not yet wired into the main training loop).
* **JS Reference Suite (`reference/`, `scripts/`):** Standalone Node.js utilities for stepping real Bonk.io replays ([`harvest.mjs`](reference/harvest.mjs)), testing determinism ([`gates.mjs`](reference/gates.mjs)), and benchmarking ([`bench.mjs`](reference/bench.mjs)).

---

## Quickstart

> **Important Build Note:** The Rust source files are currently archived as `.txt` files (`rust/src/lib.rs (ARCHIVED).txt` and `physics.rs (ARCHIVED).txt`), so building from source via Cargo/Maturin will fail until they are restored. However, a **prebuilt Linux x86-64 binary** (`wdb_engine.cpython-311-x86_64-linux-gnu.so`) is included for immediate use.

### 1. Run with the Prebuilt Extension (Linux x86-64, Python 3.11)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip numpy torch

# Run from the repository root so checkpoints save to ./latest.pt (gitignored)
# instead of overwriting the tracked python/latest.pt file
PYTHONPATH=python python -m wdb.main
```

To run the experimental baseline evaluator against your `latest.pt` checkpoint:

```bash
PYTHONPATH=python python python/eval_baseline.py
```

### 2. Building from Source (Once Rust Sources Are Restored)

If you restore `rust/src/lib.rs` and `rust/src/physics.rs` on any platform with a stable Rust toolchain:

```bash
pip install numpy torch maturin
pip install ./python
```

---

## Reference & Replay Tools

The JavaScript utilities require **Node.js (v20+)** and run with zero external npm dependencies:

```bash
# Harvest per-player JSONL frames from a Bonk.io replay
node reference/harvest.mjs 30 1 reference/replays/321746.json

# Verify replay determinism and snapshot/resume behavior
node reference/gates.mjs

# Benchmark replay stepping speed (envs, ticks, replay path)
node reference/bench.mjs 8 500 reference/replays/321746.json

# Generate synthetic traces for Rust parity testing
node scripts/record_reference.js
```

---

## Current Limitations & Roadmap

* [x] Custom Rust simulation & prebuilt CPython 3.11 extension (`wdb_engine`)
* [x] Stage-0 PPO training loop (solo ball survival/control)
* [x] Node.js replay harvesting and determinism verification (`reference/`)
* [ ] Restore active `.rs` files in `rust/src/` for cross-platform builds and CI parity
* [ ] Unify data schemas to convert harvested JS replay frames into normalized 37-dim observations
* [ ] Wire `--replay` behavioral cloning and `league.py` self-play opponents into `main.py`
* [ ] Live browser/client bridge for testing in online Bonk.io matches

---

## Contributing & Disclaimer

**Contributing:** Since this project is in early active development, APIs, observation schemas, and simulation physics are still evolving. Feel free to open an issue or submit a pull request if you want to help restore the cross-platform Rust build, unify replay schemas, or experiment with reward shaping!

**Disclaimer:** `wdb-bot` is an independent research and hobby project built for educational purposes. It is not affiliated with or endorsed by Bonk.io, and it is strictly intended for offline simulation and local experimentation.
