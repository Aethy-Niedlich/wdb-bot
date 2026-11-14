"""Python client for the Node headless Bonk bridge. Node owns ALL engine state."""
import json, subprocess
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from . import bridge_protocol as bp

@dataclass
class WorkerConfig:
    K: int = 8
    vocab: list[int] | None = None
    node_bin: str = "node"
    worker_script: str = "reference/bridge_worker.mjs"
    cwd: str | None = None
    extra_args: list[str] = field(default_factory=list)

class NodeWorker:
    def __init__(self, cfg: WorkerConfig):
        self.cfg = cfg
        self.proc = None
        self.meta = None
        self.obs_dim = None
        self.vocab_path = None

    def start(self):
        args = [self.cfg.node_bin, self.cfg.worker_script, "--serve", str(self.cfg.K)]
        if self.cfg.vocab is not None:
            self.vocab_path = Path(self.cfg.cwd or ".") / f".vocab_{id(self)}.json"
            self.vocab_path.write_text(json.dumps({"schema": "vocab.v1", "vocab": self.cfg.vocab}))
            args.extend(["--vocab", str(self.vocab_path)])
        args.extend(self.cfg.extra_args)

        self.proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, cwd=self.cfg.cwd, bufsize=0,
        )
        tag, payload = bp.read_frame(self.proc.stdout)
        if tag != bp.TAG_STATS:
            raise RuntimeError(f"expected STATS handshake, got tag {tag}")
        self.meta = json.loads(payload)
        self.obs_dim = int(self.meta["obs_dim"])
        if self.meta["K"] != self.cfg.K:
            raise RuntimeError(f"worker K={self.meta['K']} != requested {self.cfg.K}")
        return self

    def _read_until_obs(self, copy: bool = True):
        while True:
            frame = bp.read_frame(self.proc.stdout)
            if frame is None:
                raise RuntimeError("Worker process closed stdout unexpectedly")
            tag, payload = frame
            if tag == bp.TAG_OBS:
                return bp.decode_obs(payload, self.cfg.K, self.obs_dim, copy=copy)
            elif tag == bp.TAG_STATS:
                pass
            else:
                raise RuntimeError(f"expected OBS or STATS, got tag {tag}")

    def step(self, actions, copy: bool = True):
        assert len(actions) == self.cfg.K
        self.proc.stdin.write(bp.encode_step(actions))
        self.proc.stdin.flush()
        return self._read_until_obs(copy=copy)

    def reset(self, flags, payloads, copy: bool = True):
        self.proc.stdin.write(bp.encode_reset(flags, payloads))
        self.proc.stdin.flush()
        return self._read_until_obs(copy=copy)

    def close(self):
        if self.proc and self.proc.poll() is None:
            try: self.proc.stdin.close()
            except Exception: pass
            self.proc.terminate()
            try: self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired: self.proc.kill()
        if self.vocab_path and self.vocab_path.exists():
            self.vocab_path.unlink()

    def __enter__(self): return self.start()
    def __exit__(self, *a): self.close()


class NodeWorkerPool:
    """W workers, each with K envs. Total envs = W*K."""
    def __init__(self, W: int, cfg: WorkerConfig):
        self.W, self.cfg, self.workers = W, cfg, []

    def start(self):
        self.workers = [NodeWorker(self.cfg).start() for _ in range(self.W)]
        return self

    def step(self, actions_by_worker):
        assert len(actions_by_worker) == self.W
        return [w.step(a) for w, a in zip(self.workers, actions_by_worker)]

    def reset(self, flags_by_worker, payloads_by_worker):
        return [w.reset(f, p) for w, f, p in
                zip(self.workers, flags_by_worker, payloads_by_worker)]

    def close(self):
        for w in self.workers: w.close()

    def __enter__(self): return self.start()
    def __exit__(self, *a): self.close()


class NodeVecEnv:
    """Thin VecEnv adapter over NodeWorkerPool."""
    def __init__(self, pool: NodeWorkerPool, obs_dim: int, vocab: list[int] | None = None):
        self.pool = pool
        self.W = pool.W
        self.K = pool.cfg.K
        self.n = self.W * self.K
        self.obs_dim = obs_dim
        self.vocab = vocab
        
        self._ep_ret = np.zeros(self.n, dtype=np.float32)
        self._ep_len = np.zeros(self.n, dtype=np.int32)
        self._ep_start_tick = np.zeros(self.n, dtype=np.int32)
        self.obs = None

    def reset(self, descriptors: list[dict] | None = None):
        if descriptors is None:
            descriptors = []
            for i in range(self.n):
                if i % 2 == 0:
                    descriptors.append({"kind": "round_start", "map_id": 80622, "seed": 42 + i, "ego_team": 2})
                else:
                    descriptors.append({"kind": "replay", "path": "reference/replays/321746.json", "tick": 300, "ego_slot": 14})
                    
        assert len(descriptors) == self.n
        flags = [[1] * self.K for _ in range(self.W)]
        payloads = [[descriptors[w * self.K + k] for k in range(self.K)] for w in range(self.W)]
        
        results = self.pool.reset(flags, payloads)
        self.obs = np.concatenate([r["obs"] for r in results], axis=0)
        self._ep_ret.fill(0)
        self._ep_len.fill(0)
        # Track starting rl for the ledger
        for i in range(self.n):
            self._ep_start_tick[i] = results[i // self.K]["rl"][i % self.K]
        return self.obs

    def step(self, actions):
        a = np.asarray(actions, dtype=np.int32).reshape(self.W, self.K)
        results = self.pool.step([list(a[w]) for w in range(self.W)])
        
        obs = np.concatenate([r["obs"] for r in results], axis=0)
        rew = np.concatenate([r["reward"] for r in results]).astype(np.float32)
        term = np.concatenate([r["done"] for r in results]).astype(np.float32)
        trunc = np.concatenate([r["truncated"] for r in results]).astype(np.float32)
        rl = np.concatenate([r["rl"] for r in results]).astype(np.int32)

        self._ep_ret += rew
        self._ep_len += 1
        
        infos = {"ep_returns": [], "ep_lengths": [], "ledger": [],
                 "win_goal": [], "win_death": [], "win_timeout": [], "loss_death": []}
        ended = (term > 0) | (trunc > 0)
        
        for i in np.nonzero(ended)[0]:
            infos["ep_returns"].append(float(self._ep_ret[i]))
            infos["ep_lengths"].append(int(self._ep_len[i]))
            
            # Determine reason and outcome
            reason = "goal" if term[i] > 0 else "time"
            # Note: win_goal currently relies on reward heuristic (stub reward=0 for now)
            outcome = "win" if self._ep_ret[i] > 0 else "loss" if self._ep_ret[i] < 0 else "draw"
            
            if term[i] > 0 and self._ep_ret[i] >= 50:
                infos["win_goal"].append(i)
                outcome = "win_goal"
            elif term[i] > 0 and self._ep_ret[i] <= -5:
                infos["loss_death"].append(i)
                outcome = "loss_death"
            elif trunc[i] > 0:
                infos["win_timeout"].append(i)
                outcome = "timeout"

            infos["ledger"].append({
                "env_id": i,
                "reset_type": "round_start" if i % 2 == 0 else "replay",
                "start_rl": int(self._ep_start_tick[i]),
                "len": int(self._ep_len[i]),
                "reason": reason,
                "outcome": outcome
            })

            self._ep_ret[i] = 0.0
            self._ep_len[i] = 0
            self._ep_start_tick[i] = rl[i]
            
        self.obs = obs
        return obs, rew, term, trunc, infos