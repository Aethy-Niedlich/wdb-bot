"""BC data loader for WDB. Consumes reference/demos/*.jsonl + *.meta.json.

Loader is OBS-BUILDER-AGNOSTIC: yields (Clip, slot, Frame) tuples. Materialize
with to_samples(obs_builder, obs_dim), where obs_builder is `wdb.obs.build_obs`
(once C2 passes) or a stub for plumbing tests.

Schema (verified against reference/harvest.mjs):
  frame  = {t, rl, obs{x,y,vx,vy,a,av,a1a,lhid,lht,swing}, act|null,
            ball|null, scores, capZones, others[]}
  meta   = {tag, mid, rl_start, ticks, stop, decision_tick, round_length, frames{}}

Action mask bit layout (matches engine `inputs` byte):
  1=left, 2=right, 4=up, 8=down, 16=action(heavy), 32=action2(grapple)
"""
from __future__ import annotations
import json, logging
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter
from typing import Iterator, Callable

import numpy as np

log = logging.getLogger(__name__)

# ---------- 6-bit action mask ----------
_BIT = {"left": 1, "right": 2, "up": 4, "down": 8, "action": 16, "action2": 32}
BIT_ORDER = ("left", "right", "up", "down", "action", "action2")

def act_to_mask(act: dict | None) -> int | None:
    if act is None:
        return None
    m = 0
    for k, b in _BIT.items():
        if act.get(k):
            m |= b
    return m

def mask_to_act(m: int) -> dict:
    return {k: bool(m & b) for k, b in _BIT.items()}

# ---------- Frame / Clip ----------
@dataclass
class Frame:
    t: int
    rl: int
    obs_raw: dict
    act_mask: int | None
    ball: dict | None
    scores: list
    capzones: list
    others: list
    slot: int
    team: int | None = None    # NEW: filled by harvest.mjs patch; None if old data

@dataclass
class Clip:
    tag: str
    mid: int
    rl_start: int
    ticks: int
    stop: str
    decision_tick: int | None
    round_length: int | None
    slots: list[int]
    frames: dict[int, list[Frame]] = field(default_factory=dict)
    source: str = "reference"

    def labeled_count(self) -> int:
        return sum(1 for s in self.slots for f in self.frames[s]
                   if f.act_mask is not None)

# ---------- Dataset ----------
_COUNTDOWN_TICKS = 91   # Qwen brief; countdown is the first ~91 round-ticks
_OWN_TAG_PREFIXES = ("own-", "own_")

def _infer_source(tag: str) -> str:
    for p in _OWN_TAG_PREFIXES:
        if tag.startswith(p):
            return "own"
    return "reference"

def _load_jsonl(path: Path, slot: int) -> list[Frame]:
    frames = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            frames.append(Frame(
                t=d["t"], rl=d["rl"],
                obs_raw=d["obs"],
                act_mask=act_to_mask(d.get("act")),
                ball=d.get("ball"),
                scores=d.get("scores", []),
                capzones=d.get("capZones", []),
                others=d.get("others", []),
                slot=slot,
                team=d.get("team"),           # present only after harvest patch
            ))
    return frames

class BCDataset:
    """Loader + filter + vocabulary freezer for BC training.

    Filters applied at iter time:
      - drop unlabeled frames (act == null)
      - drop countdown frames (rl < countdown_ticks; proxy for missing ftu)
      - drop post-decision frames (t > decision_tick or rl >= round_length)
      - drop out-of-vocab masks (post-freeze only; log & count, never expand)
    """

    def __init__(
        self,
        demo_dir: str | Path,
        *,
        own_weight: float = 0.3,
        countdown_ticks: int = _COUNTDOWN_TICKS,
        mask_countdown: bool = True,
        mask_post_decision: bool = True,
        drop_unlabeled: bool = True,
        vocab: list[int] | None = None,
    ):
        self.demo_dir = Path(demo_dir)
        self.own_weight = own_weight
        self.countdown_ticks = countdown_ticks
        self.mask_countdown = mask_countdown
        self.mask_post_decision = mask_post_decision
        self.drop_unlabeled = drop_unlabeled
        self.vocab = list(vocab) if vocab is not None else None
        self._vocab_set: set[int] | None = None
        self._vocab_index: dict[int, int] = {}
        self.clips: list[Clip] = []
        self._drops = {"unlabeled": 0, "countdown": 0,
                       "post_decision": 0, "out_of_vocab": 0}

    # ---- load ----
    def load(self) -> "BCDataset":
        if not self.demo_dir.exists():
            raise FileNotFoundError(self.demo_dir)
        metas = sorted(self.demo_dir.glob("*.meta.json"))
        if not metas:
            raise FileNotFoundError(f"no *.meta.json in {self.demo_dir}")
        for mp in metas:
            meta = json.loads(mp.read_text())
            tag = meta.get("tag", mp.name.replace(".meta.json", ""))
            slots = [int(s) for s in meta.get("frames", {}).keys()]
            clip = Clip(
                tag=tag,
                mid=int(meta["mid"]) if meta.get("mid") is not None else -1,
                rl_start=int(meta["rl_start"]),
                ticks=int(meta["ticks"]),
                stop=str(meta.get("stop", "")),
                decision_tick=(int(meta["decision_tick"])
                               if meta.get("decision_tick") is not None else None),
                round_length=(int(meta["round_length"])
                              if meta.get("round_length") is not None else None),
                slots=slots,
                source=meta.get("source") or _infer_source(tag),
            )
            for slot in slots:
                p = self.demo_dir / f"{tag}_p{slot}.jsonl"
                if not p.exists():
                    log.warning("missing jsonl: %s", p)
                    continue
                clip.frames[slot] = _load_jsonl(p, slot)
            clip.slots = [s for s in clip.slots if s in clip.frames]
            # Sanity: rl at t=0 should equal rl_start (skip if absent)
            for s in clip.slots:
                if clip.frames[s]:
                    f0 = clip.frames[s][0]
                    if f0.rl != clip.rl_start:
                        log.warning("%s slot %d: frame rl(%d) != meta rl_start(%d)",
                                    tag, s, f0.rl, clip.rl_start)
                    break
            self.clips.append(clip)

        n_frames = sum(len(v) for c in self.clips for v in c.frames.values())
        n_lab = sum(c.labeled_count() for c in self.clips)
        log.info("loaded %d clips | %d frames | %d labeled",
                 len(self.clips), n_frames, n_lab)

        if self.vocab is None:
            self._freeze_vocab()
        else:
            self._vocab_set = set(self.vocab)
            self._vocab_index = {m: i for i, m in enumerate(self.vocab)}
        return self

    # ---- vocab ----
    def _freeze_vocab(self, cap: int = 32) -> list[int]:
        ctr: Counter[int] = Counter()
        for c in self.clips:
            for s in c.slots:
                for f in c.frames[s]:
                    if f.act_mask is not None:
                        ctr[f.act_mask] += 1
        ctr[0] = max(ctr.get(0, 0), 1)   # neutral always present
        vocab = sorted({m for m, _ in ctr.most_common(cap)})
        if 0 not in vocab:
            vocab.append(0)
        vocab = sorted(set(vocab))
        if len(vocab) > cap:
            log.warning("vocab exceeds cap %d after neutral; truncating", cap)
            vocab = vocab[:cap]
        self.vocab = vocab
        self._vocab_set = set(vocab)
        self._vocab_index = {m: i for i, m in enumerate(vocab)}
        log.info("froze vocab: %d masks %s", len(vocab), vocab)
        return vocab

    def save_vocab(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(
            {"schema": "vocab.v1", "vocab": self.vocab}, indent=1))

    @classmethod
    def load_vocab(cls, path: str | Path) -> list[int]:
        d = json.loads(Path(path).read_text())
        if d.get("schema") != "vocab.v1":
            log.warning("vocab schema mismatch: %s", d.get("schema"))
        return list(d["vocab"])

    # ---- filters ----
    def _keep(self, clip: Clip, f: Frame) -> bool:
        if self.drop_unlabeled and f.act_mask is None:
            self._drops["unlabeled"] += 1
            return False
        if self.mask_countdown and f.rl < self.countdown_ticks:
            self._drops["countdown"] += 1
            return False
        if self.mask_post_decision:
            if clip.decision_tick is not None and f.t > clip.decision_tick:
                self._drops["post_decision"] += 1
                return False
            if clip.round_length is not None and f.rl >= clip.round_length:
                self._drops["post_decision"] += 1
                return False
        if self._vocab_set is not None and f.act_mask is not None \
                and f.act_mask not in self._vocab_set:
            self._drops["out_of_vocab"] += 1
            return False
        return True

    def iter_samples(self) -> Iterator[tuple[Clip, int, Frame]]:
        for c in self.clips:
            for s in c.slots:
                for f in c.frames[s]:
                    if self._keep(c, f):
                        yield c, s, f

    # ---- materialization ----
    def to_samples(
        self,
        obs_builder: Callable[[Clip, int, Frame], np.ndarray],
        obs_dim: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """X (n, obs_dim) float32, y (n,) int64 vocab id, w (n,) float32 weight.
        obs_builder(clip, slot, frame) -> float32[obs_dim].
        Use wdb.obs.build_obs once C2 passes; a stub is OK for plumbing tests."""
        X, y, w = [], [], []
        for c, s, f in self.iter_samples():
            x = np.asarray(obs_builder(c, s, f), dtype=np.float32)
            if x.shape != (obs_dim,):
                raise ValueError(f"obs_builder returned {x.shape}, expected ({obs_dim},)")
            X.append(x)
            y.append(self._vocab_index[f.act_mask])
            w.append(self.own_weight if c.source == "own" else 1.0)
        if not X:
            return (np.zeros((0, obs_dim), np.float32),
                    np.zeros((0,), np.int64),
                    np.zeros((0,), np.float32))
        return (np.stack(X), np.asarray(y, np.int64),
                np.asarray(w, np.float32))

    # ---- leakage gate (Qwen) ----
    def leakage_check(
        self,
        obs_builder: Callable[[Clip, int, Frame], np.ndarray],
        obs_dim: int,
        thresh: float = 0.98,
    ) -> dict:
        """Single-feature threshold classifier per input bit. Flag accuracy > thresh.
        Meaningful only with the REAL obs_builder; stub obs (all zeros) will
        trivially return empty flags — that is not a pass."""
        X, y, _ = self.to_samples(obs_builder, obs_dim)
        if X.shape[0] == 0:
            return {"flagged": [], "n_samples": 0, "note": "no samples"}
        masks = np.array([self.vocab[int(i)] for i in y], dtype=np.int64)
        bits = np.stack([(masks >> k) & 1 for k in range(6)], axis=1)   # (n,6)
        flagged = []
        for fi in range(obs_dim):
            col = X[:, fi]
            med = np.median(col)
            pred = (col > med).astype(np.int64)
            for bi in range(6):
                tgt = bits[:, bi]
                if tgt.min() == tgt.max():
                    continue
                acc = float((pred == tgt).mean())
                if acc >= thresh:
                    flagged.append({"feature": fi, "bit": BIT_ORDER[bi],
                                    "accuracy": acc})
        return {"flagged": flagged, "n_samples": int(X.shape[0]),
                "threshold": thresh}

    # ---- introspection ----
    def stats(self) -> dict:
        usage = Counter()
        for c in self.clips:
            for s in c.slots:
                for f in c.frames[s]:
                    if f.act_mask is not None:
                        usage[f.act_mask] += 1
        return {
            "n_clips": len(self.clips),
            "vocab": list(self.vocab or []),
            "vocab_usage": dict(usage),
            "drops": dict(self._drops),
            "clips": [{"tag": c.tag, "mid": c.mid, "slots": c.slots,
                       "source": c.source, "rl_start": c.rl_start,
                       "decision_tick": c.decision_tick,
                       "frames": {s: len(c.frames[s]) for s in c.slots}}
                      for c in self.clips],
        }