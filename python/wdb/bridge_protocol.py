"""Length-prefixed frame codec for the Node<->Python bridge.
Frame: uint32 LE total_len, then [tag:u8][payload:bytes]. total_len = 1 + len(payload)."""
import json
import struct
import numpy as np

TAG_STEP  = 0x01
TAG_RESET = 0x02
TAG_OBS   = 0x03
TAG_STATS = 0x04

_HDR = struct.Struct('<I')

def frame(tag: int, payload: bytes) -> bytes:
    return _HDR.pack(1 + len(payload)) + bytes([tag]) + payload

def read_frame(rfile) -> tuple[int, bytes] | None:
    hdr = rfile.read(4)
    if len(hdr) < 4:
        return None
    (n,) = _HDR.unpack(hdr)
    body = rfile.read(n)
    if len(body) < n:
        raise EOFError(f"short read: {len(body)}/{n}")
    return body[0], body[1:]

def encode_step(actions: list[int]) -> bytes:
    K = len(actions)
    return frame(TAG_STEP, struct.pack(f'<{K}i', *actions))

def encode_reset(flags: list[int], payloads: list[dict]) -> bytes:
    K = len(flags)
    out = bytearray(struct.pack(f'<{K}B', *flags))
    it = iter(payloads)
    for f in flags:
        if f:
            b = json.dumps(next(it)).encode('utf-8')
            out += struct.pack('<I', len(b)) + b
    return frame(TAG_RESET, bytes(out))

def decode_obs(payload: bytes, K: int, obs_dim: int, copy: bool = True) -> dict:
    nf = K * obs_dim
    need = 4*nf + 4*K + K + K + 4*K
    if len(payload) != need:
        raise ValueError(f"OBS payload size {len(payload)} != {need}")
    off = 0
    obs = np.frombuffer(payload, '<f4', nf, off).reshape(K, obs_dim); off += 4*nf
    rew = np.frombuffer(payload, '<f4', K, off).copy();                    off += 4*K
    done = np.frombuffer(payload, 'u1', K, off).copy();                    off += K
    trunc = np.frombuffer(payload, 'u1', K, off).copy();                   off += K
    rl = np.frombuffer(payload, '<i4', K, off).copy()
    if copy:
        obs = obs.copy()
    return {"obs": obs, "reward": rew, "done": done, "truncated": trunc, "rl": rl}