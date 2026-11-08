"""
Parity gate: for every golden trace, the Rust engine must produce a
bit-identical next_state. No tolerance. No approximation.
"""
import json
import pytest
from wdb._engine import RustEnv

REFERENCE = json.load(open("reference/frames.jsonl"))

@pytest.mark.parametrize("trace", REFERENCE)
def test_bit_parity(trace):
    env = RustEnv(seed=trace["seed"])
    got = env.step_json(trace["state"], trace["action"])
    assert got == trace["next_state"], (
        f"Parity break at t={trace['state']['t']}\n"
        f"expected: {trace['next_state']}\n"
        f"got:      {got}"
    )
