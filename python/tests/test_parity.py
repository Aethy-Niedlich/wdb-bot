import json
import os
import math
import pytest
from wdb._engine import RustEnv

# Resolve path to wdb-bot/reference/frames.jsonl
REF_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "reference", "frames.jsonl")

with open(REF_PATH) as f:
    REFERENCE = [json.loads(line) for line in f]

def assert_dict_close(d1, d2, tol=1e-9):
    """Compares two dicts, allowing a tiny tolerance for floats."""
    assert d1.keys() == d2.keys()
    for k in d1:
        if isinstance(d1[k], float) or isinstance(d2[k], float):
            # Check if floats are close enough to be considered identical
            assert math.isclose(d1[k], d2[k], abs_tol=tol), \
                f"Float mismatch on '{k}': expected {d2[k]}, got {d1[k]}"
        else:
            assert d1[k] == d2[k], f"Int mismatch on '{k}': expected {d2[k]}, got {d1[k]}"

@pytest.mark.parametrize("trace", REFERENCE)
def test_bit_parity(trace):
    env = RustEnv(seed=trace["seed"])
    # Convert the python dict to a JSON string for Rust
    got_str = env.step_json(json.dumps(trace["state"]), trace["action"])
    # Parse it back to a dict to compare
    got = json.loads(got_str)
    
    try:
        assert_dict_close(got, trace["next_state"])
    except AssertionError as e:
        raise AssertionError(
            f"Parity break at t={trace['state']['t']}\n"
            f"Error: {e}"
        )