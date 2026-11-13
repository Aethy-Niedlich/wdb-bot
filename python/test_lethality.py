from wdb.wdb_engine import RustEnv

def test_lethality():
    with open("WDB_live_map.json", "r") as f:
        map_json = f.read()
        
    env = RustEnv(seed=42)
    state = env.reset(42, 0, map_json)
    
    # Test 1: Teleport P1 onto ball (gap = 0)
    bx, by = state["ball"]["x"], state["ball"]["y"]
    env.set_p1_pos(bx, by)
    state, _, done = env.step([1,1,0,0], [1,1,0,0])
    assert done and not state["p1"]["alive"], "Failed: Teleport onto ball did not kill."
    print("PASS: Teleport onto ball killed P1.")

    # Test 2: Boundary (gap = 0.5eps) -> Should die
    env.reset(42, 0, map_json)
    eps = 0.02
    r_sum = state["ball"]["radius"] + state["p1"]["radius"]
    env.set_p1_pos(bx - (r_sum + 0.5*eps), by)
    state, _, done = env.step([1,1,0,0], [1,1,0,0])
    assert done and not state["p1"]["alive"], "Failed: 0.5eps did not kill."
    print("PASS: 0.5eps killed P1.")

    # Test 3: Boundary (gap = 2eps) -> Should NOT die
    env.reset(42, 0, map_json)
    env.set_p1_pos(bx - (r_sum + 2*eps), by)
    state, _, done = env.step([1,1,0,0], [1,1,0,0])
    assert not done and state["p1"]["alive"], "Failed: 2eps killed P1."
    print("PASS: 2eps survived P1.")

if __name__ == "__main__":
    test_lethality()