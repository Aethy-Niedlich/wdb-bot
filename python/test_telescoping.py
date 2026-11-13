from wdb.wdb_engine import RustEnv
import numpy as np
GAMMA = 0.99  # must equal physics.rs GAMMA

def test_telescoping(n_eps=50):
    with open("WDB_live_map.json") as f:
        map_json = f.read()
    worst = 0.0
    for ep in range(n_eps):
        env = RustEnv(seed=ep)
        state = env.reset(ep, 0, map_json)
        phi_0, total, t = state["phi"], 0.0, 0
        for _ in range(300):
            a = [np.random.randint(0,3), np.random.randint(0,3),
                 np.random.randint(0,2), np.random.randint(0,2)]
            state, _, done = env.step(a, [1,1,0,0])
            total += (GAMMA ** t) * state["reward_shaping"]   # DISCOUNTED
            t += 1
            if done: break
        phi_T = state["phi"]                       # 0 if terminated, else Φ(s_T)
        worst = max(worst, abs(total - ((GAMMA ** t) * phi_T - phi_0)))
    print(f"max telescoping error / {n_eps} eps: {worst:.2e}")
    assert worst < 1e-6
    print("PASS")

if __name__ == "__main__":
    test_telescoping()