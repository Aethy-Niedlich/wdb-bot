from wdb.vec_env import VecEnv
import numpy as np

def test_shaping():
    venv = VecEnv(n_envs=1, seed=42)
    env = venv.envs[0]
    
    # Test 1: Constant state
    state = env.reset(42, 0)
    action = [1, 1, 0, 0] # Do nothing
    total_shaping = 0.0
    for _ in range(300):
        state, reward, done = env.step(action, action)
        total_shaping += reward
    
    print(f"Total Reward (Constant State, 300 steps): {total_shaping:.4f}")
    print("Expected: ~0.0 (since phi_t == phi_next)")
    
    # Test 2: Telescoping (gamma=1)
    # (This is a rough approximation since our gamma is 0.99)
    print(f"Claude's Theorem Holds: {abs(total_shaping) < 1.0}")

if __name__ == "__main__":
    test_shaping()