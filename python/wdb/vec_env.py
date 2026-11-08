import numpy as np
from wdb._engine import RustEnv

class VecEnv:
    def __init__(self, n_envs=8, seed=0):
        self.n = n_envs
        self.envs = [RustEnv(seed=seed + i) for i in range(n_envs)]
        self.obs_dim = 5
        self.obs = np.zeros((self.n, self.obs_dim), dtype=np.float32)
        self._reset_all()

    def _state_to_obs(self, state_dict):
        return np.array([
            state_dict["x"], state_dict["y"], 
            state_dict["vx"], state_dict["vy"], 
            state_dict["t"] / 100.0
        ], dtype=np.float32)

    def _reset_all(self):
        for i in range(self.n):
            state = self.envs[i].reset()
            self.obs[i] = self._state_to_obs(state)

    def step(self, actions):
        rewards = np.zeros(self.n, dtype=np.float32)
        dones = np.zeros(self.n, dtype=np.float32)
        
        for i in range(self.n):
            state, reward, done = self.envs[i].step(actions[i].tolist())
            rewards[i] = reward
            dones[i] = float(done)
            
            if done:
                state = self.envs[i].reset()
            
            self.obs[i] = self._state_to_obs(state)
            
        return self.obs.copy(), rewards, dones