import numpy as np
from wdb._engine import RustEnv

class VecEnv:
    def __init__(self, n_envs=8, seed=0):
        self.n = n_envs
        self.envs = [RustEnv(seed=seed + i) for i in range(n_envs)]
        
        # Expanded observation space (12 variables)
        self.obs_dim = 12 
        self.obs = np.zeros((self.n, self.obs_dim), dtype=np.float32)
        self._reset_all()

    def _state_to_obs(self, state_dict):
        p1 = state_dict["p1"]
        p2 = state_dict["p2"]
        
        # We feed the AI absolute data for itself, and relative data for the opponent
        return np.array([
            p1["x"] / 10.0, 
            p1["y"] / 10.0, 
            p1["vx"] / 10.0, 
            p1["vy"] / 10.0, 
            p1["angle"],
            p1["angular_velocity"],
            (p2["x"] - p1["x"]) / 10.0, # Relative X
            (p2["y"] - p1["y"]) / 10.0, # Relative Y
            (p2["vx"] - p1["vx"]) / 10.0, # Relative VX
            (p2["vy"] - p1["vy"]) / 10.0, # Relative VY
            p2["angle"],
            state_dict["t"] / 100.0
        ], dtype=np.float32)

    def _reset_all(self):
        for i in range(self.n):
            state = self.envs[i].reset()
            self.obs[i] = self._state_to_obs(state)

    def step(self, actions_p1):
        rewards = np.zeros(self.n, dtype=np.float32)
        dones = np.zeros(self.n, dtype=np.float32)
        
        for i in range(self.n):
            # For now, Player 2 takes random actions (we will train it later)
            action_p2 = [np.random.randint(0, 3), np.random.randint(0, 3), np.random.randint(0, 2)]
            
            state, reward, done = self.envs[i].step(actions_p1[i].tolist(), action_p2)
            rewards[i] = reward
            dones[i] = float(done)
            
            if done:
                state = self.envs[i].reset()
            
            self.obs[i] = self._state_to_obs(state)
            
        return self.obs.copy(), rewards, dones