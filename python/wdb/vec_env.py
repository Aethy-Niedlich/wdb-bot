import numpy as np
import math
from wdb.wdb_engine import RustEnv

class VecEnv:
    def __init__(self, n_envs=8, seed=0):
        self.n = n_envs
        self.envs = [RustEnv(seed=seed + i) for i in range(n_envs)]
        
        # 17 Dimensions total
        self.obs_dim = 17 
        self.obs = np.zeros((self.n, self.obs_dim), dtype=np.float32)
        self._reset_all()

    def _state_to_obs(self, state_dict):
        p1 = state_dict["p1"]
        p2 = state_dict["p2"]
        
        # Claude's Variable 1: Radial/Edge Geometry
        R = 10.0
        d = math.hypot(p1["x"], p1["y"])
        u_x, u_y = (-p1["x"]/d, -p1["y"]/d) if d > 0.001 else (0.0, 0.0)
        margin = (R - d) / R
        
        # Claude's Variable 2: Velocity decomposed
        v_rad = p1["vx"] * u_x + p1["vy"] * u_y
        v_tan = p1["vx"] * (-u_y) + p1["vy"] * u_x
        
        # Claude's Variable 3: Relative opponent kinematics
        dp_x, dp_y = p2["x"] - p1["x"], p2["y"] - p1["y"]
        dv_x, dv_y = p2["vx"] - p1["vx"], p2["vy"] - p1["vy"]
        closing_speed = dp_x * dv_x + dp_y * dv_y
        
        return np.array([
            p1["x"] / 10.0, p1["y"] / 10.0, 
            p1["vx"] / 10.0, p1["vy"] / 10.0, 
            p1["angle"], p1["angular_velocity"],
            u_x, u_y, margin,
            v_rad / 10.0, v_tan / 10.0,
            dp_x / 10.0, dp_y / 10.0, 
            dv_x / 10.0, dv_y / 10.0, 
            closing_speed / 10.0,
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
            action_p2 = [np.random.randint(0, 3), np.random.randint(0, 3), np.random.randint(0, 2), np.random.randint(0, 2)]
            
            state, _, done = self.envs[i].step(actions_p1[i].tolist(), action_p2)
            # EXTRACT REWARD FROM RUST
            rewards[i] = state["current_reward"]
            dones[i] = float(done)
            
            if done:
                state = self.envs[i].reset()
            
            self.obs[i] = self._state_to_obs(state)
            
        return self.obs.copy(), rewards, dones