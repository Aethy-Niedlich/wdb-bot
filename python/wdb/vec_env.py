import numpy as np
import math
from wdb.wdb_engine import RustEnv

class VecEnv:
    def __init__(self, n_envs=8, seed=0):
        self.n = n_envs
        self.envs = [RustEnv(seed=seed + i) for i in range(n_envs)]
        self.obs_dim = 17 
        self.obs = np.zeros((self.n, self.obs_dim), dtype=np.float32)
        
        # Claude's Obs Normalization Check
        self.obs_min = np.full(self.obs_dim, np.inf, dtype=np.float32)
        self.obs_max = np.full(self.obs_dim, -np.inf, dtype=np.float32)
        
        self._reset_all()

    def _state_to_obs(self, state_dict):
        p1 = state_dict["p1"]
        p2 = state_dict["p2"]
        
        # DYNAMIC map data from Rust
        R = state_dict["map_radius"]
        cx = state_dict["map_center_x"]
        cy = state_dict["map_center_y"]
        
        # Variable 1: Radial/Edge Geometry (Dynamic Center)
        dx = p1["x"] - cx
        dy = p1["y"] - cy
        d = math.hypot(dx, dy)
        u_x, u_y = (-dx/d, -dy/d) if d > 0.001 else (0.0, 0.0)
        margin = (R - d) / R
        
        # Variable 2: Velocity decomposed
        v_rad = p1["vx"] * u_x + p1["vy"] * u_y
        v_tan = p1["vx"] * (-u_y) + p1["vy"] * u_x
        
        # Variable 3: Relative opponent kinematics
        dp_x, dp_y = p2["x"] - p1["x"], p2["y"] - p1["y"]
        dv_x, dv_y = p2["vx"] - p1["vx"], p2["vy"] - p1["vy"]
        closing_speed = dp_x * dv_x + dp_y * dv_y
        
        obs = np.array([
            p1["x"] / R, p1["y"] / R, 
            p1["vx"] / 10.0, p1["vy"] / 10.0, 
            p1["angle"], p1["angular_velocity"],
            u_x, u_y, margin,
            v_rad / 10.0, v_tan / 10.0,
            dp_x / R, dp_y / R, 
            dv_x / 10.0, dv_y / 10.0, 
            closing_speed / 10.0,
            state_dict["t"] / 100.0
        ], dtype=np.float32)
        
        # Claude's Obs Normalization Check
        self.obs_min = np.minimum(self.obs_min, obs)
        self.obs_max = np.maximum(self.obs_max, obs)
        return obs
        
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
            rewards[i] = state["current_reward"]
            dones[i] = float(done)
            
            if done:
                state = self.envs[i].reset()
            
            self.obs[i] = self._state_to_obs(state)
            
        return self.obs.copy(), rewards, dones

    def print_obs_check(self):
        print("\n=== CLAUDE'S OBS NORMALIZATION CHECK ===")
        print("Dim | Min Value | Max Value")
        for i in range(self.obs_dim):
            print(f" {i:2d} | {self.obs_min[i]:9.4f} | {self.obs_max[i]:9.4f}")
        print("========================================\n")