import numpy as np
import math
from wdb.wdb_engine import RustEnv

class RunningMeanStd:
    """Tracks running mean/std for observation normalization."""
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        
        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count

class VecEnv:
    def __init__(self, n_envs=8, seed=0):
        self.n = n_envs
        self.envs = [RustEnv(seed=seed + i) for i in range(n_envs)]
        self.obs_dim = 17 
        self.obs = np.zeros((self.n, self.obs_dim), dtype=np.float32)
        
        # Claude's Running Normalizer (Skip Dims 6,7,16 which are unit vectors/binary)
        self.rms = RunningMeanStd(shape=(self.obs_dim,))
        self.norm_mask = np.ones(self.obs_dim, dtype=bool)
        self.norm_mask[[6, 7, 16]] = False # Do not normalize these
        
        # Claude's Tracking
        self.obs_min = np.full(self.obs_dim, np.inf, dtype=np.float32)
        self.obs_max = np.full(self.obs_dim, -np.inf, dtype=np.float32)
        self.obs_sum = np.zeros(self.obs_dim, dtype=np.float64)
        self.obs_sq_sum = np.zeros(self.obs_dim, dtype=np.float64)
        self.step_count = 0
        
        self._reset_all()

    def _raw_state_to_obs(self, state_dict):
        p1 = state_dict["p1"]
        p2 = state_dict["p2"]
        
        R = state_dict["map_radius"]
        cx = state_dict["map_center_x"]
        cy = state_dict["map_center_y"]
        
        dx = p1["x"] - cx
        dy = p1["y"] - cy
        d = math.hypot(dx, dy)
        u_x, u_y = (-dx/d, -dy/d) if d > 0.001 else (0.0, 0.0)
        margin = (R - d) / R
        
        v_rad = p1["vx"] * u_x + p1["vy"] * u_y
        v_tan = p1["vx"] * (-u_y) + p1["vy"] * u_x
        
        dp_x, dp_y = p2["x"] - p1["x"], p2["y"] - p1["y"]
        dv_x, dv_y = p2["vx"] - p1["vx"], p2["vy"] - p1["vy"]
        
        # Claude's Fix 1: Scale closing speed by 2*v_max and clip
        closing_speed = dp_x * dv_x + dp_y * dv_y
        cs_scaled = np.clip(closing_speed / 20.0, -3.0, 3.0)
        
        return np.array([
            p1["x"] / R, p1["y"] / R, 
            p1["vx"] / 10.0, p1["vy"] / 10.0, 
            p1["angle"], p1["angular_velocity"],
            u_x, u_y, margin,
            v_rad / 10.0, v_tan / 10.0,
            dp_x / R, dp_y / R, 
            dv_x / 10.0, dv_y / 10.0, 
            cs_scaled,
            state_dict["t"] / 100.0
        ], dtype=np.float32)

    def _process_obs(self, raw_obs):
        # Update normalizer with BATCH
        self.rms.update(raw_obs)
        
        # Apply normalization
        norm_obs = raw_obs.copy()
        norm_obs[:, self.norm_mask] = (raw_obs[:, self.norm_mask] - self.rms.mean[self.norm_mask]) / np.sqrt(self.rms.var[self.norm_mask] + 1e-8)
        norm_obs = np.clip(norm_obs, -5.0, 5.0)
        
        # Tracking
        self.obs_min = np.minimum(self.obs_min, raw_obs.min(axis=0))
        self.obs_max = np.maximum(self.obs_max, raw_obs.max(axis=0))
        self.obs_sum += raw_obs.sum(axis=0)
        self.obs_sq_sum += (raw_obs ** 2).sum(axis=0)
        self.step_count += self.n
        
        return norm_obs

    def _reset_all(self):
        raw_obs = np.zeros((self.n, self.obs_dim), dtype=np.float32)
        for i in range(self.n):
            state = self.envs[i].reset()
            raw_obs[i] = self._raw_state_to_obs(state)
        self.obs = self._process_obs(raw_obs)

    def step(self, actions_p1):
        rewards = np.zeros(self.n, dtype=np.float32)
        dones = np.zeros(self.n, dtype=np.float32)
        raw_obs = np.zeros((self.n, self.obs_dim), dtype=np.float32)
        
        for i in range(self.n):
            action_p2 = [np.random.randint(0, 3), np.random.randint(0, 3), np.random.randint(0, 2), np.random.randint(0, 2)]
            state, _, done = self.envs[i].step(actions_p1[i].tolist(), action_p2)
            rewards[i] = state["current_reward"]
            dones[i] = float(done)
            
            if done:
                state = self.envs[i].reset()
            
            raw_obs[i] = self._raw_state_to_obs(state)
            
        self.obs = self._process_obs(raw_obs)
        return self.obs.copy(), rewards, dones

    def print_obs_check(self):
        print("\n=== CLAUDE'S OBS NORMALIZATION CHECK ===")
        print("Dim | Min Value | Max Value | Std Dev")
        mean = self.obs_sum / self.step_count
        var = (self.obs_sq_sum / self.step_count) - (mean ** 2)
        std = np.sqrt(np.maximum(var, 0))
        
        for i in range(self.obs_dim):
            print(f" {i:2d} | {self.obs_min[i]:9.4f} | {self.obs_max[i]:9.4f} | {std[i]:9.4f}")
        print("========================================\n")