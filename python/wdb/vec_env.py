import numpy as np
import math
import os
import json
from wdb.wdb_engine import RustEnv
from wdb.opponents import ScriptedChaser 

class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.norm_mask = np.ones(37, dtype=bool)
        self.norm_mask[[4, 5, 22, 27, 28, 35, 36]] = False
    def update(self, x):
        batch_mean = np.mean(x, axis=0); batch_var = np.var(x, axis=0); batch_count = x.shape[0]
        delta = batch_mean - self.mean; tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count; m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.mean = new_mean; self.var = M2 / tot_count; self.count = tot_count

class VecEnv:
    def __init__(self, n_envs=8, seed=0, stage=0):
        self.n = n_envs
        self.stage = stage 
        self.envs = [RustEnv(seed=seed + i) for i in range(n_envs)]
        
        map_path = os.path.join(os.path.dirname(__file__), "..", "..", "WDB_live_map.json")
        with open(map_path, "r") as f:
            self.map_json = f.read()
        for env in self.envs:
            env.load_map(self.map_json)
            
        self.obs_dim = 37 # 32 + 5
        self.obs = np.zeros((self.n, self.obs_dim), dtype=np.float32)
        
        self.chaser = ScriptedChaser()
        self.current_raw_states = [None] * self.n
        
        self.rms = RunningMeanStd(shape=(self.obs_dim,))
        self.norm_mask = np.ones(37, dtype=bool)
        self.norm_mask[[4, 5, 22, 27, 28, 35, 36]] = False 
        
        self.obs_min = np.full(self.obs_dim, np.inf, dtype=np.float32)
        self.obs_max = np.full(self.obs_dim, -np.inf, dtype=np.float32)
        self.obs_sum = np.zeros(self.obs_dim, dtype=np.float64)
        self.obs_sq_sum = np.zeros(self.obs_dim, dtype=np.float64)
        self.step_count = 0
        self.ep_rews = np.zeros(self.n, dtype=np.float32)
        self.ep_lens = np.zeros(self.n, dtype=np.float32)
        self.opponents = np.zeros(self.n, dtype=int) 
        self._reset_all()

    def _raw_state_to_obs(self, state_dict):
        p1 = state_dict["p1"]; p2 = state_dict["p2"]; ball = state_dict["ball"]
        R = state_dict["map_radius"]; cx = state_dict["map_center_x"]; cy = state_dict["map_center_y"]
        goal_x = state_dict["goal_x"]; goal_y = state_dict["goal_y"]
        
        dx = p1["x"] - cx; dy = p1["y"] - cy
        d = math.hypot(dx, dy)
        u_x, u_y = (-dx/d, -dy/d) if d > 0.001 else (0.0, 0.0)
        margin = (R - d) / R
        
        v_rad = p1["vx"] * u_x + p1["vy"] * u_y
        v_tan = p1["vx"] * (-u_y) + p1["vy"] * u_x
        dp_x, dp_y = p2["x"] - p1["x"], p2["y"] - p1["y"]
        dv_x, dv_y = p2["vx"] - p1["vx"], p2["vy"] - p1["vy"]
        closing_speed = dp_x * dv_x + dp_y * dv_y
        cs_scaled = np.clip(closing_speed / 20.0, -3.0, 3.0)
        
        if ball["exists"]:
            b_dp_x, b_dp_y = ball["x"] - p1["x"], ball["y"] - p1["y"]
            b_dv_x, b_dv_y = ball["vx"] - p1["vx"], ball["vy"] - p1["vy"]
            b_dist = math.hypot(b_dp_x, b_dp_y)
            surf_gap = (b_dist - ball["radius"] - p1["radius"]) / R
            closing = -(b_dp_x * b_dv_x + b_dp_y * b_dv_y) / (max(b_dist, 0.001) * 10.0)
            
            h = 25.0; g_min = 999.0; t_star = 0.0; is_grounded = p1["y"] <= 2.1
            for k in range(11):
                t_k = k * (h / 10.0) / 60.0
                bx = ball["x"] + ball["vx"] * t_k
                by = ball["y"] + ball["vy"] * t_k - 0.5 * 1.47 * (t_k**2) # C3 FIX
                px = p1["x"] + p1["vx"] * t_k
                py = p1["y"] if is_grounded else p1["y"] + p1["vy"] * t_k - 0.5 * 1.47 * (t_k**2) # C3 FIX
                gap = math.hypot(bx-px, by-py) - ball["radius"] - p1["radius"]
                if gap < g_min: g_min = gap; t_star = t_k * 60.0 / h
            
            b_exists = 1.0
            d_0 = 3.0 * ball["radius"]
            pred_gap = min(g_min / d_0, 1.0) 
            
            goal_dx = goal_x - p1["x"]; goal_dy = goal_y - p1["y"]
            ball_goal_dx = goal_x - ball["x"]; ball_goal_dy = goal_y - ball["y"]
        else:
            b_dp_x = b_dp_y = b_dv_x = b_dv_y = surf_gap = closing = pred_gap = t_star = 0.0
            b_exists = 0.0
            goal_dx = goal_x - p1["x"]; goal_dy = goal_y - p1["y"]
            ball_goal_dx = 0.0; ball_goal_dy = 0.0
            
        grapple_flag = float(p1["grapple_attached"])
        anchor_dx = (p1["anchor_x"] - p1["x"]) / R if p1["grapple_attached"] else 0.0
        anchor_dy = (p1["anchor_y"] - p1["y"]) / R if p1["grapple_attached"] else 0.0
        rope_len = p1["rope_length"] / R if p1["grapple_attached"] else 0.0
        
        g_dx = state_dict["grapple_target_dx"] / R
        g_dy = state_dict["grapple_target_dy"] / R
        g_dist = state_dict["grapple_target_dist"] / R
        g_is_ball = state_dict["grapple_is_ball"]
        g_in_range = state_dict["grapple_in_range"]

        return np.array([
            p1["x"] / R, p1["y"] / R, p1["vx"] / 10.0, p1["vy"] / 10.0, 
            u_x, u_y, margin, v_rad / 10.0, v_tan / 10.0,
            dp_x / R, dp_y / R, dv_x / 10.0, dv_y / 10.0, cs_scaled,
            b_dp_x / R, b_dp_y / R, np.clip(b_dv_x / 10.0, -3, 3), np.clip(b_dv_y / 10.0, -3, 3),
            surf_gap, closing, pred_gap, t_star, b_exists,
            goal_dx / R, goal_dy / R, ball_goal_dx / R, ball_goal_dy / R, 1.0,
            grapple_flag, anchor_dx, anchor_dy, rope_len,
            g_dx, g_dy, g_dist, g_is_ball, g_in_range
        ], dtype=np.float32)

    def _process_obs(self, raw_obs):
        self.rms.update(raw_obs)
        norm_obs = raw_obs.copy()
        norm_obs[:, self.norm_mask] = (raw_obs[:, self.norm_mask] - self.rms.mean[self.norm_mask]) / np.sqrt(self.rms.var[self.norm_mask] + 1e-8)
        norm_obs = np.clip(norm_obs, -5.0, 5.0)
        self.obs_min = np.minimum(self.obs_min, raw_obs.min(axis=0))
        self.obs_max = np.maximum(self.obs_max, raw_obs.max(axis=0))
        self.obs_sum += raw_obs.sum(axis=0)
        self.obs_sq_sum += (raw_obs ** 2).sum(axis=0)
        self.step_count += self.n
        return norm_obs

    def _reset_all(self):
        raw_obs = np.zeros((self.n, self.obs_dim), dtype=np.float32)
        for i in range(self.n):
            r = np.random.rand()
            if r < 0.33: self.opponents[i] = 0
            elif r < 0.66: self.opponents[i] = 1
            else: self.opponents[i] = 2

            state = self.envs[i].reset(np.random.randint(0, 10000), self.stage, self.map_json)
            self.current_raw_states[i] = state
            raw_obs[i] = self._raw_state_to_obs(state)
        self.obs = self._process_obs(raw_obs)

    def step(self, actions_p1):
        rewards = np.zeros(self.n, dtype=np.float32)
        terminateds = np.zeros(self.n, dtype=np.float32)
        truncateds = np.zeros(self.n, dtype=np.float32)
        raw_obs = np.zeros((self.n, self.obs_dim), dtype=np.float32)
        infos = {}
        
        for i in range(self.n):
            if self.stage == 0:
                action_p2 = [1, 1, 0, 0]
            elif self.opponents[i] == 0:
                action_p2 = [np.random.randint(0, 3), np.random.randint(0, 3), np.random.randint(0, 2), np.random.randint(0, 2)]
            elif self.opponents[i] == 1:
                action_p2 = [1, 1, 0, 0]
            else:
                action_p2 = self.chaser.get_action(self.current_raw_states[i])

            state, _, _ = self.envs[i].step(actions_p1[i].tolist(), action_p2)
            self.current_raw_states[i] = state 
            
            rewards[i] = state["current_reward"]
            terminateds[i] = float(state["terminated"])
            truncateds[i] = float(state["truncated"])
            self.ep_rews[i] += rewards[i]
            self.ep_lens[i] += 1
            
            done = state["terminated"] or state["truncated"]
            if done:
                p1_alive = state["p1"]["alive"]
                p2_alive = state["p2"]["alive"]
                ball_exists = state["ball"]["exists"]
                
                if self.stage == 0:
                    if not ball_exists: infos.setdefault("win_goal", []).append(1.0)
                    elif state["truncated"]: infos.setdefault("win_timeout", []).append(1.0)
                elif not p2_alive and p1_alive:
                    if not ball_exists: infos.setdefault("win_goal", []).append(1.0)
                    else: infos.setdefault("win_death", []).append(1.0)
                elif state["truncated"]:
                    infos.setdefault("win_timeout", []).append(1.0)
                elif not p1_alive:
                    infos.setdefault("loss_death", []).append(1.0)
                    
                infos.setdefault("ep_returns", []).append(self.ep_rews[i])
                infos.setdefault("ep_lengths", []).append(self.ep_lens[i])
                infos.setdefault("rew_shaping", []).append(state["reward_shaping"])
                infos.setdefault("rew_terminal", []).append(state["reward_terminal"])
                infos.setdefault("rew_attach", []).append(state["reward_attach"])
                infos.setdefault("phi", []).append(state["phi"])
                infos.setdefault("spawn_success", []).append(state["spawn_success"])
                self.ep_rews[i] = 0.0; self.ep_lens[i] = 0.0
                
                state = self.envs[i].reset(np.random.randint(0, 10000), self.stage, self.map_json)
                self.current_raw_states[i] = state
            
            raw_obs[i] = self._raw_state_to_obs(state)
            
        self.obs = self._process_obs(raw_obs)
        return self.obs.copy(), rewards, terminateds, truncateds, infos

    def print_obs_check(self):
        print("\n=== OBS NORMALIZATION CHECK ===")
        print("Dim | Min Value | Max Value | Std Dev")
        mean = self.obs_sum / self.step_count
        var = (self.obs_sq_sum / self.step_count) - (mean ** 2)
        std = np.sqrt(np.maximum(var, 0))
        for i in range(self.obs_dim):
            print(f" {i:2d} | {self.obs_min[i]:9.4f} | {self.obs_max[i]:9.4f} | {std[i]:9.4f}")
        print("========================================\n")