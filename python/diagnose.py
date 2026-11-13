import torch
import numpy as np
from wdb.policy import ActorCritic
from wdb.vec_env import VecEnv

def run_eval(policy, mode="deterministic", target_episodes=64):
    device = torch.device("cpu")
    venv = VecEnv(n_envs=16, seed=123)
    
    stats = {"win_goal": 0, "win_death": 0, "win_timeout": 0, "loss_death": 0}
    lengths, returns = [], []
    episodes_collected = 0
    max_steps = 10000 # Safety cap
    step_counter = 0
    
    while episodes_collected < target_episodes and step_counter < max_steps:
        obs = torch.as_tensor(venv.obs, device=device)
        
        if mode == "deterministic":
            actions, _, _ = policy.act(obs, deterministic=True)
        elif mode == "stochastic":
            actions, _, _ = policy.act(obs, deterministic=False)
        elif mode == "random":
            actions = np.array([[np.random.randint(0, 3), np.random.randint(0, 3), np.random.randint(0, 2), np.random.randint(0, 2)] for _ in range(16)])
        elif mode == "zero":
            actions = np.zeros((16, 4), dtype=np.int64)
            
        _, _, _, _, infos = venv.step(actions)
        step_counter += 1
        
        if "ep_returns" in infos:
            episodes_collected += len(infos["ep_returns"])
            returns.extend(infos["ep_returns"])
            lengths.extend(infos["ep_lengths"])
            if "win_goal" in infos: stats["win_goal"] += len(infos["win_goal"])
            if "win_death" in infos: stats["win_death"] += len(infos["win_death"])
            if "win_timeout" in infos: stats["win_timeout"] += len(infos["win_timeout"])
            if "loss_death" in infos: stats["loss_death"] += len(infos["loss_death"])

    total_wins = stats["win_goal"] + stats["win_death"] + stats["win_timeout"]
    win_rate = total_wins / len(returns) if returns else 0
    print(f"\n--- {mode.upper()} EVAL ({len(returns)} eps) ---")
    print(f"Win Rate: {win_rate*100:.1f}% | Return: {np.mean(returns):.2f} | Length: {np.mean(lengths):.1f}")
    print(f"Breakdown -> Goal: {stats['win_goal']}, Opp Death: {stats['win_death']}, Timeout: {stats['win_timeout']}, Loss: {stats['loss_death']}")

if __name__ == "__main__":
    venv = VecEnv(n_envs=16, seed=123)
    target_dim = venv.obs_dim
    checkpoint = torch.load("latest.pt", map_location="cpu", weights_only=False)
    saved_dim = checkpoint["meta"]["obs_dim"]
    
    policy = ActorCritic(obs_dim=target_dim)
    if saved_dim < target_dim:
        print(f"Migrating checkpoint from {saved_dim} dims to {target_dim} dims...")
        old_state = checkpoint["model_state"]
        new_state = policy.state_dict()
        new_state["trunk.0.weight"][:, :saved_dim] = old_state["trunk.0.weight"]
        policy.load_state_dict(new_state)
    else:
        policy.load_state_dict(checkpoint["model_state"])
        
    policy.eval()
    run_eval(policy, "random")
    run_eval(policy, "zero")
    run_eval(policy, "stochastic")
    run_eval(policy, "deterministic")