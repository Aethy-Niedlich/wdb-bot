import torch
import numpy as np
from wdb.policy import ActorCritic
from wdb.vec_env import VecEnv

def evaluate():
    device = torch.device("cpu")
    venv = VecEnv(n_envs=16, seed=123) 
    
    # The fix: explicitly set weights_only=False
    checkpoint = torch.load("latest.pt", map_location=device, weights_only=False)
    policy = ActorCritic(obs_dim=checkpoint["meta"]["obs_dim"]).to(device)
    policy.load_state_dict(checkpoint["model_state"])
    policy.eval()
    
    ep_returns, ep_lengths, ep_wins = [], [], []
    
    print("Running 64 evaluation episodes...")
    for _ in range(64): 
        obs = torch.as_tensor(venv.obs, device=device)
        actions_np, _, _ = policy.act(obs, deterministic=True)
        _, _, _, _, infos = venv.step(actions_np)
        
        if "ep_returns" in infos: ep_returns.extend(infos["ep_returns"])
        if "ep_lengths" in infos: ep_lengths.extend(infos["ep_lengths"])
        if "ep_wins" in infos: ep_wins.extend(infos["ep_wins"])
        
    print("\n=== BASELINE EVAL RESULTS ===")
    print(f"Mean Return: {np.mean(ep_returns):.2f}")
    print(f"Mean Length: {np.mean(ep_lengths):.1f}")
    print(f"Win Rate:    {np.mean(ep_wins)*100:.1f}%")
    print("=============================")

if __name__ == "__main__":
    evaluate()