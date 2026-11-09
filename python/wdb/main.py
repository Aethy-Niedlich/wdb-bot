import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from wdb.policy import ActorCritic
from wdb.vec_env import VecEnv
from wdb.imitation import load_replay, train_bc
from wdb.league import League

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=str, default="", help="Path to replay JSONL file for BC")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    venv = VecEnv(n_envs=16, seed=42)
    policy = ActorCritic(obs_dim=venv.obs_dim).to(device)

    if args.replay and os.path.exists(args.replay):
        obs_np, act_np = load_replay(args.replay)
        policy = train_bc(policy, obs_np, act_np, device, epochs=5)
        torch.save(policy.state_dict(), "bc_pretrained.pt")
        print("BC Pretraining complete. Starting RL fine-tuning...\n")
    else:
        print("No replay data provided. Starting RL from scratch...\n")

    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    league = League(policy, max_past_models=10)

    rollout_steps = 128
    epochs = 4
    gamma = 0.99
    gae_lambda = 0.95
    clip_eps = 0.2
    batch_size = 256

    obs = torch.as_tensor(venv.obs, device=device)
    
    print("Starting PPO training loop...")
    for update in range(1, 21):  
        if update > 1 and update % 5 == 0:
            league.snapshot()

        obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []
        
        for _ in range(rollout_steps):
            actions_np, logp_np, value_np = policy.act(obs)
            
            # --- THE FIX: Convert to Tensors immediately ---
            actions = torch.as_tensor(actions_np, device=device)
            logp = torch.as_tensor(logp_np, device=device)
            value = torch.as_tensor(value_np, device=device)
            
            obs_np, rewards, dones = venv.step(actions_np)
            
            obs_buf.append(obs)
            act_buf.append(actions)
            logp_buf.append(logp)
            val_buf.append(value)
            rew_buf.append(torch.as_tensor(rewards, device=device, dtype=torch.float32))
            done_buf.append(torch.as_tensor(dones, device=device, dtype=torch.float32))
            
            obs = torch.as_tensor(obs_np, device=device)
            
        with torch.no_grad():
            _, last_val = policy(obs)
            last_val = last_val.squeeze(-1)
            
            adv_buf = torch.zeros((rollout_steps, venv.n), device=device)
            last_gae = 0
            for t in reversed(range(rollout_steps)):
                next_val = last_val if t == rollout_steps - 1 else val_buf[t+1]
                delta = rew_buf[t] + gamma * next_val * (1 - done_buf[t]) - val_buf[t]
                adv_buf[t] = last_gae = delta + gamma * gae_lambda * (1 - done_buf[t]) * last_gae
                
            returns = adv_buf + torch.stack(val_buf)
            
        obs_flat = torch.stack(obs_buf).reshape(-1, venv.obs_dim)
        act_flat = torch.stack(act_buf).reshape(-1, 4)
        logp_flat = torch.stack(logp_buf).reshape(-1)
        adv_flat = adv_buf.reshape(-1)
        ret_flat = returns.reshape(-1)
        
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        
        for _ in range(epochs):
            indices = np.random.permutation(rollout_steps * venv.n)
            for start in range(0, rollout_steps * venv.n, batch_size):
                mb_idx = indices[start:start+batch_size]
                
                logp, entropy, val = policy.evaluate(obs_flat[mb_idx], act_flat[mb_idx])
                ratio = torch.exp(logp - logp_flat[mb_idx])
                
                s1 = ratio * adv_flat[mb_idx]
                s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_flat[mb_idx]
                
                pi_loss = -torch.min(s1, s2).mean()
                v_loss = ((val - ret_flat[mb_idx]) ** 2).mean()
                loss = pi_loss + 0.5 * v_loss - 0.01 * entropy.mean()
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
                
        print(f"Update {update}/20 | Loss: {loss.item():.4f} | Value Loss: {v_loss.item():.4f}")

    print("Saving final model...")
    torch.save(policy.state_dict(), "latest.pt")
    print("Done!")

if __name__ == "__main__":
    main()