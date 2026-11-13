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

    # STAGE 0: Ball only (No P2)
    venv = VecEnv(n_envs=16, seed=42, stage=0) 
    
    # We must initialize with 32 dims since our VecEnv is 32
    policy = ActorCritic(obs_dim=venv.obs_dim).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    league = League(policy, max_past_models=10)

    rollout_steps = 128
    epochs = 4
    gamma = 0.99
    gae_lambda = 0.95
    clip_eps = 0.2
    batch_size = 256

    obs = torch.as_tensor(venv.obs, device=device)
    
    print("Starting PPO training loop (Stage 0: Ball Only)...")
    for update in range(1, 51):  
        if update > 1 and update % 5 == 0:
            league.snapshot()

        obs_buf, act_buf, logp_buf, val_buf, rew_buf, term_buf, trunc_buf = [], [], [], [], [], [], []
        ep_returns, ep_lengths, ep_wins = [], [], []
        win_goals, win_deaths, win_timeouts, losses = 0, 0, 0, 0
        
        for _ in range(rollout_steps):
            actions_np, logp_np, value_np = policy.act(obs)
            actions = torch.as_tensor(actions_np, device=device)
            logp = torch.as_tensor(logp_np, device=device)
            value = torch.as_tensor(value_np, device=device)
            
            obs_np, rewards, terminateds, truncateds, infos = venv.step(actions_np)
            
            if "ep_returns" in infos: ep_returns.extend(infos["ep_returns"])
            if "ep_lengths" in infos: ep_lengths.extend(infos["ep_lengths"])
            if "win_goal" in infos: win_goals += len(infos["win_goal"])
            if "win_death" in infos: win_deaths += len(infos["win_death"])
            if "win_timeout" in infos: win_timeouts += len(infos["win_timeout"])
            if "loss_death" in infos: losses += len(infos["loss_death"])
            
            obs_buf.append(obs)
            act_buf.append(actions)
            logp_buf.append(logp)
            val_buf.append(value)
            rew_buf.append(torch.as_tensor(rewards, device=device, dtype=torch.float32))
            term_buf.append(torch.as_tensor(terminateds, device=device, dtype=torch.float32))
            trunc_buf.append(torch.as_tensor(truncateds, device=device, dtype=torch.float32))
            
            obs = torch.as_tensor(obs_np, device=device)
            
        with torch.no_grad():
            _, last_val = policy(obs)
            last_val = last_val.squeeze(-1)
            
            adv_buf = torch.zeros((rollout_steps, venv.n), device=device)
            last_gae = 0
            for t in reversed(range(rollout_steps)):
                next_val = last_val if t == rollout_steps - 1 else val_buf[t+1]
                delta = rew_buf[t] + gamma * next_val * (1 - term_buf[t]) - val_buf[t]
                done_t = torch.clamp(term_buf[t] + trunc_buf[t], max=1.0)
                adv_buf[t] = last_gae = delta + gamma * gae_lambda * (1 - done_t) * last_gae
                
            returns = adv_buf + torch.stack(val_buf)
            
        obs_flat = torch.stack(obs_buf).reshape(-1, venv.obs_dim)
        act_flat = torch.stack(act_buf).reshape(-1, 4) # REVERTED TO 4
        logp_flat = torch.stack(logp_buf).reshape(-1)
        adv_flat = adv_buf.reshape(-1)
        ret_flat = returns.reshape(-1)
        val_flat = torch.stack(val_buf).reshape(-1)
        
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        
        var_y = torch.var(ret_flat)
        explained_var = 1 - torch.var(ret_flat - val_flat) / (var_y + 1e-8)
        
        clip_frac_sum = 0.0
        approx_kl_sum = 0.0
        head_ents = [0.0, 0.0, 0.0, 0.0]
        
        for _ in range(epochs):
            indices = np.random.permutation(rollout_steps * venv.n)
            for start in range(0, rollout_steps * venv.n, batch_size):
                mb_idx = indices[start:start+batch_size]
                
                logp, entropies, val = policy.evaluate(obs_flat[mb_idx], act_flat[mb_idx])
                ratio = torch.exp(logp - logp_flat[mb_idx])
                
                clip_frac_sum += (torch.abs(ratio - 1.0) > clip_eps).float().mean().item()
                approx_kl_sum += ((ratio - 1) - torch.log(ratio)).mean().item()
                
                for i in range(4):
                    head_ents[i] += entropies[i].mean().item()
                
                s1 = ratio * adv_flat[mb_idx]
                s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_flat[mb_idx]
                pi_loss = -torch.min(s1, s2).mean()
                v_loss = ((val - ret_flat[mb_idx]) ** 2).mean()
                total_ent = sum(entropies)
                loss = pi_loss + 0.5 * v_loss - 0.01 * total_ent.mean()
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
        
        num_batches = epochs * (rollout_steps * venv.n // batch_size)
        clip_frac = clip_frac_sum / num_batches
        approx_kl = approx_kl_sum / num_batches
        mean_ents = [h / num_batches for h in head_ents]
        
        mean_ret = np.mean(ep_returns) if ep_returns else 0.0
        mean_len = np.mean(ep_lengths) if ep_lengths else 0.0
        total_eps = win_goals + win_deaths + win_timeouts + losses
        mean_win = (win_goals + win_deaths + win_timeouts) / total_eps if total_eps > 0 else 0.0
                
        print(f"Upd {update:2d}/50 | Ret: {mean_ret:5.2f} | Len: {mean_len:5.0f} | Win%: {mean_win*100:4.1f} | KL: {approx_kl:.3f} | Clip: {clip_frac:.2f} | ExpVar: {explained_var.item():.2f}")
        print(f"     -> Term: Goal:{win_goals} Death:{win_deaths} Time:{win_timeouts} Loss:{losses}")
        print(f"     -> Ent: Move:{mean_ents[0]:.2f} Jump:{mean_ents[1]:.2f} Heavy:{mean_ents[2]:.2f} Grapple:{mean_ents[3]:.2f}")

    print("Saving final model and normalizer stats...")
    checkpoint = {
        "meta": {"obs_dim": venv.obs_dim, "version": "v6_real_grapple"},
        "model_state": policy.state_dict(),
        "obs_mean": venv.rms.mean,
        "obs_var": venv.rms.var
    }
    torch.save(checkpoint, "latest.pt")
    print("Done!")
    
    venv.print_obs_check()

if __name__ == "__main__":
    main()