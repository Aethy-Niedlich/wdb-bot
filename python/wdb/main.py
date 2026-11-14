import argparse
import os
import numpy as np
import torch
import torch.nn as nn

from wdb.config import Config
from wdb.policy import ActorCritic
from wdb.node_env import NodeWorkerPool, NodeVecEnv, WorkerConfig
from wdb.league import League
from wdb.data import BCDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=str, default="", help="Path to BC demos dir/jsonl")
    parser.add_argument("--updates", type=int, default=1, help="Number of PPO updates")
    args = parser.parse_args()

    cfg = Config()
    tc = cfg.train
    device = torch.device(tc.device if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} | W={tc.W} K={tc.K} n={tc.W*tc.K}")

    if args.replay:
        if args.replay.endswith('.json') and 'demos' not in args.replay:
            raise ValueError("--replay expects a demos JSONL or directory, not a raw replay JSON.")
        ds = BCDataset(args.replay).load()
        print(f"[BC] Loaded {len(ds.clips)} clips. Vocab: {ds.vocab}")

    vocab = list(range(32))
    pool = NodeWorkerPool(
        W=tc.W,
        cfg=WorkerConfig(K=tc.K, vocab=vocab, cwd=".")
    ).start()
    
    obs_dim = pool.workers[0].obs_dim
    print(f"[HANDSHAKE] Node worker PID(s): {[w.proc.pid for w in pool.workers]}")
    print(f"[HANDSHAKE] obs_dim={obs_dim}, K={tc.K}")

    venv = NodeVecEnv(pool, obs_dim=obs_dim, vocab=vocab)
    
    descs = []
    for i in range(venv.n):
        if i % 2 == 0:
            descs.append({"kind": "round_start", "map_id": 80622, "seed": 42 + i, "ego_team": 2})
        else:
            descs.append({"kind": "replay", "path": "reference/replays/321746.json", "tick": 300, "ego_slot": 14})
            
    obs = venv.reset(descs)
    print(f"[HANDSHAKE] First obs shape: {obs.shape}")

    policy = ActorCritic(obs_dim=obs_dim, n_actions=tc.n_actions).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=tc.lr)
    
    action_mask = torch.zeros(tc.n_actions, dtype=torch.bool, device=device)
    action_mask[:len(vocab)] = True

    obs = torch.as_tensor(obs, device=device)
    
    gamma = tc.gamma_entry
    
    print(f"Starting PPO smoke test ({args.updates} updates)...")
    for update in range(1, args.updates + 1):
        obs_buf, act_buf, logp_buf, val_buf, rew_buf, term_buf, trunc_buf = [], [], [], [], [], [], []
        ep_returns, ep_lengths = [], []
        ledger = []
        win_goals = win_deaths = win_timeouts = losses = 0
        total_terms = 0
        switch_count = 0
        prev_actions = np.zeros(venv.n, dtype=np.int32)

        for _ in range(tc.rollout_steps):
            actions_np, logp_np, value_np = policy.act(obs, action_mask=action_mask)
            actions = torch.as_tensor(actions_np, device=device, dtype=torch.long)
            logp = torch.as_tensor(logp_np, device=device, dtype=torch.float32)
            value = torch.as_tensor(value_np, device=device, dtype=torch.float32)

            switch_count += (actions_np != prev_actions).sum()
            prev_actions = actions_np

            obs_np, rewards, terminateds, truncateds, infos = venv.step(actions_np)
            
            ep_returns.extend(infos["ep_returns"])
            ep_lengths.extend(infos["ep_lengths"])
            ledger.extend(infos["ledger"])
            win_goals += len(infos["win_goal"])
            win_deaths += len(infos["win_death"])
            win_timeouts += len(infos["win_timeout"])
            losses += len(infos["loss_death"])
            total_terms += terminateds.sum().item()

            obs_buf.append(obs)
            act_buf.append(actions)
            logp_buf.append(logp)
            val_buf.append(value)
            rew_buf.append(torch.as_tensor(rewards, device=device, dtype=torch.float32))
            term_buf.append(torch.as_tensor(terminateds, device=device, dtype=torch.float32))
            trunc_buf.append(torch.as_tensor(truncateds, device=device, dtype=torch.float32))

            obs = torch.as_tensor(obs_np, device=device)

        if len(ep_lengths) == 0 or (sum(ep_lengths) == 0 and total_terms == 0):
            raise RuntimeError(
                "DEAD-ENV GUARD TRIGGERED: Zero episode lengths and zero terminations in this rollout."
            )

        with torch.no_grad():
            _, last_val = policy(obs, action_mask=action_mask)
            adv_buf = torch.zeros((tc.rollout_steps, venv.n), device=device)
            last_gae = 0
            for t in reversed(range(tc.rollout_steps)):
                next_val = last_val if t == tc.rollout_steps - 1 else val_buf[t + 1]
                delta = rew_buf[t] + gamma * next_val * (1 - term_buf[t]) - val_buf[t]
                done_t = torch.clamp(term_buf[t] + trunc_buf[t], max=1.0)
                adv_buf[t] = last_gae = delta + gamma * tc.gae_lambda * (1 - done_t) * last_gae
            returns = adv_buf + torch.stack(val_buf)

        obs_flat = torch.stack(obs_buf).reshape(-1, obs_dim)
        act_flat = torch.stack(act_buf).reshape(-1)
        logp_flat = torch.stack(logp_buf).reshape(-1)
        adv_flat = adv_buf.reshape(-1)
        ret_flat = returns.reshape(-1)
        val_flat = torch.stack(val_buf).reshape(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        var_y = torch.var(ret_flat)
        explained_var = 1 - torch.var(ret_flat - val_flat) / (var_y + 1e-8)

        clip_frac_sum = 0.0
        approx_kl_sum = 0.0
        joint_ent_sum = 0.0

        for _ in range(tc.epochs):
            indices = np.random.permutation(tc.rollout_steps * venv.n)
            for start in range(0, tc.rollout_steps * venv.n, tc.batch_size):
                mb = indices[start : start + tc.batch_size]
                logp, ent, val = policy.evaluate(obs_flat[mb], act_flat[mb], action_mask=action_mask)
                ratio = torch.exp(logp - logp_flat[mb])
                clip_frac_sum += (torch.abs(ratio - 1.0) > tc.clip_eps).float().mean().item()
                approx_kl_sum += ((ratio - 1) - torch.log(ratio)).mean().item()
                joint_ent_sum += ent.mean().item()

                s1 = ratio * adv_flat[mb]
                s2 = torch.clamp(ratio, 1 - tc.clip_eps, 1 + tc.clip_eps) * adv_flat[mb]
                pi_loss = -torch.min(s1, s2).mean()
                v_loss = ((val - ret_flat[mb]) ** 2).mean()
                loss = pi_loss + tc.value_coef * v_loss - tc.entropy_coef * ent.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), tc.max_grad_norm)
                optimizer.step()

        num_batches = tc.epochs * (tc.rollout_steps * venv.n // tc.batch_size)
        clip_frac = clip_frac_sum / num_batches
        approx_kl = approx_kl_sum / num_batches
        mean_ent = joint_ent_sum / num_batches
        switch_rate = switch_count / (tc.rollout_steps * venv.n)

        mean_ret = float(np.mean(ep_returns)) if ep_returns else 0.0
        mean_len = float(np.mean(ep_lengths)) if ep_lengths else 0.0
        total_eps = win_goals + win_deaths + win_timeouts + losses
        
        # FIXED: Win% is only actual wins (Goals), not timeouts
        mean_win = (win_goals / total_eps) if total_eps > 0 else 0.0

        print(f"Upd {update:2d}/{args.updates} | Ret: {mean_ret:6.2f} | Len: {mean_len:5.1f} "
              f"| Win%: {mean_win*100:4.1f} | KL: {approx_kl:.3f} "
              f"| ExpVar: {explained_var.item():.2f} | Ent: {mean_ent:.3f} "
              f"| Switch: {switch_rate:.3f}")
        print(f"     -> Term: Goal:{win_goals} Death:{win_deaths} Time:{win_timeouts} Loss:{losses}")
        print(f"     -> Ledger (first 5):")
        for entry in ledger[:5]:
            print(f"        {entry}")

    print("Saving checkpoint...")
    checkpoint = {
        "meta": {
            "obs_dim": obs_dim,
            "obs_schema": cfg.obs.SCHEMA_ID,
            "n_actions": tc.n_actions,
            "vocab": list(vocab),
            "vocab_version": 1,
            "gamma": gamma,
            "version": "v7_node_bridge",
        },
        "model_state": policy.state_dict(),
    }
    torch.save(checkpoint, "latest.pt")
    print("Done!")
    pool.close()

if __name__ == "__main__":
    main()