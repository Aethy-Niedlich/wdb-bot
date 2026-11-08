import json
import torch
import torch.nn as nn
import numpy as np
from wdb.policy import ActorCritic

def load_replay(path):
    obs_list, act_list = [], []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            obs_list.append(data["obs"])
            act_list.append(data["action"])
    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.int64)

def train_bc(policy, obs_np, act_np, device, epochs=5, batch_size=256):
    print(f"Starting Behavioral Cloning on {len(obs_np)} frames...")
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    obs = torch.as_tensor(obs_np, device=device)
    acts = torch.as_tensor(act_np, device=device)
    
    dataset_size = len(obs)
    
    for epoch in range(epochs):
        indices = np.random.permutation(dataset_size)
        epoch_loss = 0.0
        
        for start in range(0, dataset_size, batch_size):
            mb_idx = indices[start:start+batch_size]
            mb_obs = obs[mb_idx]
            mb_acts = acts[mb_idx]
            
            logits, _ = policy(mb_obs)
            
            # Factorized cross-entropy loss for the 3 action heads
            loss = 0
            for i in range(3):
                loss += nn.functional.cross_entropy(logits[i], mb_acts[:, i])
            loss = loss / 3.0
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(mb_idx)
            
        print(f"BC Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/dataset_size:.4f}")
        
    return policy