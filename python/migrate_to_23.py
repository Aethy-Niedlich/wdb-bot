import torch
from wdb.policy import ActorCritic

def migrate():
    old_ckpt = torch.load("latest.pt", map_location="cpu", weights_only=False)
    old_state = old_ckpt["model_state"]
    
    # 14 -> 28 dims (9 ball + 5 goal)
    new_policy = ActorCritic(obs_dim=28)
    new_state = new_policy.state_dict()
    
    # CLAUDE'S FIX: Drop indices 4, 5, 16 from the old 17-dim weights
    keep_indices = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    
    old_w = old_state["trunk.0.weight"]
    new_w = new_state["trunk.0.weight"]
    
    # Extract only the 14 valid columns
    valid_old_w = old_w[:, keep_indices]
    new_w[:, :14] = valid_old_w
    new_state["trunk.0.weight"] = new_w
    
    new_policy.load_state_dict(new_state)
    
    # Extend normalizer (slice old, pad with 0 mean, 1 std)
    old_mean = old_ckpt["obs_mean"][keep_indices]
    old_var = old_ckpt["obs_var"][keep_indices]
    new_mean = torch.cat([torch.tensor(old_mean), torch.zeros(14)]) # 14 new dims
    new_var = torch.cat([torch.tensor(old_var), torch.ones(14)])
    
    checkpoint = {
        "meta": {"obs_dim": 28, "version": "v4_goal"},
        "model_state": new_policy.state_dict(),
        "obs_mean": new_mean.numpy(),
        "obs_var": new_var.numpy()
    }
    torch.save(checkpoint, "latest.pt")
    print("Migrated model to 28 dims (dropped 4, 5, 16).")

if __name__ == "__main__":
    migrate()