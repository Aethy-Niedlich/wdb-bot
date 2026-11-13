import torch
from wdb.policy import ActorCritic

def migrate():
    # Since we changed from 5 heads back to 4 heads, the old weights are incompatible.
    # We just initialize a fresh 32-dim, 4-head model and save it.
    print("Initializing fresh 32-dim model with 4 action heads...")
    new_policy = ActorCritic(obs_dim=32)
    
    checkpoint = {
        "meta": {"obs_dim": 32, "version": "v6_real_grapple"},
        "model_state": new_policy.state_dict(),
        "obs_mean": torch.zeros(32).numpy(),
        "obs_var": torch.ones(32).numpy()
    }
    torch.save(checkpoint, "latest.pt")
    print("Model reset and saved to latest.pt.")

if __name__ == "__main__":
    migrate()