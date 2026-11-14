import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 32, hidden: int = 512):
        super().__init__()
        self.n_actions = n_actions
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs, action_mask=None):
        h = self.trunk(obs)
        logits = self.actor(h)
        if action_mask is not None:
            # Mask out unused vocab slots (e.g., if vocab size < 32)
            logits = logits.masked_fill(~action_mask, float('-inf'))
        value = self.critic(h).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, obs, action_mask=None, deterministic=False):
        logits, value = self.forward(obs, action_mask)
        d = torch.distributions.Categorical(logits=logits)
        a = d.probs.argmax(-1) if deterministic else d.sample()
        return a.cpu().numpy(), d.log_prob(a).cpu().numpy(), value.cpu().numpy()

    def evaluate(self, obs, actions, action_mask=None):
        logits, value = self.forward(obs, action_mask)
        d = torch.distributions.Categorical(logits=logits)
        # Returns (logp, entropy, value) where entropy is a single tensor, not a list
        return d.log_prob(actions), d.entropy(), value