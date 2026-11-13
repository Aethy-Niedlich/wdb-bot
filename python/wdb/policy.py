import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dims=(3,3,2,2), hidden=512):
        super().__init__()
        self.action_dims = action_dims
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.actor_heads = nn.ModuleList([nn.Linear(hidden, d) for d in action_dims])
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = self.trunk(obs)
        logits = [head(h) for head in self.actor_heads]
        value = self.critic(h).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        logits, value = self.forward(obs)
        dists = [torch.distributions.Categorical(logits=l) for l in logits]
        actions = torch.stack(
            [d.probs.argmax(-1) if deterministic else d.sample() for d in dists], dim=-1)
        logp = sum(d.log_prob(actions[:, i]) for i, d in enumerate(dists))
        return actions.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy()

    def evaluate(self, obs, actions):
        logits, value = self.forward(obs)
        dists = [torch.distributions.Categorical(logits=l) for l in logits]
        logp = sum(d.log_prob(actions[:, i]) for i, d in enumerate(dists))
        entropies = [d.entropy() for d in dists]
        return logp, entropies, value