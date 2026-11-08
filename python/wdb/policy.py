import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, obs_dim=5, hidden=128):
        super().__init__()
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        # Actor heads for [move, jump, heavy]
        self.actor_heads = nn.ModuleList([nn.Linear(hidden, 3) for _ in range(3)])
        # Critic head
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        logits = [head(h) for head in self.actor_heads]
        value = self.critic(h)
        return logits, value

    @torch.no_grad()
    def act(self, x):
        logits, value = self.forward(x)
        dists = [torch.distributions.Categorical(logits=l) for l in logits]
        actions = torch.stack([d.sample() for d in dists], dim=-1)
        logp = sum(d.log_prob(actions[:, i]) for i, d in enumerate(dists))
        return actions, logp, value.squeeze(-1)

    def evaluate(self, x, actions):
        logits, value = self.forward(x)
        dists = [torch.distributions.Categorical(logits=l) for l in logits]
        logp = sum(d.log_prob(actions[:, i]) for i, d in enumerate(dists))
        entropy = sum(d.entropy() for d in dists)
        return logp, entropy, value.squeeze(-1)