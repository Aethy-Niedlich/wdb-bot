import torch
import copy
import random
from wdb.policy import ActorCritic

class League:
    def __init__(self, main_policy, max_past_models=10):
        self.main_policy = main_policy
        self.max_past_models = max_past_models
        # We store the state dictionaries (weights), not the models themselves, to save memory
        self.past_models = [] 

    def snapshot(self):
        """Saves a frozen copy of the current policy into the league."""
        # Detach and clone the weights to the CPU so they don't take up GPU memory
        state_dict = {k: v.cpu().clone() for k, v in self.main_policy.state_dict().items()}
        self.past_models.append(state_dict)
        
        if len(self.past_models) > self.max_past_models:
            self.past_models.pop(0)
            
        print(f"📸 League Snapshot taken. Pool size: {len(self.past_models)}")

    def get_opponent(self):
        """Returns a frozen policy to play against, or None for self-play."""
        if not self.past_models:
            return None
            
        # 50% chance to fight self, 50% chance to fight a past version
        if random.random() < 0.5:
            return None
            
        # Load a random past model
        past_state_dict = random.choice(self.past_models)
        opponent = ActorCritic(obs_dim=self.main_policy.trunk[0].in_features)
        opponent.load_state_dict(past_state_dict)
        opponent.eval()
        return opponent