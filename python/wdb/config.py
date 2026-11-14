from dataclasses import dataclass, field

@dataclass
class ObsConfig:
    # obs.v1 schema (Qwen's layout)
    SCHEMA_ID: str = "obs.v1"
    EGO_DIM: int = 17
    ENTITY_FEAT: int = 11
    N_ENTITIES: int = 2
    HAZARD_DIM: int = 2
    RAYS: int = 24
    RAY_CHANNELS: int = 3
    GRAPPLE_DIM: int = 5
    CONTEXT_DIM: int = 4
    TOTAL_DIM: int = 122

@dataclass
class TrainConfig:
    # Bridge / Env
    W: int = 2                      # Node workers
    K: int = 8                      # envs per worker (total = W*K)
    
    # PPO Hyperparams
    rollout_steps: int = 128
    gae_lambda: float = 0.97        # Qwen: lambda=0.97
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    epochs: int = 4
    batch_size: int = 256
    max_grad_norm: float = 0.5
    
    # Gamma Schedule
    gamma_entry: float = 0.995      # Start at BC->PPO entry
    gamma_stage_1v1: float = 0.9989
    gamma_stage_teams: float = 0.9993
    gamma_transition_updates: int = 10
    
    # Action / Checkpoint
    n_actions: int = 32             # Fixed capacity
    seed: int = 0
    device: str = "cuda"

@dataclass
class Config:
    obs: ObsConfig = field(default_factory=ObsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)