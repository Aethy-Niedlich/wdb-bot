from dataclasses import dataclass, field

@dataclass
class ObsConfig:
    SELF_DIM: int = 14
    OPP_DIM: int = 19
    HISTORY_OFFSETS: tuple = (0, 1, 3, 7, 15, 30, 60)
    BUTTON_HISTORY: int = 7
    N_RAYS: int = 16
    RAY_STACK: int = 4
    META_DIM: int = 5

@dataclass
class TrainConfig:
    n_envs: int = 512
    rollout_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    epochs: int = 4
    minibatch: int = 4096
    max_grad_norm: float = 0.5
    fraction_self: float = 1/3
    fraction_past: float = 1/3
    fraction_exploiter: float = 1/3
    snapshot_every_updates: int = 25
    past_pool_size: int = 20
    bc_epochs: int = 5
    bc_lr: float = 1e-3
    seed: int = 0
    device: str = "cuda"

@dataclass
class Config:
    obs: ObsConfig = field(default_factory=ObsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    # REVERTED TO 4 ACTIONS
    action_dims: tuple = (3, 3, 2, 2) 