from wdb._engine import RustEnv

class CloudWDBEnv(RustEnv):
    """Same Env protocol as MockWDBEnv — swap in one line in VecEnv."""
    pass
