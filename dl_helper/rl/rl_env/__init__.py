import gymnasium as gym
try:
    import ale_py
    gym.register_envs(ale_py)
except:
    pass

from ray.tune.registry import register_env
from dl_helper.rl.rl_env.breakout_env import BreakoutEnv
from dl_helper.rl.rl_env.cartpole_env import CartPoleEnv
from dl_helper.rl.rl_env.lob_env import LOB_trade_env


class Register:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        # 注册环境
        register_env(BreakoutEnv.REG_NAME, lambda config={}: BreakoutEnv())
        # print(CartPoleEnv.__dict__)
        # register_env(CartPoleEnv.REG_NAME, lambda config={}: CartPoleEnv())
        register_env(LOB_trade_env.REG_NAME, lambda config={}: LOB_trade_env())

# 实例化
_ = Register()