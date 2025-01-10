import gymnasium as gym
try:
    import ale_py
    gym.register_envs(ale_py)
except:
    pass

if __name__ == "__main__":
    env = gym.make('ALE/Breakout-v5')
    obs, info = env.reset()
    print(obs.shape)
    print(info)
