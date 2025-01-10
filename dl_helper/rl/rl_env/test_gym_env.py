import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

if __name__ == "__main__":
    env = gym.make('ALE/Breakout-v5', obs_type='grayscale', frameskip=1)
    obs, info = env.reset()
    print(obs.shape)
    print(info)
