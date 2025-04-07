"""This is a simple example demonstrating how to clone the behavior of an expert.

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.util.logger import HierarchicalLogger
from imitation.util import logger as imit_logger

from py_ext.tool import log, init_logger, logger
from py_ext.lzma import decompress, compress_folder
from py_ext.alist import alist

from dl_helper.rl.rl_utils import plot_bc_train_progress

train_folder = train_title = f'bc_test'
init_logger(train_title, home=train_folder, timestamp=False)

custom_logger = imit_logger.configure(
    folder=train_folder,
    format_strs=["csv", "stdout"],
)

rng = np.random.default_rng(0)
env = make_vec_env(
    "seals:seals/CartPole-v0",
    rng=rng,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
)

def train_expert():
    # note: use `download_expert` instead to download a pretrained, competent expert
    log("Training a expert.")
    expert = PPO(
        policy=MlpPolicy,
        env=env,
        seed=0,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        n_steps=64,
    )
    expert.learn(1_000)  # Note: change this to 100_000 to train a decent expert.
    return expert


def download_expert():
    log("Downloading a pretrained expert.")
    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        env_name="seals-CartPole-v0",
        venv=env,
    )
    return expert


def sample_expert_transitions():
    # expert = train_expert()  # uncomment to train your own expert
    expert = download_expert()

    log("Sampling expert transitions.")
    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=500, min_episodes=None),
        rng=rng,
    )
    return rollout.flatten_trajectories(rollouts)


transitions = sample_expert_transitions()
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
    custom_logger=custom_logger,
)

evaluation_env = make_vec_env(
    "seals:seals/CartPole-v0",
    rng=rng,
    env_make_kwargs={"render_mode": "human"},  # for rendering
)

log("Evaluating the untrained policy.")
reward, _ = evaluate_policy(
    bc_trainer.policy,  # type: ignore[arg-type]
    evaluation_env,
    n_eval_episodes=3,
    render=True,  # comment out to speed up
)
log(f"Reward before training: {reward}")
    
log("Training a policy using Behavior Cloning")

total_epochs = 100
checkpoint_interval = 5
for i in range(total_epochs // checkpoint_interval):
    bc_trainer.policy.train()
    bc_trainer.train(n_epochs=checkpoint_interval)
    # 保存模型
    bc_trainer.policy.save(os.path.join(train_folder, f"bc_policy"))
    progress_file = os.path.join(train_folder, f"progress.csv")
    progress_file_all = os.path.join(train_folder, f"progress_all.csv")
    # 验证模型
    reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
    log(f"Reward after training: {reward_after_training}")
    if os.path.exists(progress_file_all):
        df_progress = pd.read_csv(progress_file_all)
    else:
        df_progress = pd.DataFrame()
    df_new = pd.read_csv(progress_file).iloc[len(df_progress):]
    df_progress = pd.concat([df_progress, df_new])
    df_progress.loc[df_progress['bc/epoch'] == i * checkpoint_interval-1, 'val/mean_reward'] = reward_after_training
    df_progress.ffill(inplace=True)
    df_progress.to_csv(progress_file_all, index=False)
    # 训练进度可视化
    plot_bc_train_progress(train_folder, df_progress=df_progress)
    # 打包
    zip_file = f'{train_folder}.7z'
    if os.path.exists(zip_file):
        os.remove(zip_file)
    compress_folder(train_folder, zip_file, 9, inplace=False)
    log('compress_folder done')
    # 上传更新到alist
    ALIST_UPLOAD_FOLDER = 'rl_learning_process'
    client = alist(os.environ.get('ALIST_USER'), os.environ.get('ALIST_PWD'))
    upload_folder = f'/{ALIST_UPLOAD_FOLDER}/'
    client.mkdir(upload_folder)
    client.upload(zip_file, upload_folder)
    log('upload done')