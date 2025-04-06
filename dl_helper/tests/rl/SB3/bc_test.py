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

def plot_bc_train_progress(train_file, train_folder):
    """
    图1 绘制 bc/loss / bc/loss平滑
    图2 绘制 bc/entropy / bc/entropy平滑
    图3 绘制 bc/neglogp / bc/neglogp平滑
    图4 绘制 bc/l2_norm / bc/l2_norm平滑
    竖向排列，对齐 x 轴
    """

    # 读取训练进度文件
    df = pd.read_csv(train_file)

    # 定义平滑函数
    def smooth_data(data, window_size=10):
        return data.rolling(window=window_size, min_periods=1).mean()

    # 创建绘图，4 个子图竖向排列，共享 x 轴
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(10, 12), sharex=True)  # 宽度 10，高度 12

    # 数据长度，用于对齐 x 轴
    data_len = len(df)

    # 图 1: bc/loss
    if 'bc/loss' in df.columns:
        axs[0].plot(df['bc/loss'], label=f'loss({df.iloc[-1]["bc/loss"]:.2f})', alpha=0.5)
        axs[0].plot(smooth_data(df['bc/loss']), label='smoothed', linewidth=2)
        axs[0].set_title('BC Loss')
        axs[0].set_ylabel('Loss')
        axs[0].legend()
        axs[0].grid(True)

    # 图 2: bc/entropy
    if 'bc/entropy' in df.columns:
        axs[1].plot(df['bc/entropy'], label=f'entropy({df.iloc[-1]["bc/entropy"]:.2f})', alpha=0.5)
        axs[1].plot(smooth_data(df['bc/entropy']), label='smoothed', linewidth=2)
        axs[1].set_title('BC Entropy')
        axs[1].set_ylabel('Entropy')
        axs[1].legend()
        axs[1].grid(True)

    # 图 3: bc/neglogp
    if 'bc/neglogp' in df.columns:
        axs[2].plot(df['bc/neglogp'], label=f'neglogp({df.iloc[-1]["bc/neglogp"]:.2f})', alpha=0.5)
        axs[2].plot(smooth_data(df['bc/neglogp']), label='smoothed', linewidth=2)
        axs[2].set_title('BC Negative Log Probability')
        axs[2].set_ylabel('Neglogp')
        axs[2].legend()
        axs[2].grid(True)

    # 图 4: bc/l2_norm
    if 'bc/l2_norm' in df.columns:
        axs[3].plot(df['bc/l2_norm'], label=f'l2_norm({df.iloc[-1]["bc/l2_norm"]:.2f})', alpha=0.5)
        axs[3].plot(smooth_data(df['bc/l2_norm']), label='smoothed', linewidth=2)
        axs[3].set_title('BC L2 Norm')
        axs[3].set_xlabel('Batch')  # x 轴表示批次
        axs[3].set_ylabel('L2 Norm')
        axs[3].legend()
        axs[3].grid(True)

    # 设置统一的 x 轴范围
    for ax in axs:
        ax.set_xlim(0, data_len - 1)

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(os.path.join(train_folder, 'training_plots.png'), dpi=300)
    plt.close()


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
        rollout.make_sample_until(min_timesteps=None, min_episodes=1),
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

total_epochs = 500
checkpoint_interval = 50
for epoch in range(total_epochs):
    bc_trainer.train(n_epochs=checkpoint_interval)
    # 保存模型
    bc_trainer.policy.save(os.path.join(train_folder, f"bc_policy"))
    # 训练进度可视化
    plot_bc_train_progress(os.path.join(train_folder, f"progress.csv"), train_folder)
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

log("Evaluating the trained policy.")
reward, _ = evaluate_policy(
    bc_trainer.policy,  # type: ignore[arg-type]
    evaluation_env,
    n_eval_episodes=3,
    render=True,  # comment out to speed up
)
log(f"Reward after training: {reward}")