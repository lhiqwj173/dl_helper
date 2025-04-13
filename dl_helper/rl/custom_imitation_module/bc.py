import dataclasses
import itertools
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)

import gymnasium as gym
import numpy as np
import torch as th
import tqdm

from torch.optim import lr_scheduler

from stable_baselines3.common import policies, torch_layers, utils, vec_env

from imitation.algorithms import base as algo_base
from imitation.data import rollout, types
from imitation.policies import base as policy_base
from imitation.util import logger as imit_logger
from imitation.util import util
from imitation.algorithms.bc import BC, RolloutStatsComputer, BatchIteratorWithEpochEndCallback, enumerate_batches

class BCWithLRScheduler(BC):
    """支持学习率调度器的行为克隆 (Behavioral Cloning with LR Scheduler)。

    通过监督学习从观察-动作对中恢复策略，并支持动态调整学习率。
    """

    def __init__(
        self,
        *,
        observation_space: gym.Space,
        action_space: gym.Space,
        rng: np.random.Generator,
        policy: Optional[policies.ActorCriticPolicy] = None,
        demonstrations: Optional[algo_base.AnyTransitions] = None,
        batch_size: int = 32,
        minibatch_size: Optional[int] = None,
        optimizer_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        ent_weight: float = 1e-3,
        l2_weight: float = 0.0,
        device: Union[str, th.device] = "auto",
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        lr_scheduler_cls: Optional[Type[lr_scheduler._LRScheduler]] = None,     # 新增：调度器类
        lr_scheduler_kwargs: Optional[Mapping[str, Any]] = None,                # 新增：调度器参数
        lr_scheduler_step_frequency: str = 'batch',                             # 新增参数：学习率调度器更新频率
    ):
        """初始化 BCWithLRScheduler。

        Args:
            ...（原有参数保持不变，此处省略详细描述）
            lr_scheduler_cls: 使用的学习率调度器类（如 lr_scheduler.StepLR），可选。
            lr_scheduler_kwargs: 调度器的参数字典（如 {'step_size': 10, 'gamma': 0.1}），可选。
            scheduler_step_frequency: 学习率调度器的更新频率，可选值为'epoch'或'batch'
        """
        # 调用父类的初始化方法，保持原有功能
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            rng=rng,
            policy=policy,
            demonstrations=demonstrations,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            ent_weight=ent_weight,
            l2_weight=l2_weight,
            device=device,
            custom_logger=custom_logger,
        )
        # 如果指定了调度器类，则初始化调度器
        if lr_scheduler_cls is not None:
            self.lr_scheduler = lr_scheduler_cls(self.optimizer, **(lr_scheduler_kwargs or {}))
        else:
            self.lr_scheduler = None
        self.lr_scheduler_step_frequency = lr_scheduler_step_frequency

    def train(
        self,
        *,
        n_epochs: Optional[int] = None,
        n_batches: Optional[int] = None,
        on_epoch_end: Optional[Callable[[], None]] = None,
        on_batch_end: Optional[Callable[[], None]] = None,
        log_interval: int = 500,
        log_rollouts_venv: Optional[vec_env.VecEnv] = None,
        log_rollouts_n_episodes: int = 5,
        progress_bar: bool = True,
        reset_tensorboard: bool = False,
    ):
        """训练模型，支持学习率调度。

        Args:
            ...（原有参数保持不变，此处省略详细描述）
        """
        if reset_tensorboard:
            self._bc_logger.reset_tensorboard_steps()
        self._bc_logger.log_epoch(0)

        compute_rollout_stats = RolloutStatsComputer(log_rollouts_venv, log_rollouts_n_episodes)

        def _on_epoch_end(epoch_number: int):
            if tqdm_progress_bar is not None:
                total_num_epochs_str = f"of {n_epochs}" if n_epochs is not None else ""
                tqdm_progress_bar.display(f"Epoch {epoch_number} {total_num_epochs_str}", pos=1)
            self._bc_logger.log_epoch(epoch_number + 1)

            # 新增：如果有调度器，则更新学习率
            if self.lr_scheduler is not None and self.lr_scheduler_step_frequency=='epoch':
                self.lr_scheduler.step()

            if on_epoch_end is not None:
                on_epoch_end()

        mini_per_batch = self.batch_size // self.minibatch_size
        n_minibatches = n_batches * mini_per_batch if n_batches is not None else None

        assert self._demo_data_loader is not None
        demonstration_batches = BatchIteratorWithEpochEndCallback(
            self._demo_data_loader, n_epochs, n_minibatches, _on_epoch_end
        )
        batches_with_stats = enumerate_batches(demonstration_batches)
        tqdm_progress_bar: Optional[tqdm.tqdm] = None

        if progress_bar:
            batches_with_stats = tqdm.tqdm(batches_with_stats, unit="batch", total=n_minibatches)
            tqdm_progress_bar = batches_with_stats

        def process_batch():
            self.optimizer.step()

            # 新增：如果有调度器，则更新学习率
            if self.lr_scheduler is not None and self.lr_scheduler_step_frequency=='batch':
                self.lr_scheduler.step()
                
            self.optimizer.zero_grad()

            if batch_num % log_interval == 0:
                # 新增：记录学习率
                lr = self.optimizer.param_groups[0]['lr']
                self._bc_logger._logger.record(f"bc/lr", lr)

                rollout_stats = compute_rollout_stats(self.policy, self.rng)
                self._bc_logger.log_batch(
                    batch_num, minibatch_size, num_samples_so_far, training_metrics, rollout_stats
                )

            if on_batch_end is not None:
                on_batch_end()

        self.optimizer.zero_grad()
        for (batch_num, minibatch_size, num_samples_so_far), batch in batches_with_stats:
            obs_tensor = types.map_maybe_dict(
                lambda x: util.safe_to_tensor(x, device=self.policy.device),
                types.maybe_unwrap_dictobs(batch["obs"]),
            )
            acts = util.safe_to_tensor(batch["acts"], device=self.policy.device)
            training_metrics = self.loss_calculator(self.policy, obs_tensor, acts)

            loss = training_metrics.loss * minibatch_size / self.batch_size
            loss.backward()

            batch_num = batch_num * self.minibatch_size // self.batch_size
            if num_samples_so_far % self.batch_size == 0:
                process_batch()
        if num_samples_so_far % self.batch_size != 0:
            batch_num += 1
            process_batch()
