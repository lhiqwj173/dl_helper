import dataclasses
import itertools
import pickle
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

import os
import gymnasium as gym
import numpy as np
import torch as th
import tqdm

from torch.optim import lr_scheduler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from stable_baselines3.common import policies, torch_layers, utils, vec_env
from stable_baselines3.common.utils import get_device

from imitation.algorithms import base as algo_base
from imitation.data import rollout, types
from imitation.policies import base as policy_base
from imitation.util import logger as imit_logger
from imitation.util import util
from imitation.algorithms.bc import BC, RolloutStatsComputer, BatchIteratorWithEpochEndCallback, enumerate_batches

class BCWithLRScheduler(BC):
    """支持学习率调度器和性能指标评估的行为克隆 (Behavioral Cloning with LR Scheduler and Metrics).

    通过监督学习从观察-动作对中恢复策略，并支持动态调整学习率和评估多种性能指标。
    """

    def __init__(
        self,
        *,
        observation_space: gym.Space,
        action_space: gym.Space,
        rng: np.random.Generator,
        policy: Optional[policies.ActorCriticPolicy] = None,
        demonstrations: Optional[algo_base.AnyTransitions] = None,
        demonstrations_val: Optional[algo_base.AnyTransitions] = None,  # 新增：验证数据集
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
        average: str = 'weighted',                                              # 新增参数：多分类指标的平均方式
    ):
        """初始化 BCWithLRScheduler。

        Args:
            ...（原有参数保持不变，此处省略详细描述）
            demonstrations_val: 用于验证的观察-动作对，可选。如果提供，将用于模型验证。
            lr_scheduler_cls: 使用的学习率调度器类（如 lr_scheduler.StepLR），可选。
            lr_scheduler_kwargs: 调度器的参数字典（如 {'step_size': 10, 'gamma': 0.1}），可选。
            lr_scheduler_step_frequency: 学习率调度器的更新频率，可选值为'epoch'或'batch'
            average: 多分类指标的平均方式，可选值为'micro', 'macro', 'weighted'等
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
        
        # 新增：指标计算的相关参数
        self.average = average
        
        # 检查是否为离散动作空间
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)
        if not self.is_discrete:
            print("警告：性能指标(accuracy, precision等)仅适用于离散动作空间")
        
        # 获取动作空间大小，用于判断是否为二分类
        if self.is_discrete:
            self.n_classes = action_space.n
            if self.n_classes == 2:
                print(f"检测到二分类问题(n_classes={self.n_classes})，将使用二分类指标计算方式")
                
        # 新增：初始化验证集数据加载器
        self._val_data_loader = None
        if demonstrations_val is not None:
            self._make_val_data_loader(demonstrations_val)

        # 训练的进度
        self.train_loop_idx = 0

    def save(self, save_folder):
        """保存当前模型的状态，包括策略参数和优化器状态。"""
        # 保存参数
        self.policy.save(os.path.join(save_folder, "policy"))

        # 保存其他状态
        other_state_dict = {
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'train_loop_idx': self.train_loop_idx,
        }
        th.save(other_state_dict, os.path.join(save_folder, "other_state.pth"))

    def load(self, load_folder):
        """加载模型的状态，包括策略参数和优化器状态。"""
        # 加载模型参数
        device = get_device()
        saved_variables = th.load(os.path.join(load_folder, "policy"), map_location=device)
        # Load weights
        self.policy.load_state_dict(saved_variables["state_dict"])
        self.policy.to(device)

        # 加载其他状态
        other_state_dict = th.load(os.path.join(load_folder, "other_state.pth"))
        self.optimizer.load_state_dict(other_state_dict['optimizer'])
        if self.lr_scheduler and other_state_dict['lr_scheduler'] is not None:
            self.lr_scheduler.load_state_dict(other_state_dict['lr_scheduler'])
        self.train_loop_idx = other_state_dict['train_loop_idx']

    def _make_val_data_loader(self, demonstrations: algo_base.AnyTransitions) -> None:
        if self._val_data_loader is not None:
            # 清理已有的数据加载器
            del self._val_data_loader
            self._val_data_loader = None

        self._val_data_loader = th.utils.data.DataLoader(
            demonstrations,
            batch_size=1024,# 验证集batch_size，加快验证速度
            shuffle=False,
        )

    def set_demonstrations_val(self, demonstrations: algo_base.AnyTransitions) -> None:
        self._make_val_data_loader(demonstrations)

    def set_demonstrations(self, demonstrations: algo_base.AnyTransitions) -> None:
        if self._demo_data_loader is not None:
            # 清理已有的数据加载器
            del self._demo_data_loader
            self._demo_data_loader = None

        self._demo_data_loader = algo_base.make_data_loader(
            demonstrations,
            self.minibatch_size,
        )

    def _get_predicted_actions(self, policy, obs_tensor):
        """获取模型预测的动作
        
        Args:
            policy: 策略模型
            obs_tensor: 观察张量
            
        Returns:
            torch.Tensor: 预测的动作索引
        """
        # 获取模型预测分布
        dist = policy.get_distribution(obs_tensor)
        
        # 根据分布类型获取预测动作
        # CategoricalDistribution可能使用probs或logits存储参数
        if hasattr(dist, 'logits'):
            # 如果有logits属性
            pred_acts = th.argmax(dist.logits, dim=1)
        elif hasattr(dist, 'probs'):
            # 如果有probs属性
            pred_acts = th.argmax(dist.probs, dim=1)
        else:
            # 如果都没有，尝试使用分布的mode方法
            pred_acts = dist.mode()
            
        return pred_acts

    def __compute_metrics(self, pred_np, true_np):
        """计算性能指标：accuracy, precision, recall, f1_score
        """
        # 计算准确率
        accuracy = accuracy_score(true_np, pred_np)
        
        # 根据是否为多分类问题选择适当的参数
        if self.n_classes > 2:
            # 多分类情况下使用average参数
            precision = precision_score(true_np, pred_np, average=self.average, zero_division=0)
            recall = recall_score(true_np, pred_np, average=self.average, zero_division=0)
            f1 = f1_score(true_np, pred_np, average=self.average, zero_division=0)
        else:
            # 二分类情况下，不使用average参数
            precision = precision_score(true_np, pred_np, zero_division=0)
            recall = recall_score(true_np, pred_np, zero_division=0)
            f1 = f1_score(true_np, pred_np, zero_division=0)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        return metrics

    def _compute_metrics(self, policy, obs_tensor, acts):
        """计算性能指标：accuracy, precision, recall, f1_score
        
        Args:
            policy: 策略模型
            obs_tensor: 观察张量
            acts: 实际动作张量
            
        Returns:
            dict: 包含各项指标的字典
        """
        # 只在离散动作空间下计算这些指标
        if not self.is_discrete:
            return {}
            
        # 将策略设置为评估模式，禁用dropout等
        policy.set_training_mode(False)
        
        with th.no_grad():
            # 获取模型预测
            # pred_acts = self._get_predicted_actions(policy, obs_tensor)
            pred_acts = policy._predict(obs_tensor)
            
            # 转为numpy计算指标
            pred_np = pred_acts.cpu().numpy()
            true_np = acts.cpu().numpy()
            
            metrics = self.__compute_metrics(pred_np, true_np)
        
        # 恢复策略为训练模式
        policy.set_training_mode(True)
            
        return metrics
        
    def validate(self) -> Dict[str, float]:
        """在验证集上评估模型性能
        
        Args:
            batch_size: 验证时使用的批次大小，若为None则使用训练批次大小
            
        Returns:
            Dict[str, float]: 包含各项指标的字典，至少包含'loss'，若为离散动作空间
                              还包含'accuracy', 'precision', 'recall', 'f1'
        """
        if self._val_data_loader is None:
            raise ValueError("未提供验证集数据，请在初始化时通过demonstrations_val参数提供验证数据")
        
        # 收集所有预测、真实标签和损失
        all_preds = []
        all_true = []
        total_loss = 0
        num_samples = 0
        
        # 将策略设置为评估模式
        self.policy.set_training_mode(False)
        
        with th.no_grad():
            for batch in self._val_data_loader:
                obs = types.map_maybe_dict(
                    lambda x: util.safe_to_tensor(x, device=self.policy.device),
                    types.maybe_unwrap_dictobs(batch["obs"]),
                )
                acts = util.safe_to_tensor(batch["acts"], device=self.policy.device)
                
                # 计算损失
                metrics = self.loss_calculator(self.policy, obs, acts)
                total_loss += metrics.loss.item() * len(acts)
                num_samples += len(acts)
                
                # 对于离散动作空间，收集预测和真实标签用于计算指标
                if self.is_discrete:
                    pred_acts = self.policy._predict(obs)
                    # pred_acts = self._get_predicted_actions(self.policy, obs)
                    all_preds.append(pred_acts.cpu().numpy())
                    all_true.append(acts.cpu().numpy())
        
        # 恢复策略为训练模式
        self.policy.set_training_mode(True)

        # FOR DEBUG
        pickle.dump((all_preds, all_true, total_loss, num_samples), open('val_data.pkl', 'wb'))
        
        # 计算平均损失
        avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
        
        # 初始化结果字典
        result = {'loss': avg_loss}
        
        # 对于离散动作空间，计算其他指标
        if self.is_discrete and all_preds:
            # 合并所有批次的结果
            all_preds = np.concatenate(all_preds)
            all_true = np.concatenate(all_true)
            
            metrics = self.__compute_metrics(all_preds, all_true)
            
            # 添加到结果字典
            result.update(metrics)
        
        return result

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
        validate_each_epoch: bool = True,       # 新增：每个epoch结束后是否验证
    ):
        """训练模型，支持学习率调度和性能指标评估。

        Args:
            ...（原有参数保持不变，此处省略详细描述）
            validate_each_epoch: 是否在每个epoch结束时在验证集上验证，默认为True
        """
        # 转为训练模式
        self.policy.train()

        if reset_tensorboard:
            self._bc_logger.reset_tensorboard_steps()
        self._bc_logger.log_epoch(0)

        compute_rollout_stats = RolloutStatsComputer(log_rollouts_venv, log_rollouts_n_episodes)

        def _on_epoch_end(epoch_number: int):
            if tqdm_progress_bar is not None:
                total_num_epochs_str = f"of {n_epochs}" if n_epochs is not None else ""
                tqdm_progress_bar.display(f"Epoch {epoch_number} {total_num_epochs_str}", pos=1)
            self._bc_logger.log_epoch(epoch_number + 1)

            # 如果有调度器，则更新学习率
            if self.lr_scheduler is not None and self.lr_scheduler_step_frequency=='epoch':
                self.lr_scheduler.step()
                # 记录当前学习率
                lr = self.optimizer.param_groups[0]['lr']
                self._bc_logger._logger.record("bc/lr", lr)

            # 统计训练集上的指标
            pred_np = np.concatenate(self.all_preds)
            true_np = np.concatenate(self.all_true)
            metrics = self.__compute_metrics(pred_np, true_np)
            for metric_name, value in metrics.items():
                self._bc_logger._logger.record(f"bc/{metric_name}", value)

            # 记录训练集损失
            self._bc_logger._logger.record("bc/train_loss", self.all_loss / (pred_np.shape[0]))

            # FOR DEBUG
            pickle.dump((self.all_preds, self.all_true, self.all_loss, pred_np.shape[0]), open('train_data.pkl', 'wb'))

            # 在每个epoch结束时在验证集上验证
            if validate_each_epoch and self._val_data_loader is not None:
                val_metrics = self.validate()
                
                # 记录验证集指标
                self._bc_logger._logger.record("bc/val_loss", val_metrics['loss'])
                if self.is_discrete:
                    self._bc_logger._logger.record("bc/val_accuracy", val_metrics.get('accuracy', 0))
                    self._bc_logger._logger.record("bc/val_precision", val_metrics.get('precision', 0))
                    self._bc_logger._logger.record("bc/val_recall", val_metrics.get('recall', 0))
                    self._bc_logger._logger.record("bc/val_f1", val_metrics.get('f1', 0))

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

            # 如果有调度器，则更新学习率
            if self.lr_scheduler is not None and self.lr_scheduler_step_frequency=='batch':
                self.lr_scheduler.step()
                
            self.optimizer.zero_grad()

            if batch_num % log_interval == 0:
                # 记录学习率
                lr = self.optimizer.param_groups[0]['lr']
                self._bc_logger._logger.record("bc/lr", lr)

                rollout_stats = compute_rollout_stats(self.policy, self.rng)
                self._bc_logger.log_batch(
                    batch_num, minibatch_size, num_samples_so_far, training_metrics, rollout_stats
                )

            if on_batch_end is not None:
                on_batch_end()

        self.optimizer.zero_grad()
        self.all_preds = []
        self.all_true = []
        self.all_loss = 0.0
        
        for (batch_num, minibatch_size, num_samples_so_far), batch in batches_with_stats:
            # 确保策略处于训练模式
            self.policy.set_training_mode(True)
            
            obs_tensor = types.map_maybe_dict(
                lambda x: util.safe_to_tensor(x, device=self.policy.device),
                types.maybe_unwrap_dictobs(batch["obs"]),
            )
            acts = util.safe_to_tensor(batch["acts"], device=self.policy.device)
            training_metrics = self.loss_calculator(self.policy, obs_tensor, acts)

            # 获取模型预测
            self.policy.set_training_mode(False)
            with th.no_grad():
                pred_acts = self.policy._predict(obs_tensor)
            self.policy.set_training_mode(True)
            # 保存当前批次数据，用于计算指标
            self.all_preds.append(pred_acts.cpu().numpy())
            self.all_true.append(acts.cpu().numpy())

            loss = training_metrics.loss * minibatch_size / self.batch_size
            loss.backward()
            self.all_loss += (training_metrics.loss * minibatch_size).item()

            batch_num = batch_num * self.minibatch_size // self.batch_size
            if num_samples_so_far % self.batch_size == 0:
                process_batch()
                
        if num_samples_so_far % self.batch_size != 0:
            batch_num += 1
            process_batch()

        # 记录loop结束
        self.train_loop_idx += 1