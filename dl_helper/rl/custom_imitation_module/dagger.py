import time, os, psutil
import pandas as pd
from collections import deque
from typing import Deque
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.dagger import (
    Optional, 
    rollout, np, DAggerTrainer, vec_env, types, policies, Tuple, List, serialize, logging, NeedsDemosException, th_data,
    Mapping, Any,
)

from dl_helper.rl.custom_imitation_module.rollout import KEYS
from dl_helper.rl.rl_utils import plot_bc_train_progress, CustomCheckpointCallback, check_gradients
from dl_helper.tool import report_memory_usage, in_windows

from py_ext.tool import log

def calculate_sample_size_bytes(sample):
    total = 0
    log("各字段内存占用（字节）：")
    for k, v in sample.items():
        size = v.nbytes
        log(f"  {k}: {size} B")
        total += size
    log(f"=> 单条样本总计: {total} B\n")
    return total

def get_max_rows(sample_size_bytes, reserved_gb=2):
    """
    参数：
        sample_size_bytes: 单条样本占用字节数
        reserved_gb: 保留的系统内存，单位GB（避免内存占满）
    返回：
        可分配的最大样本行数
    """
    available = psutil.virtual_memory().available
    reserved_bytes = reserved_gb * 1024 ** 3
    usable = max(available - reserved_bytes, 0)
    max_rows = usable // sample_size_bytes
    log(f"当前系统可用内存: {available / 1024**2:.2f} MB")
    log(f"预留内存: {reserved_gb} GB")
    log(f"可用于分配的内存: {usable / 1024**2:.2f} MB")
    log(f"最多可分配样本条数: {max_rows}\n")
    return int(max_rows)

def initialize_dataset(spec, num_rows):
    data = {}
    for key, (shape, dtype) in spec.items():
        new_shape = (num_rows, *shape[1:])  # 替换首维为行数
        if dtype == object:
            arr = np.empty(new_shape, dtype=dtype)  # object用empty
        else:
            arr = np.zeros(new_shape, dtype=dtype)
        data[key] = arr
        log(f"初始化 {key}，形状: {arr.shape}，类型: {arr.dtype}")
    return data

class SimpleDAggerTrainer(DAggerTrainer):

    MEMORY_THRESHOLD = 10  # 可用内存不足 10GB 就切换为 deque 模式

    def __init__(
        self,
        *,
        venv: vec_env.VecEnv,
        scratch_dir: types.AnyPath,
        expert_policy: policies.BasePolicy,
        rng: np.random.Generator,
        **dagger_trainer_kwargs,
    ):
        """Builds SimpleDAggerTrainer.  
        修改:   
            1. 不允许使用 初始数据 expert_trajs  
            2. 固定 beta 为0.0, 全部使用模仿策略生成轨迹  

        Args:
            venv: Vectorized training environment. Note that when the robot
                action is randomly injected (in accordance with `beta_schedule`
                argument), every individual environment will get a robot action
                simultaneously for that timestep.
            scratch_dir: Directory to use to store intermediate training
                information (e.g. for resuming training).
            expert_policy: The expert policy used to generate synthetic demonstrations.
            rng: Random state to use for the random number generator.

        Raises:
            ValueError: The observation or action space does not match between
                `venv` and `expert_policy`.
        """
        super().__init__(
            venv=venv,
            scratch_dir=scratch_dir,
            rng=rng,
            beta_schedule=lambda round_num: 0.0,# 固定使用模仿策略生成轨迹
            **dagger_trainer_kwargs,
        )
        self.expert_policy = expert_policy
        if expert_policy.observation_space != self.venv.observation_space:
            raise ValueError(
                "Mismatched observation space between expert_policy and venv",
            )
        if expert_policy.action_space != self.venv.action_space:
            raise ValueError("Mismatched action space between expert_policy and venv")
        
        # 样本数据 dict
        self.transitions_dict = None
        self.full = False   # 是否已经满了
        self.cur_idx = 0    # 可以写入的样本索引
        
    def _load_all_demos(self) -> Tuple[types.Transitions, List[int]]:
        """
        载入最新的样本
        1. 若 self.transitions_dict 未初始化，按照系统的可用内存初始化固定的大小
        2. 遍历在 self.cur_idx 处写入新的样本
        3. 若 self.cur_idx 超过了最大容量，则从头开始覆盖，设置 self.full 为 True
        """
        new_transitions_length = 0
        for round_num in range(self._last_loaded_round + 1, self.round_num + 1):
            round_dir = self._demo_dir_path_for_round(round_num)
            demo_paths = self._get_demo_paths(round_dir)

            for path in demo_paths:
                demo = serialize.load(path)[0]

                # 转为 transitions
                transitions = rollout.flatten_trajectories([demo])

                # 检查初始化
                if self.transitions_dict is None:
                    # 获取系统可用内存
                    mem = psutil.virtual_memory().available
                    # 计算单条数据的占用大小
                    single_data_dict = {}
                    for key in KEYS:
                        d = getattr(transitions, key)
                        single_data_dict[key] = [[1] + list(d.shape[1:]), d.dtype]
                    # 根据单行数据，计算最大行数
                    sample = {key: np.zeros(shape, dtype=dtype) for key, (shape, dtype) in single_data_dict.items()}
                    sample_size = calculate_sample_size_bytes(sample)
                    max_rows = get_max_rows(sample_size)
                    # 初始化数据集
                    self.transitions_dict = initialize_dataset(single_data_dict, max_rows)

                capacity = self.transitions_dict[KEYS[0]].shape[0]  # 缓冲区容量
                t_length = transitions.acts.shape[0]  # 待写入数据长度
                new_transitions_length += t_length

                log(f'capacity: {capacity}, t_length: {t_length}, cur_idx: {self.cur_idx}, full: {self.full}')
                # 写入数据
                # 情况1：写入数据比容量大 → 只保留最后 capacity 条
                if t_length > capacity:
                    offset = t_length - capacity
                    for key in KEYS:
                        data = getattr(transitions, key)
                        transitions_data = data[offset:]
                        self.transitions_dict[key][:] = transitions_data
                    self.cur_idx = 0
                    self.full = True
                else:
                    # 正常或跨越写入
                    begin = self.cur_idx
                    # 第一段：从当前 cur_idx 到缓冲区尾部
                    first_part_len = min(capacity - begin, t_length)
                    # 第二段（如果需要）：从头开始继续写
                    second_part_len = t_length - first_part_len
                    for key in KEYS:
                        data = getattr(transitions, key)
                        try:
                            self.transitions_dict[key][begin:begin + first_part_len] = data[:first_part_len]
                        except Exception as e:
                            log(f"写入数据失败，key: {key}, begin: {begin}, first_part_len: {first_part_len}")
                            log(f"data: {data.shape}")
                            log(f"transitions_dict: {self.transitions_dict[key].shape}")
                            raise e
                        if second_part_len > 0:
                            self.transitions_dict[key][0:second_part_len] = data[first_part_len:]
                            self.full = True
                    # 更新状态
                    self.cur_idx = (begin + t_length) % capacity

                # 释放内存
                del transitions
                del demo  

        log(f"Loaded {new_transitions_length} transitions")

    def extend_and_update(
        self,
        bc_train_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """Extend internal batch of data and train BC.

        Specifically, this method will load new transitions (if necessary), train
        the model for a while, and advance the round counter. If there are no fresh
        demonstrations in the demonstration directory for the current round, then
        this will raise a `NeedsDemosException` instead of training or advancing
        the round counter. In that case, the user should call
        `.create_trajectory_collector()` and use the returned
        `InteractiveTrajectoryCollector` to produce a new set of demonstrations for
        the current interaction round.

        Arguments:
            bc_train_kwargs: Keyword arguments for calling `BC.train()`. If
                the `log_rollouts_venv` key is not provided, then it is set to
                `self.venv` by default. If neither of the `n_epochs` and `n_batches`
                keys are provided, then `n_epochs` is set to `self.DEFAULT_N_EPOCHS`.

        Returns:
            New round number after advancing the round counter.
        """
        if bc_train_kwargs is None:
            bc_train_kwargs = {}
        else:
            bc_train_kwargs = dict(bc_train_kwargs)

        user_keys = bc_train_kwargs.keys()
        if "log_rollouts_venv" not in user_keys:
            bc_train_kwargs["log_rollouts_venv"] = self.venv

        if "n_epochs" not in user_keys and "n_batches" not in user_keys:
            bc_train_kwargs["n_epochs"] = self.DEFAULT_N_EPOCHS

        log("Loading demonstrations")
        demo_dir = self._demo_dir_path_for_round()
        demo_paths = self._get_demo_paths(demo_dir) if demo_dir.is_dir() else []
        if len(demo_paths) == 0:
            raise NeedsDemosException(
                f"No demos found for round {self.round_num} in dir '{demo_dir}'. "
                f"Maybe you need to collect some demos? See "
                f".create_trajectory_collector()",
            )
        # 更新载入训练数据
        if self._last_loaded_round < self.round_num:
            self._load_all_demos()

            # 将数据转换为 transitions
            transitions = types.Transitions(**self.transitions_dict)

            if len(transitions) < self.batch_size:
                raise ValueError(
                    "Not enough transitions to form a single batch: "
                    f"self.batch_size={self.batch_size} > "
                    f"len(transitions)={len(transitions)}",
                )
            
            data_loader = th_data.DataLoader(
                transitions,
                self.batch_size,
                drop_last=True,
                shuffle=True,
                collate_fn=types.transitions_collate_fn,
            )
            self.bc_trainer.set_demonstrations(data_loader)
            self._last_loaded_round = self.round_num
        log(f"Training at round {self.round_num}")
        self.bc_trainer.train(**bc_train_kwargs)
        self.round_num += 1
        log(f"New round number is {self.round_num}")

        # 清除训练数据
        self.bc_trainer.clear_demonstrations()

        # 清理数据
        del data_loader
        del transitions
        
        return self.round_num

    def train(
        self,
        total_timesteps: int,
        train_folder, 
        train_title,
        train_folder_manager,
        eval_env, 
        progress_file, 
        progress_file_all,
        *,
        rollout_round_min_episodes: int = 3,
        rollout_round_min_timesteps: int = 500,
        bc_train_kwargs: Optional[dict] = None,
    ) -> None:
        """Train the DAgger agent.

        The agent is trained in "rounds" where each round consists of a dataset
        aggregation step followed by BC update step.

        During a dataset aggregation step, `self.expert_policy` is used to perform
        rollouts in the environment but there is a `1 - beta` chance (beta is
        determined from the round number and `self.beta_schedule`) that the DAgger
        agent's action is used instead. Regardless of whether the DAgger agent's action
        is used during the rollout, the expert action and corresponding observation are
        always appended to the dataset. The number of environment steps in the
        dataset aggregation stage is determined by the `rollout_round_min*` arguments.

        During a BC update step, `BC.train()` is called to update the DAgger agent on
        all data collected so far.

        Args:
            total_timesteps: The number of timesteps to train inside the environment.
                In practice this is a lower bound, because the number of timesteps is
                rounded up to finish the minimum number of episodes or timesteps in the
                last DAgger training round, and the environment timesteps are executed
                in multiples of `self.venv.num_envs`.
            rollout_round_min_episodes: The number of episodes the must be completed
                completed before a dataset aggregation step ends.
            rollout_round_min_timesteps: The number of environment timesteps that must
                be completed before a dataset aggregation step ends. Also, that any
                round will always train for at least `self.batch_size` timesteps,
                because otherwise BC could fail to receive any batches.
            bc_train_kwargs: Keyword arguments for calling `BC.train()`. If
                the `log_rollouts_venv` key is not provided, then it is set to
                `self.venv` by default. If neither of the `n_epochs` and `n_batches`
                keys are provided, then `n_epochs` is set to `self.DEFAULT_N_EPOCHS`.
        """
        total_timestep_count = 0
        round_num = 0

        # 初始化进度数据
        if os.path.exists(progress_file_all):
            df_progress = pd.read_csv(progress_file_all)
        else:
            df_progress = pd.DataFrame()

        while total_timestep_count < total_timesteps:
            collector = self.create_trajectory_collector()
            round_episode_count = 0
            round_timestep_count = 0

            sample_until = rollout.make_sample_until(
                min_timesteps=max(rollout_round_min_timesteps, self.batch_size),
                min_episodes=rollout_round_min_episodes,
            )

            trajectories = rollout.generate_trajectories(
                policy=self.expert_policy,
                venv=collector,
                sample_until=sample_until,
                deterministic_policy=False,
                rng=collector.rng,
            )

            for traj in trajectories:
                self._logger.record_mean(
                    "dagger/mean_episode_reward",
                    np.sum(traj.rews),
                )
                round_timestep_count += len(traj)
                total_timestep_count += len(traj)

            round_episode_count += len(trajectories)

            self._logger.record("dagger/total_timesteps", total_timestep_count)
            self._logger.record("dagger/round_num", round_num)
            self._logger.record("dagger/round_episode_count", round_episode_count)
            self._logger.record("dagger/round_timestep_count", round_timestep_count)

            # `logger.dump` is called inside BC.train within the following fn call:
            # 默认会训练 self.DEFAULT_N_EPOCHS(4) 个EPOCHS
            self.extend_and_update(bc_train_kwargs)
            round_num += 1

            # # 检查梯度
            # check_gradients(self.bc_trainer)

            # 验证模型
            _t = time.time()
            eval_env.val()
            val_reward, _ = evaluate_policy(self.bc_trainer.policy, eval_env)
            eval_env.train()
            train_reward, _ = evaluate_policy(self.bc_trainer.policy, eval_env)
            log(f"train_reward: {train_reward}, val_reward: {val_reward}, 验证耗时: {time.time() - _t:.2f} 秒")

            # 合并到 progress_all.csv
            latest_ts = df_progress.iloc[-1]['timestamp'] if len(df_progress) > 0 else 0
            df_new = pd.read_csv(progress_file)
            df_new = df_new.loc[df_new['timestamp'] > latest_ts, :]
            df_new['bc/epoch'] += round_num * self.DEFAULT_N_EPOCHS
            df_new['bc/mean_reward'] = np.nan
            df_new['bc/val_mean_reward'] = np.nan
            df_new.loc[df_new.index[-1], 'bc/mean_reward'] = train_reward
            df_new.loc[df_new.index[-1], 'bc/val_mean_reward'] = val_reward
            df_progress = pd.concat([df_progress, df_new]).reset_index(drop=True)
            df_progress.ffill(inplace=True)
            df_progress.to_csv(progress_file_all, index=False)

            # 当前点是否是最优的 checkpoint
            # 使用 recall 判断
            if 'bc/recall' in list(df_progress):
                bset_recall = df_progress['bc/recall'].max()
                best_epoch = df_progress.loc[df_progress['bc/recall'] == bset_recall, 'bc/epoch'].values[0]
                is_best = df_progress.iloc[-1]['bc/epoch'] == best_epoch
            else:
                is_best = False

            # 训练进度可视化
            plot_bc_train_progress(train_folder, df_progress=df_progress, title=train_title)

            if not in_windows():
                # 保存模型
                train_folder_manager.checkpoint(self.bc_trainer, best=is_best)