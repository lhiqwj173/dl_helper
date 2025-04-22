import time, os, psutil
import pandas as pd
from collections import deque
from typing import Deque
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.dagger import Optional, rollout, np, DAggerTrainer, vec_env, types, policies, Tuple, List, serialize, logging

from dl_helper.rl.rl_utils import plot_bc_train_progress, CustomCheckpointCallback, check_gradients
from dl_helper.tool import report_memory_usage, in_windows

from py_ext.tool import log

class SimpleDAggerTrainer(DAggerTrainer):

    MEMORY_THRESHOLD = 0.85  # 内存超过 85% 就切换为受控模式

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
        
    def _load_all_demos(self) -> Tuple[types.Transitions, List[int]]:
        """
        使用 deque 控制内存使用
        1. 如果内存超过阈值，则使用 deque 替换 list
        2. deque 的最大长度为当前所有 demo 的总步数
        3. 每次添加 demo 时，检查当前总步数是否超过最大步数限制
        4. 如果超过，则不断移除旧数据，直到总步数不超过最大步数限制
        """
        num_demos_by_round = []

        for round_num in range(self._last_loaded_round + 1, self.round_num + 1):
            round_dir = self._demo_dir_path_for_round(round_num)
            demo_paths = self._get_demo_paths(round_dir)

            for path in demo_paths:
                demo = serialize.load(path)[0]  # 假设 demo 是一条 trajectory，list of transitions

                if isinstance(self._all_demos, list):
                    self._all_demos.append(demo)

                    # 检查内存使用
                    mem_used_ratio = psutil.virtual_memory().percent / 100.0
                    if mem_used_ratio > self.MEMORY_THRESHOLD:
                        print(f"[Memory Warning] RAM usage {mem_used_ratio*100:.1f}%, switching to deque...")

                        # 计算总步数
                        # 作为最大步数限制
                        total_steps = sum(len(d) for d in self._all_demos)
                        self._max_allowed_steps = total_steps

                        # 用 deque 替换 list，限制总步数
                        new_buffer = deque()
                        step_count = 0
                        for d in reversed(self._all_demos):  # 从最新数据向前填充
                            step_count += len(d)
                            new_buffer.appendleft(d)

                        self._all_demos.clear()
                        self._all_demos = new_buffer
                        self._total_steps = step_count
                else:
                    # self._all_demos 已经是 deque
                    self._all_demos.append(demo)
                    self._total_steps += len(demo)

                    # 控制容量：如果超出当前最大步数，则不断移除旧数据
                    while self._total_steps > self._max_allowed_steps:
                        removed = self._all_demos.popleft()
                        self._total_steps -= len(removed)

            num_demos_by_round.append(len(demo_paths))

        logging.info(f"Loaded {len(self._all_demos)} total demos")
        demo_transitions = rollout.flatten_trajectories(self._all_demos)
        return demo_transitions, num_demos_by_round

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