import time, os, psutil, shutil, pickle
import pandas as pd
from collections import deque
from typing import Deque
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.dagger import (
    Optional, 
    rollout, np, DAggerTrainer, vec_env, types, policies, Tuple, List, serialize, logging, NeedsDemosException, th_data, pathlib, 
    InteractiveTrajectoryCollector, Callable, VecEnvStepReturn, 
    Mapping, Any,
    util,uuid,
    
)

from dl_helper.rl.custom_imitation_module.rollout import KEYS
from dl_helper.rl.rl_utils import plot_bc_train_progress, CustomCheckpointCallback, check_gradients
from dl_helper.tool import report_memory_usage, in_windows

from py_ext.tool import log

TEST_REST_GB = 27

from memory_profiler import profile
import objgraph
import reprlib
import gc, sys
gc.set_debug(gc.DEBUG_SAVEALL)  # 记录无法回收的对象
def debug_mem():
    log('*'* 60)
    obj_list = []
    for obj in gc.get_objects():
        size = sys.getsizeof(obj)
        obj_list.append((obj, size))

    sorted_objs = sorted(obj_list, key=lambda x: x[1], reverse=True)

    msg = ['']
    for obj, size in sorted_objs[:10]:
        msg.append(f'OBJ:{id(obj)} TYPE:{type(obj)} SIZE:{size/1024/1024:.2f}MB REPR:{str(obj)[:200]}')
        referrers = gc.get_referrers(obj)
        for ref in referrers:
            msg.append(f'   {str(ref)[:300]}')

    msg_str = '\n'.join(msg)
    log(msg_str)


def take_snapshot(types):
    """
    获取指定 types 的当前对象集合。
    返回 {type_name: set(对象id)} 结构，
    对于 dict 类型，会过滤掉包含指定键的字典对象。
    """
    snapshot = {}
    
    # 需要过滤的 dict 键名称列表
    dict_filter = {'objs', 'after', 'dict', 'tuple'}

    for t in types:
        objs = objgraph.by_type(t)

        # 对于 dict 类型，排除包含指定键的字典对象
        if t == 'dict':
            filtered_objs = []
            for d in objs:
                try:
                    keys = set(d.keys())
                except Exception:
                    # 如果对象无法当作 dict 处理，跳过
                    continue
                # 如果字典的键与过滤列表无交集，则保留
                if not (keys & dict_filter):
                    filtered_objs.append(d)
            objs = filtered_objs

        snapshot[t] = set(id(o) for o in objs)

    return snapshot

def diff_snapshot(before, after):
    """
    比较前后两个快照，找出新增的对象 id
    返回 {type_name: list(新增对象)}
    """
    growth = {}
    for t in before:
        new_ids = after[t] - before[t]
        if new_ids:
            # 重新从 id 找回对象
            objs = [o for o in objgraph.by_type(t) if id(o) in new_ids]
            growth[t] = objs
    return growth

snapshot = None
count = 0
def debug_growth():
    global snapshot, count
    
    result = objgraph.growth()
    if result:
        width = max(len(name) for name, _, _ in result)
        for name, count, delta in result:
            log('%-*s%9d %+9d\n' % (width, name, count, delta))

    # watch_types = ['tuple', 'dict']
    watch_types = ['dict']
    before = snapshot
    after = take_snapshot(watch_types)

    if before is not None:
        growth = diff_snapshot(before, after)
        for t, objs in growth.items():
            log(f"\n类型 {t} 新增了 {len(objs)} 个对象")
            log(f"当前内存使用: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
            for i, o in enumerate(objs):
                # log(f"处理对象 {i}, 类型: {type(o)}")
                try:
                    log(f"  + {reprlib.repr(o)[:200]}")
                except Exception as e:
                    log(f"  + 无法转换对象 {o}, 错误: {e}")

                backrefs = gc.get_referrers(o)
                for ref in backrefs:
                    log(f"      > 引用: {reprlib.repr(ref)[:200]}")
        
        snapshot.clear()
        snapshot.update(after)
    else:
        snapshot = after

class policy_eval_collector(vec_env.VecEnvWrapper):
    """
    使用策略生成轨迹
    记录策略动作 / 专家动作 序列
    """
    _last_obs: Optional[np.ndarray]

    def __init__(
        self,
        venv: vec_env.VecEnv,
        get_robot_acts: Callable[[np.ndarray], np.ndarray],
        rng: np.random.Generator,
    ) -> None:
        """Builds InteractiveTrajectoryCollector.

        Args:
            venv: vectorized environment to sample trajectories from.
            get_robot_acts: get robot actions that can be substituted for
                human actions. Takes a vector of observations as input & returns a
                vector of actions.
            rng: random state for random number generation.
        """
        super().__init__(venv)
        self.get_robot_acts = get_robot_acts
        self._last_obs = None
        self._is_reset = False
        self.rng = rng

        # 记录动作
        self.expert_acts = []
        self.policy_acts = []

    def get_action_sequences(self):
        """
        获取专家动作和策略动作序列
        """
        return np.concatenate(self.expert_acts), np.concatenate(self.policy_acts)

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        """Set the seed for the DAgger random number generator and wrapped VecEnv.

        The DAgger RNG is used along with `self.beta` to determine whether the expert
        or robot action is forwarded to the wrapped VecEnv.

        Args:
            seed: The random seed. May be None for completely random seeding.

        Returns:
            A list containing the seeds for each individual env. Note that all list
            elements may be None, if the env does not return anything when seeded.
        """
        self.rng = np.random.default_rng(seed=seed)
        return list(self.venv.seed(seed))

    def reset(self) -> np.ndarray:
        """Resets the environment.

        Returns:
            obs: first observation of a new trajectory.
        """
        obs = self.venv.reset()
        assert isinstance(obs, np.ndarray)
        self._last_obs = obs
        self._is_reset = True
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        """
        actions: 专家动作
        """
        assert self._is_reset, "call .reset() before .step()"
        assert self._last_obs is not None

        # 记录专家动作
        self.expert_acts.append(actions)

        # 记录策略动作
        policy_acts = self.get_robot_acts(self._last_obs)
        self.policy_acts.append(policy_acts)

        # 使用策略动作交互
        self.venv.step_async(policy_acts)

    def step_wait(self) -> VecEnvStepReturn:
        """Returns observation, reward, etc after previous `step_async()` call.

        Returns:
            Observation, reward, dones (is terminal?) and info dict.
        """
        next_obs, rews, dones, infos = self.venv.step_wait()
        assert isinstance(next_obs, np.ndarray)
        self._last_obs = next_obs
        return next_obs, rews, dones, infos

def _save_dagger_demo(
    trajectory: types.Trajectory,
    trajectory_index: int,
    save_dir: types.AnyPath,
    rng: np.random.Generator,
    prefix: str = "",
) -> None:
    save_dir = util.parse_path(save_dir)
    assert isinstance(trajectory, types.Trajectory)
    actual_prefix = f"{prefix}-" if prefix else ""
    randbits = int.from_bytes(rng.bytes(16), "big")
    random_uuid = uuid.UUID(int=randbits, version=4).hex
    filename = f"{actual_prefix}dagger-demo-{trajectory_index}-{random_uuid}.npz"
    npz_path = save_dir / filename
    assert (
        not npz_path.exists()
    ), "The following DAgger demonstration path already exists: {0}".format(npz_path)

    # 保存轨迹
    # serialize.save(npz_path, [trajectory])

    # 保存 transitions
    transitions = rollout.flatten_trajectories([trajectory])
    pickle.dump(transitions, open(npz_path, 'wb'))

    logging.info(f"Saved demo at '{npz_path}'")

class InteractiveTransitionsCollector(InteractiveTrajectoryCollector):
    def step_wait(self) -> VecEnvStepReturn:
        """Returns observation, reward, etc after previous `step_async()` call.

        Stores the transition, and saves trajectory as demo once complete.

        Returns:
            Observation, reward, dones (is terminal?) and info dict.
        """
        next_obs, rews, dones, infos = self.venv.step_wait()
        assert isinstance(next_obs, np.ndarray)
        assert self.traj_accum is not None
        assert self._last_user_actions is not None
        self._last_obs = next_obs
        fresh_demos = self.traj_accum.add_steps_and_auto_finish(
            obs=next_obs,
            acts=self._last_user_actions,
            rews=rews,
            infos=infos,
            dones=dones,
        )
        for traj_index, traj in enumerate(fresh_demos):
            _save_dagger_demo(traj, traj_index, self.save_dir, self.rng)

        return next_obs, rews, dones, infos
    
def calculate_sample_size_bytes(sample):
    total = 0
    log("各字段内存占用（字节）：")
    for k, v in sample.items():
        size = v.nbytes
        log(f"  {k}: {size} B")
        total += size
    log(f"=> 单条样本总计: {total} B\n")
    return total

def get_max_rows(sample_size_bytes, reserved_gb=TEST_REST_GB):
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
            arr.fill('') 
        else:
            arr = np.zeros(new_shape, dtype=dtype)
            arr.fill(1.0) 
        data[key] = arr
        print(f"初始化 {key}，形状: {arr.shape}，类型: {arr.dtype}")
    return data

class SimpleDAggerTrainer(DAggerTrainer):

    def __init__(
        self,
        *,
        venv: vec_env.VecEnv,
        env_objs, 
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
        
        # 实际的环境对象列表
        self.env_objs = env_objs

        # 样本数据 dict
        self.transitions_dict = None
        self.full = False   # 是否已经满了
        self.cur_idx = 0    # 可以写入的样本索引
        self.capacity = 0   # 缓冲区容量

    def _init_transitions_dict(self, transitions):
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
        self.capacity = self.transitions_dict[KEYS[0]].shape[0]  # 缓冲区容量

    @profile(precision=4,stream=open('_copy_data.log','w+'))
    def _copy_data(self, transitions):
        t_length = transitions.acts.shape[0]  # 待写入数据长度

        log(f'capacity: {self.capacity}, t_length: {t_length}, cur_idx: {self.cur_idx}, full: {self.full}')
        # 写入数据
        # 情况1：写入数据比容量大 → 只保留最后 capacity 条
        if t_length > self.capacity:
            offset = t_length - self.capacity
            for key in KEYS:
                data = getattr(transitions, key)
                # transitions_data = data[offset:]
                # self.transitions_dict[key][:] = transitions_data
                np.copyto(
                    self.transitions_dict[key],
                    data[offset:]
                )
            self.cur_idx = 0
            self.full = True
        else:
            # 正常或跨越写入
            begin = self.cur_idx
            # 第一段：从当前 cur_idx 到缓冲区尾部
            first_part_len = min(self.capacity - begin, t_length)
            # 第二段（如果需要）：从头开始继续写
            second_part_len = t_length - first_part_len
            for key in KEYS:
                data = getattr(transitions, key)
                # 使用 np.copyto 避免生成切片临时数组
                np.copyto(
                    self.transitions_dict[key][begin:begin+first_part_len],
                    data[:first_part_len]
                )
                if second_part_len > 0:
                    np.copyto(
                        self.transitions_dict[key][0:second_part_len],
                        data[first_part_len:]
                    )
                    self.full = True
            # 更新状态
            self.cur_idx = (begin + t_length) % self.capacity

        return t_length

    @profile(precision=4,stream=open('_handle_demo_path.log','w+'))
    def _handle_demo_path(self, path):
        log(f'load demo: {path}')
        log(f"[before demo load] 系统可用内存: {psutil.virtual_memory().available / (1024**3):.2f} GB")

        # # 载入轨迹
        # demo = serialize.load(path)
        # # 转为 transitions
        # transitions = rollout.flatten_trajectories(demo)
        # del demo

        # 载入 transitions
        transitions = pickle.load(open(path, 'rb'))

        # 检查初始化
        if self.transitions_dict is None:
            self._init_transitions_dict(transitions)

        # 拷贝数据
        t_length = self._copy_data(transitions)

        # 释放内存
        del transitions

        # gc.collect()
        # for g in gc.garbage:
        #     log(f'garbage: {g}')

        # debug_growth()

        log(f"[after demo done] 系统可用内存: {psutil.virtual_memory().available / (1024**3):.2f} GB")
        return t_length

    @profile(precision=4,stream=open('_handle_demo_paths.log','w+'))
    def _handle_demo_paths(self, demo_paths):
        new_transitions_length =0
        for path in demo_paths:
            new_transitions_length += self._handle_demo_path(path)

        return new_transitions_length

    @profile(precision=4,stream=open('_load_all_demos.log','w+'))
    def _load_all_demos(self) -> Tuple[types.Transitions, List[int]]:
        """
        载入最新的样本
        1. 若 self.transitions_dict 未初始化，按照系统的可用内存初始化固定的大小
        2. 遍历在 self.cur_idx 处写入新的样本
        3. 若 self.cur_idx 超过了最大容量，则从头开始覆盖，设置 self.full 为 True
        """
        new_transitions_length = 0

        # debug_growth()

        log(f"_load_all_demos 系统可用内存: {psutil.virtual_memory().available / (1024**3):.2f} GB")

        # 检查可写性
        if self.transitions_dict:
            for key in KEYS:
                target_array = self.transitions_dict[key]
                # log(f"检查 key '{key}' 的 writeable 标志: {target_array.flags.writeable}")
                if not target_array.flags.writeable:
                    log(f"尝试将 key '{key}' 的数组重新设为可写。")
                    target_array.flags.writeable = True # 强制设为 True (但这可能掩盖根本原因)

        for round_num in range(self._last_loaded_round + 1, self.round_num + 1):
            round_dir = self._demo_dir_path_for_round(round_num)
            demo_paths = self._get_demo_paths(round_dir)

            new_transitions_length += self._handle_demo_paths(demo_paths)

        log(f"_load_all_demos done 系统可用内存: {psutil.virtual_memory().available / (1024**3):.2f} GB")
        log(f"Loaded new transitions {new_transitions_length}, total: {self.cur_idx if not self.full else self.capacity}")

    @profile(precision=4,stream=open('extend_and_update.log','w+'))
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
            log(f"[extend_and_update] 系统可用内存: {psutil.virtual_memory().available / (1024**3):.2f} GB")

            self._last_loaded_round = self.round_num

            if self.full:
                log(f"数据满了，开始训练")
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
                # 设置训练数据
                self.bc_trainer.set_demonstrations(data_loader)
                log(f"Training at round {self.round_num}")
                # bc 训练
                self.bc_trainer.train(**bc_train_kwargs)
                # 清除训练数据
                self.bc_trainer.clear_demonstrations()
                # 清理数据
                del data_loader
                del transitions
            else:
                log(f"数据未满，当前样本数: {self.cur_idx}")

            self.round_num += 1
            log(f"New round number is {self.round_num}")
        
        return self.round_num

    @profile(precision=4,stream=open('train.log','w+'))
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
        # for debug
        rollout_round_min_timesteps = 500

        total_timestep_count = 0
        round_num = 0

        # 初始化进度数据
        if os.path.exists(progress_file_all):
            df_progress = pd.read_csv(progress_file_all)
        else:
            df_progress = pd.DataFrame()

        while total_timestep_count < total_timesteps:
            log(f"[train 0] 系统可用内存: {psutil.virtual_memory().available / (1024**3):.2f} GB")

            if (TEST_REST_GB - 1) > psutil.virtual_memory().available / (1024**3):
                log(f'内存超出限制（{TEST_REST_GB - 1}）GB, 退出')
                return

            round_episode_count = 0
            round_timestep_count = 0

            collector = self.create_trajectory_collector()
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
            log(f"[train 1] 系统可用内存: {psutil.virtual_memory().available / (1024**3):.2f} GB")

            for traj in trajectories:
                self._logger.record_mean(
                    "dagger/mean_episode_reward",
                    np.sum(traj.rews),
                )
                round_timestep_count += len(traj)
                total_timestep_count += len(traj)
            # log(f"[train 03] 系统可用内存: {psutil.virtual_memory().available / (1024**3):.2f} GB")

            round_episode_count += len(trajectories)

            self._logger.record("dagger/total_timesteps", total_timestep_count)
            self._logger.record("dagger/round_num", round_num)
            self._logger.record("dagger/round_episode_count", round_episode_count)
            self._logger.record("dagger/round_timestep_count", round_timestep_count)

            # log(f"[train 1] 系统可用内存: {psutil.virtual_memory().available / (1024**3):.2f} GB")
            # del trajectories
            # del sample_until
            # del collector
            # log(f"[train 2] 系统可用内存: {psutil.virtual_memory().available / (1024**3):.2f} GB")

            # `logger.dump` is called inside BC.train within the following fn call:
            # 默认会训练 self.DEFAULT_N_EPOCHS(4) 个EPOCHS
            self.extend_and_update(bc_train_kwargs)
            round_num += 1
        
            # # 检查梯度
            # check_gradients(self.bc_trainer)

            # gc.collect()
            # debug_mem()

            # 检查是否发生训练（数据是否满了）
            if not self.full:
                continue

            # # 验证模型
            # collector = policy_eval_collector(
            #     venv=self.venv, 
            #     get_robot_acts=lambda acts: self.bc_trainer.policy.predict(acts)[0],
            #     rng=self.rng,
            # )
            # sample_until = rollout.make_sample_until(
            #     min_timesteps=max(rollout_round_min_timesteps, self.batch_size),
            #     min_episodes=rollout_round_min_episodes,
            # )
            # trajectories = rollout.generate_trajectories(
            #     policy=self.expert_policy,
            #     venv=collector,
            #     sample_until=sample_until,
            #     deterministic_policy=False,
            #     rng=collector.rng,
            # )
            # # 获取动作序列
            # expert_acts, policy_acts = collector.get_action_sequences()

            # _t = time.time()
            # eval_env.val()
            # val_reward, _ = evaluate_policy(self.bc_trainer.policy, eval_env)
            # eval_env.train()
            # train_reward, _ = evaluate_policy(self.bc_trainer.policy, eval_env)
            # log(f"train_reward: {train_reward}, val_reward: {val_reward}, 验证耗时: {time.time() - _t:.2f} 秒")

            # # 合并到 progress_all.csv
            # latest_ts = df_progress.iloc[-1]['timestamp'] if len(df_progress) > 0 else 0
            # df_new = pd.read_csv(progress_file)
            # df_new = df_new.loc[df_new['timestamp'] > latest_ts, :]
            # df_new['bc/epoch'] += round_num * self.DEFAULT_N_EPOCHS
            # df_new['bc/mean_reward'] = np.nan
            # df_new['bc/val_mean_reward'] = np.nan
            # df_new.loc[df_new.index[-1], 'bc/mean_reward'] = train_reward
            # df_new.loc[df_new.index[-1], 'bc/val_mean_reward'] = val_reward
            # df_progress = pd.concat([df_progress, df_new]).reset_index(drop=True)
            # df_progress.ffill(inplace=True)
            # df_progress.to_csv(progress_file_all, index=False)

            # # 当前点是否是最优的 checkpoint
            # # 使用 recall 判断
            # if 'bc/recall' in list(df_progress):
            #     bset_recall = df_progress['bc/recall'].max()
            #     best_epoch = df_progress.loc[df_progress['bc/recall'] == bset_recall, 'bc/epoch'].values[0]
            #     is_best = df_progress.iloc[-1]['bc/epoch'] == best_epoch
            # else:
            #     is_best = False

            # # 训练进度可视化
            # plot_bc_train_progress(train_folder, df_progress=df_progress, title=train_title)

            # for debug
            is_best = False
            for file in os.listdir(r'/kaggle/working/'):
                if file.endswith('.log'):
                    shutil.copy2(os.path.join(r'/kaggle/working/', file), os.path.join(train_folder, file))

            if not in_windows():
                # 保存模型
                train_folder_manager.checkpoint(self.bc_trainer, best=is_best)

