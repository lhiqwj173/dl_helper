from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import numpy as np
from imitation.data import types
from dl_helper.rl.rl_env.lob_trade.lob_const import ACTION_BUY, ACTION_SELL

def balance_rollout(r):

    obs = r.obs
    acts = r.acts

    # 查找 obs[-2](pos) == 1 且 act==ACTION_SELL 的数量
    cond = (obs[:-1, -2] == 1) & (acts==ACTION_SELL)
    pos_sell_indices = np.where(cond)[0]
    # print(f"持仓卖出: {len(pos_sell_indices)} / {len(acts)} : {pos_sell_indices}")

    # 查找 obs[-2](pos) == 1 且 act==ACTION_BUY 的数量
    cond = (obs[:-1, -2] == 1) & (acts==ACTION_BUY)
    pos_buy_indices = np.where(cond)[0]
    # print(f"持仓买入: {len(pos_buy_indices)} / {len(acts)}")
    # 随机 pos_sell_indices 相同的个数
    pos_buy_indices = np.random.choice(pos_buy_indices, size=len(pos_sell_indices), replace=False) if len(pos_buy_indices) else []
    # print(f"持仓买入(随机): {len(pos_buy_indices)} / {len(acts)} : {pos_buy_indices}")

    # 查找 obs[-2](pos) == 0 且 act==ACTION_BUY 的数量
    cond = (obs[:-1, -2] == 0) & (acts==ACTION_BUY)
    blank_buy_indices = np.where(cond)[0]
    # print(f"空仓买入: {len(blank_buy_indices)} / {len(acts)} : {blank_buy_indices}")

    # 查找 obs[-2](pos) == 0 且 act==ACTION_SELL 的数量
    cond = (obs[:-1, -2] == 0) & (acts==ACTION_SELL)
    blank_sell_indices = np.where(cond)[0]
    # print(f"空仓卖出: {len(blank_sell_indices)} / {len(acts)}")
    # 随机 blank_buy_indices 相同的个数
    blank_sell_indices = np.random.choice(blank_sell_indices, size=len(blank_buy_indices), replace=False) if len(blank_sell_indices) else []
    # print(f"空仓卖出(随机): {len(blank_sell_indices)} / {len(acts)} : {blank_sell_indices}")

    # 合并所有的索引
    wait_concat_list = [i for i in [pos_sell_indices, pos_buy_indices, blank_buy_indices, blank_sell_indices] if len(i)]
    all_indices = np.concatenate(wait_concat_list) if wait_concat_list else np.array([], dtype=int)
    all_indices = np.sort(all_indices)
    # print(f'取用索引: {all_indices}')

    # 修改 rollout
    _obs = r.obs[all_indices]
    _next_obs = r.obs[all_indices + 1]
    _acts = r.acts[all_indices]
    _infos = r.infos[all_indices] if r.infos is not None else None
    _rews = r.rews[all_indices]

    # print(f'obs: {_obs}')
    # print(f'next_obs: {_next_obs}')
    # print(f'acts: {_acts}')
    # print(f'infos: {_infos}')
    # print(f'rews: {_rews}')

    return _obs, _next_obs, _acts, _infos, _rews


class rollouts_filter:

    def __init__(self) -> None:
        # mypy struggles without Any annotation here.
        # The necessary constraints are enforced above.
        keys = ["obs", "next_obs", "acts", "dones", "infos"]
        self.parts: Mapping[str, List[Any]] = {key: [] for key in keys}
        self._length = 0

    def add_rollouts(self, trajectories: Iterable[types.Trajectory]):
        def all_of_type(key, desired_type):
            return all(
                isinstance(getattr(traj, key), desired_type) for traj in trajectories
            )

        assert all_of_type("obs", types.DictObs) or all_of_type("obs", np.ndarray)
        assert all_of_type("acts", np.ndarray)

        for i, traj in enumerate(trajectories):
            # print(i)
            _obs, _next_obs, _acts, _infos, _rews = balance_rollout(traj)
            if len(_obs) == 0:
                continue
            else:
                self._length += len(_obs)

            self.parts["acts"].append(_acts)
            self.parts["obs"].append(_obs)
            self.parts["next_obs"].append(_next_obs)

            dones = np.zeros(len(_acts), dtype=bool)
            dones[-1] = traj.terminal
            self.parts["dones"].append(dones)

            if _infos is None:
                infos = np.array([{}] * len(_acts))
            else:
                infos = _infos
            self.parts["infos"].append(infos)

    def __len__(self):
        return self._length

    def flatten_trajectories(
        self,
    ) -> types.Transitions:
        """Flatten a series of trajectory dictionaries into arrays.

        Args:
            trajectories: list of trajectories.

        Returns:
            The trajectories flattened into a single batch of Transitions.
        """

        cat_parts = {
            key: types.concatenate_maybe_dictobs(part_list)
            for key, part_list in self.parts.items()
        }
        lengths = set(map(len, cat_parts.values()))
        assert len(lengths) == 1, f"expected one length, got {lengths}"
        return types.Transitions(**cat_parts)
