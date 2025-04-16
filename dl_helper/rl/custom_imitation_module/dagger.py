from imitation.algorithms.dagger import SimpleDAggerTrainer as _SimpleDAggerTrainer
from imitation.algorithms.dagger import Optional, rollout, np

class SimpleDAggerTrainer(_SimpleDAggerTrainer):
    def train(
        self,
        total_timesteps: int,
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
            self.extend_and_update(bc_train_kwargs)
            round_num += 1

