from ray.rllib.algorithms.ppo import PPOConfig

from dl_helper.rl.costum_rllib_module.client_learner import ClientLearnerGroup

class ClientPPOConfig(PPOConfig):
    def build_learner_group(
        self,
        *,
        env = None,
        spaces = None,
        rl_module_spec = None,
    ):
        """
        使用自定义的 ClientLearnerGroup
        """
        if rl_module_spec is None:
            rl_module_spec = self.get_multi_rl_module_spec(env=env, spaces=spaces)

        learner_group = ClientLearnerGroup(config=self.copy(), module_spec=rl_module_spec)
        return learner_group