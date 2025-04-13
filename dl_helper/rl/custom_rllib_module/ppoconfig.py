from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.learner.learner_group import LearnerGroup

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

        if hasattr(self, '_extra_learner_group_class'):
            learner_group = self._extra_learner_group_class(config=self.copy(), module_spec=rl_module_spec, **self._extra_learner_group_kwargs)
        else:
            learner_group = LearnerGroup(config=self.copy(), module_spec=rl_module_spec)
        return learner_group
    
    def extra_config(self, learner_group_class=LearnerGroup, learner_group_kwargs={}):
        """
        额外的自定义配置
        """
        self._extra_learner_group_class = learner_group_class
        self._extra_learner_group_kwargs = learner_group_kwargs
        return self

