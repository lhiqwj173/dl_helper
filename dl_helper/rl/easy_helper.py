class algo_base:
    def __init__(self, **kwargs):
        self._training_kwargs = kwargs

    @property
    def training_kwargs(self):
        return self._training_kwargs
    
    def _update_kwargs(self, kwargs):
        for k, v in kwargs.items():
            self._training_kwargs[k] = v

class PPO(algo_base):
    @property
    def algo(self):
        return "PPO"

class IMPALA(algo_base):
    @property
    def algo(self):
        return "IMPALA"

class APPO(algo_base):
    def __init__(self, **kwargs):
        self._training_kwargs = {
            'grad_clip': 30.0,
        }

        self._update_kwargs(kwargs)

    @property
    def algo(self):
        return "APPO"

class DQN(algo_base):

    @property
    def algo(self):
        return "DQN"
    
class Rainbow_DQN(DQN):
    def __init__(self, **kwargs):
        self._training_kwargs = {
            "target_network_update_freq": 500,
            'replay_buffer_config': {
                "type": "PrioritizedEpisodeReplayBuffer",
                "capacity": 60000,
                "alpha": 0.5,
                "beta": 0.5,
            },
            # "replay_buffer_config": {
            #     "_enable_replay_buffer_api": False,
            #     "type": "ReplayBuffer",
            #     "type": "PrioritizedReplayBuffer",
            #     "capacity": 50000,
            #     "prioritized_replay_alpha": 0.6,
            #     "prioritized_replay_beta": 0.4,
            #     "prioritized_replay_eps": 1e-6,
            #     "replay_sequence_length": 1
            # },
            "epsilon": [[0, 1.0], [1000000, 0.1]],
            "adam_epsilon": 1e-8,
            "grad_clip": 40.0,
            "num_steps_sampled_before_learning_starts": 10000,
            "tau": 1,
            "num_atoms": 51,
            "v_min": -10.0,
            "v_max": 10.0,
            "noisy": True,
            "sigma0": 0.5,
            "dueling": True,
            "hiddens": [512],
            "double_q": True,
            "n_step": 3,
        }

        self._update_kwargs(kwargs)

class Double_DQN(DQN):
    def __init__(self, **kwargs):
        self._training_kwargs = {
            'double_q': True,
        }

        self._update_kwargs(kwargs)

class Dueling_DQN(DQN):
    def __init__(self, **kwargs):
        self._training_kwargs = {
            'dueling': True,
        }

        self._update_kwargs(kwargs)

class DQN_PER(DQN):
    def __init__(self, **kwargs):
        self._training_kwargs = {
            'replay_buffer_config': {
                "type": "PrioritizedEpisodeReplayBuffer",
                "capacity": 60000,
                "alpha": 0.5,
                "beta": 0.5,
            },
        }

        self._update_kwargs(kwargs)

class Noisy_DQN(DQN):
    def __init__(self, **kwargs):
        self._training_kwargs = {
            'noisy': True,
        }
        
        self._update_kwargs(kwargs)

class DQN_C51(DQN):
    def __init__(self, **kwargs):
        self._training_kwargs = {
            'num_atoms': 51,
            'v_min': -10.0,
            'v_max': 10.0,
        }
        
        self._update_kwargs(kwargs)



