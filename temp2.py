from pprint import pprint

from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    .environment("CartPole-v1")
    .env_runners(num_env_runners=1)
)

algo = config.build()

for i in range(10):
    result = algo.train()
    result.pop("config")
    pprint(result)

    if i % 5 == 0:
        checkpoint_dir = algo.save_to_path()
        print(f"Checkpoint saved in directory {checkpoint_dir}")