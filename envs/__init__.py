from envs.sc2_gym_env import SC2GymEnv
from envs.sc2_dzb_gym_env import SC2DZBGymEnv
from gym.envs.registration import register

register(
    id='sc2-gym-env-v0',
    entry_point='envs.sc2_gym_env:SC2GymEnv',
)
register(
    id='sc2-dzb-gym-env-v0',
    entry_point='envs.sc2_dzb_gym_env:SC2DZBGymEnv',
)

