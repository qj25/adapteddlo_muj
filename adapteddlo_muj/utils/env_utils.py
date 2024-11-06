from typing import Callable
import gymnasium as gym
# import gym
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor


def make_env(
    env_id: str, rank: int, seed: int = 0, log_path=None
) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        # print(gym.__version__)
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        if log_path is not None:
            env = Monitor(env, log_path)
        return env

    set_random_seed(seed)
    return _init