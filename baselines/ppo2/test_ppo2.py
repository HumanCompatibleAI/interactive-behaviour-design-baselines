import os
import sys
import unittest
from collections import namedtuple

import numpy as np
from gym.spaces import Box

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(repo_path)

import global_variables
from baselines.ppo2.ppo2 import Runner
from reward_switcher import RewardSelector, RewardSource

Model = namedtuple('Model', 'initial_state')
Env = namedtuple('Env', 'observation_space num_envs reset')


class TestProcessRewards(unittest.TestCase):
    def test_process_rewards(self):
        num_envs = 16
        n_steps = 128
        model = Model(None)
        observation_space = Box(low=-1.0, high=1.0, shape=(84, 84, 4))
        env = Env(observation_space, num_envs, lambda: None)
        runner = Runner(env=env, model=model, nsteps=n_steps, gamma=None, lam=None)

        global_variables.reward_selector = RewardSelector(classifiers=None, reward_predictor=None)
        global_variables.reward_selector.set_reward_source(RewardSource.ENV)

        mb_obs = np.random.rand(n_steps, num_envs, *observation_space.shape)
        mb_rewards = np.random.rand(n_steps, num_envs)
        returned_rewards = runner.process_rewards(mb_obs, mb_rewards)
        np.testing.assert_array_equal(returned_rewards, mb_rewards)

    def test_reshape(self):
        num_envs = 16
        n_envs = 128
        mb_obs = np.random.rand(n_envs, num_envs, 84, 84, 4)
        mb_rewards = np.random.rand(n_envs, num_envs)
        mb_obs_flat = np.reshape(mb_obs, (-1,) + mb_obs.shape[2:])
        mb_rewards_flat = np.reshape(mb_rewards, (-1,))
        for step_n in range(n_envs):
            for env_n in range(num_envs):
                actual = (mb_obs[step_n, env_n], mb_rewards[step_n, env_n])
                i = step_n * (num_envs) + env_n
                expected = (mb_obs_flat[i], mb_rewards_flat[i])
                assert np.array_equal(actual[0], expected[0]) and actual[1] == expected[1]


if __name__ == '__main__':
    unittest.main()
