import numpy as np
import pytz
import gym
from stable_baselines.results_plotter import ts2xy
from stable_baselines.bench.monitor import load_results

class CustomRewardAndDoneEnv(gym.Wrapper):
    def __init__(self, env):
        super(CustomRewardAndDoneEnv, self).__init__(env)

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        reward = reward / 10

        if info["life"] < 2:
            done = True

        return state, reward, done, info