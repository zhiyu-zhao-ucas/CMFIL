import numpy as np
import torch
from torch.distributions import Categorical
from gym import spaces
import gym
from common.util import wrapper, inverse_wrapper
import copy
import pandas as pd


class ExtendTraffic(gym.Env):
    def __init__(self, horizon, state):
        super(ExtendTraffic, self).__init__()
        self.action_space = spaces.Discrete(6)  # 动作空间
        self.observation_space = spaces.Discrete(6)  # 状态空间
        self.name = 'Fish'
        self.mean_field = np.array(np.ones(self.observation_space.n)/self.observation_space.n)
        # state is int, but output is a vector
        # self.state = np.random.choice(np.arange(self.observation_space.n), p=np.ones(self.observation_space.n)/self.observation_space.n)
        self.state = state
        self.count = 0
        self.horizon = horizon

    def _reset(self):
        self.mean_field = np.array(np.ones(self.observation_space.n)/self.observation_space.n)
        self.count = 0
        # self.state = np.random.choice(np.arange(self.observation_space.n), p=np.ones(self.observation_space.n)/self.observation_space.n)
        # self.state = 0
        return wrapper(self.state, self.observation_space.n)

    def _step(self, action):
        next_obs, obs = self._get_observation(action['action'])
        # self.signal(action['signal'])
        self.evolve(action['prob'])
        reward = self._get_reward(action['action'])
        expert_data = None
        self.count += 1
        done = self._get_done()
        return next_obs, reward, done, expert_data

    def _get_observation(self, action):
        old_state = wrapper(self.state, self.observation_space.n)
        # self.state = int(action)
        return wrapper(self.state, self.observation_space.n), old_state

    def _get_reward(self, action):
        return self.mean_field[int(action)]

    def _get_done(self):
        return self.count >= self.horizon

    def evolve(self, prob):
        self.mean_field = prob
    
    def expert_data(self, obs):
        exp_signal = torch.from_numpy(np.random.choice([0, 1], 1, p=[0.5, 0.5]))
        exp_signal = 0
        exp_action = Categorical(probs=torch.tensor([2 / 3, 1 / 3])).sample().unsqueeze(-1) if exp_signal == 0 else \
            Categorical(probs=torch.tensor([1 / 3, 2 / 3])).sample().unsqueeze(-1)
        expert = torch.zeros(5)
        expert[:2] = torch.from_numpy(obs)
        expert[:2] = torch.from_numpy(wrapper(self.state, self.observation_space.n))
        expert[np.random.choice([0, 1], p=[0.5, 0.5])] = 1.
        expert[10] = exp_signal
        expert[3] = self.count
        expert[4] = exp_action
        return expert
