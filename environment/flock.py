from cmath import exp
import numpy as np
import torch
from torch.distributions import Categorical
from gym import spaces
import gym
from common.util import wrapper, inverse_wrapper
import copy
import signatory


class Flock(gym.Env):
    def __init__(self, horizon):
        super(Flock, self).__init__()
        self.action_space = spaces.Discrete(4)  # 动作空间
        self.observation_space = spaces.Discrete(4)  # 状态空间
        self.name = 'Fish'
        self.mean_field = np.array(np.ones(self.observation_space.n)/self.observation_space.n)
        # state is int, but output is a vector
        self.state = np.zeros(self.observation_space.n)
        self.count = 0
        self.horizon = horizon
        self.his = np.array([-1])
        self.sign = torch.zeros(1, 3)

    def _reset(self):
        self.mean_field = np.array(np.ones(self.observation_space.n)/self.observation_space.n)
        self.count = 0
        self.state = np.zeros(self.observation_space.n)
        self.his = np.array([-1])
        self.sign = torch.zeros(1, 3)
        obs = copy.deepcopy(self.state)
        sign = np.array(self.sign.squeeze())
        out = np.concatenate((obs, sign))
        self.obs = out
        return copy.deepcopy(self.obs)

    def _step(self, action):
        next_obs, obs = self._get_observation(action['action'])
        obs = np.concatenate((obs, np.array(self.sign.squeeze())))
        # self.signal(action['signal'])
        self.evolve(action['prob'], action['signal'])
        reward = self._get_reward(action['action'])
        expert_data = self.expert_data(obs)
        next_obs = np.concatenate((next_obs, np.array(self.sign.squeeze())))
        self.count += 1
        self.obs = next_obs[:5]
        done = self._get_done()
        return next_obs, reward, done, expert_data

    def _get_observation(self, action):
        old_state = copy.deepcopy(self.state)
        self.state = wrapper(int(action), self.observation_space.n) + old_state
        return wrapper(int(action), self.observation_space.n) + old_state, old_state

    def _get_reward(self, action):
        return self.mean_field[int(action)]

    def _get_done(self):
        return self.count >= self.horizon

    # def signal(self, signal):
    #     if signal ==  0:
    #         self.mean_field = np.array([2/3, 1/3])
    #     elif signal == 1:
    #         self.mean_field = np.array([1/3, 2/3])

    def evolve(self, prob, signal):
        self.mean_field = prob
        self.his = np.concatenate((self.his, [signal]))
        self.sign = signatory.signature(torch.from_numpy(self.his).type(torch.FloatTensor).view(1, -1, 1), 3)
    
    def expert_data(self, obs):
        exp_signal = torch.from_numpy(np.random.choice([0, 1, 2, 3], 1, p=[0.25, 0.25, 0.25, 0.25]))
        # exp_action = Categorical(probs=torch.tensor([2 / 3, 1 / 3])).sample().unsqueeze(-1) if exp_signal == 0 else \
        #     Categorical(probs=torch.tensor([1 / 3, 2 / 3])).sample().unsqueeze(-1)
        exp_action = exp_signal
        expert = torch.zeros(10)
        # print(obs, expert[:3])
        expert[:7] = torch.from_numpy(obs)
        # expert[:2] = torch.from_numpy(wrapper(self.state, self.observation_space.n))
        # expert[np.random.choice([0, 1], p=[0.5, 0.5])] = 1.
        expert[7] = exp_signal
        expert[8] = self.count
        expert[9] = exp_action
        return expert
