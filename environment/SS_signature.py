from cmath import sin
import numpy as np
import torch
from gym import spaces
import gym
from common.util import wrapper, inverse_wrapper
from torch.distributions import Categorical
import copy
import signatory


class Sequential_Squeeze_sign(gym.Env):
    def __init__(self, horizon=2):
        super(Sequential_Squeeze_sign, self).__init__()
        self.action_space = spaces.Discrete(2)  # 动作空间
        self.observation_space = spaces.Discrete(2)  # 状态空间
        self.name = 'Sequential_Squeeze_sign'
        self.mean_field = np.array(np.ones(self.observation_space.n)/self.observation_space.n)
        # state is int, but output is a vector
        self.state = np.random.choice([0, 1], p=np.ones(self.observation_space.n)/self.observation_space.n)
        self.count = 0
        self.horizon = horizon
        self.his = np.array([-1])
        self.sign = torch.zeros(1, 3)
        self.out = np.concatenate((wrapper(self.state, self.observation_space.n), np.array(self.sign.squeeze())))

    def _reset(self):
        self.mean_field = np.array(np.ones(self.observation_space.n)/self.observation_space.n)
        self.count = 0
        self.state = np.random.choice([0, 1], p=np.ones(self.observation_space.n) / self.observation_space.n)
        self.his = np.array([-1])
        self.sign = torch.zeros(1, 3)
        obs = wrapper(self.state, self.observation_space.n)
        sign = np.array(self.sign.squeeze())
        out = np.concatenate((obs, sign))
        self.obs = out
        return out
        # return wrapper(self.state, self.observation_space.n)

    def _step(self, action):
        next_obs, obs = self._get_observation(action['action'])
        # self.signal(action['signal'])
        self.evolve(action['prob'], action['signal'])
        reward = self._get_reward(action['action'])
        expert_data = self.expert_data(obs)
        self.count += 1
        done = self._get_done()
        next_obs = np.concatenate((next_obs, np.array(self.sign.squeeze())))
        self.obs = next_obs[:5]
        return next_obs, reward, done, expert_data

    def _get_observation(self, action):
        # obs = wrapper(self.state, self.observation_space.n)
        if int(action) == 0:
            self.state = Categorical(probs=torch.tensor([3 / 4, 1 / 4])).sample().unsqueeze(-1)
        else:
            self.state = Categorical(probs=torch.tensor([1 / 4, 3 / 4])).sample().unsqueeze(-1)
        return wrapper(self.state, self.observation_space.n), copy.deepcopy(self.obs)

    def _get_reward(self, action):
        return self.mean_field[int(action)]

    def _get_done(self):
        return self.count >= self.horizon

    # def signal(self, signal):
    #     if signal ==  0:
    #         self.mean_field = np.array([2/3, 1/3])
    #     elif signal == 1:
    #         self.mean_field = np.array([1/3, 2/3])

    def expert_data(self, obs):
        if self.count == 0:
            exp_signal = torch.from_numpy(np.random.choice([0, 1], 1, p=[0.6, 0.4]))
            exp_action = Categorical(probs=torch.tensor([2 / 3, 1 / 3])).sample().unsqueeze(-1) if exp_signal == 0 else \
                Categorical(probs=torch.tensor([1 / 3, 2 / 3])).sample().unsqueeze(-1)
        else:
            exp_signal = torch.from_numpy(np.random.choice([0, 1], 1, p=[0.4, 0.6]))
            exp_action = 0 if exp_signal == 0 else 1
        expert = torch.zeros(8)
        expert[:5] = torch.from_numpy(obs)
        # expert[:5] = torch.from_numpy(wrapper(self.state, self.observation_space.n))
        expert[np.random.choice([0, 1], p=[0.5, 0.5])] = 1.
        expert[5] = exp_signal
        expert[6] = self.count
        expert[7] = exp_action
        return expert

    def evolve(self, prob, signal):
        self.mean_field[0] = 0.75 * prob[0] + 0.25 * self.mean_field[0]
        self.mean_field[1] = 1 - self.mean_field[0]
        self.his = np.concatenate((self.his, [signal]))
        self.sign = signatory.signature(torch.from_numpy(self.his).type(torch.FloatTensor).view(1, -1, 1), 3)
