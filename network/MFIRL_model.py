import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torch.nn.utils as U
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import gym
from gym import spaces


class RewardModel(nn.Module):
    def __init__(self, state_shape, action_shape, mf_shape, num_of_units):
        super(RewardModel, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_c1 = nn.Linear(state_shape + action_shape + mf_shape, num_of_units)
        self.linear_c2 = nn.Linear(num_of_units, num_of_units)
        self.linear_c = nn.Linear(num_of_units, 1)

        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, state_input, action_input, mf_input):
        """
        input_g: input_global, input features of all agents
        """
        x_cat = self.LReLU(self.linear_c1(torch.cat([state_input, action_input.squeeze(), mf_input], dim=0)))
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        #print('value', value)
        return value


class ShapingModel(nn.Module):
    def __init__(self, state_shape, mf_shape, num_of_units):
        super(ShapingModel, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_c1 = nn.Linear(state_shape + mf_shape, num_of_units)
        self.linear_c2 = nn.Linear(num_of_units, num_of_units)
        self.linear_c = nn.Linear(num_of_units, 1)

        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, state_input, mf_input):
        """
        input_g: input_global, input features of all agents
        """
        x_cat = self.LReLU(self.linear_c1(torch.cat([state_input, mf_input], dim=1)))
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        #print('value', value)
        return value