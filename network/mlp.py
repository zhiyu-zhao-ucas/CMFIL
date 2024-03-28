import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Normal


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(self.input_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, self.output_size)
        self.activation = nn.LeakyReLU()
        self.apply(weights_init_)

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        action_scores = self.linear3(x)
        return F.sigmoid(action_scores)

    def get_logits(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        action_scores = self.linear3(x)
        return action_scores
