from cmath import exp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.distributions import Categorical
from network.mlp import MLP
from common.buffer import Replay_buffer as buffer


class Discriminator(object):
    def __init__(self, args):
        self.input_size = args.dis_obs_space
        self.output_size = args.dis_output_size
        self.buffer_size = args.dis_buffer_size
        self.batch_size = args.dis_batch_size
        self.obs_space = args.obs_space
        self.mlp = MLP(input_size=self.input_size,
                                           output_size=self.output_size)
        self.opt = torch.optim.Adam(self.mlp.parameters(), lr=args.dis_lr)
        self.max_grad_norm = args.dis_max_grad_norm
        # self.decayRate = 0.96
        # self.my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.opt, gamma=self.decayRate)
        trajectory_property = ["exp_pair", "action"]
        self.memory = buffer(self.buffer_size, trajectory_property)
        self.memory.init_item_buffers()
        self.loss = torch.tensor(0)

    def get_reward(self, state, action, expert):
        # exp_signal = Categorical(probs=torch.tensor([0.5, 0.5])).sample().unsqueeze(-1)
        # exp_signal = torch.from_numpy(np.random.choice([0, 1], 1, p=[0.5, 0.5]))
        # exp_action = Categorical(probs=torch.tensor([2 / 3, 1 / 3])).sample().unsqueeze(-1) if exp_signal == 0 else \
        #     Categorical(probs=torch.tensor([1 / 3, 2 / 3])).sample().unsqueeze(-1)
        # expert_ = torch.zeros(self.input_size)
        # expert_[:3] = torch.from_numpy(state)
        # expert_[2] = exp_signal
        # expert_[4] = exp_action
        # expert[np.random.choice([0, 1], p=[0.5, 0.5])] = 1.
        # expert[:2] = torch.from_numpy(state[:2])
        # expert[2] = exp_signal
        # expert[3] = exp_action
        # expert[np.random.choice([0, 1], p=[0.5, 0.5])] = 1.
        # if state.shape[0] == action.shape[0]:
        pair = torch.tensor(np.concatenate((state, [action])), dtype=torch.float).view(1, -1)
        # else:
        #     pair = torch.tensor(np.concatenate((state, action)), dtype=torch.float).view(1, -1)
        reward = (-1) * torch.log(self.mlp(pair)).squeeze().detach()
        self.add_experience({"action": pair, "exp_pair": expert})
        return reward

    def update(self):
        data = self.memory.sample(self.batch_size)

        transitions = {
            "exp_pair": np.array(data['exp_pair']),
            "action_pair": np.array(data['action']),
        }

        expert = torch.tensor(transitions["exp_pair"], dtype=torch.float).detach()
        pair = torch.tensor(transitions["action_pair"], dtype=torch.float)
        gp = self.gradient_penalty(expert, pair)
        exp_score = self.mlp.get_logits(expert)
        gen_score = self.mlp.get_logits(pair)
        self.loss = torch.nn.functional.binary_cross_entropy_with_logits(
            exp_score, torch.zeros_like(exp_score)
        ) \
                    + torch.nn.functional.binary_cross_entropy_with_logits(
            gen_score, torch.ones_like(gen_score) * 0.9
        ) \
                    + gp
        self.true_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            exp_score, torch.zeros_like(exp_score)
        ).detach().numpy()
        self.fake_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            gen_score, torch.ones_like(gen_score) * 0.9
        ).detach().numpy()
        self.gp = gp.detach().numpy()
        self.opt.zero_grad()
        self.loss.backward()
        # nn.utils.clip_grad_norm_(self.mlp.parameters(), self.max_grad_norm)
        self.opt.step()
        # self.my_lr_scheduler.step()

    def add_experience(self, output):
        agent_id = 0
        for k, v in output.items():
            self.memory.insert(k, agent_id, v)

    def gradient_penalty(self, xr, xf):
        """

        :param D:
        :param xr: [b, 2]
        :param xf: [b, 2]
        :return:
        """
        # [b, 1]
        # t = torch.rand(xr.shape[0], 1)
        # print(t * xr)
        # [b, 1] => [b, 2]  broadcasting so t is the same for x1 and x2
        # t = t.expand_as(torch.tensor(xr.shape[0]))
        # interpolation
        # mid = t * xr + (1 - t) * xf
        mid = xr
        # set it to require grad info
        mid.requires_grad_()

        pred = self.mlp(mid)
        grads = autograd.grad(outputs=pred, inputs=mid,
                            grad_outputs=torch.ones_like(pred),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

        gp = torch.pow(grads.norm(2, dim=1), 2).mean()

        return gp


class WGANgp(object):
    def __init__(self, args):
        self.input_size = args.dis_obs_space
        self.output_size = args.dis_output_size
        self.buffer_size = args.dis_buffer_size
        self.batch_size = args.dis_batch_size
        self.obs_space = args.obs_space
        self.mlp = MLP(input_size=self.input_size,
                                           output_size=self.output_size)
        self.opt = torch.optim.Adam(self.mlp.parameters(), lr=args.dis_lr)
        self.max_grad_norm = args.dis_max_grad_norm
        # self.decayRate = 0.96
        # self.my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.opt, gamma=self.decayRate)
        trajectory_property = ["exp_pair", "action"]
        self.memory = buffer(self.buffer_size, trajectory_property)
        self.memory.init_item_buffers()
        self.loss = torch.tensor(0)

    def get_reward(self, state, action, expert):
        # exp_signal = Categorical(probs=torch.tensor([0.5, 0.5])).sample().unsqueeze(-1)
        # exp_signal = torch.from_numpy(np.random.choice([0, 1], 1, p=[0.5, 0.5]))
        # exp_action = Categorical(probs=torch.tensor([2 / 3, 1 / 3])).sample().unsqueeze(-1) if exp_signal == 0 else \
        #     Categorical(probs=torch.tensor([1 / 3, 2 / 3])).sample().unsqueeze(-1)
        # expert_ = torch.zeros(self.input_size)
        # expert_[:3] = torch.from_numpy(state)
        # expert_[2] = exp_signal
        # expert_[4] = exp_action
        # expert[np.random.choice([0, 1], p=[0.5, 0.5])] = 1.
        # expert[:2] = torch.from_numpy(state[:2])
        # expert[2] = exp_signal
        # expert[3] = exp_action
        # expert[np.random.choice([0, 1], p=[0.5, 0.5])] = 1.

        pair = torch.tensor(np.concatenate((state, [action])), dtype=torch.float).view(1, -1)
        reward = (-1) * self.mlp(pair).squeeze().detach()
        self.add_experience({"action": pair, "exp_pair": expert})
        return reward

    def update(self):
        data = self.memory.sample(self.batch_size)

        transitions = {
            "exp_pair": np.array(data['exp_pair']),
            "action_pair": np.array(data['action']),
        }

        expert = torch.tensor(transitions["exp_pair"], dtype=torch.float)
        pair = torch.tensor(transitions["action_pair"], dtype=torch.float)
        gp = self.gradient_penalty(expert, pair)
        exp_score = self.mlp.get_logits(expert)
        gen_score = self.mlp.get_logits(pair)
        self.loss = - gen_score.mean() + exp_score.mean()\
                    + 0.01 * gp
        self.true_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            exp_score, torch.zeros_like(exp_score)
        ).detach().numpy()
        self.fake_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            gen_score, torch.ones_like(gen_score) * 0.9
        ).detach().numpy()
        self.gp = gp.detach().numpy()
        self.opt.zero_grad()
        self.loss.backward()
        # nn.utils.clip_grad_norm_(self.mlp.parameters(), self.max_grad_norm)
        self.opt.step()
        for p in self.mlp.parameters():
            p.data.clamp_(-0.1, 0.1)
        # self.my_lr_scheduler.step()

    def add_experience(self, output):
        agent_id = 0
        for k, v in output.items():
            self.memory.insert(k, agent_id, v)

    def gradient_penalty(self, xr, xf):
        """

        :param D:
        :param xr: [b, 2]
        :param xf: [b, 2]
        :return:
        """
        # [b, 1]
        t = torch.rand(xr.shape[0], 1)
        # print(t * xr)
        # [b, 1] => [b, 2]  broadcasting so t is the same for x1 and x2
        # t = t.expand_as(torch.tensor(xr.shape[0]))
        # interpolation
        mid = t * xr + (1 - t) * xf
        # set it to require grad info
        mid.requires_grad_()

        pred = self.mlp(mid)
        grads = autograd.grad(outputs=pred, inputs=mid,
                            grad_outputs=torch.ones_like(pred),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

        gp = torch.pow(grads.norm(2, dim=1), 2).mean()

        return gp

class WGAN(object):
    def __init__(self, input_size, output_size, buffer_size, lr, batch_size=128, max_grad_norm=0.2):
        self.input_size = input_size
        self.output_size = output_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.mlp = MLP(input_size=self.input_size,
                                           output_size=self.output_size)
        self.opt = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        self.max_grad_norm = max_grad_norm
        # self.decayRate = 0.96
        # self.my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.opt, gamma=self.decayRate)
        trajectory_property = ["exp_pair", "action"]
        self.memory = buffer(self.buffer_size, trajectory_property)
        self.memory.init_item_buffers()
        self.loss = torch.tensor(0)

    def get_reward(self, state, action):
        exp_signal = Categorical(probs=torch.tensor([0.5, 0.5])).sample().unsqueeze(-1)
        exp_action = Categorical(probs=torch.tensor([2 / 3, 1 / 3])).sample().unsqueeze(-1) if exp_signal == 0 else \
            Categorical(probs=torch.tensor([1 / 3, 2 / 3])).sample().unsqueeze(-1)
        expert = torch.zeros(self.input_size)
        expert[2] = exp_signal
        expert[3] = exp_action
        expert[np.random.choice([0, 1], p=[0.5, 0.5])] = 1.
        pair = torch.tensor(np.concatenate((state, [action])), dtype=torch.float).view(1, -1)
        reward = self.mlp(pair).squeeze().detach()
            # torch.log(self.mlp(pair)).squeeze().detach()
        self.add_experience({"action": pair, "exp_pair": expert})
        return reward

    def update(self):
        data = self.memory.sample(self.batch_size)

        transitions = {
            "exp_pair": np.array(data['exp_pair']),
            "action_pair": np.array(data['action']),
        }

        expert = torch.tensor(transitions["exp_pair"], dtype=torch.float)
        pair = torch.tensor(transitions["action_pair"], dtype=torch.float)
        exp_score = self.mlp.get_logits(expert)
        gen_score = self.mlp.get_logits(pair)
        self.loss = - exp_score.mean() + gen_score.mean()
        self.true_loss = - exp_score.mean()
        self.fake_loss = gen_score.mean()
        total_grad = 0
        parameters = [p for p in self.mlp.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.norm(2)
            total_grad += param_norm
        self.loss += total_grad
            # print(param_norm)
        #     torch.nn.functional.binary_cross_entropy_with_logits(
        #     exp_score, torch.ones_like(exp_score) * 0.9
        # ) \
        #             + torch.nn.functional.binary_cross_entropy_with_logits(
        #     gen_score, torch.zeros_like(gen_score) * 0.1
        # )
        self.opt.zero_grad()
        self.loss.backward()
        # nn.utils.clip_grad_norm_(self.mlp.parameters(), self.max_grad_norm)
        self.opt.step()
        for p in self.mlp.parameters():
            p.data.clamp_(-0.1, 0.1)
        # self.my_lr_scheduler.step()

    def add_experience(self, output):
        agent_id = 0
        for k, v in output.items():
            self.memory.insert(k, agent_id, v)
