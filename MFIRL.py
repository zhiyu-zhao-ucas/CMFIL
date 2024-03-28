import torch
import torch.nn.utils as U
from tqdm import tqdm
from network.MFIRL_model import RewardModel, ShapingModel


class MFIRL:
    def __init__(self, obs_dim, act_dim, horizon, max_epoch=700, lr=1e-4, mf_dim=3):
        self.reward_model = RewardModel(action_shape=act_dim, state_shape=obs_dim, mf_shape=mf_dim, num_of_units=64)
        self.shaping_model = ShapingModel(mf_shape=mf_dim, num_of_units=64, state_shape=obs_dim)
        self.max_epoch = max_epoch
        self.lr = lr
        self.horizon = horizon
        self.optimizer1 = torch.optim.Adam(self.reward_model.parameters(), lr=lr)
        self.optimizer2 = torch.optim.Adam(self.shaping_model.parameters(), lr=lr)

    def learn(self, state, mf, action):
        for epoch in tqdm(range(self.max_epoch)):
            reward = []
            for i in range(len(state)):
                reward_per_step = [self.reward_model(torch.tensor(state[i][t], dtype=torch.float),
                                                     torch.tensor(action[i][t], dtype=torch.float).unsqueeze(dim=0),
                                                     torch.tensor(mf[t], dtype=torch.float))
                                   for t in range(self.horizon)]
                reward_per_sample = torch.sum(torch.cat(reward_per_step, dim=0).reshape((1, -1)))
                reward_shaping_per_sample = reward_per_sample + self.shaping_model(
                    torch.tensor(state[i][-1], dtype=torch.float).unsqueeze(dim=0),
                    torch.tensor(mf[-1], dtype=torch.float).unsqueeze(dim=0)) \
                                            - self.shaping_model(
                    torch.tensor(state[i][0], dtype=torch.float).unsqueeze(dim=0),
                    torch.tensor(mf[-1], dtype=torch.float).unsqueeze(dim=0))
                reward.append(reward_shaping_per_sample)
            expect_reward = torch.sum(torch.cat(reward, dim=0).reshape((1, -1))) / self.max_epoch

            logZ = torch.log(torch.sum(torch.exp(torch.cat(reward, dim=0).reshape((1, -1)))))
            loss = - expect_reward + logZ
            loss.backward()
            U.clip_grad_norm_(self.reward_model.parameters(), 0.5)
            U.clip_grad_norm_(self.shaping_model.parameters(), 0.5)
            self.optimizer1.step()
            self.optimizer2.step()

    def save(self):
        torch.save(self.reward_model.state_dict(), 'chen_malware.pth')