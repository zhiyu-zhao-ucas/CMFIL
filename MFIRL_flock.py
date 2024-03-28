import numpy as np
import torch
from environment.flock import Flock
from common import util
from tqdm import tqdm
import os
from MFIRL import MFIRL
from rl_algo import SAC
from common import util
import datetime
import scipy.stats
import signatory
import copy
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (5, 3.09) # 单位是inches
import seaborn as sns


def get_trajectories(size=100):
    env = Flock(2)
    states = np.zeros((size, env.horizon, 9))
    actions = np.zeros((size, env.horizon, env.action_space.n))
    his = np.array([-1])
    sign = torch.zeros(1, 3)
    Gt = 0
    done = False
    # t = 0
    for j in range(size):
        state = np.zeros(env.observation_space.n)
        for i in range(env.horizon):
            signal = np.random.choice([0, 1, 2, 3], p=[0.25, 0.25, 0.25, 0.25])
            action = signal
            if i > 0:
                state += util.wrapper(int(action), env.observation_space.n)
            states[j][i] = np.concatenate((state, np.array(sign.squeeze()), [signal], [env.count]))
            actions[j][i] = util.wrapper(action, env.action_space.n)
    return states, actions


def save(save_path, policy, name):
    base_path = os.path.join(save_path, 'MFIRL')
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    model_critic_path = os.path.join(base_path, name + ".pth")
    torch.save(policy, model_critic_path)
    print("successfully saved at "+str(model_critic_path))


def smooth(data, sm=100):
    smooth_data = copy.deepcopy(data)
    smooth_data[0] = data[0]
    smooth_data = np.convolve(np.ones(sm)/sm, data, 'valid')
    return smooth_data


sns.set(style="whitegrid")


if __name__ == "__main__":
    units = 64
    learning_rate = 0.0001
    max_epoch = 1070
    size = 100
    run_dir, log_dir = util.make_logpath('RPS', 'MFIRL')
    data_length = 10000
    mf00_list = []
    mf10_list = []
    mf20_list = []
    mf01_list = []
    mf11_list = []
    mf21_list = []
    mf02_list = []
    mf12_list = []
    mf22_list = []
    mf03_list = []
    mf13_list = []
    mf23_list = []
    for i in range(3):
        mf00 = np.zeros(data_length)
        mf10 = np.zeros(data_length)
        mf20 = np.zeros(data_length)
        mf01 = np.zeros(data_length)
        mf11 = np.zeros(data_length)
        mf21 = np.zeros(data_length)
        mf02 = np.zeros(data_length)
        mf12 = np.zeros(data_length)
        mf22 = np.zeros(data_length)
        mf03 = np.zeros(data_length)
        mf13 = np.zeros(data_length)
        mf23 = np.zeros(data_length)
        rho_data = np.zeros(data_length)
        states, actions = get_trajectories(size)
        mf = np.sum(states, axis=0)[:,:4]
        mf /= size
        agent = MFIRL(9, 4, 2, max_epoch=300, mf_dim=4)
        agent.learn(states, mf, actions)
        reward_recover = agent.reward_model.eval()
        dir = "config/flock.yaml"
        config_dict = util.load_config(dir)
        paras = util.get_paras_from_dict(config_dict)
        RL_agent = SAC(paras)
        env = Flock(2)
        obs = env.reset()
        signal = np.random.choice(np.arange(paras.signal_dim), p=RL_agent.rho_sample())
        obs = np.concatenate((obs, [signal], [env.count]))
        Gt = 0
        done = False
        policy = []
        for data_id in tqdm(range(data_length)):
            iter = 0
            while not done:
                iter += 1
                action = RL_agent.choose_action(obs)
                action.update({"signal": signal})
                next_obs, reward, done, info = env.step(action)
                reward = reward_recover(torch.tensor(obs, dtype=torch.float), torch.tensor([util.wrapper(action['action'], env.action_space.n)], dtype=torch.float), torch.tensor(env.mean_field, dtype=torch.float)).detach().numpy()
                signal = np.random.choice(np.arange(paras.signal_dim), p=RL_agent.rho_sample())
                next_obs = np.concatenate((next_obs, [signal], [env.count]))
                RL_agent.add_experience(
                    {"states": obs, "states_next": next_obs, "rewards": reward, "dones": np.float32(done)})
                obs = next_obs
                Gt += reward
            obs = env.reset()
            done = False
            Gt = 0
            signal = np.random.choice(np.arange(paras.signal_dim), p=RL_agent.rho_sample())
            obs = np.concatenate((obs, [signal], [env.count]))
            RL_agent.learn()
            obs_test = copy.deepcopy(obs)
            obs_test[-2] = 0
            rho_data[data_id] = RL_agent.rho_sample()[0]
            obs_test = copy.deepcopy(obs)
            obs_test[-2] = 0
            mf00[data_id] = RL_agent.choose_action(obs_test, train=False)['prob'][0]
            mf10[data_id] = RL_agent.choose_action(obs_test, train=False)['prob'][1]
            mf20[data_id] = RL_agent.choose_action(obs_test, train=False)['prob'][2]
            obs_test[-2] = 1
            mf01[data_id] = RL_agent.choose_action(obs_test, train=False)['prob'][0]
            mf11[data_id] = RL_agent.choose_action(obs_test, train=False)['prob'][1]
            mf21[data_id] = RL_agent.choose_action(obs_test, train=False)['prob'][2]
            obs_test[-2] = 2
            mf02[data_id] = RL_agent.choose_action(obs_test, train=False)['prob'][0]
            mf12[data_id] = RL_agent.choose_action(obs_test, train=False)['prob'][1]
            mf22[data_id] = RL_agent.choose_action(obs_test, train=False)['prob'][2]
            obs_test[-2] = 3
            mf03[data_id] = RL_agent.choose_action(obs_test, train=False)['prob'][0]
            mf13[data_id] = RL_agent.choose_action(obs_test, train=False)['prob'][1]
            mf23[data_id] = RL_agent.choose_action(obs_test, train=False)['prob'][2]
        save(run_dir, mf00, "mf00" + " " + str(i))
        save(run_dir, mf10, "mf10" + " " + str(i))
        save(run_dir, mf20, "mf20" + " " + str(i))
        save(run_dir, mf01, "mf01" + " " + str(i))
        save(run_dir, mf11, "mf11" + " " + str(i))
        save(run_dir, mf21, "mf21" + " " + str(i))
        save(run_dir, mf02, "mf02" + " " + str(i))
        save(run_dir, mf12, "mf12" + " " + str(i))
        save(run_dir, mf22, "mf22" + " " + str(i))
        save(run_dir, mf03, "mf03" + " " + str(i))
        save(run_dir, mf13, "mf13" + " " + str(i))
        save(run_dir, mf23, "mf23" + " " + str(i))
        print("saved!")
