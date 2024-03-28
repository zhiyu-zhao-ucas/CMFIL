import numpy as np
import torch
from environment.flock import Flock
from common import util
from tqdm import tqdm
import os
from MFAIRL import MFAIRL
from rl_algo import SAC
from common import util
import datetime
import scipy.stats
import signatory
import copy
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (5, 3.09)
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


def get_agent_trajectories(agent, size=100):
    env = Flock(2)
    states = np.zeros((size, env.horizon, 9))
    actions = np.zeros((size, env.horizon, env.action_space.n))
    policies = np.zeros_like(actions)
    his = np.array([-1])
    sign = torch.zeros(1, 3)
    Gt = 0
    done = False
    # t = 0
    for j in range(size):
        state = np.zeros(env.observation_space.n)
        for i in range(env.horizon):
            signal = np.random.choice([0, 1, 2, 3], p=[0.25, 0.25, 0.25, 0.25])
            state1 = np.concatenate((state, np.array(sign.squeeze()), [signal], [env.count]))
            action = agent.choose_action(state1, train=False)
            if i > 0:
                state += util.wrapper(int(action['action']), env.observation_space.n)
            states[j][i] = np.concatenate((state, np.array(sign.squeeze()), [signal], [env.count]))
            actions[j][i] = util.wrapper(action['action'], env.action_space.n)
            policies[j][i] = action['prob']
    return states, actions, policies


def save(save_path, policy, name):
    base_path = os.path.join(save_path, 'MFAIRL')
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


sns.set_theme(style="whitegrid")


if __name__ == "__main__":
    size = 3
    run_dir, log_dir = util.make_logpath('RPS', 'MFAIRL')
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
        agent = MFAIRL(9, 4, 2, max_epoch=200, mf_dim=4)
        reward_recover = agent.reward_model.eval()
        dir = "config/flock.yaml"
        config_dict = util.load_config(dir)
        paras = util.get_paras_from_dict(config_dict)
        RL_agent = SAC(paras)
        env = Flock(2)
        obs = env._reset()
        signal = np.random.choice(np.arange(paras.signal_dim), p=[0.25, 0.25, 0.25, 0.25])
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
                next_obs, reward, done, info = env._step(action)
                reward = reward_recover(torch.tensor(obs, dtype=torch.float), torch.tensor([util.wrapper(action['action'], env.action_space.n)], dtype=torch.float), torch.tensor(env.mean_field, dtype=torch.float)).detach().numpy()
                signal = np.random.choice(np.arange(paras.signal_dim), p=RL_agent.rho_sample())
                next_obs = np.concatenate((next_obs, [signal], [env.count]))
                RL_agent.add_experience(
                    {"states": obs, "states_next": next_obs, "rewards": reward, "dones": np.float32(done)})
                obs = next_obs
                Gt += reward
            obs = env._reset()
            done = False
            Gt = 0
            signal = np.random.choice(np.arange(paras.signal_dim), p=RL_agent.rho_sample())
            obs = np.concatenate((obs, [signal], [env.count]))
            RL_agent.learn()
            states, actions = get_trajectories(size)
            mf = np.sum(states, axis=0)[:,:4]
            mf /= size
            agent.learn(states, mf, actions)
            states, actions, policies = get_agent_trajectories(RL_agent, size)
            mf = np.sum(states, axis=0)[:,:4]
            mf /= size
            agent.learn(states, mf, actions, policy_logit=policies, expert_data=False)
            reward_recover = agent.reward_model.eval()
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
        mf00_list.append(mf00)
        mf10_list.append(mf10)
        mf20_list.append(mf20)
        mf01_list.append(mf01)
        mf11_list.append(mf11)
        mf21_list.append(mf21)
        mf02_list.append(mf02)
        mf12_list.append(mf12)
        mf22_list.append(mf22)
        mf03_list.append(mf03)
        mf13_list.append(mf13)
        mf23_list.append(mf23)

    e0 = np.vstack((mf00_list[0], mf00_list[1], mf00_list[2]))
    se = scipy.stats.sem(e0, axis=0)
    In = np.arange(smooth(e0.mean(0)).shape[0])
    plt.fill_between(In, smooth(e0.mean(0) - se), smooth(e0.mean(0) + se), alpha=0.2)
    plt.plot(In, smooth(e0.mean(0)), label='$\pi(a=0|s=\cdot, z=0)$')

    print(e0[:,-1].round(5), np.mean(e0[:,-1]), se[-1].round(5), smooth(e0.mean(0))[-1].round(5))

    e1 = np.vstack((mf10_list[0], mf10_list[1], mf10_list[2]))
    se = scipy.stats.sem(e1, axis=0)
    In = np.arange(smooth(e1.mean(0)).shape[0])
    plt.fill_between(In, smooth(e1.mean(0) - se), smooth(e1.mean(0) + se), alpha=0.2)
    plt.plot(In, smooth(e1.mean(0)), label='$\pi(a=1|s=\cdot, z=0)$')
    print(e1[:,-1].round(5), np.mean(e1[:,-1]).round(5), se[-1].round(5), smooth(e1.mean(0))[-1].round(5))

    e2 = np.vstack((mf20_list[0], mf20_list[1], mf20_list[2]))
    se = scipy.stats.sem(e2, axis=0)
    In = np.arange(smooth(e2.mean(0)).shape[0])
    # plt.subplot(223)
    # plt.subplot(121)
    plt.fill_between(In, smooth(e2.mean(0) - se), smooth(e2.mean(0) + se), alpha=0.2)
    plt.plot(In, smooth(e2.mean(0)), label='$\pi(a=2|s=\cdot, z=0)$')
    print(e2[:,-1].round(5), np.mean(e2[:,-1]).round(5), se[-1].round(5), smooth(e2.mean(0))[-1].round(5))


    e3 = 1.0 - e0 - e1 - e2
    se = scipy.stats.sem(e3, axis=0)
    In = np.arange(smooth(e3.mean(0)).shape[0])
    # plt.subplot(224)
    # plt.subplot(121)
    plt.fill_between(In, smooth(e3.mean(0) - se), smooth(e3.mean(0) + se), alpha=0.2)
    plt.plot(In, smooth(e3.mean(0)), label='$\pi(a=3|s=\cdot, z=0)$')
    # plt.tight_layout() 
    plt.xlabel("game plays")
    plt.ylabel("policy")
    plt.legend()
    plt.tight_layout() 
    plt.show()
    plt.savefig("figure/flock1.pdf")
    print(e3[:,-1].round(5), np.mean(e3[:,-1]).round(5), se[-1].round(5), smooth(e3.mean(0))[-1].round(5))
    print("_____________z=0_____________")
    plt.cla()

    e0 = np.vstack((mf01_list[0], mf01_list[1], mf01_list[2]))
    se = scipy.stats.sem(e0, axis=0)
    In = np.arange(smooth(e0.mean(0)).shape[0])
    plt.fill_between(In, smooth(e0.mean(0) - se), smooth(e0.mean(0) + se), alpha=0.2)
    plt.plot(In, smooth(e0.mean(0)), label='$\pi(a=0|s=\cdot, z=1)$')
    print(e0[:,-1].round(5), np.mean(e0[:,-1]).round(5), se[-1].round(5), smooth(e0.mean(0))[-1].round(5))


    e1 = np.vstack((mf11_list[0], mf11_list[1], mf11_list[2]))
    se = scipy.stats.sem(e1, axis=0)
    In = np.arange(smooth(e1.mean(0)).shape[0])
    plt.fill_between(In, smooth(e1.mean(0) - se), smooth(e1.mean(0) + se), alpha=0.2)
    plt.plot(In, smooth(e1.mean(0)), label='$\pi(a=1|s=\cdot, z=1)$')
    print(e1[:,-1].round(5), np.mean(e1[:,-1]).round(5), se[-1].round(5), smooth(e1.mean(0))[-1].round(5))

    e2 = np.vstack((mf21_list[0], mf21_list[1], mf21_list[2]))
    se = scipy.stats.sem(e2, axis=0)
    In = np.arange(smooth(e2.mean(0)).shape[0])
    plt.fill_between(In, smooth(e2.mean(0) - se), smooth(e2.mean(0) + se), alpha=0.2)
    plt.plot(In, smooth(e2.mean(0)), label='$\pi(a=2|s=\cdot, z=1)$')
    print(e2[:,-1].round(5), np.mean(e2[:,-1]).round(5), se[-1].round(5), smooth(e2.mean(0))[-1].round(5))


    e3 = 1.0 - e0 - e1 - e2
    se = scipy.stats.sem(e3, axis=0)
    In = np.arange(smooth(e3.mean(0)).shape[0])
    plt.fill_between(In, smooth(e3.mean(0) - se), smooth(e3.mean(0) + se), alpha=0.2)
    plt.plot(In, smooth(e3.mean(0)), label='$\pi(a=3|s=\cdot, z=1)$')
    print(e3[:,-1].round(5), np.mean(e3[:,-1]).round(5), se[-1].round(5), smooth(e3.mean(0))[-1].round(5))
    plt.xlabel("game plays")
    plt.ylabel("policy")
    plt.legend()
    plt.tight_layout() 
    plt.show()
    plt.savefig("figure/flock2.pdf")
    print("_____________z=1_____________")
    plt.cla()



    e0 = np.vstack((mf02_list[0], mf02_list[1], mf02_list[2]))
    se = scipy.stats.sem(e0, axis=0)
    In = np.arange(smooth(e0.mean(0)).shape[0])
    plt.fill_between(In, smooth(e0.mean(0) - se), smooth(e0.mean(0) + se), alpha=0.2)
    plt.plot(In, smooth(e0.mean(0)), label='$\pi(a=0|s=\cdot, z=2)$')
    print(e0[:,-1].round(5), np.mean(e0[:,-1]).round(5), se[-1].round(5), smooth(e0.mean(0))[-1].round(5))

    e1 = np.vstack((mf12_list[0], mf12_list[1], mf12_list[2]))
    se = scipy.stats.sem(e1, axis=0)
    In = np.arange(smooth(e1.mean(0)).shape[0])
    plt.fill_between(In, smooth(e1.mean(0) - se), smooth(e1.mean(0) + se), alpha=0.2)
    plt.plot(In, smooth(e1.mean(0)), label='$\pi(a=1|s=\cdot, z=2)$')
    print(e1[:,-1].round(5), np.mean(e1[:,-1]).round(5), se[-1].round(5), smooth(e1.mean(0))[-1].round(5))

    e2 = np.vstack((mf22_list[0], mf22_list[1], mf22_list[2]))
    se = scipy.stats.sem(e2, axis=0)
    In = np.arange(smooth(e2.mean(0)).shape[0])
    plt.fill_between(In, smooth(e2.mean(0) - se), smooth(e2.mean(0) + se), alpha=0.2)
    plt.plot(In, smooth(e2.mean(0)), label='$\pi(a=2|s=\cdot, z=2)$')
    print(e2[:,-1].round(5), np.mean(e2[:,-1]).round(5), se[-1].round(5), smooth(e2.mean(0))[-1].round(5))


    e3 = 1.0 - e0 - e1 - e2
    se = scipy.stats.sem(e3, axis=0)
    In = np.arange(smooth(e3.mean(0)).shape[0])
    plt.fill_between(In, smooth(e3.mean(0) - se), smooth(e3.mean(0) + se), alpha=0.2)
    plt.plot(In, smooth(e3.mean(0)), label='$\pi(a=3|s=\cdot, z=2)$')
    plt.tight_layout() 
    plt.xlabel("game plays")
    plt.ylabel("policy")
    plt.legend()
    plt.tight_layout() 
    plt.show()
    plt.savefig("figure/flock3.pdf")
    print(e3[:,-1].round(5), np.mean(e3[:,-1]).round(5), se[-1].round(5), smooth(e3.mean(0))[-1].round(5))
    print("_____________z=2_____________")
    plt.cla()

    e0 = np.vstack((mf03_list[0], mf03_list[1], mf03_list[2]))
    se = scipy.stats.sem(e0, axis=0)
    In = np.arange(smooth(e0.mean(0)).shape[0])
    plt.fill_between(In, smooth(e0.mean(0) - se), smooth(e0.mean(0) + se), alpha=0.2)
    plt.plot(In, smooth(e0.mean(0)), label='$\pi(a=0|s=\cdot, z=2)$')
    print(e0[:,-1].round(5), np.mean(e0[:,-1]).round(5), se[-1].round(5), smooth(e0.mean(0))[-1].round(5))

    e1 = np.vstack((mf13_list[0], mf13_list[1], mf13_list[2]))
    se = scipy.stats.sem(e1, axis=0)
    In = np.arange(smooth(e1.mean(0)).shape[0])
    plt.fill_between(In, smooth(e1.mean(0) - se), smooth(e1.mean(0) + se), alpha=0.2)
    plt.plot(In, smooth(e1.mean(0)), label='$\pi(a=1|s=\cdot, z=2)$')
    print(e1[:,-1].round(5), np.mean(e1[:,-1]).round(5), se[-1].round(5), smooth(e1.mean(0))[-1].round(5))

    e2 = np.vstack((mf23_list[0], mf23_list[1], mf23_list[2]))
    se = scipy.stats.sem(e2, axis=0)
    In = np.arange(smooth(e2.mean(0)).shape[0])
    plt.fill_between(In, smooth(e2.mean(0) - se), smooth(e2.mean(0) + se), alpha=0.2)
    plt.plot(In, smooth(e2.mean(0)), label='$\pi(a=2|s=\cdot, z=2)$')
    print(e2[:,-1].round(5), np.mean(e2[:,-1]).round(5), se[-1].round(5), smooth(e2.mean(0))[-1].round(5))


    e3 = 1.0 - e0 - e1 - e2
    se = scipy.stats.sem(e3, axis=0)
    In = np.arange(smooth(e3.mean(0)).shape[0])
    plt.fill_between(In, smooth(e3.mean(0) - se), smooth(e3.mean(0) + se), alpha=0.2)
    plt.plot(In, smooth(e3.mean(0)), label='$\pi(a=3|s=\cdot, z=2)$')
    plt.tight_layout() 
    plt.xlabel("game plays")
    plt.ylabel("policy")
    plt.legend()
    plt.tight_layout() 
    plt.show()
    plt.savefig("figure/flock4.pdf")
    print(e3[:,-1].round(5), np.mean(e3[:,-1]).round(5), se[-1].round(5), smooth(e3.mean(0))[-1].round(5))
    print("_____________z=3_____________")
    plt.cla()