from math import ceil, floor
import numpy as np
import torch
from environment.flock import Flock
from common import util
from rl_algo import SAC
from discriminator import Discriminator, WGAN
from tqdm import tqdm
import copy
import datetime
import os


def save(save_path, policy, name, num):
    base_path = os.path.join(save_path, 'flock_data' + num)
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    model_critic_path = os.path.join(base_path, name + ".pth")
    torch.save(policy, model_critic_path)
    print("successfully saved at "+str(model_critic_path))


if __name__ == '__main__':
        use_wandb = True
        dir = "config/flock.yaml"
        config_dict = util.load_config(dir)
        paras = util.get_paras_from_dict(config_dict)
        run_dir, log_dir = util.make_logpath('flock', 'MFIL')
        agent = SAC(paras)
        env = Flock(paras.horizon)
        episode = 10000
        data_length = ceil(episode/10)
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
        rho_data0 = np.zeros(data_length)
        rho_data1 = np.zeros(data_length)
        rho_data2 = np.zeros(data_length)
        discriminator = Discriminator(paras)
        ret = []
        obs = env.reset()
        Gt = 0
        done = False
        signal = np.random.choice(np.arange(paras.signal_dim), p=agent.rho_sample())
        obs = np.concatenate((obs, [signal], [env.count]))
        print("-----------------------log_dir", log_dir, "-----------------------")
        for _ in tqdm(range(episode)):
            for con in range(64):
                while not done:
                    action = agent.choose_action(obs)
                    action.update({"signal": signal})
                    next_obs, reward, done, expert_data = env.step(action)
                    d_score = discriminator.get_reward(obs, action['action'], expert_data)
                    signal = np.random.choice(np.arange(paras.signal_dim), p=agent.rho_sample())
                    next_obs = np.concatenate((next_obs, [signal], [env.count]))
                    agent.add_experience(
                        {"states": obs, "states_next": next_obs, "rewards": d_score, "dones": np.float32(done)})
                    obs = next_obs
                    Gt += reward
                obs = env.reset()
                done = False
                ret.append(Gt)
                Gt = 0
                signal = np.random.choice(np.arange(paras.signal_dim), p=agent.rho_sample())
                obs = np.concatenate((obs, [signal], [env.count]))
            discriminator.update()
            agent.learn()
            loss = 0
            if _ % 10 == 0:
                data_id = floor(_/10)
                rho_data0[data_id] = agent.rho_sample()[0]
                rho_data1[data_id] = agent.rho_sample()[1]
                rho_data2[data_id] = agent.rho_sample()[2]
                obs_test = copy.deepcopy(obs)
                obs_test[-2] = 0
                mf00[data_id] = agent.choose_action(obs_test, train=False)['prob'][0]
                mf10[data_id] = agent.choose_action(obs_test, train=False)['prob'][1]
                mf20[data_id] = agent.choose_action(obs_test, train=False)['prob'][2]
                obs_test[-2] = 1
                mf01[data_id] = agent.choose_action(obs_test, train=False)['prob'][0]
                mf11[data_id] = agent.choose_action(obs_test, train=False)['prob'][1]
                mf21[data_id] = agent.choose_action(obs_test, train=False)['prob'][2]
                obs_test[-2] = 2
                mf02[data_id] = agent.choose_action(obs_test, train=False)['prob'][0]
                mf12[data_id] = agent.choose_action(obs_test, train=False)['prob'][1]
                mf22[data_id] = agent.choose_action(obs_test, train=False)['prob'][2]
                obs_test[-2] = 3
                mf03[data_id] = agent.choose_action(obs_test, train=False)['prob'][0]
                mf13[data_id] = agent.choose_action(obs_test, train=False)['prob'][1]
                mf23[data_id] = agent.choose_action(obs_test, train=False)['prob'][2]
        save(run_dir, mf00, "mf00" + " " + str(datetime.datetime.now().minute), str(iter))
        save(run_dir, mf10, "mf10" + " " + str(datetime.datetime.now().minute), str(iter))
        save(run_dir, mf20, "mf20" + " " + str(datetime.datetime.now().minute), str(iter))
        save(run_dir, mf01, "mf01" + " " + str(datetime.datetime.now().minute), str(iter))
        save(run_dir, mf11, "mf11" + " " + str(datetime.datetime.now().minute), str(iter))
        save(run_dir, mf21, "mf21" + " " + str(datetime.datetime.now().minute), str(iter))
        save(run_dir, mf02, "mf02" + " " + str(datetime.datetime.now().minute), str(iter))
        save(run_dir, mf12, "mf12" + " " + str(datetime.datetime.now().minute), str(iter))
        save(run_dir, mf22, "mf22" + " " + str(datetime.datetime.now().minute), str(iter))
        save(run_dir, mf03, "mf03" + " " + str(datetime.datetime.now().minute), str(iter))
        save(run_dir, mf13, "mf13" + " " + str(datetime.datetime.now().minute), str(iter))
        save(run_dir, mf23, "mf23" + " " + str(datetime.datetime.now().minute), str(iter))
        save(run_dir, rho_data0, "rho_data0" + " " + str(datetime.datetime.now().minute), str(iter))
        save(run_dir, rho_data1, "rho_data1" + " " + str(datetime.datetime.now().minute), str(iter))
        save(run_dir, rho_data2, "rho_data2" + " " + str(datetime.datetime.now().minute), str(iter))
        print("saved!")

