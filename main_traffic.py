from math import ceil
import numpy as np
import torch
from common import util
from environment.extend_traffic import ExtendTraffic
from rl_algo import SAC
from discriminator import Discriminator
from tensorboardX import SummaryWriter
from tqdm import tqdm
import copy
import datetime
import os
import pandas as pd
import argparse


loc_list = ["Lewisham", "Hammersmith", "Ealing", "Redbridge", "Enfield", "Ben"]
Lewisham = pd.read_csv("traffic_data/Lewisham.csv")
Hammersmith = pd.read_csv("traffic_data/Hammersmith and Fulham.csv")
Ealing = pd.read_csv("traffic_data/Ealing.csv")
Redbridge = pd.read_csv("traffic_data/Redbridge.csv")
Enfield = pd.read_csv("traffic_data/Enfield.csv")
Ben = pd.read_csv("traffic_data/big.csv")


def relabel(x, loc_list=loc_list):
    for i in loc_list:
        if i in x:
            return loc_list.index(i)


def read_data(df, ori_loc, loc_list):
    full = [df[df["Destination Display Name"].str.contains(i)][["Origin Display Name", "Destination Display Name"]] for i in loc_list if not i == ori_loc]
    full = pd.concat(full).reset_index(drop=True)
    full["Destination Display Name"] = full["Destination Display Name"].map(relabel)
    full["Origin Display Name"] = full["Origin Display Name"].map(relabel)
    # df[df["Destination Display Name"].str.contains(des_loc)][["Origin Display Name", "Destination Display Name"]]
    return full


def save(save_path, policy, name, num):
    base_path = os.path.join(save_path, 'traffic_data1' + num)
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    model_critic_path = os.path.join(base_path, name + ".pth")
    torch.save(policy, model_critic_path)
    print("successfully saved at " + str(model_critic_path))

# print([i for i in loc_list if not i == loc_list[3]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', default="Lewisham", type=str)
    args = parser.parse_args()

    print("================== args: ", args)
    data0 = read_data(eval(args.city), args.city, loc_list=loc_list)
    use_wandb = True
    dir = "config/extend_traffic.yaml"
    config_dict = util.load_config(dir)
    paras = util.get_paras_from_dict(config_dict)
    for iteration_count in range(3):
        run_dir, log_dir = util.make_logpath('traffic', args.city)
        writer = SummaryWriter(str(run_dir))
        agent = SAC(paras)
        env = ExtendTraffic(1, loc_list.index(args.city))
        episode = 10000
        data_length = ceil(episode / 10)
        mf00 = np.zeros(data_length)
        mf01 = np.zeros(data_length)
        mf02 = np.zeros(data_length)
        mf03 = np.zeros(data_length)
        mf04 = np.zeros(data_length)
        mf05 = np.zeros(data_length)
        discriminator = Discriminator(paras)
        obs = env.reset()
        Gt = 0
        done = False
        signal = np.random.choice(np.arange(paras.signal_dim), p=agent.rho_sample())
        obs = np.concatenate((obs, [signal], [env.count]))
        for _ in tqdm(range(episode)):
            for con in range(len(data0)):
                while not done:
                    action = agent.choose_action(obs)
                    action.update({"signal": signal})
                    next_obs, reward, done, no = env.step(action)
                    expert_action = data0.iloc[con].values
                    expert_state = util.wrapper(expert_action[0], env.observation_space.n)
                    expert_action = expert_action[1]
                    expert_data = np.concatenate([expert_state, [0.0], [env.count - 1], [expert_action]])
                    d_score = discriminator.get_reward(obs, action["action"], expert_data)
                    signal = np.random.choice(np.arange(paras.signal_dim), p=agent.rho_sample())
                    next_obs = np.concatenate((next_obs, [signal], [env.count]))
                    agent.add_experience(
                        {"states": obs, "states_next": next_obs, "rewards": d_score, "dones": np.float32(done)})
                    obs = next_obs
                obs = env.reset()
                done = False
                signal = np.random.choice(np.arange(paras.signal_dim), p=agent.rho_sample())
                obs = np.concatenate((obs, [signal], [env.count]))
            discriminator.update()
            agent.learn()
            if _ % 10 == 0:
                test_obs = copy.deepcopy(obs)
                test_obs[1:6] = 0.0
                test_obs[0] = 1.0
                test_action = agent.choose_action(obs, train=False)['prob']
                writer.add_scalars('policy', global_step=_,
                                tag_scalar_dict={"policy0": test_action[0],
                                                    "policy1": test_action[1],
                                                    "policy2": test_action[2],
                                                    "policy3": test_action[3],
                                                    "policy4": test_action[4],
                                                    "policy5": test_action[5]})
                writer.add_scalars("d_loss", global_step=_,
                                    tag_scalar_dict={"true_loss": discriminator.true_loss,
                                                    "fake_loss": discriminator.fake_loss,
                                                    "gp": discriminator.gp})
                writer.add_scalars("g_loss", global_step=_,
                                    tag_scalar_dict={"a_loss": agent.a_loss,
                                                    "c_loss": agent.c_loss})
                mf00[ceil(_/10)] = test_action[0]
                mf01[ceil(_/10)] = test_action[1]
                mf02[ceil(_/10)] = test_action[2]
                mf03[ceil(_/10)] = test_action[3]
                mf04[ceil(_/10)] = test_action[4]
                mf05[ceil(_/10)] = test_action[5]

        save(run_dir, mf00, "mf00" + " " + args.city + str(datetime.datetime.now().minute), str(iteration_count))
        save(run_dir, mf01, "mf01" + " " + args.city + str(datetime.datetime.now().minute), str(iteration_count))
        save(run_dir, mf02, "mf02" + " " + args.city + str(datetime.datetime.now().minute), str(iteration_count))
        save(run_dir, mf03, "mf03" + " " + args.city + str(datetime.datetime.now().minute), str(iteration_count))
        save(run_dir, mf04, "mf04" + " " + args.city + str(datetime.datetime.now().minute), str(iteration_count))
        save(run_dir, mf05, "mf05" + " " + args.city + str(datetime.datetime.now().minute), str(iteration_count))
        print("saved!")
