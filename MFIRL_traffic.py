import numpy as np
import torch
from environment.extend_traffic import ExtendTraffic
from tqdm import tqdm
import os
from MFIRL import MFIRL
from rl_algo import SAC
from common import util
import pandas as pd
import argparse


loc_list = ["Lewisham", "Hammersmith", "Ealing", "Redbridge", "Enfield", "Ben"]
Lewisham = pd.read_csv("Lewisham.csv")
Hammersmith = pd.read_csv("Hammersmith and Fulham.csv")
Ealing = pd.read_csv("Ealing.csv")
Redbridge = pd.read_csv("Redbridge.csv")
Enfield = pd.read_csv("Enfield.csv")
Ben = pd.read_csv("big.csv")

def get_trajectories(city_index, size=100):
    env = ExtendTraffic(1, city_index)
    state = np.zeros((size, env.horizon, env.observation_space.n))
    actions = np.zeros((size, env.horizon, env.action_space.n))
    obs = env.reset()
    state[0][0] = obs
    Gt = 0
    done = False
    t = 0
    dist = np.zeros(6)
    data0 = read_data(eval(args.city), args.city, loc_list=loc_list)
    for id, prop in zip(list(data0.iloc[:, -1].value_counts(normalize=True).index), list(data0.iloc[:, -1].value_counts(normalize=True).values)):
        dist[id] = prop
    for j in range(size):
        while not done:
            action = np.random.choice([0, 1, 2, 3, 4, 5], p=dist)
            actions[j][t] = util.wrapper(action, env.action_space.n)
            action = {'action': action}
            action.update({"signal": 0})
            action.update({"prob": dist})
            next_obs, reward, done, info = env.step(action)
            if t+1 < env.horizon:
                state[j][t+1] = next_obs
            signal = 0
            next_obs = np.concatenate((next_obs, [signal]))
            obs = next_obs
            Gt += reward
            t += 1
        obs = env.reset()
        t = 0
        state[j][0] = obs
        done = False
    return state, actions

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


def save(save_path, policy, name):
    base_path = os.path.join(save_path, 'MFIRL')
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    model_critic_path = os.path.join(base_path, name + ".pth")
    torch.save(policy, model_critic_path)
    print("successfully saved at "+str(model_critic_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', default="Ealing", type=str)
    args = parser.parse_args()

    print("================== args: ", args)
    data0 = read_data(eval(args.city), args.city, loc_list=loc_list)
    use_wandb = True
    dir = "config/extend_traffic.yaml"
    config_dict = util.load_config(dir)
    units = 64
    learning_rate = 0.0001
    max_epoch = 1070
    size = 100
    run_dir, log_dir = util.make_logpath('extend_traffic', 'MFIRL')
    datalength = 10000
    for i in range(3):
        mf0 = np.zeros(datalength)
        mf1 = np.zeros(datalength)
        mf2 = np.zeros(datalength)
        mf3 = np.zeros(datalength)
        mf4 = np.zeros(datalength)
        mf5 = np.zeros(datalength)
        states, actions = get_trajectories(city_index=loc_list.index(args.city), size=size)
        mf = np.sum(states, axis=0)
        den = np.sum(mf, axis=1)
        mf /= size
        agent = MFIRL(6, 6, 1, max_epoch=300, mf_dim=6)
        agent.learn(states, mf, actions)
        reward_recover = agent.reward_model.eval()
        
        dir = "config/extend_traffic.yaml"
        config_dict = util.load_config(dir)
        paras = util.get_paras_from_dict(config_dict)
        RL_agent = SAC(paras)
        env = ExtendTraffic(1, loc_list.index(args.city))
        obs = env.reset()
        obs = np.concatenate([obs, [0], [env.count]])
        Gt = 0
        done = False
        policy = []
        signal = 0
        for _ in tqdm(range(datalength)):
            iter = 0
            while not done:
                iter += 1
                action = RL_agent.choose_action(obs)
                next_obs, reward, done, info = env.step(action)
                reward = reward_recover(torch.tensor(obs[:6], dtype=torch.float), torch.tensor([util.wrapper(action['action'], env.action_space.n)], dtype=torch.float), torch.tensor(env.mean_field, dtype=torch.float)).detach().numpy()
                mf0[_] = action['prob'][0]
                mf1[_] = action['prob'][1]
                mf2[_] = action['prob'][2]
                mf3[_] = action['prob'][3]
                mf4[_] = action['prob'][4]
                mf5[_] = action['prob'][5]
                next_obs = np.concatenate((next_obs, [signal], [env.count]))
                RL_agent.add_experience(
                    {"states": obs, "states_next": next_obs, "rewards": reward, "dones": np.float32(done)})
                obs = next_obs
                Gt += reward
            obs = env.reset()
            obs = np.concatenate([obs, [0], [env.count]])
            done = False
            Gt = 0
            RL_agent.learn()
        save(run_dir, mf0, "mf0" + str(i) + args.city)
        save(run_dir, mf1, "mf1" + str(i) + args.city)
        save(run_dir, mf2, "mf2" + str(i) + args.city)
        save(run_dir, mf3, "mf3" + str(i) + args.city)
        save(run_dir, mf4, "mf4" + str(i) + args.city)
        save(run_dir, mf5, "mf5" + str(i) + args.city)
