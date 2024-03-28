from math import ceil, floor
import numpy as np
import torch
from environment.fish import Fish
from common import util
from rl_algo import SAC
from discriminator import Discriminator
from tqdm import tqdm
import copy
import os


def save(save_path, policy, name):
    base_path = os.path.join(save_path, 'fish_wgangp_data2')
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    model_critic_path = os.path.join(base_path, name + ".pth")
    torch.save(policy, model_critic_path)
    print("successfully saved at "+str(model_critic_path))


if __name__ == '__main__':
    for iterations_num in range(3): 
        use_wandb = True
        dir = "config/fish.yaml"
        config_dict = util.load_config(dir)
        paras = util.get_paras_from_dict(config_dict)
        print('local:', paras)
        print('finetune: ', paras)
        run_dir, log_dir = util.make_logpath('fish', 'new')
        agent = SAC(paras)
        env = Fish(2)
        episode = 10000
        data_length = ceil(episode/10)
        mf0_t0_data = np.zeros(data_length)
        mf1_t0_data = np.zeros(data_length)
        mf0_t1_data = np.zeros(data_length)
        mf1_t1_data = np.zeros(data_length)
        rho_data = np.zeros(data_length)
        discriminator = Discriminator(paras)
        ret = []
        obs = env.reset()
        Gt = 0
        done = False
        signal = np.random.choice(np.arange(paras.signal_dim), p=agent.rho_sample())
        obs = np.concatenate((obs, [signal], [env.count]))
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
            data_id = floor(_ / 10)
            if _ % 10 == 0:
                rho_data[data_id] = agent.rho_sample()[0]
                obs_test = copy.deepcopy(obs)
                obs_test[-2] = 0
                mf0_t0_data[data_id] = agent.choose_action(obs_test, train=False)['prob'][0]
                obs_test[-2] = 1
                mf1_t0_data[data_id] = agent.choose_action(obs_test, train=False)['prob'][0]
        save(run_dir, mf1_t0_data, "mf1_t0_data" + str(iterations_num))
        save(run_dir, mf0_t0_data, "mf0_t0_data" + str(iterations_num))
        save(run_dir, rho_data, "rho_data" + str(iterations_num))
        print("saved!")

