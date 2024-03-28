import numpy as np
import torch
from environment.fish import Fish
from common import util
from tqdm import tqdm
import os
from MFIRL import MFIRL
from rl_algo import SAC
from common import util
import datetime
import signatory
import copy


def get_trajectories(size=100):
    env = Fish(1)
    states = np.zeros((size, env.horizon, 4))
    actions = np.zeros((size, env.horizon, env.action_space.n))
    Gt = 0
    done = False
    # t = 0
    for j in range(size):
        for i in range(env.horizon):
            state = util.wrapper(np.random.choice([0, 1], p=np.ones(env.observation_space.n) / env.observation_space.n), env.observation_space.n)
            signal = np.random.choice([0, 1], p=[0.5, 0.5])
            states[j][i] = np.concatenate((state, [signal], [env.count]))
            action = np.random.choice([0, 1], p=[3/4, 1/4]) if signal == 0 else np.random.choice([0, 1], p=[1/3, 2/3])
            actions[j][i] = util.wrapper(action, env.action_space.n)
    return states, actions


def save(save_path, policy, name):
    base_path = os.path.join(save_path, 'MFIRL')
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    model_critic_path = os.path.join(base_path, name + ".pth")
    torch.save(policy, model_critic_path)
    print("successfully saved at "+str(model_critic_path))


if __name__ == "__main__":
    units = 64
    learning_rate = 0.0001
    max_epoch = 1070
    size = 100
    run_dir, log_dir = util.make_logpath('fish', 'MFIRL')
    data_length = 10000
    for i in range(3):
        mf0_t0_data = np.zeros(data_length)
        mf1_t0_data = np.zeros(data_length)
        mf0_t1_data = np.zeros(data_length)
        mf1_t1_data = np.zeros(data_length)
        states, actions = get_trajectories(size)
        mf = np.sum(states, axis=0)[:,:2]
        mf /= size
        agent = MFIRL(4, 2, 1, max_epoch=300, mf_dim=2)
        agent.learn(states, mf, actions)
        reward_recover = agent.reward_model.eval()
        dir = "config/fish.yaml"
        config_dict = util.load_config(dir)
        paras = util.get_paras_from_dict(config_dict)
        RL_agent = SAC(paras)
        env = Fish(1)
        obs = env.reset()
        signal = np.random.choice(np.arange(paras.signal_dim), p=RL_agent.rho_sample())
        obs = np.concatenate((obs, [signal], [env.count]))
        Gt = 0
        done = False
        policy = []
        for _ in tqdm(range(10000)):
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
            obs_test[-1] = 0
            obs_test[-2] = 0
            mf0_t0_data[_] = RL_agent.choose_action(obs_test, train=False)['prob'][0]
            obs_test[-2] = 1
            mf1_t0_data[_] = RL_agent.choose_action(obs_test, train=False)['prob'][0]
        save(run_dir, mf1_t0_data, "mf1_t0_data" + str(datetime.datetime.now().minute))
        save(run_dir, mf0_t0_data, "mf0_t0_data" + str(datetime.datetime.now().minute))
        print("saved!")
