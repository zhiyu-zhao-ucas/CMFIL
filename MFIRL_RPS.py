import numpy as np
import torch
from environment.RPS import RPS
from common import util
from tqdm import tqdm
import os
from MFIRL import MFIRL
from rl_algo import SAC
from common import util
import datetime


def get_trajectories(size=100):
    env = RPS(10)
    state = np.zeros((size, env.horizon, env.observation_space.n))
    actions = np.zeros((size, env.horizon, env.action_space.n))
    obs = env.reset()
    state[0][0] = obs
    Gt = 0
    done = False
    t = 0
    A = np.matrix([[4, 1, -4], [7, -6, -2], [1, 1, 1]])
    b = np.matrix([[0], [0], [1]])
    dist = np.array(A.I @ b).T.squeeze()
    for j in range(size):
        while not done:
            action = np.random.choice([0, 1, 2], p=dist)
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
    run_dir, log_dir = util.make_logpath('RPS', 'MFIRL')
    datalength = 10000
    for i in range(3):
        mf0 = torch.zeros(datalength)
        mf1 = torch.zeros(datalength)
        mf2 = torch.zeros(datalength)
        states, actions = get_trajectories(size)
        mf = np.sum(states, axis=0)
        den = np.sum(mf, axis=1)
        mf /= size
        agent = MFIRL(3, 3, 10, max_epoch=200)
        agent.learn(states, mf, actions)
        reward_recover = agent.reward_model.eval()
        dir = "config/fish_MFIRL.yaml"
        config_dict = util.load_config(dir)
        paras = util.get_paras_from_dict(config_dict)
        RL_agent = SAC(paras)
        env = RPS(10)
        obs = env.reset()
        Gt = 0
        done = False
        policy = []
        for _ in tqdm(range(datalength)):
            iter = 0
            while not done:
                iter += 1
                action = RL_agent.choose_action(obs)
                next_obs, reward, done, info = env.step(action)
                reward = reward_recover(torch.tensor(obs, dtype=torch.float), torch.tensor([util.wrapper(action['action'], env.action_space.n)], dtype=torch.float), torch.tensor(env.mean_field, dtype=torch.float)).detach().numpy()
                mf0[_] += action['prob'][0]
                mf1[_] += action['prob'][1]
                mf2[_] += action['prob'][2]
                # policy.append(action['prob'][0])
                RL_agent.add_experience(
                    {"states": obs, "states_next": next_obs, "rewards": reward, "dones": np.float32(done)})
                obs = next_obs
                Gt += reward
            mf0[_] /= iter
            mf1[_] /= iter
            mf2[_] /= iter
            obs = env.reset()
            done = False
            Gt = 0
            RL_agent.learn()
            # state = np.random.choice([0, 1, 2], p=np.ones(3) / 3)
            # action = RL_agent.choose_action(state, False)
        # policies.append(policy)
        save(run_dir, mf0, "mf0" + str(datetime.datetime.now().microsecond))
        save(run_dir, mf1, "mf1" + str(datetime.datetime.now().microsecond))
        save(run_dir, mf2, "mf2" + str(datetime.datetime.now().microsecond))
    # policies = np.array(policies)
