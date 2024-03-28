import numpy as np
import torch
from environment.SS_signature import Sequential_Squeeze_sign
from common import util
from tqdm import tqdm
import os
from MFAIRL import MFAIRL
from rl_algo import SAC
from common import util
import datetime
import signatory
import copy


def get_trajectories(size=100):
    env = Sequential_Squeeze_sign()
    states = np.zeros((size, env.horizon, 7))
    actions = np.zeros((size, env.horizon, env.action_space.n))
    Gt = 0
    done = False
    # t = 0
    for j in range(size):
        sign = np.zeros((1, 3))
        his = np.array([-1])
        for i in range(env.horizon):
            if i == 0:
                state = util.wrapper(np.random.choice([0, 1], p=np.ones(env.observation_space.n) / env.observation_space.n), env.observation_space.n)
            else:
                if int(action) == 0:
                    state = np.random.choice([0, 1], p=[3 / 4, 1 / 4])
                else:
                    state = np.random.choice([0, 1], p=[1 / 4, 3 / 4])
                state = util.wrapper(state, env.observation_space.n)
            state = np.concatenate((state, np.array(sign.squeeze())))
            signal = np.random.choice([0, 1], p=[0.5, 0.5])
            states[j][i] = np.concatenate((state, [signal], [env.count]))
            if i == 0:
                action = np.random.choice([0, 1], p=[2/3, 1/3]) if signal == 0 else np.random.choice([0, 1], p=[1/3, 2/3])
            else:
                action = 0 if signal == 0 else 1
            actions[j][i] = util.wrapper(action, env.action_space.n)
            his = np.concatenate((his, [signal]))
            sign = signatory.signature(torch.from_numpy(his).type(torch.FloatTensor).view(1, -1, 1), 3)
    return states, actions


def get_agent_trajectories(agent, size=100):
    env = Sequential_Squeeze_sign()
    states = np.zeros((size, env.horizon, 7))
    actions = np.zeros((size, env.horizon, env.action_space.n))
    policies = np.zeros_like(actions)
    Gt = 0
    done = False
    # t = 0
    for j in range(size):
        sign = np.zeros((1, 3))
        his = np.array([-1])
        for i in range(env.horizon):
            if i == 0:
                state = util.wrapper(np.random.choice([0, 1], p=np.ones(env.observation_space.n) / env.observation_space.n), env.observation_space.n)
            else:
                if int(action['action']) == 0:
                    state = np.random.choice([0, 1], p=[3 / 4, 1 / 4])
                else:
                    state = np.random.choice([0, 1], p=[1 / 4, 3 / 4])
                state = util.wrapper(state, env.observation_space.n)
            state = np.concatenate((state, np.array(sign.squeeze())))
            signal = np.random.choice([0, 1], p=[0.5, 0.5])
            states[j][i] = np.concatenate((state, [signal], [env.count]))
            action = RL_agent.choose_action(states[j][i], train=False)
            actions[j][i] = util.wrapper(action['action'], env.action_space.n)
            policies[j][i] = action['prob']
            his = np.concatenate((his, [signal]))
            sign = signatory.signature(torch.from_numpy(his).type(torch.FloatTensor).view(1, -1, 1), 3)
    return states, actions, policies


def save(save_path, policy, name):
    base_path = os.path.join(save_path, 'MFAIRL')
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    model_critic_path = os.path.join(base_path, name + ".pth")
    torch.save(policy, model_critic_path)
    print("successfully saved at "+str(model_critic_path))


if __name__ == "__main__":
    size = 3
    run_dir, log_dir = util.make_logpath('SS_sign', 'MFAIRL')
    data_length = 10000
    for i in range(3):
        mf0_t0_data = np.zeros(data_length)
        mf1_t0_data = np.zeros(data_length)
        mf0_t1_data = np.zeros(data_length)
        mf1_t1_data = np.zeros(data_length)
        agent = MFAIRL(7, 2, 2, max_epoch=200, mf_dim=2)
        reward_recover = agent.reward_model.eval()
        dir = "config/SS_sign.yaml"
        config_dict = util.load_config(dir)
        paras = util.get_paras_from_dict(config_dict)
        RL_agent = SAC(paras)
        env = Sequential_Squeeze_sign()
        obs = env._reset()
        signal = np.random.choice(np.arange(paras.signal_dim), p=[0.5, 0.5])
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
            mf = np.sum(states, axis=0)[:,:2]
            mf /= size
            agent.learn(states, mf, actions)
            states, actions, policies = get_agent_trajectories(RL_agent, size=100)
            mf = np.sum(states, axis=0)[:,:2]
            mf /= size
            agent.learn(states, mf, actions, policy_logit=policies, expert_data=False)
            reward_recover = agent.reward_model.eval()
            obs_test = copy.deepcopy(obs)
            obs_test[-1] = 0
            obs_test[-2] = 0
            mf0_t0_data[_] = RL_agent.choose_action(obs_test, train=False)['prob'][0]
            obs_test[-2] = 1
            mf1_t0_data[_] = RL_agent.choose_action(obs_test, train=False)['prob'][0]
            obs_test[-1] = 1
            mf1_t1_data[_] = RL_agent.choose_action(obs_test, train=False)['prob'][0]
            obs_test[-2] = 0
            mf0_t1_data[_] = RL_agent.choose_action(obs_test, train=False)['prob'][0]
        save(run_dir, mf1_t1_data, "mf1_t1_data" + str(datetime.datetime.now().minute))
        save(run_dir, mf1_t0_data, "mf1_t0_data" + str(datetime.datetime.now().minute))
        save(run_dir, mf0_t1_data, "mf0_t1_data" + str(datetime.datetime.now().minute))
        save(run_dir, mf0_t0_data, "mf0_t0_data" + str(datetime.datetime.now().minute))
        print("saved!")