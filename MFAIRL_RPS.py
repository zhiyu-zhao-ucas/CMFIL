import numpy as np
import torch
from environment.RPS import RPS
from common import util
from tqdm import tqdm
import os
from MFAIRL import MFAIRL
from rl_algo import SAC
from common import util
import datetime


def get_trajectories(size=100):
    env = RPS(10)
    state = np.zeros((size, env.horizon, env.observation_space.n))
    actions = np.zeros((size, env.horizon, env.action_space.n))
    obs = env._reset()
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
            action.update({"signal": 1})
            action.update({"prob": dist})
            next_obs, reward, done, info = env._step(action)
            if t+1 < env.horizon:
                state[j][t+1] = next_obs
            signal = np.random.choice([0, 1], p=[0.5, 0.5])
            next_obs = np.concatenate((next_obs, [signal]))
            obs = next_obs
            Gt += reward
            t += 1
        obs = env._reset()
        t = 0
        state[j][0] = obs
        done = False
    return state, actions


def get_trajectories_from_agent(RL_agent, num_trajectories=1000):
    env = RPS(10)
    # Initialize arrays to store trajectories
    num_actions = env.action_space.n
    num_states = env.observation_space.n
    states = np.zeros((num_trajectories, env.horizon, num_states))
    actions = np.zeros((num_trajectories, env.horizon, num_actions))
    policies = np.zeros((num_trajectories, env.horizon, num_actions))

    for traj_idx in tqdm(range(num_trajectories)):
        obs = env._reset()
        done = False
        Gt = 0
        iter_count = 0

        while not done:
            iter_count += 1
            action = RL_agent.choose_action(obs, train=False)  # Get action from the RL agent
            next_obs, reward, done, info = env._step(action)
            reward = reward_recover(torch.tensor(obs, dtype=torch.float), 
                                    torch.tensor([util.wrapper(action['action'], num_actions)], dtype=torch.float), 
                                    torch.tensor(env.mean_field, dtype=torch.float)).detach().numpy()
            
            
            # Store trajectory data
            states[traj_idx, iter_count-1] = obs
            actions[traj_idx, iter_count-1] = util.wrapper(action['action'], num_actions)
            policies[traj_idx, iter_count-1] = action['prob']
            
            # Add experience to the RL agent
            RL_agent.add_experience({
                "states": obs,
                "states_next": next_obs,
                "rewards": reward,
                "dones": np.float32(done)
            })
            
            obs = next_obs

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
    run_dir, log_dir = util.make_logpath('RPS', 'MFAIRL')
    for i in range(3):
        mf0 = torch.zeros(1000)
        mf1 = torch.zeros(1000)
        mf2 = torch.zeros(1000)
        agent = MFAIRL(3, 3, 10, max_epoch=100)
        reward_recover = agent.reward_model.eval()
        dir = "config/fish_MFAIRL.yaml"
        config_dict = util.load_config(dir)
        paras = util.get_paras_from_dict(config_dict)
        RL_agent = SAC(paras)
        env = RPS(10)
        obs = env._reset()
        Gt = 0
        done = False
        policy = []
        for _ in tqdm(range(1000)):
            iter = 0
            while not done:
                iter += 1
                action = RL_agent.choose_action(obs)
                next_obs, reward, done, info = env._step(action)
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
            obs = env._reset()
            done = False
            Gt = 0
            RL_agent.learn()
            states, actions = get_trajectories(size)
            mf = np.sum(states, axis=0)
            den = np.sum(mf, axis=1)
            mf /= size
            agent.learn(states, mf, actions)
            states1, actions1, policies1 = get_trajectories_from_agent(RL_agent, num_trajectories=100)
            mf1 = np.sum(states1, axis=0)
            den = np.sum(mf1, axis=1)
            mf1 /= size
            agent.learn(states1, mf1, actions1, policy_logit=policies1, expert_data=False)
        save(run_dir, mf0, "mf0" + str(datetime.datetime.now().microsecond))
        save(run_dir, mf1, "mf1" + str(datetime.datetime.now().microsecond))
        save(run_dir, mf2, "mf2" + str(datetime.datetime.now().microsecond))
