from math import ceil, floor
import numpy as np
import torch
from environment.SS_signature import Sequential_Squeeze_sign
from common import util
from rl_algo import SAC
from discriminator import Discriminator
from tqdm import tqdm
import copy
import datetime
import os
from tensorboardX import SummaryWriter


def save(save_path, policy, name):
    base_path = os.path.join(save_path, 'SS_sign_time_rho_data')
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    model_critic_path = os.path.join(base_path, name + ".pth")
    torch.save(policy, model_critic_path)
    print("successfully saved at " + str(model_critic_path))



if __name__ == '__main__':
    for iteration in range(3):
        use_wandb = False
        dir = "config/SS_sign.yaml"
        config_dict = util.load_config(dir)
        paras = util.get_paras_from_dict(config_dict)
        run_dir, log_dir = util.make_logpath('Sequtial_Squeeze', "MFIL")
        # run_dir, log_dir = util.make_logpath('SS_sign', 'gp')
        writer = SummaryWriter(str(run_dir))
        print(run_dir)
        agent = SAC(paras)
        env = Sequential_Squeeze_sign()
        episode = 10000
        data_length = ceil(episode / 10)
        mf0_t0_data = np.zeros(data_length)
        mf1_t0_data = np.zeros(data_length)
        mf0_t1_data = np.zeros(data_length)
        mf1_t1_data = np.zeros(data_length)
        rho_data0 = np.zeros(data_length)
        rho_data1 = np.zeros(data_length)
        discriminator = Discriminator(paras)
        ret = []
        obs = env.reset()
        Gt = 0
        done = False
        signal = np.random.choice(np.arange(paras.signal_dim), p=agent.rho_sample(float(done)))
        obs = np.concatenate((obs, [signal], [env.count]))
        for _ in tqdm(range(episode)):
            for con in range(64):
                while not done:
                    action = agent.choose_action(obs)
                    action.update({"signal": signal})
                    next_obs, reward, done, expert_data = env.step(action)
                    d_score = discriminator.get_reward(obs, action['action'], expert_data)
                    # print(con, done, agent.rho_sample(float(done)))
                    signal = np.random.choice(np.arange(paras.signal_dim), p=agent.rho_sample(float(done)))
                    next_obs = np.concatenate((next_obs, [signal], [env.count]))
                    agent.add_experience(
                        {"states": obs, "states_next": next_obs, "rewards": d_score, "dones": np.float32(done)})
                    obs = next_obs
                    Gt += reward
                obs = env.reset()
                done = False
                ret.append(Gt)
                Gt = 0
                signal = np.random.choice(np.arange(paras.signal_dim), p=agent.rho_sample(float(done)))
                obs = np.concatenate((obs, [signal], [env.count]))
            discriminator.update()
            agent.learn()
            loss = 0
            if use_wandb:
                if _ % 10 == 0:
                    data_id = floor(_ / 10)
                    rho_data0[data_id] = agent.rho_sample(0.0)[0]
                    rho_data1[data_id] = agent.rho_sample(1.0)[0]
                    obs_test = copy.deepcopy(obs)
                    obs_test[-2] = 0
                    mf0_t0_data[data_id] = agent.choose_action(obs_test, train=False)['prob'][0]
                    obs_test[-2] = 1
                    mf1_t0_data[data_id] = agent.choose_action(obs_test, train=False)['prob'][0]
                    obs_test[-1] = 1
                    mf1_t1_data[data_id] = agent.choose_action(obs_test, train=False)['prob'][0]
                    obs_test[-2] = 0
                    mf0_t1_data[data_id] = agent.choose_action(obs_test, train=False)['prob'][0]
                    writer.add_scalars("mf", global_step=_,
                                       tag_scalar_dict={"t0z0": mf0_t0_data[-1],
                                                        "t0z1": mf0_t1_data[-1],
                                                        "t1z0": mf1_t0_data[-1],
                                                        "t1z1": mf1_t1_data[-1]})
                    writer.add_scalars('d_loss', global_step=_,
                                       tag_scalar_dict={'true_loss': discriminator.true_loss,
                                       'fake_loss': discriminator.fake_loss})
                    # writer.add_scalars('true_loss', global_step=_,
                                        # tag_scalar_dict={'return': discriminator.true_loss})
                    # writer.add_scalars('fake_loss', global_step=_,
                                        # tag_scalar_dict={'return': discriminator.fake_loss})
                    writer.add_scalars('reward', global_step=_,
                                       tag_scalar_dict={'return': d_score.numpy()})
                    writer.add_scalars('a_loss', global_step=_,
                                       tag_scalar_dict={'return': agent.a_loss})
                    writer.add_scalars('c_loss', global_step=_,
                                       tag_scalar_dict={'return': agent.c_loss})
                    writer.add_scalars('rho', global_step=_,
                                       tag_scalar_dict={'rho0': rho_data0[-1],
                                                        'rho1': rho_data1[-1]})
        save(run_dir, mf1_t1_data, "mf1_t1_data" + str(iteration))
        save(run_dir, mf1_t0_data, "mf1_t0_data" + str(iteration))
        save(run_dir, mf0_t1_data, "mf0_t1_data" + str(iteration))
        save(run_dir, mf0_t0_data, "mf0_t0_data" + str(iteration))
        save(run_dir, rho_data0, "rho_data0" + str(iteration))
        save(run_dir, rho_data1, "rho_data1" + str(iteration))
        print("saved!")
