import numpy as np
import os
import sys
import time

from gym.wrappers import TimeLimit
from typing import Callable
from collections import deque

from imitation_Nao_env import robotisImitationEnv
from utils.schedules import LinearDecay, ExponentialSchedule
# from utils.logger import Original_Logger

from config import hyperparameter as cfg


def sb3_PPO():
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3 import PPO

    # Use the enviroment of khr3hv Robot
    nao_env = robotisImitationEnv(agent_timestep=40)

    # max_episode_steps = sys.maxsize のときは環境からのリセットのみで動作 →　infinity Task
    # PPO の場合, max_episode_steps = 1000に設定されることが多い
    nao_env_timelimit = TimeLimit(nao_env, max_episode_steps=cfg.ep_dur_max)

    if cfg.train:

        # random_seedの固定(python, numpy, torch)
        set_random_seed(seed=cfg.seed, using_cuda=cfg.using_cuda)

        model = PPO("MlpPolicy", nao_env_timelimit, policy_kwargs=cfg.policy_kwargs, learning_rate=cfg.lr, n_steps=cfg.n_steps,
                    batch_size=cfg.batch_size, n_epochs=cfg.n_epochs, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
                    clip_range=cfg.clip_param, ent_coef=cfg.ent_coef, tensorboard_log=cfg.tensorboard_log_dir, verbose=2)
        model.learn(total_timesteps=cfg.total_timesteps)
        model.save("10_milion_step_MLP")
    else:
        model = PPO.load("10_milion_step_MLP.zip")


        for i in range(10):
            obs = nao_env_timelimit.reset()

            episode_score = 0
            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, _ = nao_env_timelimit.step(action)
                episode_score += reward

                if done:
                    print("socre", episode_score)
                    break


def sb3_RecurrentPPO():
    from stable_baselines3.common.utils import set_random_seed
    from sb3_contrib import RecurrentPPO

    # Use the enviroment of khr3hv Robot
    nao_env = robotisImitationEnv(agent_timestep=None)

    # max_episode_steps = sys.maxsize のときは環境からのリセットのみで動作 →　infinity Task
    # PPO の場合, max_episode_steps = 1000に設定されることが多い
    nao_env_timelimit = TimeLimit(nao_env, max_episode_steps=cfg.ep_dur_max)

    if cfg.train:

        # random_seedの固定(python, numpy, torch)
        set_random_seed(seed=cfg.seed, using_cuda=cfg.using_cuda)

        model = RecurrentPPO("MlpLstmPolicy", nao_env_timelimit, policy_kwargs=cfg.policy_kwargs, learning_rate=cfg.lr, n_steps=cfg.n_steps,
                             batch_size=cfg.batch_size, n_epochs=cfg.n_epochs, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
                             clip_range=cfg.clip_param, ent_coef=cfg.ent_coef, tensorboard_log=cfg.tensorboard_log_dir, verbose=2)
        model.learn(total_timesteps=cfg.total_timesteps)
        model.save("10_milion_step_RNN")


if __name__ == '__main__':
    # sb3_RecurrentPPO()
    sb3_PPO()
