import numpy as np
import wandb
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

    projectName = 'Minic_nao'

    if cfg.train:
        # Init the wandb providing the project name
        wandb_cache_dir = os.environ['HOME'] + '/.cache/'

        if cfg.use_wandb:
            # dir : cacheファイルの保存場所
            wandb.init(project=projectName, dir=wandb_cache_dir,
                       sync_tensorboard=True)

        # Change the neural net size or the activation function
        policy_kwargs = dict(  # activation_fn=th.nn.ReLU,
            net_arch=[dict(pi=[64, 64], vf=[64, 64])])

        tensorboard_log_dir = os.environ['HOME'] + '/.cache/tensorboard'

        # random_seedの固定(python, numpy, torch)
        using_cuda = False
        set_random_seed(seed=42, using_cuda=using_cuda)

        model = PPO("MlpPolicy", nao_env_timelimit, learning_rate=cfg.lr, n_steps=cfg.n_steps,
                    batch_size=cfg.batch_size, n_epochs=cfg.n_epochs, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
                    clip_range=cfg.clip_param, ent_coef=cfg.ent_coef, tensorboard_log=tensorboard_log_dir, verbose=2)
        model.learn(total_timesteps=int(2e7))
        model.save("Walking")


def sb3_RecurrentPPO():
    from stable_baselines3.common.utils import set_random_seed
    from sb3_contrib import RecurrentPPO

    # Use the enviroment of khr3hv Robot
    nao_env = robotisImitationEnv(agent_timestep=None)

    # max_episode_steps = sys.maxsize のときは環境からのリセットのみで動作 →　infinity Task
    # PPO の場合, max_episode_steps = 1000に設定されることが多い
    nao_env_timelimit = TimeLimit(nao_env, max_episode_steps=cfg.ep_dur_max)

    projectName = 'Minic_nao'

    if cfg.train:
        # Init the wandb providing the project name
        wandb_cache_dir = os.environ['HOME'] + '/.cache/'

        if cfg.use_wandb:
            # dir : cacheファイルの保存場所
            wandb.init(project=projectName, dir=wandb_cache_dir,
                       sync_tensorboard=True)

        # Change the neural net size or the activation function
        policy_kwargs = dict(  # activation_fn=th.nn.ReLU,
            net_arch=[dict(pi=[64, 64], vf=[64, 64])])

        tensorboard_log_dir = os.environ['HOME'] + '/.cache/tensorboard'

        # random_seedの固定(python, numpy, torch)
        using_cuda = False
        set_random_seed(seed=42, using_cuda=using_cuda)

        model = RecurrentPPO("MlpLstmPolicy", nao_env_timelimit, learning_rate=cfg.lr, n_steps=cfg.n_steps,
                             batch_size=cfg.batch_size, n_epochs=cfg.n_epochs, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
                             clip_range=cfg.clip_param, ent_coef=cfg.ent_coef, tensorboard_log=tensorboard_log_dir, verbose=2)
        model.learn(total_timesteps=int(2e7))
        model.save("Walking")


if __name__ == '__main__':
    sb3_RecurrentPPO()
    # sb3_PPO()
