import os
import sys
import datetime
import torch as th
import torch.nn

train = True        # Train or evaluate the model
use_wandb = True    # wandb(log管理ツール)と連携するか
using_cuda = False

tensorboard_log_dir = os.environ['HOME'] + '/.cache/tensorboard'
now_time = datetime.datetime.now()
model_dir = os.environ['HOME'] + '/.results/' + str(now_time.year) + "-" + str(now_time.month) + "-" + str(
    now_time.day) + " " + str(now_time.hour) + ":" + str(now_time.minute) + "/"  # Output dir


# Change the neural net size or the activation function
# 初期値: net_arch=[dict(pi=[64, 64], vf=[64, 64])] using tanh function
policy_kwargs = dict(  
    activation_fn=th.nn.Tanh,
    # activation_fn=th.nn.ReLU,
    net_arch=[dict(pi=[64, 64], vf=[64, 64])])
    # net_arch=[dict(pi=[128, 128], vf=[128, 128])])

# Earyly Termination
if train == True:
    ep_dur_max = 500
else:
    ep_dur_max = sys.maxsize


if use_wandb and train:
    import wandb

    projectName = 'AROB'

    # Init the wandb providing the project name
    wandb_cache_dir = os.environ['HOME'] + '/.cache/'

    # dir : cacheファイルの保存場所
    wandb.init(project=projectName, dir=wandb_cache_dir,
               sync_tensorboard=True)

"""
PPOのときのパラメータ
"""
# verbose – (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug 

total_timesteps = int(30.5e6)
n_steps = 2048        # default value: 2048
batch_size = 128      # default value: 64
n_epochs = 10         # default value: 10
lr = 1e-4             # default value: 3e-4
gamma = 0.99          # default value: 0.99
gae_lambda = 0.95     # defalut value: 0.95
clip_param = 0.1      # default value: 0.2
ent_coef = 0.0
seed = 42
verbose = 0
