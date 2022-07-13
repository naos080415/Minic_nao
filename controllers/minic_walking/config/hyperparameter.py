train = True  # Train or evaluate the model
use_wandb = True    # wandb(log管理ツール)と連携するか

# Change the neural net size or the activation function
policy_kwargs = dict(  # activation_fn=th.nn.ReLU,
    net_arch=[dict(pi=[64, 64], vf=[64, 64])])

# Earyly Termination
ep_dur_max = 1000

"""
PPOのときのパラメータ
"""

n_steps = 2048        # default value: 2048
batch_size = 128      # default value: 64
n_epochs = 10         # default value: 10
lr = 1e-4             # default value: 3e-4
gamma = 0.99          # default value: 0.99
gae_lambda = 0.95     # defalut value: 0.95
clip_param = 0.1      # default value: 0.2
ent_coef = 0.0
seed = 42
