from baselines import deepq
from coinrun import coinrunenv
import time
from mpi4py import MPI
import tensorflow as tf
from baselines.common import set_global_seeds
import coinrun.main_utils as main_utils
from coinrun import setup_utils, policies, wrappers
from coinrun.config_dqn import Config
import numpy as np
import gym

def train():
    args = setup_utils.setup_and_load()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    seed = int(time.time()) % 10000
    set_global_seeds(seed * 100 + rank)

    main_utils.setup_mpi_gpus()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    env = main_utils.Scalarize(main_utils.make_general_env(1, seed=rank))
    print("==================================")
    print("Learning rate :{}, batch size: {}".format(Config.LR, Config.BATCH_SIZE))

    act = deepq.learn(
                    env,
                    # network=Config.ARCHITECTURE,
                    network="conv_only",
                    convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                    hiddens=[256],
                    lr=Config.LR,
                    batch_size=Config.BATCH_SIZE,
                    gamma=0.99,
                    total_timesteps=Config.TOTAL_TIMESTEPS,
                    buffer_size=Config.BUFFER_SIZE,
                    print_freq=10,
                    checkpoint_freq=Config.CHECKPOINT_FREQ,
                    checkpoint_path="{}/ckpts/{}".format(Config.SAVE_PATH, Config.RUN_ID),
                    render=Config.RENDER,
                    callback=None,
                    exploration_fraction=1.0,
                    exploration_final_eps=0.1,
                    prioritized_replay=True,
                    train_freq=4,
                    learning_starts=10000,
                    target_network_update_freq=1000
                    )
    print("Saving model to {}/saved_models".format(Config.SAVE_PATH))
    act.save("{}/saved_models/{}.pkl".format(Config.SAVE_PATH, Config.RUN_ID))

if __name__ == '__main__':
    train()
