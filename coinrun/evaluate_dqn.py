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

def main():
    args = setup_utils.setup_and_load()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    seed = int(time.time()) % 10000
    set_global_seeds(seed * 100 + rank)

    main_utils.setup_mpi_gpus()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    env = main_utils.Scalarize(main_utils.make_general_env(1, seed=rank))
    act = deepq.learn(
        env,
        network=Config.ARCHITECTURE,
        total_timesteps=0,
        load_path="./saved_models/{}.pkl".format(Config.RUN_ID)
    )

    num_episodes = 100
    # while True:
    for i in range(num_episodes):
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            if Config.RENDER:
                env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)

if __name__ == '__main__':
    main()