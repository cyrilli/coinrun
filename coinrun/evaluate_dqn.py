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
    print("load path:")
    print("{}/saved_models/{}.pkl".format(Config.SAVE_PATH, Config.RUN_ID))
    act = deepq.learn(
        env,
        network="conv_only",
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        total_timesteps=0,
        # load_path="{}/saved_models/{}.pkl".format(Config.SAVE_PATH, Config.RUN_ID)
        load_path="{}/ckpts/{}/model".format(Config.SAVE_PATH, Config.RUN_ID)
    )

    num_episodes = 500
    # while True:
    episode_rew_ls = []
    for i in range(num_episodes):
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            if Config.RENDER:
                env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        episode_rew_ls.append(episode_rew)
        print("Episode reward", episode_rew)
    print("Avg episode reward", np.mean(episode_rew_ls))
    print("Var episode reward", np.std(episode_rew_ls))
if __name__ == '__main__':
    main()