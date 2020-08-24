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

class Scalarize:
    """
    Convert a VecEnv into an Env

    There is a minor difference between this and a normal Env, which is that
    the final observation (when done=True) does not exist, and instead you will
    receive the second to last observation a second time due to how the VecEnv
    interface handles resets.  In addition, you are cannot step this
    environment after done is True, since that is not possible for VecEnvs.
    """

    def __init__(self, venv) -> None:
        assert venv.num_envs == 1
        self._venv = venv
        self._waiting_for_reset = True
        self._previous_obs = None
        self.observation_space = self._venv.observation_space
        self.action_space = self._venv.action_space
        self.metadata = self._venv.metadata
        # self.spec = self._venv.spec
        self.reward_range = self._venv.reward_range

    def _process_obs(self, obs):
        if isinstance(obs, dict):
            # dict space
            scalar_obs = {}
            for k, v in obs.items():
                scalar_obs[k] = v[0]
            return scalar_obs
        else:
            return obs[0]

    def reset(self):
        self._waiting_for_reset = False
        obs = self._venv.reset()
        self._previous_obs = obs
        return self._process_obs(obs)

    def step(self, action):
        assert not self._waiting_for_reset
        final_action = action
        if isinstance(self.action_space, gym.spaces.Discrete):
            final_action = np.array([action], dtype=self._venv.action_space.dtype)
        else:
            final_action = np.expand_dims(action, axis=0)
        obs, rews, dones, infos = self._venv.step(final_action)
        if dones[0]:
            self._waiting_for_reset = True
            obs = self._previous_obs
        else:
            self._previous_obs = obs
        return self._process_obs(obs), rews[0], dones[0], infos[0]

    def render(self, mode="human"):
        if mode == "human":
            return self._venv.render(mode=mode)
        else:
            return self._venv.get_images(mode=mode)[0]

    def close(self):
        return self._venv.close()

    def seed(self, seed=None):
        return self._venv.seed(seed)

    @property
    def unwrapped(self):
        # it might make more sense to return the venv.unwrapped here
        # except that the interface is different for a venv so things are unlikely to work
        return self

    def __repr__(self):
        return f"<Scalarize venv={self._venv}>"

def train(print_freq=10):
    args = setup_utils.setup_and_load()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    seed = int(time.time()) % 10000
    set_global_seeds(seed * 100 + rank)

    main_utils.setup_mpi_gpus()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    env = Scalarize(main_utils.make_general_env(1, seed=rank))
    print("==================================")
    print("Learning rate :{}, batch size: {}".format(Config.LR, Config.BATCH_SIZE))

    act = deepq.learn(
                    env,
                    network=Config.ARCHITECTURE,
                    lr=Config.LR,
                    batch_size=Config.BATCH_SIZE,
                    total_timesteps=Config.TOTAL_TIMESTEPS,
                    buffer_size=Config.BUFFER_SIZE,
                    exploration_fraction=0.1,
                    exploration_final_eps=0.02,
                    print_freq=print_freq,
                    checkpoint_freq=Config.CHECKPOINT_FREQ,
                    checkpoint_path="./saved_models/{}".format(Config.RUN_ID),
                    load_path=None,#"~/Codes/coinrun/saved_model/{}/model".format(Config.RUN_ID),
                    render=Config.RENDER,
                    callback=None
                    )
    print("Saving model to dqn_model.pkl")
    act.save("dqn_model.pkl")

if __name__ == '__main__':
    train()