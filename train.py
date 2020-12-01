import numpy as np
import gym
import os
from arguments import get_args
from mpi4py import MPI
from rl_modules.ddpg_agent import ddpg_agent
import random
import torch

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""
def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params


def launch(args):
    # create the ddpg_agent
    env1 = gym.make(args.env1_name)
    env2 = gym.make(args.env2_name)

    # set random seeds for reproduce
    env1.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    env2.seed(args.seed + MPI.COMM_WORLD.Get_rank())

    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    # get the environment parameters
    env1_params = get_env_params(env1)
    env2_params = get_env_params(env2)

    # create the ddpg agent to interact with the environment
    ddpg_trainer = ddpg_agent(args, env1, env2, env1_params, env2_params)
    ddpg_trainer.learn()


if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    # get the params
    args = get_args()
    launch(args)
