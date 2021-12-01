import numpy as np
import gym
import os, sys
from arguments import get_args
from mpi4py import MPI
from rl_modules.ddpg_agent import ddpg_agent
import random
import torch
from envs.low_policy_env import LowPolicyEnv

"""
train the hierarchical agent, the low-level policy is integrated in the env 

"""
def get_env_params(env, c):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            'action_space': env.action_space,
            }
    # print("action shape", params['action'])
    # print("goal shape", params['goal'])
    params['max_timesteps'] = env.env._max_episode_steps // c
    print("episode length", params['max_timesteps'])
    return params

def launch(args):
    # create hierarchical env
    pretrain_env = gym.make('HandReach-v0')
    model_path = 'saved_models/HandReach-v0/model.pt'
    inner_env = gym.make(args.env_name)
    env = LowPolicyEnv(inner_env, pretrain_env, args.c, model_path, args)

    # set random seeds for reproduce
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # get the environment parameters
    env_params = get_env_params(env, args.c)
    # create the ddpg agent to interact with the environment
    ddpg_trainer = ddpg_agent(args, env, env_params, True)
    ddpg_trainer.learn()

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)
