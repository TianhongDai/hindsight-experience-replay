import numpy as np
import gym
import os, sys
from arguments import get_args_dqn
# from mpi4py import MPI
from rl_modules.dqn_agent import dqn_agent
import random
import torch
from envs.multi_level_env import MultiLevelEnv

"""
train the hierarchical agent, the low-level policy is integrated in the env 

"""
def get_env_params(env, c):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': 1,
            'action_max': 1,
            'n_action': env.action_space.n,
            }
    # print("action shape", params['action'])
    # print("goal shape", params['goal'])
    params['max_timesteps'] = env.env._max_episode_steps // c
    print("episode length", params['max_timesteps'])
    return params

def launch(args):
    # create hierarchical env
    pretrain_env = gym.make('HandManipulateBlockRotateZ-v0')
    pos_model = 'saved_models/'+ args.pos_path + '/model.pt'
    rot_model = 'saved_models/' + args.rot_path + '/model.pt'

    inner_env = gym.make(args.env_name)
    env = MultiLevelEnv(inner_env, pretrain_env,
                        pos_model, rot_model, args)

    # set random seeds for reproduce
    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device is not 'cpu':
        torch.cuda.manual_seed(args.seed)
    # get the environment parameters
    env_params = get_env_params(env, args.c)
    # create the ddpg agent to interact with the environment
    dqn_trainer = dqn_agent(args, env, env_params, True)
    dqn_trainer.learn()

if __name__ == '__main__':
    # # take the configuration for the HER
    # os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['MKL_NUM_THREADS'] = '1'
    # os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args_dqn()
    launch(args)
