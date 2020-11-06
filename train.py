from client import Client
import numpy as np
import os
from arguments import get_args
from mpi4py import MPI
from rl_modules.ddpg_agent import ddpg_agent
import random
import torch

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""
def get_env_params(client):
    obs = client.reset_env()
    action_space = client.get_action_space()

    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': action_space.shape[0],
            'action_max': action_space.high[0],
            }
    params['max_timesteps'] = client.get_max_episode_steps()
    return params

def launch(args):
    client = Client(args.server_name)
    
    # create the ddpg_agent
    client.create_env(args.env_name)
    # set random seeds for reproduce
    client.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # get the environment parameters
    env_params = get_env_params(client)
    # create the ddpg agent to interact with the environment 
    ddpg_trainer = ddpg_agent(args, client, env_params)
    ddpg_trainer.learn()
    client.close()

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)
