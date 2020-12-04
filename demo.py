import torch
from rl_modules.models import actor
from arguments import get_args
import gym
import numpy as np
from rl_modules.ddpg_agent import ddpg_agent


# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs


def demo_2_envs(env, args, env_id):
    # load the model param
    model_path = args.save_dir + args.env1_name + args.env2_name + '/model.pt'
    o_mean, o_std, g_mean, g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)

    # get the env param
    observation = env.reset()
    # get the environment params
    env_params = {'obs': ddpg_agent.inject_obs(observation['observation'], env_id, args).shape[0],
                  'goal': observation['desired_goal'].shape[0],
                  'action': env.action_space.shape[0],
                  'action_max': env.action_space.high[0],
                  }
    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()
    for i in range(args.demo_length):
        observation = env.reset()
        # start to do the demo
        obs = ddpg_agent.inject_obs(observation['observation'], env_id, args)
        g = observation['desired_goal']
        for t in range(env._max_episode_steps):
            env.render()
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()
            # put actions into the environment
            observation_new, reward, _, info = env.step(action)
            obs = ddpg_agent.inject_obs(observation_new['observation'], env_id, args)
        print('the episode is: {}, is success: {}'.format(i, info['is_success']))


if __name__ == '__main__':
    args = get_args()
    # create the environment
    env1 = gym.make(args.env1_name)
    env2 = gym.make(args.env2_name)

    print("Playing demo for {}".format(args.env1_name))
    demo_2_envs(env1, args, ddpg_agent.env1_id)
    print("Playing demo for {}".format(args.env2_name))
    demo_2_envs(env2, args, ddpg_agent.env2_id)
