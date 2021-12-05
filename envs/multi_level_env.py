import gym
import numpy as np
import torch
from rl_modules.models import actor
from arguments import get_args
import time
from envs.gym_robotics import *
import sys
sys.path.append('../')


# FINGERTIP_SITE_NAMES = [
#     'robot0:S_fftip',
#     'robot0:S_mftip',
#     'robot0:S_rftip',
#     'robot0:S_lftip',
#     'robot0:S_thtip',
# ]

class MultiLevelEnv(gym.Wrapper):
    def __init__(self,
                 env,
                 pretrain_env,  # the env to train low-level policy
                 pos_model,
                 rot_model,
                 args,
                 ):
        super().__init__(env)
        # load position policy
        self.o_mean, self.o_std, self.g_mean, self.g_std, model = torch.load(pos_model, map_location=lambda storage, loc: storage)
        # load rotation policy
        self.o_mean1, self.o_std1, self.g_mean1, self.g_std1, model1 = torch.load(rot_model,
                                                                             map_location=lambda storage, loc: storage)
        observation = pretrain_env.reset()
        # get the environment params
        env_params = {'obs': observation['observation'].shape[0],
                      'goal': observation['desired_goal'].shape[0],
                      'action': env.action_space.shape[0],
                      'action_max': env.action_space.high[0],
                      'action_space': env.action_space,
                      }
        # create the actor network
        if args.device is not 'cpu':
            cuda = True
        else:
            cuda = False
        self.actor_network_pos = actor(env_params, cuda)
        self.actor_network_pos.load_state_dict(model)
        self.actor_network_rot = actor(env_params, cuda)
        self.actor_network_rot.load_state_dict(model1)
        if cuda:
            self.actor_network_pos.cuda()
            self.actor_network_rot.cuda()
        self.actor_network_pos.eval()
        self.actor_network_rot.eval()
        self.c = args.c

        # obs space and action space for high-level env
        self.observation_space = self.env.observation_space
        self.action_space = gym.spaces.Discrete(2)  # two kinds of skills

        self.last_obs = None

    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        self.last_obs = ret
        self.init_obj_state = ret['achieved_goal']
        self.t = 0
        return ret

    # process the inputs
    def process_inputs(self, o, g, o_mean, o_std, g_mean, g_std):
        o_clip = np.clip(o, -200., 200.)
        g_clip = np.clip(g, -200., 200.)
        o_norm = np.clip((o_clip - o_mean) / (o_std), -5., 5.)
        g_norm = np.clip((g_clip - g_mean) / (g_std), -5., 5.)
        inputs = np.concatenate([o_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32)
        return inputs

    def step(self, hi_action, **kwargs):
        # action=0 is pos, action=1 is rotation
        sum_rewards = 0.
        done_final = False
        desired_goal = self.last_obs['desired_goal'].copy()
        achieved_goal = self.last_obs['achieved_goal'].copy()
        sum_d = 0.
        # The ignored goal dims are set as the current object states, match the training distribution?
        # related to the initial object state distribution in the pretrain env?
        if hi_action == 0:
            # TODO: these two implementations have similar performance on random pi_h, maybe need better pi_l
            goal = np.concatenate((desired_goal[:3], achieved_goal[3:]))
            # goal = np.concatenate((desired_goal[:3], self.init_obj_state[3:]))
            o_mean, o_std, g_mean, g_std = self.o_mean, self.o_std, self.g_mean, self.g_std
            actor_network = self.actor_network_pos
        elif hi_action == 1:
            goal = np.concatenate((achieved_goal[:3], desired_goal[3:]))
            # goal = np.concatenate((self.init_obj_state[:3], desired_goal[3:]))
            o_mean, o_std, g_mean, g_std = self.o_mean1, self.o_std1, self.g_mean1, self.g_std1
            actor_network = self.actor_network_rot
        else:
            print("hi_action:", hi_action)
            assert False

        success = 0.
        self.t += 1

        for i in range(self.c):
            cp_obs = self.last_obs
            cp_obs = cp_obs['observation']
            inputs = self.process_inputs(cp_obs, goal, o_mean, o_std, g_mean, g_std)
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()

            next_obs, reward, done, info = self.env.step(action, **kwargs)

            # calculate the distance to goal
            d_pos, d_rot = self.env.env._goal_distance(next_obs['achieved_goal'], next_obs['desired_goal'])
            sum_distance = -(10. * d_pos + d_rot)
            sum_d += sum_distance
            # print("sum_distance", sum_distance)

            # # render
            # self.env.render()

            # record success inside c steps, only if success, done_final is True
            if info['is_success']:
                success = 1.0
                done_final = True

            self.last_obs = next_obs
            sum_rewards += reward

            # if reach subgoal, return control, take high-level action
            if (hi_action == 0 and d_pos < self.env.env.distance_threshold) or \
                    (hi_action == 1 and d_rot < self.env.env.rotation_threshold):
                # done_final = (done or bool(success))
                break

            # if reach env goal, return control
            if done or success:
                # done_final = True
                break

        # # reach max steps, not set done as True
        # if self.t == self.env._max_episode_steps // self.c:
        #     done_final = True
        info["sum_d"] = sum_d
        info['is_success'] = success
        return next_obs, sum_rewards, done_final, info

    # still problematic, high-level reward is in [-c, 0], conditioned on whether reached the subgoal in c steps
    def compute_reward(self, achieved_goal, desired_goal, info):
        success = self.env.env._is_success(achieved_goal, desired_goal).astype(np.float32)
        # print("compute relabel reward in hierarchical wrapper !!!")
        return (success - 1.) * self.c


if __name__ == '__main__':

    args = get_args()

    env = gym.make(args.env_name)
    pretrain_env = gym.make('HandManipulateBlockRotateZ-v0')
    pos_model = '../saved_models/Success_HandPos_flat/model.pt'
    rot_model = '../saved_models/Success_HandManipulateBlockRotateXYZ-v0_hier_False/model.pt'
    # args = get_args()
    low_policy_env = MultiLevelEnv(env,
                 pretrain_env,  # the env to train low-level policy
                 pos_model,
                 rot_model,
                 args)

    low_policy_env.reset()
    distance_list = []
    success = 0.

    for _ in range(args.demo_length):
        observation = low_policy_env.reset()
        # action = 1
        sum_d = 0.
        for t in range(env._max_episode_steps // args.c):
            # low_policy_env.env.render()
            # time.sleep(0.1)
            action = low_policy_env.action_space.sample()
            _, _, done, info = low_policy_env.step(action)
            sum_d += info['sum_d']
            if info['is_success']:
                success += 1
                break
        distance_list.append(sum_d)
        # print("success:", done)

    print('mean cumulative distance', np.mean(distance_list))
    print("success rate", success / args.demo_length)

