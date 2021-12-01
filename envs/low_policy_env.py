import gym
import numpy as np
import torch
from rl_modules.models import actor
from arguments import get_args


FINGERTIP_SITE_NAMES = [
    'robot0:S_fftip',
    'robot0:S_mftip',
    'robot0:S_rftip',
    'robot0:S_lftip',
    'robot0:S_thtip',
]

class LowPolicyEnv(gym.Wrapper):
    def __init__(self,
                 env,
                 pretrain_env,   # the env to train low-level policy
                 c,   # low-level policy length
                 model_path,
                 args,
                 ):
        super().__init__(env)
        # load policy
        self.o_mean, self.o_std, self.g_mean, self.g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)
        observation = pretrain_env.reset()
        # get the environment params
        env_params = {'obs': observation['observation'].shape[0],
                      'goal': observation['desired_goal'].shape[0],
                      'action': env.action_space.shape[0],
                      'action_max': env.action_space.high[0],
                      'action_space': env.action_space,
                      }
        # create the actor network
        self.actor_network = actor(env_params, args.cuda)
        self.actor_network.load_state_dict(model)
        if args.cuda:
            self.actor_network.cuda()
        self.actor_network.eval()
        self.c = c

        # obs space and action space for high-level env
        self.observation_space = self.env.observation_space
        meet_pos = pretrain_env.palm_xpos + np.array([0.0, -0.09, 0.05])
        meet = np.tile(meet_pos, 5)
        initial_goal = pretrain_env.initial_goal.copy()
        diff = meet - initial_goal
        idxs = np.where(diff < 0)
        for id in idxs[0]:
            meet[id], initial_goal[id] = initial_goal[id], meet[id]

        self.action_space = gym.spaces.Box(low=initial_goal, high=meet)

        self.last_obs = None


    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)

        self.last_obs = ret
        return ret

    # get finger pos
    def _get_achieved_goal(self):
        goal = [self.env.sim.data.get_site_xpos(name) for name in FINGERTIP_SITE_NAMES]
        return np.array(goal).flatten()

    # preprocess obs
    def preprocess(self, obs):
        obs = obs[:48]  # hard code her, only applicable to hand env
        finger_pos = self._get_achieved_goal()
        obs_new = np.concatenate((obs, finger_pos))
        return obs_new

    # process the inputs
    def process_inputs(self, o, g):
        o_clip = np.clip(o, -200., 200.)
        g_clip = np.clip(g, -200., 200.)
        o_norm = np.clip((o_clip - self.o_mean) / (self.o_std), -5., 5.)
        g_norm = np.clip((g_clip - self.g_mean) / (self.g_std), -5., 5.)
        inputs = np.concatenate([o_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32)
        return inputs



    def step(self, hi_action, **kwargs):
        hi_action = hi_action.copy()
        sum_rewards = 0.
        done_final = False

        for i in range(self.c):
            cp_obs = self.last_obs
            cp_obs = cp_obs['observation']
            cp_obs = self.preprocess(cp_obs)
            inputs = self.process_inputs(cp_obs, hi_action)
            with torch.no_grad():
                pi = self.actor_network(inputs)
            action = pi.detach().numpy().squeeze()

            next_obs, reward, done, info = self.env.step(action, **kwargs)

            self.last_obs = next_obs
            sum_rewards += reward
            if done:
                done_final = True
                break
        return next_obs, sum_rewards, done_final, info


if __name__ == '__main__':

    env = gym.make('HandManipulateBlockRotateZ-v0')
    pretrain_env = gym.make('HandReach-v0')
    model_path = '../saved_models/HandReach-v0/model.pt'
    args = get_args()
    low_policy_env = LowPolicyEnv(env, pretrain_env, 10, model_path, args)

    low_policy_env.reset()

    for _ in range(20):
        observation = low_policy_env.reset()
        # # pretrained goals
        # action = pretrain_env.env._sample_goal()

        # # interpolation goals
        # action = low_policy_env.action_space.sample()

        for t in range(env._max_episode_steps // 10):
            low_policy_env.env.render()
            action = low_policy_env.action_space.sample()
            _, _, done, info = low_policy_env.step(action)
            if done:
                print(info['is_success'])
                break

