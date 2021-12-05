import os
import sys

sys.path.append('../')

from datetime import datetime
from rl_modules.models import *
from rl_modules.replay_buffer import replay_buffer
from her_modules.her import her_sampler
import random
from tensorboardX import SummaryWriter
import torch
import numpy as np
from rl_modules.utils import LinearSchedule


class dqn_agent:
    def __init__(self, args, env, env_params, hier):
        self.args = args
        self.device = args.device
        self.env = env
        self.env_params = env_params
        self.action_n = env.action_space.n
        print("action_n", self.action_n)

        self.init_qnets()
        self.start_epoch = 0

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        current_time = "{}_hier_{}".format(current_time, hier)
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name + current_time)
        if not os.path.exists(self.model_path) and args.save:
            os.mkdir(self.model_path)
            # define tensorboard writer
            log_dir = self.model_path + '/tb'
            self.writer = SummaryWriter(log_dir)

        # load the weights into the target networks
        self.targetQ_network.load_state_dict(self.Q_network.state_dict())
        # create the optimizer
        self.q_optim = torch.optim.Adam(self.Q_network.parameters(), lr=self.args.lr)
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)

        # random exploration schedule
        self.eps = LinearSchedule(20000, 0.1)

    def init_qnets(self):
        self.Q_network = Qnet(self.env_params).to(self.device)
        self.targetQ_network = Qnet(self.env_params).to(self.device)

    def learn(self):
        eval_interval = 100
        train_success = []
        train_in_success = []
        for epoch in range(self.start_epoch, self.args.n_epochs):
            ep_obs, ep_ag, ep_g, ep_actions, ep_r, ep_done = [], [], [], [], [], []
            observation = self.env.reset()
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']
            success = 0.
            in_success = 0.
            for t in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    inputs = self._preproc_inputs(obs, g)
                    action = self.explore_policy(inputs, epoch)
                    # feed the actions into the environment
                observation_new, r, done, info = self.env.step(action.squeeze(0))
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                # append rollouts
                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
                ep_g.append(g.copy())
                ep_actions.append(action.copy())
                ep_r.append(r)
                ep_done.append(done)
                # re-assign the observation
                obs = obs_new
                ag = ag_new
                # record success
                if info['is_success']:
                    in_success = 1
            if info['is_success']:
                success = 1.
            train_success.append(success)
            train_in_success.append(in_success)
            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())
            mb_obs = np.array([ep_obs])
            mb_ag = np.array([ep_ag])
            mb_g = np.array([ep_g])
            mb_actions = np.array([ep_actions])
            mb_r = np.array([ep_r])
            mb_r = mb_r.reshape(-1, 1)
            # print("mb_r", mb_r)
            mb_done = np.array([ep_done])
            mb_done = mb_done.reshape(-1, 1)
            self.buffer.store_episode_r([mb_obs, mb_ag, mb_g, mb_actions, mb_r, mb_done])
            # update network after 200 episode
            if epoch > 2 * eval_interval:
                # update with 10 batches
                for n_batch in range(self.env_params['max_timesteps']):
                    self._update_network()
                # hard update the network every 10 episodes
                if epoch % 10 == 0:
                    # hard update the network
                    self._hard_update_target_network(self.targetQ_network, self.Q_network)
            # start to do the evaluation
            if epoch % eval_interval == 0:
                success_rate, in_success_rate = self._eval_agent()
                print('[{}] DQN, epoch is: {}, eval success is: {:.3f}, eval in success is: {:.3f}, train success is: {:.3f}, '
                      'train in success is: {:.3f}, eps: {:.3f}'.format(
                    datetime.now(), epoch // eval_interval,
                    success_rate, in_success_rate, np.mean(train_success[-eval_interval:]),
                    np.mean(train_in_success[-eval_interval:]), self.eps.value(epoch)))
                if self.args.save:
                    if epoch % 1000 == 0:
                        print("model_path:", self.model_path)
                    torch.save([self.Q_network.state_dict()], \
                               self.model_path + '/q_model.pt')
                    # torch.save(self.buffer, self.model_path + '/buffer.pt')
                    self.writer.add_scalar('eval/success', success_rate, epoch // eval_interval)
                    self.writer.add_scalar('eval/in_success', in_success_rate, epoch // eval_interval)
                    self.writer.add_scalar('train/success', np.mean(train_success[-eval_interval:]), epoch // eval_interval)
                    self.writer.add_scalar('train/in_success', np.mean(train_in_success[-eval_interval:]), epoch // eval_interval)
                    self.writer.add_scalar('eps', self.eps.value(epoch),
                                           epoch // eval_interval)

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        inputs = np.concatenate([obs, g])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(self.device)
        return inputs

    def explore_policy(self, inputs, epoch):
        q_value = self.Q_network(inputs)
        # print("q_value", q_value)
        best_actions = q_value.max(1)[1].cpu().numpy()
        # print("best_actions", best_actions)
        # eps is linear decayed from 1 to 0.1
        eps = self.eps.value(epoch)
        if random.random() < eps:
            best_actions[0] = np.random.randint(self.action_n)
        return best_actions

    def test_policy(self, inputs):
        q_value = self.Q_network(inputs)
        best_actions = q_value.max(1)[1].cpu().numpy()
        return best_actions

    # hard update
    def _hard_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = o, g
        transitions['obs_next'], transitions['g_next'] = o_next, g

        # start to do the update
        obs_cur = transitions['obs']
        g_cur = transitions['g']
        cont_obs = np.concatenate((obs_cur, g_cur), axis=1)
        # print("cont_obs", cont_obs.shape)
        obs_next = transitions['obs_next']
        g_next = transitions['g_next']
        cont_obs_next = np.concatenate((obs_next, g_next), axis=1)
        # ag_next = transitions['ag_next']

        # done
        not_done = (1 - transitions['done'])
        not_done = torch.tensor(not_done, dtype=torch.int32).to(self.device)
        # print("not done", not_done)

        # transfer them into the tensor
        obs_cur = torch.tensor(cont_obs, dtype=torch.float32).to(self.device)
        obs_next = torch.tensor(cont_obs_next, dtype=torch.float32).to(self.device)
        # ag_next = torch.tensor(ag_next, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.long).to(self.device)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32).to(self.device)
        # calculate the target Q value function
        with torch.no_grad():
            next_q_values = self.targetQ_network(obs_next)
            # NOT use double dqn here
            target_action = next_q_values.max(1)[1].unsqueeze(1)
            # print("next q before", next_q_values)
            next_q_value = next_q_values.gather(1, target_action)
            # print("next q after", next_q_value)
            next_q_value = next_q_value.detach()
            target_q_value = r_tensor + self.args.gamma * next_q_value * not_done
            target_q_value = target_q_value.detach()
            # clip target value, since the reward is between -c and 0
            clip_return = self.args.c / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0.)

        real_q_value = self.Q_network(obs_cur).gather(1, actions_tensor)
        td_loss = (real_q_value - target_q_value).pow(2).mean()
        # This is a regularization, but maybe unnecessary.
        # forward_loss = (self.Q_network(obs_cur, ag_next).gather(1, actions_tensor)).pow(2).mean()
        # td_loss += 0.5 * forward_loss
        self.q_optim.zero_grad()
        td_loss.backward()
        self.q_optim.step()

    def _eval_agent(self, policy=None):
        if policy is None:
            policy = self.test_policy

        total_success_rate = []
        success_list = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            success = 0.
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    inputs = self._preproc_inputs(obs, g)
                    actions = policy(inputs)
                observation_new, _, _, info = self.env.step(actions.squeeze(0))
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
                if info['is_success']:
                    success = 1.0
            total_success_rate.append(per_success_rate)
            success_list.append(success)
        total_success_rate = np.array(total_success_rate)
        # only calculate the success of the final state
        global_success_rate = np.mean(total_success_rate[:, -1])
        in_success_rate = np.mean(success_list)
        return global_success_rate, in_success_rate
