import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer import replay_buffer
from rl_modules.models import actor, critic
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler
from train_mode import TrainMode
from typing import Tuple
import wandb

"""
ddpg with HER (MPI-version)

"""


class ddpg_agent:
    def __init__(self, args, env1, env2, env1_params, env2_params):
        self.args = args
        self.env1 = env1
        self.env2 = env2
        self.env1_params = env1_params
        self.env1_params['obs'] += 1
        self.env2_params = env2_params
        self.env2_params['obs'] += 1

        self.env1_id = 0.0
        self.env2_id = 1.0

        self.train_mode = TrainMode(args.training_mode)

        # store weights and biases API key if in args
        if self.args.wandb_api_key is not None:
            os.environ["WANDB_API_KEY"] = self.args.wandb_api_key
        # if key is present set a flag to enable the functionality
        self.use_wandb_log = os.environ.get("WANDB_API_KEY") is not None

        # create the network
        assert env1_params == env2_params  # TODO: make sure to check for equality
        self.actor_network = actor(env1_params)

        self.critic_network = critic(env1_params)

        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)

        # build up the target network
        self.actor_target_network = actor(env1_params)
        self.critic_target_network = critic(env1_params)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # her sampler
        self.her_module1 = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env1.compute_reward)
        self.her_module2 = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env2.compute_reward)

        # create the replay buffer
        self.buffer1 = replay_buffer(self.env1_params, self.args.buffer_size, self.her_module1.sample_her_transitions)
        self.buffer2 = replay_buffer(self.env2_params, self.args.buffer_size, self.her_module2.sample_her_transitions)

        # create the normalizer TODO: See if we need two of these
        self.o_norm = normalizer(size=env1_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env1_params['goal'], default_clip_range=self.args.clip_range)

        # create the dict for storing the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)

            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env1_name + self.args.env2_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

    def get_env1_set(self):
        return self.env1, self.env1_params, self.buffer1, self.her_module1, self.args.env1_name, self.env1_id

    def get_env2_set(self):
        return self.env2, self.env2_params, self.buffer2, self.her_module2, self.args.env2_name, self.env2_id

    def get_env(self, curr_epoch: int) -> Tuple:
        progress_percent = curr_epoch / self.args.n_epochs

        if self.train_mode == TrainMode.FirstThenSecond:
            if progress_percent < 0.5:
                return self.get_env1_set()
            else:
                return self.get_env2_set()
        elif self.train_mode == TrainMode.SecondThenFirst:
            if progress_percent < 0.5:
                return self.get_env2_set()
            else:
                return self.get_env1_set()
        else:  # interlaced
            if curr_epoch % 2 == 0:
                return self.get_env1_set()
            else:
                return self.get_env2_set()

    def inject_obs(self, obs, env_id):
        """Will inject the env_id to the observation if args.train_baseline is False. Otherwise will do nothing."""
        if self.args.train_baseline:
            return obs
        return np.append(obs, env_id)

    def learn(self):
        """
        train the network

        """
        
        # setup weights and biases
        if self.use_wandb_log and MPI.COMM_WORLD.Get_rank() == 0:
            wandb.init(project="Input-Based DDPG + HER Dual Fetch Model")
            config = wandb.config
            config.env1_name = self.args.env1_name
            config.env2_name = self.args.env2_name
            config.training_mode = self.train_mode.name
            config.number_of_epochs = self.args.n_epochs

        # start to collect samples
        for epoch in range(self.args.n_epochs):
            env, env_params, buffer, her_module, env_name, env_id = self.get_env(epoch)
            for _ in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):

                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []

                    # reset the environment
                    observation = env.reset()
                    obs = observation['observation']
                    obs = self.inject_obs(obs, env_id)
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']

                    # start to collect samples
                    for t in range(env_params['max_timesteps']):
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            pi = self.actor_network(input_tensor)
                            action = self._select_actions(pi, env_params)

                        # feed the actions into the environment
                        observation_new, _, _, info = env.step(action)
                        obs_new = observation_new['observation']
                        obs_new = self.inject_obs(obs_new, env_id)
                        ag_new = observation_new['achieved_goal']

                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())

                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)

                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)

                # store the episodes
                buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions], her_module)
                for _ in range(self.args.n_batches):
                    # train the network
                    self._update_network(env_params, buffer)

                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)

            # start to do the evaluation
            success_rate = self._eval_agent(env, env_params, env_id)
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch is: {}, eval success rate is: {:.3f}, env name: {}'.format(
                    datetime.now(), epoch, success_rate, env_name))
                if self.use_wandb_log:
                    wandb.log({'Epoch': epoch, 'Success Rate': success_rate,
                               'Environment': env_name})
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std,
                            self.actor_network.state_dict()],
                           self.model_path + '/model.pt')

        # ~~~~~ Evaluation and Cloud Model Save ~~~~~~
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Training finished! Results:")

            env1_eval = self._eval_agent(self.env1, self.env1_params, self.env1_id, testing=True)
            print("{} eval success rate is: {:.5f}".format(
                self.args.env1_name, env1_eval))

            env2_eval = self._eval_agent(self.env2, self.env2_params, self.env2_id, testing=True)
            print("{} eval success rate is: {:.5f}".format(
                self.args.env2_name, env2_eval))

            if self.use_wandb_log:
                wandb.log({'{} Eval Success Rate'.format(self.args.env1_name): env1_eval,
                           '{} Eval Success Rate'.format(self.args.env2_name): env2_eval})

                wandb.save(os.path.join(self.model_path, 'model.pt'))

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)

        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs

    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi, env_params):
        action = pi.cpu().numpy().squeeze()

        # add the gaussian
        action += self.args.noise_eps * env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -env_params['action_max'], env_params['action_max'])

        # random actions...
        random_actions = np.random.uniform(low=-env_params['action_max'], high=env_params['action_max'],
                                           size=env_params['action'])

        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch, her_module):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]

        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]

        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs,
                       'ag': mb_ag,
                       'g': mb_g,
                       'actions': mb_actions,
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']

        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)

        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])

        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self, env_params, buffer):

        # sample the episodes
        transitions = buffer.sample(self.args.batch_size)

        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)

        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)

        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()

        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization

            # concatenate the stuffs
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()

            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)

        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()

        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / env_params['action_max']).pow(2).mean()

        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()

        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()

    # do the evaluation
    def _eval_agent(self, env, env_params, env_id, testing=False):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = env.reset()
            obs = self.inject_obs(observation['observation'], env_id)
            g = observation['desired_goal']
            for _ in range(env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)

                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = env.step(actions)
                obs = self.inject_obs(observation_new['observation'], env_id)
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        if testing:
            return local_success_rate
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()
