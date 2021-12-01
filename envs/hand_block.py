# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import gym
import os
from gym import spaces
from envs.gym_robotics.hand.manipulate import ManipulateEnv
import mujoco_py

MANIPULATE_BLOCK_XML = os.path.join('hand', 'manipulate_block.xml')

class HandBlockCustomEnv(ManipulateEnv):
	def __init__(self,
				 model_path=MANIPULATE_BLOCK_XML,
				 target_position='random',
				 target_rotation='xyz',
				 reward_type='sparse',
				 horizontal_wrist_constraint=1.0,
				 vertical_wrist_constraint=1.0,
				 **kwargs):
		ManipulateEnv.__init__(self,
			 model_path=MANIPULATE_BLOCK_XML,
			 target_position=target_position,
			 target_rotation=target_rotation,
			 target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
			 reward_type=reward_type,
			 **kwargs)

		self._viewers = {}

		# constraining the movement of wrist (vertical movement more important than horizontal)
		self.action_space.low[0] = -horizontal_wrist_constraint
		self.action_space.high[0] = horizontal_wrist_constraint
		self.action_space.low[1] = -vertical_wrist_constraint
		self.action_space.high[1] = vertical_wrist_constraint

		self._max_episode_steps = 100

	def _get_viewer(self, mode):
		self.viewer = self._viewers.get(mode)
		if self.viewer is None:
			if mode == 'human':
				self.viewer = mujoco_py.MjViewer(self.sim)
			elif mode == 'rgb_array':
				self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
				self._viewer_setup()
				self._viewers[mode] = self.viewer
		return self.viewer

	def _viewer_setup(self):
		body_id = self.sim.model.body_name2id('robot0:palm')
		lookat = self.sim.data.body_xpos[body_id]
		for idx, value in enumerate(lookat):
			self.viewer.cam.lookat[idx] = value
		self.viewer.cam.distance = 0.5
		self.viewer.cam.azimuth = 55.
		self.viewer.cam.elevation = -25.

	def step(self, action):
		
		def is_on_palm():
			self.sim.forward()
			cube_middle_idx = self.sim.model.site_name2id('object:center')
			cube_middle_pos = self.sim.data.site_xpos[cube_middle_idx]
			is_on_palm = (cube_middle_pos[2] > 0.04)
			return is_on_palm

		obs, reward, done, info = super().step(action)
		done = not is_on_palm()
		return obs, reward, done, info

	def render(self, mode='human', width=500, height=500):
		self._render_callback()
		if mode == 'rgb_array':
			self._get_viewer(mode).render(width, height)
			# window size used for old mujoco-py:
			data = self._get_viewer(mode).read_pixels(width, height, depth=False)
			# original image is upside-down, so flip it
			return data[::-1, :, :]
		elif mode == 'human':
			self._get_viewer(mode).render()
