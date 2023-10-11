from cgitb import enable
import time
import gym
from gym import spaces
from gym_envs.envs import robot_env

import numpy as np

class RobotPour2DEnv(robot_env.RobotEnv):
	def __init__(self, height=84, width=84, step_size=10, enable_arm=True, enable_gripper=True, enable_camera=True, camera_view='side',
				 use_depth=False, dist_threshold=0.05, random_start=True, x_limit=None, y_limit=None, z_limit=None, pitch=90, roll=-90, yaw=0, keep_gripper_closed=True):
		robot_env.RobotEnv.__init__(
			self,
			home_displacement = [1.4, -2.4, -0.1],
			height=height,
			width=width,
			step_size=step_size,
			enable_arm=enable_arm, 
			enable_gripper=enable_gripper,
			enable_camera=enable_camera,
			camera_view=camera_view,
			use_depth=use_depth,
			keep_gripper_closed=keep_gripper_closed,
			highest_start=True,
			x_limit=x_limit,
			y_limit=y_limit,
			z_limit=z_limit,
			pitch=pitch,
			roll=roll,
			yaw=yaw
		)
		self.action_space = spaces.Box(low = np.array([-1,-1,-1],dtype=np.float32), 
									   high = np.array([1, 1, 1],dtype=np.float32),
									   dtype = np.float32)
		self.dist_threshold = dist_threshold
		self.random_start = random_start

	def arm_refresh(self, reset=True):
		self.arm.clear_errors()
		self.arm.set_mode_and_state()
		self.arm.pitch = 90
		if reset:
			self.arm.reset(home=True)
		time.sleep(2)

	def reset(self):
		if not self.enable_arm:
			return np.array([0,0,0], dtype=np.float32)
		self.arm_refresh(reset=False)
		if self.random_start:
			self.arm.set_random_pos()
		time.sleep(0.4)		
		obs = {}
		obs['features'] = np.array(self.arm.get_position(), dtype=np.float32)
		obs['pixels'] = self.render(mode='rgb_array', width=self.width, height=self.height)
		return obs

	def set_position(self, pos, wait=False):
		pos = self.arm.limit_pos(pos)
		x = (pos[0] + self.arm.zero[0])*100
		y = (pos[1] + self.arm.zero[1])*100
		z = (pos[2] + self.arm.zero[2])*100
		self.arm.arm.set_position(x=x, y=y, z=z, roll=self.arm.roll, pitch=self.arm.pitch, yaw=self.arm.yaw, wait=wait)

	def step(self, action):
		new_pos = self.arm.get_position()
		new_pos[:2] += action[:2] * 0.25
		if self.enable_arm:
		
			if action[2]>0.5 and self.arm.pitch<210:
				self.arm.pitch = min(self.arm.pitch+40, 210)
				self.set_position(new_pos)
				time.sleep(3)
			elif action[2]<-0.5 and self.arm.pitch>-30:
				self.arm.pitch = max(self.arm.pitch-40, -30)
				self.set_position(new_pos)
				time.sleep(3)
			self.set_position(new_pos)
			time.sleep(0.4)
		self.reward = 0
		
		done = False
		
		info = {}
		info['is_success'] = 1 if self.reward==1 else 0

		obs = {}
		obs['features'] = np.array(self.arm.get_position(), dtype=np.float32)
		obs['pixels'] = self.render(mode='rgb_array', width=self.width, height=self.height)

		return obs, self.reward, done, info