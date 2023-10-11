import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# import utils
from agent.networks.distral_continuous import ContinuousDistral
import copy

class DistralAgent:
	def __init__(self, obs_shape, action_shape, device, lr,
				 hidden_dim, critic_target_tau, num_expl_steps,
				 update_every_steps, stddev_schedule, stddev_clip, use_tb,
				 obs_type, num_envs, gamma=0.99, alpha=0.5, beta=5, **kwargs):

		self.device = device
		self.lr = lr
		self.critic_target_tau = critic_target_tau
		self.update_every_steps = update_every_steps
		self.use_tb = use_tb
		self.stddev_schedule = stddev_schedule
		self.stddev_clip = stddev_clip
		self.num_envs = num_envs

		self.agents = [
			ContinuousDistral(
				obs_shape, action_shape, device, lr, hidden_dim,
				critic_target_tau, num_expl_steps, stddev_schedule,
				stddev_clip, use_tb, obs_type, gamma, alpha, beta
			)
			for _ in range(num_envs)
		]

		self.distill_agent = ContinuousDistral(
			obs_shape, action_shape, device, lr, hidden_dim,
			critic_target_tau, num_expl_steps, stddev_schedule,
			stddev_clip, use_tb, obs_type, gamma, alpha, beta
		)

		self.train()
		
	def __repr__(self):
		return "distral"
	
	def train(self, training=True):
		self.training = training
		for agent in self.agents:
			agent.train(training)
		self.distill_agent.train(training)


	def act(self, obs, goal, step, eval_mode):
		return self.distill_agent.act(obs, goal, step, eval_mode)

	def update(self, replay_iter, expert_replay_iter, step, bc_regularize=False, 
	    	   goal_embedding_func=None, task_id=None):
		metrics = dict()

		if step % self.update_every_steps != 0:
			return metrics

		for env_idx in range(len(replay_iter)):
			# Update task specific agent
			task_specific_metrics, (obs, goal, action) = self.agents[env_idx].update(
																self.distill_agent,
																replay_iter[env_idx],
																expert_replay_iter[env_idx],
																env_idx,
																step,
																bc_regularize=bc_regularize
															)
			metrics.update(task_specific_metrics)
			# Update distill agent
			metrics.update(
				self.distill_agent.update_distill(obs, goal, action, env_idx, step)
			)

		return metrics

	
	def save_snapshot(self):
		keys_to_save = ['agents', 'distill_agent']
		payload = {k: self.__dict__[k] for k in keys_to_save}
		return payload

	def load_snapshot(self, payload, env_idx):
		if env_idx == 0:
			for k, v in payload.items():
				self.__dict__[k] = v
			self.critic_target.load_state_dict(self.critic.state_dict())
			if self.use_encoder:
				self.encoder_target.load_state_dict(self.encoder.state_dict())

		if self.use_encoder and not self.train_encoder:
			for param in self.encoder.parameters():
				param.requires_grad = False

		# Update optimizers
		if env_idx == 0:
			if self.use_encoder and self.train_encoder:
				self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
			self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
			self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
