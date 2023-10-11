import einops
import random
import numpy as np
from collections import deque
import torch
from torch import nn
from torch.nn import functional as F

import utils
from agent.networks.encoder import Encoder
from agent.networks.mlp import MLP


class Actor(nn.Module):
	def __init__(self, repr_dim, action_shape, hidden_dim):
		super().__init__()

		self._output_dim = action_shape[0]
		if repr_dim != 512:
			goal_dim = repr_dim - 1 if repr_dim%2==1 else repr_dim
		else:
			goal_dim = repr_dim
		
		self.policy = nn.Sequential(nn.Linear(repr_dim + goal_dim, hidden_dim),
									nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, hidden_dim),
									nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, hidden_dim),
									nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, hidden_dim),
									nn.ReLU(inplace=True))
		
		self._head = MLP(in_channels=hidden_dim, hidden_channels=[self._output_dim])

		self.apply(utils.weight_init)

	def forward(self, obs, goal, std):

		obs = torch.cat([goal, obs], dim=-1)
		feat = self.policy(obs)

		mu = torch.tanh(self._head(feat))
		std = torch.ones_like(mu) * std

		dist = utils.TruncatedNormal(mu, std)
		return dist

class BCAgent:
	def __init__(self, obs_shape, action_shape, device, lr, hidden_dim, stddev_schedule, 
	      		 stddev_clip, use_tb, augment, obs_type, train_encoder=True):
		self.device = device
		self.lr = lr
		self.stddev_schedule = stddev_schedule
		self.stddev_clip = stddev_clip
		self.use_tb = use_tb
		self.augment = augment
		self.use_encoder = True if obs_type=='pixels' else False
		self.train_encoder = train_encoder

		# models
		if self.use_encoder:
			self.encoder = Encoder(obs_shape).to(device)
			repr_dim = self.encoder.repr_dim
			if not self.train_encoder:
				for param in self.encoder.parameters():
					param.requires_grad = False
		else:
			repr_dim = obs_shape[0]

		self.actor = Actor(repr_dim, action_shape, hidden_dim).to(device)

		# optimizers
		if self.use_encoder and self.train_encoder:
			self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

		# data augmentation
		self.aug = utils.RandomShiftsAug(pad=4)

		self.train()
		
	def __repr__(self):
		return "bc"
	
	def train(self, training=True):
		self.training = training
		if training:
			if self.use_encoder and self.train_encoder:
				self.encoder.train(training)
			self.actor.train(training)
		else:
			if self.use_encoder:
				self.encoder.eval()
			self.actor.eval()

	def act(self, obs, goal, step, eval_mode):
		obs = torch.as_tensor(obs, device=self.device).float()
		goal = torch.as_tensor(goal, device=self.device).float()

		obs = self.encoder(obs[None]) if self.use_encoder else obs[None]
		goal = self.encoder(goal[None]) if self.use_encoder else goal[None]
		stddev = utils.schedule(self.stddev_schedule, step)

		dist = self.actor(obs, goal, stddev)

		if eval_mode:
			action = dist.mean
		else:
			action = dist.sample(clip=None)
		return action.cpu().numpy()[0]

	def update(self, replay_iter, expert_replay_iter, step, bc_regularize=False,
			   goal_embedding_func=None, task_id=None):
		metrics = dict()

		env_idx = random.randint(0, len(expert_replay_iter)-1)
		batch = next(expert_replay_iter[env_idx])
		obs, action, goal = utils.to_torch(batch, self.device)
		action = action.float()

		if goal_embedding_func is not None and task_id is not None:
			goal = goal_embedding_func(task_id[env_idx])
			goal = torch.as_tensor(goal[None], device=self.device).repeat(obs.shape[0], 1).float()

		# augment
		if self.use_encoder and self.augment:
			obs = self.aug(obs.float())
			goal = self.aug(goal.float())
			# encode
			goal = self.encoder(goal)
			obs = self.encoder(obs)
		else:
			obs = obs.float()
			goal = goal.float()

		stddev = utils.schedule(self.stddev_schedule, step)
		dist = self.actor(obs, goal, stddev)
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)

		actor_loss = -log_prob.mean()
		
		if self.use_encoder and self.train_encoder:
			self.encoder_opt.zero_grad(set_to_none=True)
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()
		if self.use_encoder and self.train_encoder:
			self.encoder_opt.step()
		self.actor_opt.step()

		if self.use_tb:
			metrics['actor_loss'] = actor_loss.item()
			metrics['actor_logprob'] = log_prob.mean().item()
			metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

		return metrics

	def save_snapshot(self):
		keys_to_save = ['actor']
		if self.use_encoder:
			keys_to_save += ['encoder']
		payload = {k: self.__dict__[k] for k in keys_to_save}
		return payload

	def load_snapshot(self, payload, env_idx):
		for k, v in payload.items():
			self.__dict__[k] = v

		# Update optimizers
		if self.use_encoder and self.train_encoder:
			self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)