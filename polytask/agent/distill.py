import einops
import random
import logging
import numpy as np
from collections import deque
import torch
from torch import nn
from torch.nn import functional as F

import utils
from agent.networks.encoder import Encoder
from agent.networks.mlp import MLP
from agent.networks.pcgrad import PCGrad

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Actor(nn.Module):
	def __init__(self, repr_dim, action_shape, hidden_dim, goal_modality, goal_dim):
		super().__init__()

		self._output_dim = action_shape[0]
		if goal_modality == 'pixels':
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

class DistillAgent:
	def __init__(self, obs_shape, action_shape, device, lr, hidden_dim,
	      		 stddev_schedule, stddev_clip, use_tb, augment, obs_type,
				 goal_modality, goal_dim, pcgrad):
		self.device = device
		self.lr = lr
		self.stddev_schedule = stddev_schedule
		self.stddev_clip = stddev_clip
		self.use_tb = use_tb
		self.augment = augment
		self.use_encoder = True if obs_type=='pixels' else False
		self.goal_modality = goal_modality
		self.pcgrad = pcgrad

		# models
		if self.use_encoder:
			self.encoder = Encoder(obs_shape).to(device)
			repr_dim = self.encoder.repr_dim
		else:
			repr_dim = obs_shape[0]

		self.actor = Actor(repr_dim, action_shape, hidden_dim, goal_modality, goal_dim).to(device)

		# compute model sizes
		encoder_size = count_parameters(self.encoder) if self.use_encoder else 0
		actor_size = count_parameters(self.actor)
		total_size = encoder_size + actor_size
		log = logging.getLogger(__name__)
		log.info(f"Model sizes: E: {encoder_size/1e6:.2f}M, A: {actor_size/1e6:.2f}M, Total: {total_size/1e6:.2f}M")
		
		# optimizers
		params = list(self.actor.parameters()) + list(self.encoder.parameters()) if self.use_encoder else list(self.actor.parameters())
		self.opt = torch.optim.Adam(params, lr=lr)
		if self.pcgrad:
			self.opt = PCGrad(self.opt)

		# data augmentation
		self.aug = utils.RandomShiftsAug(pad=4)

		self.train()

	def __repr__(self):
		return "bc"
	
	def train(self, training=True):
		self.training = training
		if training:
			if self.use_encoder:
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
		
		# goal
		if self.goal_modality == 'pixels':
			goal = self.encoder(goal[None]) if self.use_encoder else goal[None]
		else:
			goal = goal[None]
		
		stddev = utils.schedule(self.stddev_schedule, step)
		dist = self.actor(obs, goal, stddev)

		if eval_mode:
			action = dist.mean
		else:
			action = dist.sample(clip=None)
		return action.cpu().numpy()[0]

	def update(self, replay_iter, goals, step, goal_embedding_func=None, task_id=None):
		metrics = dict()

		losses = []
		buffer_idx = [random.randint(0, len(replay_iter)-1)] if not self.pcgrad else range(len(replay_iter))

		for env_idx in buffer_idx:
			# env_idx = random.randint(0, len(replay_iter)-1)
			batch = next(replay_iter[env_idx])
			obs, action, goal = utils.to_torch(batch, self.device)
			
			if not self.use_encoder and goal_embedding_func is not None and task_id is not None:
				# For dmc
				goal = goal_embedding_func(task_id[env_idx])
				goal = torch.as_tensor(goal[None], device=self.device).repeat(obs.shape[0], 1)
			else:
				# Change goals based on modality
				if not self.goal_modality == 'pixels':
					goal = torch.tensor(goals[env_idx], device=self.device).float()
					goal = goal[None].repeat(obs.shape[0], 1)
				else:
					goal = goal.float()

			# augment
			if self.use_encoder and self.augment:
				# obs
				obs = self.aug(obs.float())
				obs = self.encoder(obs)
				#goal
				if self.goal_modality == 'pixels':
					goal = self.aug(goal.float())
					goal = self.encoder(goal)
			else:
				obs = obs.float()
				goal = goal.float()
			
			stddev = utils.schedule(self.stddev_schedule, step)
			dist = self.actor(obs, goal, stddev)
			
			# loss
			actor_loss = F.mse_loss(dist.mean, action, reduction='mean')
			losses.append(actor_loss)

		if self.pcgrad:
			self.opt.pc_backward(losses)
			self.opt.step()
		else:
			actor_loss = torch.stack(losses).mean()
			self.opt.zero_grad(set_to_none=True)
			actor_loss.backward()
			self.opt.step()

		if self.use_tb:
			metrics['actor_loss'] = actor_loss.item()

		return metrics

	def save_snapshot(self):
		keys_to_save = ['actor']
		if self.use_encoder:
			keys_to_save += ['encoder']
		payload = {k: self.__dict__[k] for k in keys_to_save}
		return payload

	def load_snapshot(self, payload, env_idx):
		if env_idx == 0:
			self.actor_expert, self.encoder_expert = [], []

		for k, v in payload.items():
			if k == 'actor':
				self.actor_expert.append(v)
			elif self.use_encoder and k == 'encoder':
				self.encoder_expert.append(v)
		
		# Turn off gradients for expert models
		for param in self.actor_expert[env_idx].parameters():
			param.requires_grad = False
		if self.use_encoder:
			for param in self.encoder_expert[env_idx].parameters():
				param.requires_grad = False
