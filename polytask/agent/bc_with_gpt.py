import einops
import numpy as np
from collections import deque
import torch
from torch import nn, optim, distributions
from torch.nn import functional as F

import utils
from agent.networks.encoder import Encoder
# from agent.networks.gpt import GPT, GPTConfig
from agent.networks.mlp import MLP


class Actor(nn.Module):
	def __init__(self, repr_dim, action_shape, hidden_dim, policy_type='gpt',
	      		 history_len=1):
		super().__init__()

		self._output_dim = action_shape[0]
		self._history_len = history_len
		self._policy_type = policy_type
		hidden_dim = 256 if policy_type=='gpt' else hidden_dim

		# GPT model
		if policy_type == 'gpt':
			self.policy = GPT(
				GPTConfig(
					block_size=30,
					input_dim=repr_dim,
					output_dim=hidden_dim,
					n_layer=6,
					n_head=6,
					n_embd=120,
				)
			)
		elif policy_type == 'mlp':
			self.policy = nn.Sequential(nn.Linear(repr_dim * 2, hidden_dim),
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

		if self._policy_type == 'gpt':
			obs = torch.cat([goal, obs], dim=1)
		elif self._policy_type == 'mlp':
			obs = torch.cat([goal, obs], dim=-1)
		feat = self.policy(obs)
		shape = feat.shape
		feat = feat.view(-1, shape[-1])

		mu = torch.tanh(self._head(feat))
		std = torch.ones_like(mu) * std

		dist = utils.TruncatedNormal(mu, std)
		return dist

class BCAgent:
	def __init__(self, obs_shape, action_shape, device, lr, hidden_dim,
	      		 stddev_schedule, stddev_clip, use_tb, augment, suite_name,
				 obs_type, policy_type, history=True, history_len=1, eval_history_len=1):
		self.device = device
		self.lr = lr
		self.stddev_schedule = stddev_schedule
		self.stddev_clip = stddev_clip
		self.use_tb = use_tb
		self.augment = augment
		self.use_encoder = True if (suite_name!="adroit" and obs_type=='pixels') else False

		# actor parameters
		self.history = history
		self.history_len = history_len
		self.eval_history_len = eval_history_len

		# models
		if self.use_encoder:
			self.encoder = Encoder(obs_shape).to(device)
			repr_dim = self.encoder.repr_dim
		else:
			repr_dim = obs_shape[0]

		self.actor = Actor(repr_dim, action_shape, hidden_dim, policy_type, self.history_len).to(device)

		# optimizers
		if self.use_encoder:
			self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

		# data augmentation
		self.aug = utils.RandomShiftsAug(pad=4)

		self.train()
		self.buffer_reset()

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

	def buffer_reset(self):
		self.observation_buffer = deque(maxlen=self.eval_history_len)

	def act(self, obs, goal, step, eval_mode):
		self.observation_buffer.append(obs)
		obs = torch.as_tensor(np.array(self.observation_buffer), device=self.device).float()
		goal = torch.as_tensor(goal, device=self.device).float()

		obs = self.encoder(obs) if self.use_encoder else obs
		goal = self.encoder(goal[None]) if self.use_encoder else goal
		stddev = utils.schedule(self.stddev_schedule, step)

		dist = self.actor(obs[None], goal[None], stddev)

		if eval_mode:
			action = dist.mean
		else:
			action = dist.sample(clip=None)
		return action.cpu().numpy()[0]

	def update(self, replay_iter, expert_replay_iter, step, bc_regularize=False, act_low=-1, act_high=1, num_envs=None):
		metrics = dict()

		batch = next(expert_replay_iter[0])
		obs, action, goal = utils.to_torch(batch, self.device)
		if self.history:
			action = einops.rearrange(action, 'b t a -> (b t) a')
		action = action.float()
		
		# augment
		if self.use_encoder and self.augment:
			if self.history:
				obs = einops.rearrange(obs, 'b t c h w -> (b t) c h w')
			obs = self.aug(obs.float())
			goal = self.aug(goal.float())
			# encode
			goal = self.encoder(goal)
			obs = self.encoder(obs)
			if self.history:
				obs = einops.rearrange(obs, '(b t) d -> b t d', t=self.history_len)
		else:
			obs = obs.float()
			goal = goal.float()

		stddev = utils.schedule(self.stddev_schedule, step)
		dist = self.actor(obs, goal, stddev)
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)

		actor_loss = -log_prob.mean()
		
		if self.use_encoder:
			self.encoder_opt.zero_grad(set_to_none=True)
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()
		if self.use_encoder:
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

	def load_snapshot(self, payload):
		for k, v in payload.items():
			self.__dict__[k] = v

		# Update optimizers
		if self.use_encoder:
			self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
