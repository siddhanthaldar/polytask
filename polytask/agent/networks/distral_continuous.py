import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np

import utils
from agent.networks.encoder import Encoder
from agent.networks.mlp import MLP
from agent.networks.pcgrad import PCGrad

class Actor(nn.Module):
	def __init__(self, obs_shape, action_shape, hidden_dim, use_encoder):
		super().__init__()

		self._output_dim = action_shape[0]
		self.use_encoder = use_encoder

		if self.use_encoder:
			self.encoder = Encoder(obs_shape)
			self.repr_dim = self.encoder.repr_dim
		else:
			self.repr_dim = obs_shape[0]

		if self.repr_dim != 512:
			goal_dim = self.repr_dim - 1 if self.repr_dim%2==1 else self.repr_dim
		else:
			goal_dim = self.repr_dim

		self.policy = nn.Sequential(nn.Linear(self.repr_dim + goal_dim, hidden_dim),
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

		if self.use_encoder:
			obs_encoder = self.encoder(obs)
			with torch.no_grad():
				goal_encoder = self.encoder(goal)
		else:
			obs_encoder = obs
			goal_encoder = goal

		obs = torch.cat([goal_encoder, obs_encoder], dim=-1)
		feat = self.policy(obs)

		mu = torch.tanh(self._head(feat))
		std = torch.ones_like(mu) * std

		dist = utils.TruncatedNormal(mu, std)
		return dist, (obs_encoder, goal_encoder)

class Critic(nn.Module):
	def __init__(self, repr_dim, action_shape, hidden_dim):
		super().__init__()
	
		if repr_dim != 512:
			goal_dim = repr_dim - 1 if repr_dim%2==1 else repr_dim
			self._repeat = (repr_dim + goal_dim) // action_shape[0]
		else:
			goal_dim = repr_dim
			self._repeat = 200

		self.Q1 = nn.Sequential(
			nn.Linear(repr_dim + goal_dim + self._repeat * action_shape[0], hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

		self.Q2 = nn.Sequential(
			nn.Linear(repr_dim + goal_dim + self._repeat * action_shape[0], hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

		self.apply(utils.weight_init)

	def forward(self, obs, goal, action):
		action = action.repeat(1,self._repeat)
		obs = torch.cat([goal, obs], dim=-1)
		h_action = torch.cat([obs, action], dim=-1)
		q1 = self.Q1(h_action)
		q2 = self.Q2(h_action)

		return q1, q2


class ContinuousDistral:
	"""
	Code borrowed and modified from
	https://github.com/giangbang/rl_codebase/blob/f4ca2af1f28c167b5fc14e7809a9e771f5083dad/rl_codebase/agents/distral/distral_continuous.py

	"""

	def __init__(self, obs_shape, action_shape, device, lr,
				 hidden_dim, critic_target_tau, num_expl_steps,
				 stddev_schedule, stddev_clip, use_tb,
				 obs_type, gamma=0.99, alpha=0.5, beta=5, **kwargs):
		self.device = device
		self.lr = lr
		self.critic_target_tau = critic_target_tau
		self.use_tb = use_tb
		self.num_expl_steps = num_expl_steps
		self.stddev_schedule = stddev_schedule
		self.stddev_clip = stddev_clip
		self.gamma = gamma # Not used
		self.alpha = alpha
		self.beta = beta
		self.use_encoder = True if obs_type=='pixels' else False

		# init actor
		self.actor = Actor(obs_shape, action_shape, hidden_dim, self.use_encoder).to(device)

		# init critic
		self.critic = Critic(self.actor.repr_dim, action_shape, hidden_dim).to(device)
		self.critic_target = Critic(self.actor.repr_dim, action_shape, hidden_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())

		# optimizers
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
		self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

		self.ent_coef = 1 / self.beta
		self.cross_ent_coef = self.alpha / self.beta

		# data augmentation
		self.aug = utils.RandomShiftsAug(pad=4)

		self.train()
		self.critic_target.train()
	
	def __repr__(self):
		return "drqv2"
	
	def train(self, training=True):
		self.training = training
		self.actor.train(training)
		self.critic.train(training)
	
	def act(self, obs, goal, step, eval_mode):
		obs = torch.as_tensor(obs, device=self.device).float()
		goal = torch.as_tensor(goal, device=self.device).float()

		stddev = utils.schedule(self.stddev_schedule, step)

		dist, _ = self.actor(obs.unsqueeze(0), goal.unsqueeze(0), stddev)

		if eval_mode:
			action = dist.mean
		else:
			action = dist.sample(clip=None)
			if step < self.num_expl_steps:
				action.uniform_(-1.0, 1.0)
		return action.cpu().numpy()[0]

	def update_critic(self, distill_policy, env_idx, obs, goal, action, reward, discount, next_obs, step):
		metrics = dict()

		with torch.no_grad():
			stddev = utils.schedule(self.stddev_schedule, step)
			dist, (next_obs_encoder, goal_encoder) = self.actor(next_obs, goal, stddev)
			next_action = dist.sample(clip=self.stddev_clip)
			target_Q1, target_Q2 = self.critic_target(next_obs_encoder, goal_encoder, next_action)
			# Next q value
			target_V = torch.min(target_Q1, target_Q2)
			# Add entropy term
			target_V = target_V - self.ent_coef * dist.log_prob(next_action).mean()
			# Get distill action
			dist_distill, _ = distill_policy.actor(next_obs, goal, stddev)
			# Add distill logprob term
			target_V = target_V + self.cross_ent_coef * dist_distill.log_prob(next_action).mean()
			target_Q = reward + (discount * target_V)

		obs_encoder = self.actor.encoder(obs) if self.use_encoder else obs
		Q1, Q2 = self.critic(obs_encoder, goal_encoder, action)

		critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

		# optimize encoder and critic
		self.critic_opt.zero_grad(set_to_none=True)
		critic_loss.backward()
		self.critic_opt.step()

		if self.use_tb:
			metrics[f'critic_target_q_env{env_idx}'] = target_Q.mean().item()
			metrics[f'critic_q1_env{env_idx}'] = Q1.mean().item()
			metrics[f'critic_q2_env{env_idx}'] = Q2.mean().item()
			metrics[f'critic_loss_env{env_idx}'] = critic_loss.item()
			
		return metrics

	def update_actor(self, distill_policy, env_idx, obs, goal, observations_bc, goals_bc, observations_qfilter, 
		  			 actions_bc, bc_regularize, step):
		metrics = dict()

		stddev = utils.schedule(self.stddev_schedule, step)

		dist, (obs_encoder, goal_encoder) = self.actor(obs, goal, stddev)
		action = dist.sample(clip=self.stddev_clip)
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)

		Q1, Q2 = self.critic(obs_encoder, goal_encoder, action)
		Q = torch.min(Q1, Q2)

		total_bc_weight = 0.0

		actor_loss = -Q.mean() * (1-total_bc_weight)

		# Add entropy term
		actor_loss += self.ent_coef * dist.log_prob(action).mean()

		# Add distill logprob
		dist_distill, _ = distill_policy.actor(obs, goal, stddev)
		distill_logprob = dist_distill.log_prob(action).mean()
		actor_loss -= self.cross_ent_coef * distill_logprob

		# optimize actor
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()
		self.actor_opt.step()

		if self.use_tb:
			metrics[f'actor_loss_env{env_idx}'] = actor_loss.item()
			metrics[f'actor_logprob_env{env_idx}'] = log_prob.mean().item()
			metrics[f'actor_ent_env{env_idx}'] = dist.entropy().sum(dim=-1).mean().item()
			metrics[f'rl_loss_env{env_idx}'] = -(Q* (1-total_bc_weight)).mean().item()

		return metrics

	def update(self, distill_policy, replay_iter, expert_replay_iter, env_idx, step, bc_regularize=False):
		metrics = dict()

		batch = next(replay_iter)
		obs, action, reward, discount, next_obs, goal = utils.to_torch(
			batch, self.device)
		
		# # augment
		if self.use_encoder:
			obs_qfilter = self.aug(obs.clone().float())
			obs = self.aug(obs.float())
			next_obs = self.aug(next_obs.float())
			goal = self.aug(goal.float())
		else:
			obs_qfilter = obs.clone().float()
			obs = obs.float()
			next_obs = next_obs.float()
			goal = goal.float()

		observations_bc, observations_qfilter, actions_bc, goals_bc = None, None, None, None

		# Copies to be returns
		obs_ret = obs.clone()
		goal_ret = goal.clone()
		action_ret = action.clone()

		if self.use_tb:
			metrics[f'batch_reward_env{env_idx}'] = reward.mean().item()

		# update critic
		metrics.update(
			self.update_critic(distill_policy, env_idx, obs, goal, action, reward, discount, next_obs, step))

		# update actor
		metrics.update(
			self.update_actor(distill_policy, env_idx, obs.detach(), goal, observations_bc, goals_bc, observations_qfilter, 
							  actions_bc, bc_regularize, step)
		)
			
		# update critic target
		utils.soft_update_params(self.critic, self.critic_target,
								 self.critic_target_tau)

		return metrics, (obs_ret, goal_ret, action_ret)

	def update_distill(self, obs, goal, action, env_idx, step):
		stddev = utils.schedule(self.stddev_schedule, step)
		dist, _ = self.actor(obs, goal, stddev)
		log_distill = dist.log_prob(action).sum(-1, keepdim=True)
		log_loss = -self.cross_ent_coef * log_distill.mean()
		# log_loss = self.log_loss(batch)

		self.actor_opt.zero_grad(set_to_none=True)
		log_loss.backward()
		self.actor_opt.step()

		metrics = dict()
		if self.use_tb:
			metrics[f'distill_loss_env{env_idx}'] = log_loss.item()

		return metrics
