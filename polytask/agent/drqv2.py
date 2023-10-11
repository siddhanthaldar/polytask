import hydra
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from agent.networks.encoder import Encoder
from agent.networks.mlp import MLP
from agent.networks.pcgrad import PCGrad
from rewarder import optimal_transport_plan, cosine_distance, euclidean_distance

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


class DrQv2Agent:
	def __init__(self, obs_shape, action_shape, device, lr, hidden_dim, critic_target_tau, num_expl_steps,
				 update_every_steps, stddev_schedule, stddev_clip, use_tb, augment, obs_type, train_encoder,
				 rewards, sinkhorn_rew_scale, update_target_every, auto_rew_scale, auto_rew_scale_factor, pcgrad):
		self.device = device
		self.lr = lr
		self.critic_target_tau = critic_target_tau
		self.update_every_steps = update_every_steps
		self.use_tb = use_tb
		self.num_expl_steps = num_expl_steps
		self.stddev_schedule = stddev_schedule
		self.stddev_clip = stddev_clip
		self.augment = augment
		self.use_encoder = True if obs_type=='pixels' else False
		self.train_encoder = train_encoder
		self.pcgrad = pcgrad

		# OT based rewards
		self.rewards = rewards
		self.sinkhorn_rew_scale = sinkhorn_rew_scale
		self.update_target_every = update_target_every
		self.auto_rew_scale = auto_rew_scale
		self.auto_rew_scale_factor = auto_rew_scale_factor

		# models
		if self.use_encoder:
			self.encoder = Encoder(obs_shape).to(device)
			self.encoder_target = Encoder(obs_shape).to(device)
			repr_dim = self.encoder.repr_dim
			if not self.train_encoder:
				for param in self.encoder.parameters():
					param.requires_grad = False
		else:
			repr_dim = obs_shape[0]

		self.actor = Actor(repr_dim, action_shape, hidden_dim).to(device)

		# For critic
		self.repr_dim = repr_dim
		self.action_shape = action_shape
		self.hidden_dim = hidden_dim

		self.critic = Critic(repr_dim, action_shape, hidden_dim).to(device)
		self.critic_target = Critic(repr_dim, action_shape, hidden_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())

		# optimizers
		# actor
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
		# critic and encoder
		params = list(self.critic.parameters()) + list(self.encoder.parameters()) if self.use_encoder and self.train_encoder else list(self.critic.parameters())
		self.critic_opt = torch.optim.Adam(params, lr=lr)
		if self.pcgrad:
			self.actor_opt = PCGrad(self.actor_opt)
			self.critic_opt = PCGrad(self.critic_opt)

		# data augmentation
		self.aug = utils.RandomShiftsAug(pad=4)

		self.train()
		self.critic_target.train()

	def __repr__(self):
		return "drqv2"
	
	def train(self, training=True):
		self.training = training
		if self.use_encoder and self.train_encoder:
			self.encoder.train(training)
		self.actor.train(training)
		self.critic.train(training)

	def act(self, obs, goal, step, eval_mode):
		obs = torch.as_tensor(obs, device=self.device).float()
		goal = torch.as_tensor(goal, device=self.device).float()

		obs = self.encoder(obs.unsqueeze(0)) if self.use_encoder else obs.unsqueeze(0)
		goal = self.encoder(goal.unsqueeze(0)) if self.use_encoder and len(goal.shape) == 3 else goal.unsqueeze(0)
		stddev = utils.schedule(self.stddev_schedule, step)

		dist = self.actor(obs, goal, stddev)

		if eval_mode:
			action = dist.mean
		else:
			action = dist.sample(clip=None)
			if step < self.num_expl_steps:
				action.uniform_(-1.0, 1.0)
		return action.cpu().numpy()[0]

	def act_eval(self, obs, goal, env_idx):
		obs = torch.as_tensor(obs, device=self.device).float()
		goal = torch.as_tensor(goal, device=self.device).float()

		obs = self.encoder_expert[env_idx](obs.unsqueeze(0)) if self.use_encoder else obs.unsqueeze(0)
		goal = self.encoder_expert[env_idx](goal.unsqueeze(0)) if self.use_encoder else goal.unsqueeze(0)
		stddev = 0.1

		dist = self.actor_expert[env_idx](obs, goal, stddev)
		action = dist.mean
		return action.cpu().numpy()[0]

	def update_critic(self, obs, goal, action, reward, discount, next_obs, step):
		metrics = dict()

		losses = []
		for env_idx in range(len(obs)):
			with torch.no_grad():
				stddev = utils.schedule(self.stddev_schedule, step)
				dist = self.actor(next_obs[env_idx], goal[env_idx], stddev)
				next_action = dist.sample(clip=self.stddev_clip)
				target_Q1, target_Q2 = self.critic_target(next_obs[env_idx], goal[env_idx], next_action)
				target_V = torch.min(target_Q1, target_Q2)
				target_Q = reward[env_idx] + (discount[env_idx] * target_V)

			Q1, Q2 = self.critic(obs[env_idx], goal[env_idx], action[env_idx])

			critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
			losses.append(critic_loss)
			
		# optimize encoder and critic
		if self.pcgrad:
			self.critic_opt.pc_backward(losses)
			self.critic_opt.step()
		else:
			self.critic_opt.zero_grad(set_to_none=True)
			critic_loss = torch.stack(losses).mean()
			critic_loss.backward()
			self.critic_opt.step()
		
		if self.use_tb:
			metrics['critic_target_q'] = target_Q.mean().item()
			metrics['critic_q1'] = Q1.mean().item()
			metrics['critic_q2'] = Q2.mean().item()
			metrics['critic_loss'] = critic_loss.item() #torch.tensor(losses).mean().item()
			
		return metrics

	def update_actor(self, obs, goal, observations_bc, goals_bc, actions_bc, bc_regularize, step):
		metrics = dict()

		stddev = utils.schedule(self.stddev_schedule, step)

		losses = []
		for env_idx in range(len(obs)):
			dist = self.actor(obs[env_idx], goal[env_idx], stddev)
			action = dist.sample(clip=self.stddev_clip)
			log_prob = dist.log_prob(action).sum(-1, keepdim=True)

			Q1, Q2 = self.critic(obs[env_idx], goal[env_idx], action)
			Q = torch.min(Q1, Q2)

			# Compute bc weight
			if not bc_regularize:
				total_bc_weight = 0.0
			else:
				# TODO: Don't call it qfilter
				bc_weight = [0.25 / len(observations_bc) for _ in range(len(observations_bc))]
				total_bc_weight = sum(bc_weight)
				
			actor_loss = -Q.mean() * (1-total_bc_weight)

			if bc_regularize and env_idx == 0:
				stddev = 0.1
				bc_loss = []
				for expert_idx in range(len(observations_bc)):
					dist_bc = self.actor(observations_bc[expert_idx], goals_bc[expert_idx], stddev)
					log_prob_bc = dist_bc.log_prob(actions_bc[expert_idx]).sum(-1, keepdim=True)
					bc_loss.append(- log_prob_bc.mean()*bc_weight[env_idx]*0.3)
					actor_loss += bc_loss[-1]
			
			losses.append(actor_loss)

		# optimize
		if self.pcgrad:
			self.actor_opt.pc_backward(losses)
			self.actor_opt.step()
		else:
			self.actor_opt.zero_grad(set_to_none=True)
			actor_loss = torch.stack(losses).mean()
			actor_loss.backward()
			self.actor_opt.step()

		if self.use_tb:
			metrics['actor_loss'] = actor_loss.item()
			metrics['actor_logprob'] = log_prob.mean().item()
			metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
			metrics['rl_loss'] = -(Q* (1-total_bc_weight)).mean().item()
			if bc_regularize:
				for env_idx in range(len(observations_bc)):
					metrics[f'bc_weight_{env_idx}'] = bc_weight[env_idx]
					metrics[f'bc_loss_{env_idx}'] = bc_loss[env_idx].item()
			
		return metrics

	def update(self, replay_iter, expert_replay_iter, step, bc_regularize=False, 
			   goal_embedding_func=None, task_id=None):
		metrics = dict()

		if step % self.update_every_steps != 0:
			return metrics
		
		buffer_idx = [random.randint(0, len(replay_iter)-1)] if not self.pcgrad else range(len(replay_iter))
		observations, actions, rewards, discounts, next_observations, goals = [], [], [], [], [], []
		observations_bc, actions_bc, goals_bc = None, None, None
		for env_idx in buffer_idx:
			batch = next(replay_iter[env_idx])
			obs, action, reward, discount, next_obs, goal = utils.to_torch(
				batch, self.device)
			action = action.float()
			
			# augment
			if self.use_encoder and self.augment:
				obs = self.aug(obs.float())
				next_obs = self.aug(next_obs.float())
				goal = self.aug(goal.float()) if len(goal.shape)==4 else goal.float()
			else:
				obs = obs.float()
				next_obs = next_obs.float()
				goal = goal.float()

			if self.use_encoder:
				# encode
				obs = self.encoder(obs)
				with torch.no_grad():
					next_obs = self.encoder(next_obs)
					if len(goal.shape)==4:
						goal = self.encoder(goal)
			
			observations.append(obs)
			actions.append(action)
			rewards.append(reward)
			discounts.append(discount)
			next_observations.append(next_obs)
			goals.append(goal)

			if bc_regularize and env_idx == 0:
				observations_bc, actions_bc, goals_bc = [], [], []
				for env_idx in range(len(expert_replay_iter)):
					batch = next(expert_replay_iter[env_idx])
					obs_bc, act_bc, goal_bc = utils.to_torch(batch, self.device)
					act_bc = act_bc.float()
					# get goal embedding
					if not self.use_encoder and goal_embedding_func is not None and task_id is not None:
						goal_bc = goal_embedding_func(task_id[env_idx])
						goal_bc = torch.as_tensor(goal_bc[None], device=self.device).repeat(obs_bc.shape[0], 1)
					# augment
					if self.use_encoder and self.augment:
						obs_bc = self.aug(obs_bc.float())
						goal_bc = self.aug(goal_bc.float()) if len(goal_bc.shape)==4 else goal_bc.float()
					else:
						obs_bc = obs_bc.float()
						goal_bc = goal_bc.float()
					obs_bc = self.encoder(obs_bc) if self.use_encoder else obs_bc
					goal_bc = self.encoder(goal_bc) if self.use_encoder and len(goal_bc.shape)==4 else goal_bc
					# Detach grads
					obs_bc = obs_bc.detach()
					goal_bc = goal_bc.detach()
					
					# Save in lists
					observations_bc.append(obs_bc)
					actions_bc.append(act_bc)
					goals_bc.append(goal_bc)
			
			if self.use_tb:
				metrics['batch_reward'] = reward.mean().item()

		# update critic
		metrics.update(
			self.update_critic(observations, goals, actions, rewards, discounts, next_observations, step))

		# update actor
		metrics.update(
			self.update_actor([obs.detach() for obs in observations], goals, observations_bc, 
					 		   goals_bc, actions_bc, bc_regularize, step)
		)
			
		# update critic target
		utils.soft_update_params(self.critic, self.critic_target,
								 self.critic_target_tau)

		return metrics


	def ot_rewarder(self, observations, demos, step):

		if step % self.update_target_every == 0:
			if self.use_encoder and self.train_encoder:
				self.encoder_target.load_state_dict(self.encoder.state_dict())
			self.target_updated = True

		scores_list = list()
		ot_rewards_list = list()
		for demo in demos:
			obs = torch.tensor(observations).to(self.device).float()
			obs = self.encoder_target(obs) if self.use_encoder else obs
			exp = torch.tensor(demo).to(self.device).float()
			exp = self.encoder_target(exp) if self.use_encoder else exp
			obs = obs.detach()
			exp = exp.detach()
			
			if self.rewards == 'sinkhorn_cosine':
				cost_matrix = cosine_distance(
					obs, exp)  # Get cost matrix for samples using critic network.
				transport_plan = optimal_transport_plan(
					obs, exp, cost_matrix, method='sinkhorn',
					niter=100).float()  # Getting optimal coupling
				ot_rewards = -self.sinkhorn_rew_scale * torch.diag(
					torch.mm(transport_plan,
							 cost_matrix.T)).detach().cpu().numpy()
				
			elif self.rewards == 'sinkhorn_euclidean':
				cost_matrix = euclidean_distance(
					obs, exp)  # Get cost matrix for samples using critic network.
				transport_plan = optimal_transport_plan(
					obs, exp, cost_matrix, method='sinkhorn',
					niter=100).float()  # Getting optimal coupling
				ot_rewards = -self.sinkhorn_rew_scale * torch.diag(
					torch.mm(transport_plan,
							 cost_matrix.T)).detach().cpu().numpy()
				
			elif self.rewards == 'cosine':
				exp = torch.cat((exp, exp[-1].unsqueeze(0)))
				ot_rewards = -(1. - F.cosine_similarity(obs, exp))
				ot_rewards *= self.sinkhorn_rew_scale
				ot_rewards = ot_rewards.detach().cpu().numpy()
				
			elif self.rewards == 'euclidean':
				exp = torch.cat((exp, exp[-1].unsqueeze(0)))
				ot_rewards = -(obs - exp).norm(dim=1)
				ot_rewards *= self.sinkhorn_rew_scale
				ot_rewards = ot_rewards.detach().cpu().numpy()
				
			else:
				raise NotImplementedError()

			scores_list.append(np.sum(ot_rewards))
			ot_rewards_list.append(ot_rewards)

		closest_demo_index = np.argmax(scores_list)
		return ot_rewards_list[closest_demo_index]


	def save_snapshot(self):
		keys_to_save = ['actor', 'critic']
		if self.use_encoder:
			keys_to_save += ['encoder']
		payload = {k: self.__dict__[k] for k in keys_to_save}
		return payload

	def load_snapshot(self, payload, env_idx):
		if env_idx == 0:
			for k, v in payload.items():
				self.__dict__[k] = v
			self.critic_target.load_state_dict(self.critic.state_dict())
			if self.use_encoder:
				self.encoder_target.load_state_dict(self.encoder.state_dict())

			if not self.train_encoder and self.use_encoder:
				for param in self.encoder.parameters():
					param.requires_grad = False

		# Update optimizers
		if env_idx == 0:
			# actor
			self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
			# critic and encoder
			params = list(self.critic.parameters()) + list(self.encoder.parameters()) if self.use_encoder and self.train_encoder else list(self.critic.parameters())
			self.critic_opt = torch.optim.Adam(params, lr=self.lr)
			if self.pcgrad:
				self.actor_opt = PCGrad(self.actor_opt)
				self.critic_opt = PCGrad(self.critic_opt)
	
	def load_snapshot_eval(self, payload, env_idx):
		if env_idx == 0:
			self.actor_expert = []
			if self.use_encoder:
				self.encoder_expert = []

		for k, v in payload.items():
			if k == 'actor':
				self.actor_expert.append(v)
				self.actor_expert[env_idx].eval()
			elif self.use_encoder and k == 'encoder':
				self.encoder_expert.append(v)
				self.encoder_expert[env_idx].eval()
		
		# Turn off gradients for expert models
		for param in self.actor_expert[env_idx].parameters():
			param.requires_grad = False
		if self.use_encoder:
			for param in self.encoder_expert[env_idx].parameters():
				param.requires_grad = False
