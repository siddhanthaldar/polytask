#!/usr/bin/env python3

import warnings
import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path

import hydra
import random
import numpy as np
import torch
from dm_env import specs

import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader, make_expert_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import pickle

warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.backends.cudnn.benchmark = True

def make_agent(obs_spec, action_spec, cfg):
	cfg.obs_shape = obs_spec[cfg.obs_type].shape
	cfg.action_shape = action_spec.shape
	return hydra.utils.instantiate(cfg)

class Workspace:
	def __init__(self, cfg):
		self.work_dir = Path.cwd()
		print(f'workspace: {self.work_dir}')

		self.cfg = cfg
		utils.set_seed_everywhere(cfg.seed)
		self.device = torch.device(cfg.device)
		self.setup()

		self.agent = make_agent(self.train_env[0].observation_spec(),
								self.train_env[0].action_spec(), cfg.agent)

		if repr(self.agent) == 'drqv2' and not self.cfg.multitask:
			self.cfg.suite.num_train_frames = self.cfg.num_train_frames_drq
		if repr(self.agent) == 'bc':
			self.cfg.suite.num_train_frames = self.cfg.num_train_frames_bc
			self.cfg.suite.num_seed_frames = 0

		if repr(self.agent)=='bc' or self.cfg.bc_regularize:
			# load expert data
			self.expert_replay_loader, self.expert_replay_iter = [], []
			if self.cfg.multitask:
				if self.cfg.suite.name == 'metaworld':
					expert_dataset = Path(self.cfg.expert_dataset) / 'multitask' / 'expert_demos.pkl'
				elif self.cfg.suite.name == 'kitchen':
					expert_dataset = Path(self.cfg.expert_dataset) / self.cfg.task_name / f'multitask.pkl'
				elif self.cfg.suite.name == 'dmc':
					expert_dataset = Path(self.cfg.expert_dataset) / self.cfg.task_name / f'expert_demos_multitask.pkl'
				self.expert_replay_loader.append(make_expert_replay_loader(
															expert_dataset, self.cfg.batch_size, 
							       							self.cfg.num_demos_per_task, self.cfg.obs_type))
				self.expert_replay_iter.append(iter(self.expert_replay_loader[-1]))
			else:
				if self.cfg.suite.name == 'metaworld':
					for task_name in self.cfg.task_name:
						expert_dataset = Path(self.cfg.expert_dataset) / task_name / 'expert_demos.pkl'
						self.expert_replay_loader.append(make_expert_replay_loader(
																	expert_dataset, self.cfg.batch_size, 
																	self.cfg.num_demos_per_task, self.cfg.obs_type))
						self.expert_replay_iter.append(iter(self.expert_replay_loader[-1]))
				elif self.cfg.suite.name == 'kitchen':
					for task in self.cfg.suite.task:
						expert_dataset = Path(self.cfg.expert_dataset) / self.cfg.task_name / f'{task}.pkl'
						self.expert_replay_loader.append(make_expert_replay_loader(
																	expert_dataset, self.cfg.batch_size, 
																	self.cfg.num_demos_per_task, self.cfg.obs_type))
						self.expert_replay_iter.append(iter(self.expert_replay_loader[-1]))
				elif self.cfg.suite.name == 'dmc':
					for idx in self.cfg.suite.task_id:
						expert_dataset = Path(self.cfg.expert_dataset) / self.cfg.task_name / f'expert_demos_{idx}.pkl'
						self.expert_replay_loader.append(make_expert_replay_loader(
																	expert_dataset, self.cfg.batch_size, 
																	self.cfg.num_demos_per_task, self.cfg.obs_type))
						self.expert_replay_iter.append(iter(self.expert_replay_loader[-1]))
		else:
			self.expert_replay_loader = [None for _ in range(len(self.cfg.expert_dataset))]
			self.expert_replay_iter = [None for _ in range(len(self.cfg.expert_dataset))]
			
		self.timer = utils.Timer()
		self._global_step = 0
		self._global_episode = 0

		# Load expert data
		self.expert_demo, self.expert_action, self.expert_reward, self.expert_goal = [], [], [], []
		if self.cfg.multitask:
			if self.cfg.suite.name == 'metaworld':
				expert_dataset = Path(self.cfg.expert_dataset) / 'multitask' / 'expert_demos.pkl'
			elif self.cfg.suite.name == 'kitchen':
				expert_dataset = Path(self.cfg.expert_dataset) / self.cfg.task_name / f'multitask.pkl'
			elif self.cfg.suite.name == 'dmc':
				expert_dataset = Path(self.cfg.expert_dataset) / self.cfg.task_name / f'expert_demos_multitask.pkl'
			with open(expert_dataset, 'rb') as f:
				data = pickle.load(f)
				expert_action = data[2]
				expert_reward = data[-1]
				if self.cfg.obs_type == 'pixels':
					expert_demo = data[0]
				elif self.cfg.obs_type == 'features':
					expert_demo = data[1]
			self.expert_demo.append(expert_demo)#[:self.cfg.num_demos_per_task])
			self.expert_action.append(expert_action)#[:self.cfg.num_demos_per_task])
			self.expert_reward.append(0)
			self.expert_goal.extend([expert_demo[i][-1] for i in range(len(expert_demo))])
		else:
			if self.cfg.suite.name == 'metaworld':
				for task_name in self.cfg.task_name:
					expert_dataset = Path(self.cfg.expert_dataset) / task_name / 'expert_demos.pkl'
					with open(expert_dataset, 'rb') as f:
						data = pickle.load(f)
						if len(data) == 5 and len(data[-1]) < self.cfg.num_demos_per_task:
							data = data[:-1]
						expert_action = data[3] if len(data) == 5 else data[2]
						expert_reward = data[-1]
						if self.cfg.obs_type == 'pixels':
							expert_demo = [np.concatenate([data[0][i], data[1][i]], axis=1) for i in range(self.cfg.num_demos_per_task)] if len(data) == 5 else data[0]
						elif self.cfg.obs_type == 'features':
							expert_demo = data[2] if len(data) == 5 else data[1]
					self.expert_demo.append(expert_demo[:self.cfg.num_demos_per_task])
					self.expert_action.append(expert_action[:self.cfg.num_demos_per_task])
					self.expert_reward.append(0)
					self.expert_goal.append([expert_demo[i][-1] for i in range(self.cfg.num_demos_per_task)])
			elif self.cfg.suite.name == 'kitchen':
				for task in self.cfg.suite.task:
					expert_dataset = Path(self.cfg.expert_dataset) / self.cfg.task_name / f'{task}.pkl'
					with open(expert_dataset, 'rb') as f:
						data = pickle.load(f)
						expert_action = data[2]
						expert_reward = data[3]
						if self.cfg.obs_type == 'pixels':
							expert_demo = data[0]
						elif self.cfg.obs_type == 'features':
							expert_demo = data[1]
					self.expert_demo.append(expert_demo[:self.cfg.num_demos_per_task])
					self.expert_action.append(expert_action[:self.cfg.num_demos_per_task])
					self.expert_reward.append(0)
					self.expert_goal.append([expert_demo[i][-1] for i in range(self.cfg.num_demos_per_task)])
			elif self.cfg.suite.name == 'dmc':
				for idx in self.cfg.suite.task_id:
					expert_dataset = Path(self.cfg.expert_dataset) / self.cfg.task_name / f'expert_demos_{idx}.pkl'
					with open(expert_dataset, 'rb') as f:
						data = pickle.load(f)
						expert_action = data[2]
						expert_reward = data[3]
						if self.cfg.obs_type == 'pixels':
							expert_demo = data[0]
						elif self.cfg.obs_type == 'features':
							expert_demo = data[1]
					self.expert_demo.append(expert_demo[:self.cfg.num_demos_per_task])
					self.expert_action.append(expert_action[:self.cfg.num_demos_per_task])
					self.expert_reward.append(0)
					self.expert_goal.append([expert_demo[i][-1] for i in range(self.cfg.num_demos_per_task)])
			
		# Store task elements
		if self.cfg.suite.name == 'kitchen':
			from suite.kitchen import TASK_ELEMENTS
			self.tasks = [TASK_ELEMENTS[i] for i in self.cfg.suite.task]

	def setup(self):
		# create logger
		self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
		# create envs
		self.train_env = hydra.utils.call(self.cfg.suite.task_make_fn)
		self.eval_env = hydra.utils.call(self.cfg.suite.task_make_fn)
		
		# obs and goal shapes
		obs_shape = self.train_env[0].observation_spec()[self.cfg.obs_type].shape
		if self.cfg.obs_type == 'pixels':
			self.goal_shape = obs_shape
		elif self.cfg.obs_type == 'features':
			self.goal_shape = self.train_env[0].observation_spec()[self.cfg.obs_type].shape[0]
			self.goal_shape = self.goal_shape - 1 if self.goal_shape%2==1 else self.goal_shape
			self.goal_shape = (self.goal_shape, )

		# create replay buffer
		obs_shape = self.train_env[0].observation_spec()[self.cfg.obs_type].shape
		goal_dtype = np.uint8 if self.cfg.suite.name in ['metaworld', 'kitchen'] else np.float32
		data_specs = [
			self.train_env[0].observation_spec()[self.cfg.obs_type],
			specs.Array(self.goal_shape, goal_dtype, 'goal'),
			self.train_env[0].action_spec(),
			specs.Array((1, ), np.float32, 'reward'),
			specs.Array((1, ), np.float32, 'discount'),
		]

		self.replay_storage, self.replay_loader, self._replay_iter = [], [], []
		for env_idx in range(len(self.train_env)):
			self.replay_storage.append(ReplayBufferStorage(data_specs,
														   self.work_dir / f'buffer_{env_idx}'))

			self.replay_loader.append(make_replay_loader(
											self.work_dir / f'buffer_{env_idx}', self.cfg.replay_buffer_size,
											self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
											self.cfg.suite.save_snapshot, self.cfg.nstep, self.cfg.suite.discount))

			self._replay_iter.append(None)

		self.video_recorder = VideoRecorder(
			self.work_dir if self.cfg.save_video else None)
		self.train_video_recorder = TrainVideoRecorder(
			self.work_dir if self.cfg.save_train_video else None)

	@property
	def global_step(self):
		return self._global_step

	@property
	def global_episode(self):
		return self._global_episode

	@property
	def global_frame(self):
		return self.global_step * self.cfg.suite.action_repeat

	@property
	def replay_iter(self):
		if self._replay_iter[self.env_idx] is None:
			self._replay_iter[self.env_idx] = iter(self.replay_loader[self.env_idx])
		return self._replay_iter
			
	def positional_goal_embedding(self, goal):
		# Using sinusoidal positional encoding for goal embedding
		# https://arxiv.org/pdf/1706.03762.pdf
		pe = np.zeros(self.goal_shape[0])
		for i in range(0, self.goal_shape[0], 2):
			pe[i] = np.sin(goal / (10000 ** ((2*i)/self.goal_shape[0])))
			pe[i+1] = np.cos(goal / (10000 ** ((2*(i+1))/self.goal_shape[0])))
		return pe.astype(np.float32)

	def eval(self):
		episode_rewards = []
		success_percentages = []
		for env_idx in range(len(self.eval_env)):
			step, episode, total_reward = 0, 0, 0
			eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)

			paths = []
			time_step = self.eval_env[env_idx].reset()
			self.video_recorder.init(self.eval_env[env_idx], enabled=(episode == 0))
			while eval_until_episode(episode):
				if self.cfg.suite.name in ['kitchen']:
					completions = set()
				elif self.cfg.suite.name in ['metaworld']:
					path = []
				
				# get goal
				if self.cfg.suite.name in ['metaworld', 'kitchen']:
					if self.cfg.multitask:
						goal_idx = np.random.randint(env_idx*self.cfg.num_demos_per_task, (env_idx+1)*self.cfg.num_demos_per_task)
						goal = self.expert_goal[goal_idx]
					else:
						goal = self.expert_goal[env_idx][np.random.randint(len(self.expert_goal[env_idx]))]
				elif self.cfg.suite.name == 'dmc':
					goal =  self.positional_goal_embedding(self.cfg.suite.task_id[env_idx])
				
				# get task id
				kwargs = {}
				if repr(self.agent) == 'progressive':
					kwargs['task_id'] = min(self.env_idx, env_idx)

				time_step = self.eval_env[env_idx].reset()
				while not time_step.last():
					with torch.no_grad(), utils.eval_mode(self.agent):
						action = self.agent.act(time_step.observation[self.cfg.obs_type],
												goal,
												self.global_step,
												eval_mode=True,
												**kwargs)
			
					time_step = self.eval_env[env_idx].step(action)

					if self.cfg.suite.name == 'kitchen':
						if len(time_step.observation['completions']) > 0:
							for c in time_step.observation['completions']:
								completions.add(c)
					elif self.cfg.suite.name == 'metaworld':
						path.append(time_step.observation['goal_achieved'])

					# record 
					self.video_recorder.record(self.eval_env[env_idx])
					total_reward += time_step.reward
					step += 1

				episode += 1

				# Update success rate
				if self.cfg.suite.name in ['kitchen']:
					# Compute success score
					score = 0
					for c in completions:
						if c == self.tasks[env_idx]:
						# if c in self.tasks:
							score += 1
						else:
							score -= 2
					paths.append(score)
				elif self.cfg.suite.name in ['metaworld']:
					paths.append(1 if np.sum(path)>10 else 0)

			self.video_recorder.save(f'env_{env_idx}_frame_{self.global_frame}.mp4')
			
			episode_rewards.append(total_reward / episode)
			success_percentages.append(np.mean(paths) if self.cfg.suite.name in ['metaworld', 'kitchen'] else episode_rewards[-1]/1000)
		
		with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
			for env_idx in range(len(self.eval_env)):
				log(f'episode_reward_env_{env_idx}', episode_rewards[env_idx])
				log(f'success_percentage_env_{env_idx}', success_percentages[env_idx])
			log('episode_reward', np.mean(episode_rewards))
			log('episode_length', step * self.cfg.suite.action_repeat / episode)
			log('episode', self.global_episode)
			log('step', self.global_step)
			if self.cfg.multitask:
				log('mean_task_success', np.mean(success_percentages))
	
	def train(self):

		if repr(self.agent) in ['bc']:
			self.cfg.suite.action_repeat = 1

		# predicates
		train_until_step = utils.Until(self.cfg.suite.num_train_frames,
									   self.cfg.suite.action_repeat)
		seed_until_step = utils.Until(self.cfg.suite.num_seed_frames,
									  self.cfg.suite.action_repeat)
		eval_every_step = utils.Every(self.cfg.suite.eval_every_frames,
									  self.cfg.suite.action_repeat)

		episode_step, episode_reward = 0, 0
		self.env_idx = 0

		time_steps = list()
		observations = list()
		actions = list()

		# init for first task for progressive
		if repr(self.agent) == 'progressive':
			self.agent.switch_task()

		if repr(self.agent) not in ['bc']:
			# get goal
			if self.cfg.suite.name in ['metaworld', 'kitchen']:
				if self.cfg.multitask:
					goal_idx = np.random.randint(self.env_idx*self.cfg.num_demos_per_task, (self.env_idx+1)*self.cfg.num_demos_per_task)
					goal = self.expert_goal[goal_idx]
				else:
					goal = self.expert_goal[self.env_idx][np.random.randint(len(self.expert_goal[self.env_idx]))]
			elif self.cfg.suite.name == 'dmc':
				goal =  self.positional_goal_embedding(self.cfg.suite.task_id[self.env_idx])

			time_step = self.train_env[self.env_idx].reset()
			time_step = time_step._replace(goal=goal)
			time_steps.append(time_step)
			observations.append(time_step.observation[self.cfg.obs_type])
			actions.append(time_step.action)
			
			if self.cfg.irl:
				if self.agent.auto_rew_scale:
					self.agent.sinkhorn_rew_scale = 1.  # Set after first episode

			self.train_video_recorder.init(time_step.observation[self.cfg.obs_type])

		# goal embedding function only for dmc
		if self.cfg.suite.name == 'dmc':
			goal_embedding_func = self.positional_goal_embedding
			task_id = self.cfg.suite.task_id
		else:
			goal_embedding_func = None
			task_id = None

		metrics = None
		while train_until_step(self.global_step):
			if repr(self.agent) not in ['bc']:
				if time_step.last():
					self._global_episode += 1
					if self._global_episode % 10 == 0:
						self.train_video_recorder.save(f'{self.global_frame}.mp4')
					# wait until all the metrics schema is populated
					observations = np.stack(observations, 0)
					actions = np.stack(actions, 0)
					if self.cfg.irl:
						new_rewards = self.agent.ot_rewarder(
							observations, self.expert_demo[self.env_idx], self.global_step)
						new_rewards_sum = np.sum(new_rewards)
					
						if self.agent.auto_rew_scale: 
							if self._global_episode == 1:
								self.agent.sinkhorn_rew_scale = self.agent.sinkhorn_rew_scale * self.agent.auto_rew_scale_factor / float(
									np.abs(new_rewards_sum))
								new_rewards = self.agent.ot_rewarder(
									observations, self.expert_demo[self.env_idx], self.global_step)
								new_rewards_sum = np.sum(new_rewards)

					for i, elt in enumerate(time_steps):
						elt = elt._replace(
							observation=time_steps[i].observation[self.cfg.obs_type])
						if self.cfg.irl:
							elt = elt._replace(reward=new_rewards[i])
						self.replay_storage[self.env_idx].add(elt)

					if metrics is not None:
						# log stats
						elapsed_time, total_time = self.timer.reset()
						episode_frame = episode_step * self.cfg.suite.action_repeat
						with self.logger.log_and_dump_ctx(self.global_frame,
														ty='train') as log:
							log('fps', episode_frame / elapsed_time)
							log('total_time', total_time)
							log('episode_reward', episode_reward)
							log('episode_length', episode_frame)
							log('episode', self.global_episode)
							log('buffer_size', len(self.replay_storage))
							log('step', self.global_step)
							log('env_idx', self.env_idx)
							if self.cfg.irl:
								log('expert_reward', 0)
								log('imitation_reward', new_rewards_sum)

					# try to save snapshot
					if self.cfg.suite.save_snapshot:
						self.save_snapshot(f"env_{self.env_idx}")
					
					# switch env
					if self.global_frame > self.cfg.num_train_frames_per_env * (self.env_idx + 1):
						self.env_idx += 1
						if self.env_idx >= len(self.train_env):
							break
						print(f"Switching to env {self.env_idx}")
						self.agent.switch_task()
						seed_until_step = utils.Until(self.global_frame + self.cfg.suite.num_seed_frames,
													  self.cfg.suite.action_repeat)
						metrics = None

					# reset env
					time_steps = list()
					observations = list()
					actions = list()
					
					# get goal
					if self.cfg.suite.name in ['metaworld', 'kitchen']:
						if self.cfg.multitask:
							goal_idx = np.random.randint(self.env_idx*self.cfg.num_demos_per_task, (self.env_idx+1)*self.cfg.num_demos_per_task)
							goal = self.expert_goal[goal_idx]
						else:
							goal = self.expert_goal[self.env_idx][np.random.randint(len(self.expert_goal[self.env_idx]))]
					elif self.cfg.suite.name == 'dmc':
						goal =  self.positional_goal_embedding(self.cfg.suite.task_id[self.env_idx])

					time_step = self.train_env[self.env_idx].reset()
					time_step = time_step._replace(goal=goal)
					time_steps.append(time_step)
					observations.append(time_step.observation[self.cfg.obs_type])
					actions.append(time_step.action) 
					self.train_video_recorder.init(time_step.observation[self.cfg.obs_type])
					episode_step = 0
					episode_reward = 0

			# try to evaluate
			if eval_every_step(self.global_step):
				self.logger.log('eval_total_time', self.timer.total_time(),
								self.global_frame)
				self.eval()
				
			# sample action
			if repr(self.agent) not in ['bc']:
				with torch.no_grad(), utils.eval_mode(self.agent):
					action = self.agent.act(time_step.observation[self.cfg.obs_type],
			     							goal,
											self.global_step,
											eval_mode=False)

			# try to update the agent
			if not seed_until_step(self.global_step):
				# Update
				metrics = self.agent.update(self.replay_iter[:self.env_idx+1], self.expert_replay_iter, self.global_step, 
											self.cfg.bc_regularize, goal_embedding_func, task_id, self.env_idx)
				metrics['env_idx'] = self.env_idx
				self.logger.log_metrics(metrics, self.global_frame, ty='train')

				if repr(self.agent) in ["bc"] and self.global_step % 1000 == 0:
					if metrics is not None:
						# log stats
						elapsed_time, total_time = self.timer.reset()
						with self.logger.log_and_dump_ctx(self.global_frame,
														ty='train') as log:
							log('total_time', total_time)
							log('step', self.global_step)
							log('episode_reward', metrics['actor_loss'])
					
					# Save snapshot
					# try to save snapshot
					if self.cfg.suite.save_snapshot and self.global_step % 5000 == 0:
						self.save_snapshot(self._global_step)

			if repr(self.agent) not in ['bc']:
				# take env step
				time_step = self.train_env[self.env_idx].step(action)
				time_step = time_step._replace(goal=goal)
				episode_reward += time_step.reward

				time_steps.append(time_step)
				observations.append(time_step.observation[self.cfg.obs_type])
				actions.append(time_step.action)

				self.train_video_recorder.record(time_step.observation[self.cfg.obs_type])
			episode_step += 1
			self._global_step += 1

	def save_snapshot(self, name=None):
		if name is None:
			name = ""
		snapshot_dir = self.work_dir / 'weights'
		snapshot_dir.mkdir(exist_ok=True, parents=True)
		snapshot = snapshot_dir / f'snapshot_{name}.pt'
		keys_to_save = ['timer', '_global_step', '_global_episode']
		payload = {k: self.__dict__[k] for k in keys_to_save}
		payload.update(self.agent.save_snapshot())
		with snapshot.open('wb') as f:
			torch.save(payload, f)

	def load_snapshot(self, snapshot, env_idx):
		with snapshot.open('rb') as f:
			payload = torch.load(f)
		agent_payload = {}
		for k, v in payload.items():
			if k not in self.__dict__:
				agent_payload[k] = v
		self.agent.load_snapshot(agent_payload, env_idx)


@hydra.main(config_path='cfgs', config_name='config_baselines')
def main(cfg):
	from train_baselines import Workspace as W
	root_dir = Path.cwd()
	workspace = W(cfg)

	# Load weights
	if cfg.load_bc:
		bc_weight = Path(cfg.bc_weight)
		if not cfg.multitask:
			if cfg.suite.name == 'metaworld':
				for env_idx, task_name in enumerate(cfg.task_name):
					snapshot = bc_weight / task_name / f'{cfg.bc_weight_name}.pt'
					if snapshot.exists():
						print(f'resuming bc: {snapshot}')
						workspace.load_snapshot(snapshot, env_idx)
			elif cfg.suite.name == 'kitchen':
				for env_idx, task in enumerate(cfg.suite.task):
					snapshot = bc_weight / cfg.task_name / f'{cfg.bc_weight_name}_{task}.pt'
					if snapshot.exists():
						print(f'resuming bc: {snapshot}')
						workspace.load_snapshot(snapshot, env_idx)
			elif cfg.suite.name == 'dmc':
				for env_idx, task_id in enumerate(cfg.suite.task_id):
					snapshot = bc_weight / cfg.task_name / f'{cfg.bc_weight_name}_{task_id}.pt'
					if snapshot.exists():
						print(f'resuming bc: {snapshot}')
						workspace.load_snapshot(snapshot, env_idx)
		else:
			if cfg.suite.name == 'metaworld':
				snapshot = bc_weight / 'multitask' / f'{cfg.bc_weight_name}.pt'
			elif cfg.suite.name in ['kitchen', 'dmc']:
				snapshot = bc_weight / cfg.task_name / f'multitask_{cfg.bc_weight_name}.pt'
			if snapshot.exists():
				print(f'resuming bc: {snapshot}')
				workspace.load_snapshot(snapshot, 0)

	workspace.train()


if __name__ == '__main__':
	main()
