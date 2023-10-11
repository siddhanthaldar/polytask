#!/usr/bin/env python3

"""
Distillation happens on expert data from given set of envs. Basically, imagine training an
expert and only storing expert data. Then, we train a distillation model on this expert data.

"""


import warnings
import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import utils
from logger import Logger
from replay_buffer import make_expert_replay_loader
from video import TrainVideoRecorder, VideoRecorder

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

		self.agent = make_agent(self.eval_env[0].observation_spec(),
								self.eval_env[0].action_spec(), cfg.agent)

		self.cfg.suite.num_train_frames = self.cfg.num_train_frames_distil

		# Goal
		self.goals = None
		if self.cfg.goal_modality == 'language':
			print("Initializing language goals ...")
			os.environ['TOKENIZERS_PARALLELISM'] = 'false'
			from sentence_transformers import SentenceTransformer
			model = SentenceTransformer('all-MiniLM-L6-v2')
			self.goals = model.encode(list(self.cfg.task_name))
		elif self.cfg.goal_modality == 'onehot':
			print("Initializing onehot goals ...")
			self.goals = np.eye(len(self.cfg.task_name))
			self.goals = np.concatenate([self.goals for _ in range(512//len(self.cfg.task_name) + 1)], axis=1)[:,:512]

			
		self.timer = utils.Timer()
		self._global_step = 0
		self._global_episode = 0
			
	def setup(self):
		# create logger
		self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
		# create envs
		self.eval_env = hydra.utils.call(self.cfg.suite.task_make_fn)

		# obs and goal shapes
		obs_shape = self.eval_env[0].observation_spec()[self.cfg.obs_type].shape
		if self.cfg.obs_type == 'pixels':
			self.goal_shape = obs_shape
		elif self.cfg.obs_type == 'features':
			self.goal_shape = self.eval_env[0].observation_spec()[self.cfg.obs_type].shape[0]
			self.goal_shape = self.goal_shape - 1 if self.goal_shape%2==1 else self.goal_shape
			self.goal_shape = (self.goal_shape, )

		# create replay buffer
		self.relabeled_replay_loader, self._relabeled_replay_iter = [], []
		if self.cfg.suite.name in ['metaworld', 'robotgym']:
			for task_name in self.cfg.task_name:
				buffer_path = Path(self.cfg.replay_buffers) / task_name / f'buffer.pkl'
				self.relabeled_replay_loader.append(make_expert_replay_loader(
												buffer_path, self.cfg.batch_size, None, self.cfg.obs_type, relabeled=True))
				self._relabeled_replay_iter.append(None)
		elif self.cfg.suite.name == 'kitchen':
			for task in self.cfg.suite.task:
				buffer_path = Path(self.cfg.replay_buffers) / task / f'buffer.pkl'
				self.relabeled_replay_loader.append(make_expert_replay_loader(
												buffer_path, self.cfg.batch_size, None, self.cfg.obs_type, relabeled=True))
				self._relabeled_replay_iter.append(None)
		elif self.cfg.suite.name == 'dmc':
			for task_id in self.cfg.suite.task_id:
				buffer_path = Path(self.cfg.replay_buffers) / f'task_{task_id}' / f'buffer.pkl'
				self.relabeled_replay_loader.append(make_expert_replay_loader(
												buffer_path, self.cfg.batch_size, None, self.cfg.obs_type, relabeled=True))
				self._relabeled_replay_iter.append(None)
			
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
		for env_idx in range(len(self._relabeled_replay_iter)):
			if self._relabeled_replay_iter[env_idx] is None:
				self._relabeled_replay_iter[env_idx] = iter(self.relabeled_replay_loader[env_idx])
		return self._relabeled_replay_iter

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
				
				# Get goal
				if self.cfg.suite.name in ['metaworld', 'kitchen']:
					if self.cfg.goal_modality in ['language', 'onehot']:
						goal = self.goals[env_idx]
					else:
						batch = next(self.replay_iter[env_idx])
						_, _, goal = utils.to_torch(batch, self.device)
						goal = goal[0]
				elif self.cfg.suite.name == 'dmc':
					goal =  self.positional_goal_embedding(self.cfg.suite.task_id[env_idx])
				
				time_step = self.eval_env[env_idx].reset()
				while not time_step.last():
					with torch.no_grad(), utils.eval_mode(self.agent):
						action = self.agent.act(time_step.observation[self.cfg.obs_type],
												goal,
												self.global_step,
												eval_mode=True)

					time_step = self.eval_env[env_idx].step(action)
					
					if self.cfg.suite.name in ['kitchen']:
						if len(time_step.observation['completions']) > 0:
							for c in time_step.observation['completions']:
								completions.add(c)
					elif self.cfg.suite.name in ['metaworld']:
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
			log('mean_success_percentage', np.mean(success_percentages))
			log('episode_length', step * self.cfg.suite.action_repeat / episode)
			log('episode', self.global_episode)
			log('step', self.global_step)
			if len(self.eval_env) > 1:
				log('mean_task_success', np.mean(success_percentages))
	
	def train(self):

		self.cfg.suite.action_repeat = 1

		# predicates
		train_until_step = utils.Until(self.cfg.suite.num_train_frames,
									   self.cfg.suite.action_repeat)
		eval_every_step = utils.Every(self.cfg.suite.eval_every_frames,
									  self.cfg.suite.action_repeat)

		episode_step = 0

		# goal embedding function only for dmc
		if self.cfg.suite.name == 'dmc':
			goal_embedding_func = self.positional_goal_embedding
			task_id = self.cfg.suite.task_id
		else:
			goal_embedding_func = None
			task_id = None

		metrics = None
		while train_until_step(self.global_step):
			# try to evaluate
			if eval_every_step(self.global_step):
				self.logger.log('eval_total_time', self.timer.total_time(),
								self.global_frame)
				self.eval()

			# update agent
			metrics = self.agent.update(self.replay_iter, self.goals, self.global_step,
							   			goal_embedding_func, task_id)
			self.logger.log_metrics(metrics, self.global_frame, ty='train')

			if self.global_step % 100 == 0:
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


@hydra.main(config_path='cfgs', config_name='config_distill')
def main(cfg):
	from train_distill import Workspace as W

	# Filter envs
	if cfg.suite.name == 'metaworld':
		cfg.task_name = cfg.task_name[:cfg.num_envs_to_distil]
	elif cfg.suite.name == 'kitchen':
		cfg.suite.task = cfg.suite.task[:cfg.num_envs_to_distil]
	elif cfg.suite.name == 'dmc':
		cfg.suite.task_id = cfg.suite.task_id[:cfg.num_envs_to_distil]

	root_dir = Path.cwd()
	workspace = W(cfg)

	# Load weights
	if cfg.suite.name in ['metaworld', 'robotgym']:
		for env_idx, task_name in enumerate(cfg.task_name):
			snapshot = Path(cfg.weight_dir) / task_name / f"{cfg.weight_name}.pt"
			if snapshot.exists():
				print(f'resuming model: {snapshot}')
				workspace.load_snapshot(snapshot, env_idx)
	elif cfg.suite.name == 'kitchen':
		for env_idx, task in enumerate(cfg.suite.task):
			snapshot = Path(cfg.weight_dir) / cfg.task_name / f'{cfg.weight_name}_{task}.pt'
			if snapshot.exists():
				print(f'resuming bc: {snapshot}')
				workspace.load_snapshot(snapshot, env_idx)
	elif cfg.suite.name == 'dmc':
		for env_idx, task_id in enumerate(cfg.suite.task_id):
			snapshot = Path(cfg.weight_dir) / cfg.task_name / f'{cfg.weight_name}_{task_id}.pt'
			if snapshot.exists():
				print(f'resuming bc: {snapshot}')
				workspace.load_snapshot(snapshot, env_idx)

	workspace.train()


if __name__ == '__main__':
	main()
