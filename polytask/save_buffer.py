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
import pickle
import numpy as np
import torch
from dm_env import specs

import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader

warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.backends.cudnn.benchmark = True

def make_agent(obs_spec, action_spec, cfg):
	cfg.obs_shape = obs_spec[cfg.obs_type].shape
	cfg.action_shape = action_spec.shape
	return hydra.utils.instantiate(cfg)

class WorkspaceIL:
	def __init__(self, cfg):
		self.work_dir = Path.cwd()
		print(f'workspace: {self.work_dir}')

		self.cfg = cfg
		utils.set_seed_everywhere(cfg.seed)
		self.device = torch.device(cfg.device)
		self.setup()

		self.agent = make_agent(self.eval_env[0].observation_spec(),
								self.eval_env[0].action_spec(), cfg.agent)
			
		self.timer = utils.Timer()
			
	def setup(self):
		# create logger
		self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
		# create envs
		self.eval_env = hydra.utils.call(self.cfg.suite.task_make_fn)

		# create replay buffer
		obs_shape = self.eval_env[0].observation_spec()[self.cfg.obs_type].shape
		data_specs = [
			self.eval_env[0].observation_spec()[self.cfg.obs_type],
			specs.Array(obs_shape, np.uint8, 'goal'),
			self.eval_env[0].action_spec(),
			specs.Array((1, ), np.float32, 'reward'),
			specs.Array((1, ), np.float32, 'discount'),
		]

		self.replay_storage, self.replay_loader, self._replay_iter = [], [], []
		if self.cfg.suite.name == 'metaworld':
			for task_name in self.cfg.task_name:
				task_name = "_".join(task_name.split('-')[:-1])
				buffer_path = Path(self.cfg.replay_buffers) / self.cfg.suite.name / task_name / f'buffer_{self.cfg.seed}'
				self.replay_storage.append(ReplayBufferStorage(data_specs,
															buffer_path))
				self.replay_loader.append(make_replay_loader(
												buffer_path, self.cfg.replay_buffer_size,
												self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
												self.cfg.suite.save_snapshot, self.cfg.nstep, self.cfg.suite.discount))
		elif self.cfg.suite.name == 'kitchen':
			for task in self.cfg.suite.task:
				buffer_path = Path(self.cfg.replay_buffers) / self.cfg.suite.name / task / f'buffer_{self.cfg.seed}'
				self.replay_storage.append(ReplayBufferStorage(data_specs,
															buffer_path))
				self.replay_loader.append(make_replay_loader(
												buffer_path, self.cfg.replay_buffer_size,
												self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
												self.cfg.suite.save_snapshot, self.cfg.nstep, self.cfg.suite.discount))
		elif self.cfg.suite.name == 'dmc':
			for task_id in self.cfg.suite.task_id:
				buffer_path = Path(self.cfg.replay_buffers) / self.cfg.suite.name / f'task_{task_id}' / f'buffer_{self.cfg.seed}'
				self.replay_storage.append(ReplayBufferStorage(data_specs,
															buffer_path))
				self.replay_loader.append(make_replay_loader(
												buffer_path, self.cfg.replay_buffer_size,
												self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
												self.cfg.suite.save_snapshot, self.cfg.nstep, self.cfg.suite.discount))
	
	def save(self):

		self.cfg.suite.action_repeat = 1

		# create save directory
		Path(self.cfg.save_replay_buffers).mkdir(exist_ok=True, parents=True)

		# save replay buffer
		if self.cfg.suite.name == 'metaworld':
			tasks = self.cfg.task_name
		elif self.cfg.suite.name == 'kitchen':
			tasks = self.cfg.suite.task
		elif self.cfg.suite.name == 'dmc':
			tasks = [f'task_{idx}' for idx in self.cfg.suite.task_id]
		for env_idx, task in enumerate(tasks):
			# make directory
			buffer_path = Path(self.cfg.save_replay_buffers) / task
			buffer_path.mkdir(exist_ok=True, parents=True)

			replay_dataset = self.replay_loader[env_idx].dataset
			replay_dataset._sample()
			ep_fns = replay_dataset._episode_fns
			
			# read dataset
			observations, actions = [], []
			for ep_fn in ep_fns:
				episode = replay_dataset._episodes[ep_fn]
				observations.append(episode['observation'])

				# get actions from model
				with torch.no_grad():
					# get goal
					goal = torch.as_tensor(episode['goal'], device=self.device).float()
					obs = torch.as_tensor(episode['observation'], device=self.device).float()
					if self.agent.use_encoder:
						goal = self.agent.encoder_expert[env_idx](goal)
						obs = self.agent.encoder_expert[env_idx](obs)
					dist = self.agent.actor_expert[env_idx](obs, goal, 0.1)
					actions.append(dist.mean.cpu().numpy())
			
			# save dataset
			dataset = [observations, actions]
			with open(buffer_path / f'buffer.pkl', 'wb') as f:
				pickle.dump(dataset, f)
			
			print(f"Saved buffer for {task} to {buffer_path}")			

	def load_snapshot(self, snapshot, env_idx):
		with snapshot.open('rb') as f:
			payload = torch.load(f)
		agent_payload = {}
		for k, v in payload.items():
			if k not in self.__dict__:
				agent_payload[k] = v
		self.agent.load_snapshot_eval(agent_payload, env_idx)


@hydra.main(config_path='cfgs', config_name='config_save_buffer')
def main(cfg):
	from save_buffer import WorkspaceIL as W

	root_dir = Path.cwd()
	workspace = W(cfg)

	# Load weights
	if cfg.suite.name == 'metaworld':
		for env_idx, task_name in enumerate(cfg.task_name):
			snapshot = Path(cfg.weight_dir) / task_name / f"{cfg.weight_name}_seed_{cfg.seed}.pt"
			if snapshot.exists():
				print(f'resuming model: {snapshot}')
				workspace.load_snapshot(snapshot, env_idx)
	elif cfg.suite.name == 'kitchen':
		for env_idx, task in enumerate(cfg.suite.task):
			snapshot = Path(cfg.weight_dir) / cfg.task_name / f'{cfg.weight_name}_{task}_seed_{cfg.seed}.pt'
			if snapshot.exists():
				print(f'resuming bc: {snapshot}')
				workspace.load_snapshot(snapshot, env_idx)
	elif cfg.suite.name == 'dmc':
		for env_idx, task_id in enumerate(cfg.suite.task_id):
			snapshot = Path(cfg.weight_dir) / cfg.task_name / f'{cfg.weight_name}_{task_id}_seed_{cfg.seed}.pt'
			if snapshot.exists():
				print(f'resuming bc: {snapshot}')
				workspace.load_snapshot(snapshot, env_idx)

	workspace.save()


if __name__ == '__main__':
	main()
