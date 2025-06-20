import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'
import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored
from omegaconf import OmegaConf

from common.parser import parse_cfg
from common.seed import set_seed
from common.mixed_buffer import MixedBuffer
from envs import make_envs
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger, print_run
import multiprocessing

import gymnasium as gym

torch.backends.cudnn.benchmark = True


def make_envs_with_colored_eval(cfg, num_envs, video_path=None, is_eval=False, logger=None):
	"""Modified environment creation that uses colored table variants for evaluation"""
	if is_eval:
		# Use the colored environment creator for evaluation
		from envs.maniskill_colored import make_envs as make_colored_envs
		return make_colored_envs(cfg, num_envs, video_path, is_eval, logger)
	else:
		# Use regular environments for training
		return make_envs(cfg, num_envs, video_path, is_eval, logger)


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
	"""
	Script for training TD-MPC2 agents with synthetic data augmentation and colored table evaluation.

	New args:
		`synthetic_data_dir`: path to synthetic data created by postprocess.py
		`synthetic_ratio`: ratio of synthetic data in training batches (default: 0.2)
		`use_colored_eval`: whether to use colored table variants for evaluation (default: True)

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task training)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`steps`: number of training/environment steps (default: 10M)
		`seed`: random seed (default: 1)

	See config.yaml for a full list of args.

	Example usage:
	```
		$ python train_with_synthetic.py task=PickCube-v1 synthetic_data_dir=./synthetic_data
		$ python train_with_synthetic.py task=PickCube-v1 synthetic_data_dir=./synthetic_data synthetic_ratio=0.3
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	assert not cfg.multitask, colored('Warning: multi-task models is not currently supported for maniskill.', 'red', attrs=['bold'])
	set_seed(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)
	
	# Handle synthetic data configuration
	synthetic_data_dir = getattr(cfg, 'synthetic_data_dir', None)
	cfg.synthetic_ratio = getattr(cfg, 'synthetic_ratio', 0.2)
	use_colored_eval = getattr(cfg, 'use_colored_eval', True)
	
	if synthetic_data_dir:
		print(colored(f'Using synthetic data from: {synthetic_data_dir}', 'green', attrs=['bold']))
		print(colored(f'Synthetic data ratio: {cfg.synthetic_ratio:.1%}', 'green', attrs=['bold']))
	
	if use_colored_eval and cfg.env_id == 'PickCube-v1':
		print(colored('Using colored table variants for evaluation', 'blue', attrs=['bold']))
	
	# Need to initiate logger before make env to wrap record episode wrapper into async vec cpu env
	manager = multiprocessing.Manager()
	video_path = cfg.work_dir / 'eval_video'
	if cfg.save_video_local:
		try:
			os.makedirs(video_path)
		except:
			pass
	logger = Logger(cfg, manager)
	
	# Init env with colored evaluation if enabled
	env = make_envs(cfg, cfg.num_envs)
	if use_colored_eval:
		eval_env = make_envs_with_colored_eval(cfg, cfg.num_eval_envs, video_path=video_path, is_eval=True, logger=logger)
	else:
		eval_env = make_envs(cfg, cfg.num_eval_envs, video_path=video_path, is_eval=True, logger=logger)
	
	print_run(cfg)
	
	# Init agent
	agent = TDMPC2(cfg)
	
	# Update wandb config, for control_mode, env_horizon, discount are set after logger init
	if logger._wandb != None:
		logger._wandb.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)
	
	# Use mixed buffer if synthetic data is provided
	if synthetic_data_dir:
		buffer = MixedBuffer(cfg, synthetic_data_dir)
	else:
		from common.buffer import Buffer
		buffer = Buffer(cfg)
		print(colored('No synthetic data provided, using regular buffer', 'yellow'))
	
	trainer_cls = OnlineTrainer # OfflineTrainer not available
	trainer = trainer_cls(
		cfg=cfg,
		env=env,
		eval_env=eval_env,
		agent=agent,
		buffer=buffer,
		logger=logger,
	)
	trainer.train()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train() 